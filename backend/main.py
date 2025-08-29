# Importing core dependencies
import os
from dotenv import load_dotenv
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import re
from urllib.parse import urlparse

# Core dependencies
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Groq for LLM
from groq import AsyncGroq

# Web scraping and search
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from tavily import TavilyClient

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration and constants"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = "llama-3.1-8b-instant"  # Fast Groq model
    MAX_SEARCH_RESULTS = 10
    MAX_SCRAPE_PAGES = 3
    CONVERSATION_WINDOW_SIZE = 10
    SESSION_TIMEOUT_HOURS = 24
    CONFIDENCE_THRESHOLD = 0.1

# =============================================================================
# DATA MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    query: str
    session_id: Optional[str] = None
    clear_history: Optional[bool] = False

class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    answer: str
    sources: List[str]
    session_id: str
    timestamp: str
    status: str
    conversation_length: int
    context_used: bool
    debug_info: Optional[Dict[str, Any]] = None

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionManager:
    """Manages user sessions and conversation history with automatic cleanup"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.last_cleanup = datetime.now()
    
    def cleanup_expired_sessions(self):
        """Remove sessions older than timeout period"""
        if datetime.now() - self.last_cleanup > timedelta(hours=1):
            current_time = datetime.now()
            expired_sessions = [
                sid for sid, data in self.sessions.items() 
                if current_time - datetime.fromisoformat(data['last_activity']) > timedelta(hours=Config.SESSION_TIMEOUT_HOURS)
            ]
            for session_id in expired_sessions:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            self.last_cleanup = current_time
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """Get existing session or create new one"""
        self.cleanup_expired_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'messages': [], 
                'created_at': datetime.now().isoformat(), 
                'last_activity': datetime.now().isoformat(), 
                'message_count': 0
            }
            logger.info(f"Created new session: {session_id}")
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, sources: List[str] = None):
        """Add message to session history with automatic trimming"""
        session = self.get_or_create_session(session_id)
        session['messages'].append({
            'role': role, 
            'content': content, 
            'timestamp': datetime.now().isoformat(), 
            'sources': sources or []
        })
        session['last_activity'] = datetime.now().isoformat()
        session['message_count'] += 1
        
        # Trim conversation if too long
        if len(session['messages']) > Config.CONVERSATION_WINDOW_SIZE * 2:
            session['messages'] = session['messages'][-Config.CONVERSATION_WINDOW_SIZE * 2:]
            logger.info(f"Trimmed conversation history for session: {session_id}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Convert session messages to format for Groq"""
        session = self.get_or_create_session(session_id)
        return [
            {"role": msg['role'], "content": msg['content']} 
            for msg in session['messages']
        ]
    
    def clear_session(self, session_id: str):
        """Clear session message history"""
        if session_id in self.sessions:
            self.sessions[session_id].update({'messages': [], 'message_count': 0})
            logger.info(f"Cleared session history: {session_id}")

# =============================================================================
# AGENT 1: QUERY ROUTER (GUARDRAIL)
# =============================================================================

class QueryRouterAgent:
    """
    First line of defense - determines if queries are relevant to Apple iPad.
    Acts as a guardrail to filter out off-topic questions with confidence scoring.
    """
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query relevance with confidence scoring"""
        try:
            logger.info(f"Classifying query: {query}")
            
            system_prompt = """You are a strict classification agent. Your task is to determine if a user's query is related to "Apple iPad" and provide a confidence score.

The query must be about iPad features, models, specifications, pricing, comparisons, troubleshooting, apps, accessories, or Apple's tablet products.

You MUST respond with ONLY a valid JSON object with exactly two keys:
1. "is_relevant": boolean (true or false)
2. "confidence": number between 0.0 and 1.0

Examples of RELEVANT queries (high confidence):
- "What's the battery life of iPad Pro?"
- "Compare iPad Air vs iPad Pro models"
- "How do I connect my iPad to Apple Pencil?"
- "What's the price of latest iPad?"
- "iPad Pro M2 chip specifications"

Examples of IRRELEVANT queries (should be false):
- "What's the weather today?"
- "Who is the CEO of Apple?" (unless specifically asking about iPad-related announcements)
- "How to cook pasta?"
- "Tell me a joke"

Your response must be ONLY the JSON object, nothing else."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = await self.groq_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response: {result}")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]*\}', result)
            if json_match:
                json_str = json_match.group()
                classification = json.loads(json_str)
            else:
                classification = json.loads(result)
            
            # Validate response structure
            if "is_relevant" in classification and "confidence" in classification:
                is_relevant = bool(classification["is_relevant"])
                confidence = float(classification["confidence"])
                
                logger.info(f"Classification result: relevant={is_relevant}, confidence={confidence}")
                return {"is_relevant": is_relevant, "confidence": confidence}
            else:
                logger.warning(f"Invalid classification structure: {classification}")
                # Fallback keyword check
                ipad_keywords = ['ipad', 'tablet', 'apple tablet', 'pad', 'generation']
                if any(keyword in query.lower() for keyword in ipad_keywords):
                    logger.info(f"Fallback classification triggered for: {query}")
                    return {"is_relevant": True, "confidence": 0.7}
                return {"is_relevant": False, "confidence": 0.0}
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in query classification: {str(e)}. Raw result: {result}")
            # Fallback keyword check
            ipad_keywords = ['ipad', 'tablet', 'apple tablet', 'pad', 'generation']
            if any(keyword in query.lower() for keyword in ipad_keywords):
                logger.info(f"Fallback classification triggered for: {query}")
                return {"is_relevant": True, "confidence": 0.7}
            return {"is_relevant": False, "confidence": 0.0}
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}. Defaulting to NOT RELEVANT.")
            # Fallback keyword check
            ipad_keywords = ['ipad', 'tablet', 'apple tablet', 'pad', 'generation']
            if any(keyword in query.lower() for keyword in ipad_keywords):
                logger.info(f"Fallback classification triggered for: {query}")
                return {"is_relevant": True, "confidence": 0.7}
            return {"is_relevant": False, "confidence": 0.0}

# =============================================================================
# AGENT 2: RESEARCH PLANNER
# =============================================================================

class ContextualResearchPlannerAgent:
    """
    Strategic planner that creates targeted search queries prioritizing official Apple sources.
    Uses conversation context to create more relevant research plans.
    """
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
    
    async def plan_research(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create context-aware research plan with official source prioritization"""
        if "price" in query.lower() or "cost" in query.lower():
            return {
                "search_queries": [
                    "iPad price 2024 Apple official site:apple.com",
                    "Apple iPad 10th generation cost specifications",
                    "iPad models pricing comparison"
                ],
                "key_topics": ["price", "cost", "specifications"],
                "priority": "high"
            }
        try:
            logger.info(f"Planning research for query: {query}")
            
            # Format chat history for context
            history_text = ""
            if chat_history:
                recent_messages = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    history_text += f"{msg['role']}: {msg['content']}\n"
            
            system_prompt = """You are a research planning agent. Create search queries to find information about Apple iPad.

You MUST respond with ONLY this JSON format:
{
  "search_queries": ["query1", "query2", "query3"],
  "key_topics": ["topic1", "topic2"],
  "priority": "high/medium/low"
}

Make 2-3 search queries that include:
1. Official Apple sources (site:apple.com)
2. iPad-specific terms
3. Current year (2024) when relevant

Example:
{"search_queries": ["iPad Pro M4 specifications site:apple.com", "Apple iPad Pro 2024 features"], "key_topics": ["specifications", "features"], "priority": "high"}"""

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add chat history context if available
            if history_text:
                messages.append({"role": "user", "content": f"Previous conversation context:\n{history_text}\nCurrent query: {query}"})
            else:
                messages.append({"role": "user", "content": query})
            
            response = await self.groq_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response: {result}") 
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                plan = json.loads(json_str)
            else:
                plan = json.loads(result)
            
            # Validate plan structure
            required_keys = ["search_queries", "key_topics", "priority"]
            if all(key in plan for key in required_keys):
                logger.info(f"Research plan: {plan}")
                return plan
            else:
                logger.warning(f"Invalid plan structure: {plan}")
                return self._get_fallback_plan(query)
                
        except Exception as e:
            logger.error(f"Error in contextual research planning: {str(e)}")
            return self._get_fallback_plan(query)
    
    def _get_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Fallback research plan when main planning fails"""
        return {
            "search_queries": [
                f"Apple iPad {query} site:apple.com", 
                f"iPad specifications {query} 2024"
            ],
            "key_topics": ["features", "specifications"],
            "priority": "medium"
        }

# =============================================================================
# WEB SCRAPING TOOL - FIXED VERSION
# =============================================================================

class PlaywrightScrapingTool:
    """
    Fixed Playwright-based web scraper with better text extraction and cleaning
    """
    
    def __init__(self):
        self.name = "playwright_scraper"
        self.description = "Scrape web pages using Playwright for dynamic content"
        self.timeout = 30000  # 30 seconds
    
    async def scrape_url(self, url: str) -> str:
        """Main scraping method with improved text extraction"""
        try:
            logger.info(f"Scraping URL: {url}")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = await context.new_page()
                
                try:
                    # Navigate to page
                    await page.goto(url, wait_until='load', timeout=self.timeout)
                    
                    # Wait for content to load
                    await page.wait_for_timeout(3000)
                    
                    # IMPROVED CONTENT EXTRACTION with better text cleaning
                    content = await page.evaluate("""
                        () => {
                            // Remove script and style elements
                            const scripts = document.querySelectorAll('script, style, nav, footer, header, .cookie, .popup');
                            scripts.forEach(el => el.remove());
                            
                            // Try to get main content area first
                            const mainSelectors = ['main', 'article', '.content', '#content', '.main-content', '.page-content'];
                            for (let selector of mainSelectors) {
                                const element = document.querySelector(selector);
                                if (element) {
                                    return element.innerText;
                                }
                            }
                            
                            // If no main content found, get clean body text
                            return document.body.innerText;
                        }
                    """)
                    
                    if content:
                        # IMPROVED TEXT CLEANING
                        cleaned_content = self._clean_extracted_text(content)
                        
                        if len(cleaned_content.strip()) > 50:
                            logger.info(f"Successfully scraped {len(cleaned_content)} characters from {url}")
                            return cleaned_content
                    
                    # Fallback: try extracting structured text
                    fallback_content = await page.evaluate("""
                        () => {
                            // Get text from paragraphs and headings, with proper spacing
                            const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, div[class*="price"], div[class*="cost"]');
                            const textArray = [];
                            
                            elements.forEach(el => {
                                const text = el.innerText.trim();
                                if (text && text.length > 10) {
                                    textArray.push(text);
                                }
                            });
                            
                            return textArray.join('\\n\\n');
                        }
                    """)
                    
                    if fallback_content:
                        cleaned_fallback = self._clean_extracted_text(fallback_content)
                        if len(cleaned_fallback.strip()) > 50:
                            logger.info(f"Fallback extraction successful: {len(cleaned_fallback)} characters")
                            return cleaned_fallback
                    
                    logger.warning(f"No substantial content extracted from {url}")
                    return f"Content from {url}: Page loaded but content extraction failed."
                    
                except PlaywrightTimeoutError:
                    logger.error(f"Timeout scraping {url}")
                    return f"Content from {url}: Page timed out during loading."
                
                finally:
                    await browser.close()
                    
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return f"Content from {url}: Error occurred while scraping - {str(e)}"
    
    def _clean_extracted_text(self, text: str) -> str:
        """Enhanced text cleaning to fix garbled content"""
        if not text:
            return ""
        
        try:
            # Step 1: Fix common encoding issues
            text = text.replace('â€™', "'")
            text = text.replace('â€œ', '"')
            text = text.replace('â€\x9d', '"')
            text = text.replace('â€"', '—')
            text = text.replace('â€¦', '...')
            
            # Step 2: Fix currency symbols and special characters
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
            text = text.replace('₹', 'Rs. ')
            text = text.replace('$', 'USD ')
            
            # Step 3: Fix spacing issues - this is the KEY fix for your problem
            # Replace multiple spaces, weird spacing patterns
            text = re.sub(r'\s+', ' ', text)
            
            # Fix patterns like "349fortheWi−Fimodeland" 
            text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
            text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Add space between letters and numbers
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)   # Add space between camelCase
            
            # Fix specific patterns for pricing
            text = re.sub(r'(\$\d+)([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'(Rs\.\s?\d+)([a-zA-Z])', r'\1 \2', text)
            
            # Step 4: Remove excessive punctuation and symbols
            text = re.sub(r'[−–—]', '-', text)  # Normalize dashes
            text = re.sub(r'[^\w\s\.\,\!\?\$\-\(\)\[\]:]', ' ', text)  # Keep only basic punctuation
            
            # Step 5: Remove navigation/menu items and common webpage clutter
            lines = text.split('\n')
            cleaned_lines = []
            
            skip_patterns = [
                r'^(menu|navigation|nav|header|footer|cookie|privacy|terms)',
                r'^(home|about|contact|login|register|sign)',
                r'^(click here|learn more|read more|view all)$',
                r'^\s*[<>{}]\s*$'
            ]
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:  # Skip very short lines
                    skip_line = False
                    for pattern in skip_patterns:
                        if re.match(pattern, line.lower()):
                            skip_line = True
                            break
                    
                    if not skip_line:
                        cleaned_lines.append(line)
            
            # Step 6: Rejoin and final cleanup
            cleaned_text = ' '.join(cleaned_lines)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Final space normalization
            
            # Limit length
            if len(cleaned_text) > 5000:
                cleaned_text = cleaned_text[:5000] + "..."
            
            return cleaned_text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text[:5000] if text else ""
    
    def _run(self, url: str) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self.scrape_url(url))
    
    async def _arun(self, url: str) -> str:
        """Async version"""
        return await self.scrape_url(url)

# =============================================================================
# AGENT 3: WEB SEARCH & SCRAPER
# =============================================================================

class WebSearchScrapeAgent:
    """
    Executes research plan by searching web and scraping content.
    Prioritizes official Apple domains and handles multiple concurrent searches.
    """
    
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY) if Config.TAVILY_API_KEY else None
        self.scraper = PlaywrightScrapingTool()
    
    async def search_and_scrape(self, research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute all search queries and scrape results concurrently"""
        if not self.tavily_client:
            logger.error("Tavily client not initialized - missing API key")
            return []
        
        search_queries = research_plan.get("search_queries", [])
        if not search_queries:
            logger.warning("No search queries provided in research plan")
            return []
        
        # Execute all searches concurrently
        tasks = [self._process_query(query) for query in search_queries]
        query_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        results = []
        for result in query_results:
            if isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
            elif isinstance(result, list):
                results.extend(result)
        
        return results

    async def _process_query(self, query: str) -> List[Dict[str, Any]]:
        """Process single search query with official source prioritization"""
        try:
            logger.info(f"Searching for: {query}")
            search_results = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=Config.MAX_SEARCH_RESULTS
            )
            logger.info(f"Search results count: {len(search_results.get('results', []))}")
        
            all_results = search_results.get("results", [])
            if not all_results:
                logger.warning(f"No search results found for query: {query}")
                return []

            # Prioritize official Apple domains
            preferred_domains = ["apple.com", "support.apple.com"]
            official_results = []
            other_results = []

            for result in all_results:
                try:
                    domain = urlparse(result['url']).netloc
                    if any(preferred in domain for preferred in preferred_domains):
                        official_results.append(result)
                    else:
                        other_results.append(result)
                except Exception:
                    other_results.append(result)

            # Prioritize official sources first
            sorted_results = official_results + other_results
            logger.info(f"Re-ranked results. Found {len(official_results)} official sources.")
            
            # Scrape top results
            results_to_scrape = sorted_results[:Config.MAX_SCRAPE_PAGES]
            
            query_results = []
            for result in results_to_scrape:
                try:
                    # First try search snippet
                    snippet_content = result.get('snippet', '') + " " + result.get('content', '')
                    
                    # If snippet is too short, scrape the page
                    if len(snippet_content.strip()) < 100:
                        logger.info(f"Snippet too short, scraping: {result['url']}")
                        scraped_content = await self.scraper.scrape_url(result['url'])
                        if scraped_content and len(scraped_content.strip()) > 50:
                            result['content'] = scraped_content
                        else:
                            result['content'] = snippet_content
                    else:
                        result['content'] = snippet_content
                    
                    if result['content'].strip():
                        query_results.append(result)
                        logger.info(f"Added result with {len(result['content'])} characters")
                    
                except Exception as e:
                    logger.error(f"Error processing result {result.get('url', '')}: {e}")
                    # Use snippet as fallback
                    result['content'] = result.get('snippet', 'Content unavailable')
                    query_results.append(result)

            logger.info(f"Processed {len(query_results)} results for query: {query}")
            return query_results
            
        except Exception as e:
            logger.error(f"Error processing search query '{query}': {str(e)}")
            return []

# =============================================================================
# AGENT 4: SYNTHESIZER WITH STREAMING
# =============================================================================

class ContextualSynthesizerCiterAgent:
    """
    Final agent that synthesizes web content into coherent answers.
    Supports real-time streaming and maintains conversation context.
    """
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
    
    async def synthesize_answer_streaming(self, query: str, web_data: List[Dict[str, Any]], chat_history: List[Dict[str, str]]):
        """Stream response in real-time with sentence-by-sentence delivery"""
        try:
            # Prepare web content for synthesis
            web_content = "\n".join([
                f"\n--- Source: {item.get('title', '')} ({item.get('url', '')}) ---\n{item.get('content', item.get('snippet', ''))}" 
                for item in web_data if item.get("url") and item.get("content", item.get("snippet"))
            ])
            
            if not web_content.strip():
                yield json.dumps({
                    "type": "sentence",
                    "content": "I apologize, but I couldn't find specific information about that topic. Could you please rephrase your question or ask about a different aspect of Apple iPad?"
                }) + "\n"
                return
            
            sources = [item['url'] for item in web_data if item.get('url')]
            
            # Format chat history
            history_text = ""
            if chat_history:
                recent_messages = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    history_text += f"{msg['role']}: {msg['content']}\n"
            
            system_prompt = """You are a Synthesizer Agent specializing in Apple iPad information.
Your role is to analyze raw web data and conversation history to create coherent, contextual answers.

IMPORTANT: Provide ONLY the answer content. Do NOT include source URLs or references in your response. The sources will be handled separately by the system.

Your response should be:
- A comprehensive, well-structured answer based on the provided web content
- Contextual and relevant to the user's query about Apple iPad
- Clear and informative
- WITHOUT any source citations, URLs, or reference mentions

Focus solely on synthesizing the information into a helpful answer about Apple iPad."""

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add chat history if available
            if history_text:
                messages.append({"role": "user", "content": f"Previous conversation:\n{history_text}"})
            
            messages.append({"role": "user", "content": f"Current Query: {query}\n\nNew Web Content:\n{web_content}"})
            
            # Stream response from Groq
            stream = await self.groq_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                stream=True,
                temperature=0.1,
                max_tokens=1000
            )
            
            accumulated_content = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    
                    # Extract complete sentences for streaming
                    sentences = self._extract_complete_sentences(accumulated_content)
                    if sentences["complete_sentences"]:
                        for sentence in sentences["complete_sentences"]:
                            if sentence.strip():
                                yield json.dumps({
                                    "type": "sentence",
                                    "content": sentence.strip()
                                }) + "\n"
                        
                        accumulated_content = sentences["remaining"]
            
            # Send remaining content
            if accumulated_content.strip():
                yield json.dumps({
                    "type": "sentence",
                    "content": accumulated_content.strip()
                }) + "\n"
            
            # Send sources and completion signal
            yield json.dumps({
                "type": "sources",
                "content": list(set(sources))
            }) + "\n"
            
            yield json.dumps({
                "type": "complete",
                "timestamp": datetime.now().isoformat(),
                "context_used": len(chat_history) > 0
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {str(e)}")
            context_note = " I'm having trouble accessing my previous responses, but I'll do my best." if chat_history else ""
            yield json.dumps({
                "type": "error",
                "content": f"I apologize, but I encountered an error processing your request.{context_note} Please try rephrasing your question about Apple iPad."
            }) + "\n"
    
    def _extract_complete_sentences(self, text: str) -> Dict[str, Any]:
        """Extract complete sentences from accumulated streaming text"""
        sentence_pattern = r'([.!?]+)'
        parts = re.split(sentence_pattern, text)
        
        complete_sentences = []
        remaining = ""
        
        i = 0
        while i < len(parts) - 1:
            sentence_content = parts[i]
            if i + 1 < len(parts):
                punctuation = parts[i + 1]
                complete_sentence = sentence_content + punctuation
                complete_sentences.append(complete_sentence)
                i += 2
            else:
                i += 1
        
        # Handle remaining incomplete sentence
        if parts:
            remaining = parts[-1] if not re.match(sentence_pattern, parts[-1]) else ""
        
        return {
            "complete_sentences": complete_sentences,
            "remaining": remaining
        }

    async def synthesize_answer(self, query: str, web_data: List[Dict[str, Any]], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Non-streaming version for backward compatibility"""
        try:
            web_content = "\n".join([
                f"\n--- Source: {item.get('title', '')} ({item.get('url', '')}) ---\n{item.get('content', item.get('snippet', ''))}" 
                for item in web_data if item.get("url") and item.get("content", item.get("snippet"))
            ])
            
            if not web_content.strip():
                return {
                    "answer": "I apologize, but I couldn't find specific information about that topic. Could you please rephrase your question or ask about a different aspect of Apple iPad?",
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "context_used": len(chat_history) > 0
                }
            
            sources = [item['url'] for item in web_data if item.get('url')]
            
            # Format chat history
            history_text = ""
            if chat_history:
                recent_messages = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    history_text += f"{msg['role']}: {msg['content']}\n"
            
            system_prompt = """You are a Synthesizer Agent specializing in Apple iPad information.
Your role is to analyze raw web data and conversation history to create coherent, contextual answers.

IMPORTANT: Provide ONLY the answer content. Do NOT include source URLs or references in your response. The sources will be handled separately by the system.

Your response should be:
- A comprehensive, well-structured answer based on the provided web content
- Contextual and relevant to the user's query about Apple iPad
- Clear and informative
- WITHOUT any source citations, URLs, or reference mentions

Focus solely on synthesizing the information into a helpful answer about Apple iPad."""

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add chat history if available
            if history_text:
                messages.append({"role": "user", "content": f"Previous conversation:\n{history_text}"})
            
            messages.append({"role": "user", "content": f"Current Query: {query}\n\nNew Web Content:\n{web_content}"})
            
            response = await self.groq_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            return {
                "answer": result,
                "sources": list(set(sources)),
                "timestamp": datetime.now().isoformat(),
                "context_used": len(chat_history) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in contextual synthesis: {str(e)}")
            context_note = " I'm having trouble accessing my previous responses, but I'll do my best." if chat_history else ""
            return {
                "answer": f"I apologize, but I encountered an error processing your request.{context_note} Please try rephrasing your question about Apple iPad.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "context_used": False
            }

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class ContextualMultiAgentOrchestrator:
    """
    Main orchestrator that coordinates all agents in a pipeline:
    1. QueryRouter: Filter irrelevant queries
    2. ResearchPlanner: Create search strategy
    3. WebSearchScraper: Gather information
    4. Synthesizer: Generate final answer
    
    Supports both streaming and traditional response modes.
    """
    
    def __init__(self):
        self.groq_client = AsyncGroq(api_key=Config.GROQ_API_KEY)
        
        # Initialize all agents
        self.router = QueryRouterAgent(self.groq_client)
        self.planner = ContextualResearchPlannerAgent(self.groq_client)
        self.searcher = WebSearchScrapeAgent()
        self.synthesizer = ContextualSynthesizerCiterAgent(self.groq_client)
        self.session_manager = SessionManager()
    
    def _create_denial_response(self, session_id: str, debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized denial response for off-topic queries"""
        session = self.session_manager.get_or_create_session(session_id)
        return {
            "answer": "I am a specialized assistant for Apple iPad devices. I can only answer questions about iPads, their features, specifications, pricing, models, troubleshooting, apps, and accessories. Please ask me something about Apple iPad.",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "context_used": False,
            "conversation_length": len(session["messages"]) // 2,
            "debug_info": debug_info
        }
    
    async def process_query_streaming(self, query: str, session_id: str, clear_history: bool = False):
        """Main streaming pipeline processing queries through all agents"""
        try:
            if clear_history:
                self.session_manager.clear_session(session_id)
            
            chat_history = self.session_manager.get_conversation_history(session_id)
            logger.info(f"Processing query for session {session_id}: {query}")
            
            debug_info = {}

            # STAGE 1: Query Classification (Guardrail)
            logger.info("Stage 1: Classifying query relevance")
            classification = await self.router.classify_query(query)
            debug_info["classification"] = classification
            
            is_relevant = classification.get("is_relevant", False)
            confidence = classification.get("confidence", 0.0)
            
            if not is_relevant or confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"DENIED: Query not relevant or low confidence (Score: {confidence})")
                self.session_manager.add_message(session_id, "user", query)
                denial_response = self._create_denial_response(session_id, debug_info)
                self.session_manager.add_message(session_id, "assistant", denial_response["answer"], [])
                
                yield json.dumps({
                    "type": "sentence",
                    "content": denial_response["answer"]
                }) + "\n"
                yield json.dumps({
                    "type": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "context_used": False,
                    "debug_info": debug_info
                }) + "\n"
                return
            
            # Send progress updates
            yield json.dumps({"type": "status", "content": "Planning research..."}) + "\n"
            
            # STAGE 2: Research Planning
            logger.info("Stage 2: Planning research")
            research_plan = await self.planner.plan_research(query, chat_history)
            debug_info["research_plan"] = research_plan
            
            yield json.dumps({"type": "status", "content": "Searching and gathering information..."}) + "\n"
            
            # STAGE 3: Web Search and Scraping
            logger.info("Stage 3: Executing web search and scraping")
            web_data = await self.searcher.search_and_scrape(research_plan)
            debug_info["web_results"] = len(web_data)
            
            yield json.dumps({"type": "status", "content": "Analyzing information and generating response..."}) + "\n"
            
            # STAGE 4: Synthesis with Streaming
            logger.info("Stage 4: Synthesizing answer with streaming")
            self.session_manager.add_message(session_id, "user", query)
            
            accumulated_response = ""
            sources = []
            
            async for chunk in self.synthesizer.synthesize_answer_streaming(query, web_data, chat_history):
                chunk_data = json.loads(chunk.strip())
                chunk_type = chunk_data.get("type")
                
                if chunk_type == "sentence":
                    accumulated_response += chunk_data.get("content", "") + " "
                    yield chunk
                elif chunk_type == "sources":
                    sources = chunk_data.get("content", [])
                    yield chunk
                elif chunk_type == "complete":
                    chunk_data["debug_info"] = debug_info
                    yield json.dumps(chunk_data) + "\n"
                    break
                elif chunk_type == "error":
                    yield chunk
                    break
            
            # Store the complete response
            if accumulated_response.strip():
                self.session_manager.add_message(session_id, "assistant", accumulated_response.strip(), sources)
            
            logger.info(f"Successfully processed query with {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"Critical error in orchestration: {str(e)}", exc_info=True)
            self.session_manager.add_message(session_id, "user", query)
            error_response = f"I apologize, but I encountered a technical error. Please try asking your question about Apple iPad again."
            self.session_manager.add_message(session_id, "assistant", error_response, [])
            
            yield json.dumps({
                "type": "error",
                "content": error_response
            }) + "\n"

    async def process_query(self, query: str, session_id: str, clear_history: bool = False):
        """Traditional non-streaming processing for backward compatibility"""
        try:
            if clear_history:
                self.session_manager.clear_session(session_id)
            
            chat_history = self.session_manager.get_conversation_history(session_id)
            logger.info(f"Processing query for session {session_id}: {query}")
            
            debug_info = {}

            # STAGE 1: Query Classification (Guardrail)
            classification = await self.router.classify_query(query)
            debug_info["classification"] = classification
            
            is_relevant = classification.get("is_relevant", False)
            confidence = classification.get("confidence", 0.0)
            
            if not is_relevant or confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"DENIED: Query not relevant or low confidence (Score: {confidence})")
                self.session_manager.add_message(session_id, "user", query)
                denial_response = self._create_denial_response(session_id, debug_info)
                self.session_manager.add_message(session_id, "assistant", denial_response["answer"], [])
                return denial_response
            
            # STAGE 2: Research Planning
            research_plan = await self.planner.plan_research(query, chat_history)
            debug_info["research_plan"] = research_plan
            
            # STAGE 3: Web Search and Scraping
            web_data = await self.searcher.search_and_scrape(research_plan)
            debug_info["web_results"] = len(web_data)
            
            # STAGE 4: Synthesis
            final_result = await self.synthesizer.synthesize_answer(query, web_data, chat_history)
            
            self.session_manager.add_message(session_id, "user", query)
            self.session_manager.add_message(session_id, "assistant", final_result["answer"], final_result["sources"])
            
            session = self.session_manager.get_or_create_session(session_id)
            final_result["conversation_length"] = len(session["messages"]) // 2
            final_result["debug_info"] = debug_info
            
            logger.info(f"Successfully processed query with {len(final_result['sources'])} sources")
            return final_result
            
        except Exception as e:
            logger.error(f"Critical error in orchestration: {str(e)}", exc_info=True)
            self.session_manager.add_message(session_id, "user", query)
            error_response = {
                "answer": f"I apologize, but I encountered a technical error. Please try asking your question about Apple iPad again.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "context_used": False,
                "conversation_length": 0,
                "debug_info": {"error": str(e)}
            }
            self.session_manager.add_message(session_id, "assistant", error_response["answer"], [])
            return error_response

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Contextual Multi-Agent Apple iPad Chatbot", version="1.0.0-streaming")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the main orchestrator
orchestrator = ContextualMultiAgentOrchestrator()

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Traditional non-streaming endpoint for backward compatibility"""
    try:
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        result = await orchestrator.process_query(
            query=request.query,
            session_id=session_id,
            clear_history=request.clear_history or False
        )
        return ChatResponse(status="success", session_id=session_id, **result)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-stream")
async def ask_question_stream(request: ChatRequest):
    """Real-time streaming endpoint for enhanced user experience"""
    try:
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        async def generate_stream():
            """Generator function for streaming responses"""
            try:
                # Send session ID first
                yield json.dumps({
                    "type": "session_id",
                    "content": session_id
                }) + "\n"
                
                # Stream the orchestrated response
                async for chunk in orchestrator.process_query_streaming(
                    query=request.query,
                    session_id=session_id,
                    clear_history=request.clear_history or False
                ):
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}", exc_info=True)
                yield json.dumps({
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }) + "\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up stream: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "Apple iPad Chatbot is online and ready to help!"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/debug-config")
async def debug_config():
    """Debug endpoint to check API configuration"""
    return {
        "groq_key_present": bool(Config.GROQ_API_KEY and len(Config.GROQ_API_KEY) > 10),
        "tavily_key_present": bool(Config.TAVILY_API_KEY and len(Config.TAVILY_API_KEY) > 10),
        "model_name": Config.MODEL_NAME
    }

@app.post("/test-scraper")
async def test_scraper(request: dict):
    """Debug endpoint to test scraping"""
    url = request.get("url", "https://www.apple.com/ipad/")
    scraper = PlaywrightScrapingTool()
    content = await scraper._arun(url)
    return {
        "url": url, 
        "content_length": len(content),
        "content_preview": content[:500]
    }

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Validate required API keys
    if not Config.GROQ_API_KEY or not Config.TAVILY_API_KEY:
        raise ValueError("API keys for GROQ and TAVILY must be set in .env file")
    
    # Start the FastAPI server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
