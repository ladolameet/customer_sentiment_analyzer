import os

# 🔥 FULLY DISABLE OPENAI + EMBEDDINGS
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "http://localhost"
os.environ["CREWAI_DISABLE_OPENAI"] = "true"
os.environ["CREWAI_EMBEDDINGS_PROVIDER"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["CREWAI_LLM_PROVIDER"] = "groq"


import streamlit as st
import pandas as pd
import requests, json, os, io, textwrap, re, time, hashlib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import urllib3
from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Optional
from crewai.tools import tool

# =====================================================
# HIDE STREAMLIT DEFAULT UI (HEADER / SHARE / MENU)
# =====================================================
st.markdown("""
<style>
/* Encourage full-width layout */
.block-container {
    padding-top: 1rem;
}

/* Hide Streamlit header (Share, GitHub, Menu) */
header {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==================== LANGGRAPH AGENT CONFIGURATION ====================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
# ==================== GROQ LLM ====================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

NEGATIVE_KEYWORDS = ['bad', 'poor', 'worst', 'terrible', 'horrible', 'awful', 
                    'disappointed', 'waste', 'avoid', 'cheated', 'broken', 
                    'defective', 'useless', 'junk', 'scam', 'fake']


# ==================== AGENT CLASSES ====================
class ScrapingAgent:
    """Agent responsible for web scraping reviews"""
    def __init__(self):
        self.api_key = os.getenv("SCRAPINGDOG_API_KEY", "")
        self.max_pages = 2
        self.max_reviews = 50
    def scrape_reviews(self, url: str) -> Dict:
        """Agent task: Scrape reviews from multiple pages"""
        if not self.api_key:
            return json.dumps({"raw_reviews": [], "product_title": "Product", "error": "❌ ScrapingDog API Key missing"})
        
        all_reviews = []
        product_title = ""
        
        for page in range(1, self.max_pages + 1):
            page_url = self._generate_page_url(url, page)
            page_reviews, title = self._scrape_single_page(page_url)
            
            if page_reviews:
                all_reviews.extend(page_reviews)
                if title and not product_title:
                    product_title = title
                
                if len(page_reviews) < 5:
                    break
                
                if page < self.max_pages:
                    time.sleep(1)
            else:
                if page == 1:
                    break
                else:
                    break
        
        # Clean and deduplicate reviews
        cleaned_reviews = []
        for review in all_reviews:
            cleaned_text = self._clean_review_text(review['text'])
            if cleaned_text and len(cleaned_text) >= 20:
                review['text'] = cleaned_text
                cleaned_reviews.append(review)
        
        unique_reviews = self._deduplicate_reviews(cleaned_reviews)
        return json.dumps({
            "raw_reviews": unique_reviews[:self.max_reviews],
            "product_title": product_title,
            "error": None
        })

    
    def _scrape_single_page(self, url):
        """Scrape reviews from a single page"""
        try:
            response = requests.get(
                "https://api.scrapingdog.com/scrape",
                params={
                    'api_key': self.api_key, 
                    'url': url, 
                    'dynamic': 'false', 
                    'render_js': 'false'
                },
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                timeout=60, 
                verify=False
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                product_title = self._extract_product_title(soup)
                
                all_reviews = []
                all_reviews.extend(self._extract_amazon_reviews(soup))
                all_reviews.extend(self._extract_generic_reviews(soup))
                all_reviews.extend(self._extract_json_ld_reviews(soup))
                all_reviews.extend(self._extract_text_pattern_reviews(soup))
                all_reviews.extend(self._extract_review_sections(soup))
                
                return all_reviews, product_title
                
        except Exception as e:
            return [], f"Error: {str(e)}"
        
        return [], ""
    
    def _clean_review_text(self, text):
        """Clean review text by removing metadata, names, dates, etc."""
        if not text:
            return text
        
        # Remove common patterns
        patterns_to_remove = [
            # Amazon patterns
            r'Reviewed in .*? on \d+ \w+ \d+',
            r'Reviewed in .*? on \w+ \d+, \d+',
            r'Size:.*?(?=\n|$)',
            r'Colou?r:.*?(?=\n|$)',
            r'Pattern:.*?(?=\n|$)',
            r'Style:.*?(?=\n|$)',
            r'Verified Purchase.*?(?=\n|$)',
            r'\d+\.\d+ out of \d+ stars?',
            r'\d+/\d+ stars?',
            r'★+',
            r'⭐+',
            r'\d+ people found this helpful',
            r'Helpful',
            r'Report',
            r'Read more',
            r'Translate review to English',
            r'Images in this review',
            r'The media could not be loaded\..*?(?=\n|$)',
            
            # General patterns
            r'^[A-Z][a-z]+ [A-Z][a-z]+',  # Names at start
            r'by [A-Z][a-z]+ [A-Z][a-z]+',  # "by John Doe"
            r'on \w+ \d+,? \d+',  # Dates
            r'\d+ \w+ ago',  # "2 days ago"
            r'Posted on \w+ \d+, \d+',  # "Posted on January 1, 2024"
            r'Updated on \w+ \d+, \d+',  # "Updated on January 1, 2024"
            r'This review is from:.*?(?=\n|$)',
            r'Purchased on:.*?(?=\n|$)',
            r'Was this review helpful\?.*?(?=\n|$)',
            
            # Flipkart patterns
            r'Certified Buyer,.*?(?=\n|$)',
            r'\d+ days ago',
            r'READ MORE',
            r'COMMENT',
            r'LIKE',
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove extra whitespace and normalize
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Remove empty quotes and common prefixes
        cleaned_text = re.sub(r'^"|"$', '', cleaned_text)
        cleaned_text = re.sub(r'^Review: ', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^Comment: ', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^Feedback: ', '', cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def _extract_amazon_reviews(self, soup):
        """Extract Amazon-style reviews and clean them"""
        reviews = []
        selectors = [
            'div[data-hook="review"]',
            'div.a-section.review',
            'div.review',
            'div[data-component-type="review"]',
            'div.review-text-content'
        ]
        
        for selector in selectors:
            for container in soup.select(selector):
                try:
                    # Find review text
                    text_elem = container.select_one('span[data-hook="review-body"], .review-text-content span, .review-text')
                    if not text_elem:
                        # Try to get any text content
                        text_elem = container
                    
                    raw_text = text_elem.get_text(strip=True, separator=' ') if text_elem else ""
                    
                    if not raw_text or len(raw_text) < 20:
                        continue
                    
                    # Clean the text
                    cleaned_text = self._clean_review_text(raw_text)
                    if not cleaned_text or len(cleaned_text) < 20:
                        continue
                    
                    # Extract rating
                    rating_elem = container.select_one('i[data-hook="review-star-rating"] span, span.a-icon-alt, i.a-icon-star')
                    rating = 0
                    if rating_elem:
                        rating_text = rating_elem.get_text(strip=True)
                        rating = self._parse_rating_text(rating_text)
                    
                    if rating == 0:
                        # Try to find rating in parent container
                        parent = container.parent
                        while parent and rating == 0:
                            rating_text = parent.get_text()
                            rating = self._parse_rating_text(rating_text)
                            parent = parent.parent
                        
                        if rating == 0:
                            rating = self._guess_rating_from_text(cleaned_text)
                    
                    reviews.append({
                        'text': cleaned_text[:400],
                        'rating': rating,
                        'platform': 'Amazon',
                        'source': 'amazon',
                        'raw_text': raw_text[:500]  # Keep raw for debugging
                    })
                except Exception as e:
                    continue
        
        return reviews
    
    def _extract_generic_reviews(self, soup):
        """Extract generic reviews from various selectors"""
        reviews = []
        selectors = [
            'div.review-content', 'div.review-body', 'div.review-text',
            'div.comment-content', 'div.comment-body', 'div.comment-text',
            'div.feedback-content', 'div.feedback-body', 'div.feedback-text',
            'div.testimonial-content', 'div.testimonial-body', 'div.testimonial-text',
            'div.customer-review-content', 'div.user-review-content',
            'div.product-review-content', 'section.review p',
            'article.review p', 'li.review p',
            '.review-description', '.review-comment', '.user-comment'
        ]
        
        for selector in selectors:
            for container in soup.select(selector):
                try:
                    raw_text = container.get_text(strip=True, separator=' ')
                    if 30 < len(raw_text) < 600:
                        cleaned_text = self._clean_review_text(raw_text)
                        if not cleaned_text or len(cleaned_text) < 20:
                            continue
                        
                        rating = self._parse_rating_from_element(container)
                        if rating == 0:
                            # Look for rating in nearby elements
                            parent = container.parent
                            siblings = container.find_previous_siblings() + container.find_next_siblings()
                            all_elements = [parent] + siblings if parent else siblings
                            
                            for elem in all_elements[:5]:  # Check first 5 nearby elements
                                if elem:
                                    rating = self._parse_rating_text(elem.get_text())
                                    if rating > 0:
                                        break
                        
                        if rating == 0:
                            rating = self._guess_rating_from_text(cleaned_text)
                        
                        reviews.append({
                            'text': cleaned_text[:400],
                            'rating': rating,
                            'platform': 'Generic',
                            'source': 'generic',
                            'raw_text': raw_text[:500]
                        })
                except:
                    continue
        
        return reviews
    
    def _extract_json_ld_reviews(self, soup):
        """Extract reviews from JSON-LD structured data"""
        reviews = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                review_data = None
                
                if isinstance(data, dict):
                    review_data = data.get('review')
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'review' in item:
                            review_data = item.get('review')
                            break
                
                if review_data:
                    if isinstance(review_data, list):
                        for review in review_data:
                            text = review.get('reviewBody') or review.get('description') or review.get('text')
                            
                            if text and len(text) > 20:
                                cleaned_text = self._clean_review_text(text)
                                if not cleaned_text or len(cleaned_text) < 20:
                                    continue
                                
                                rating = review.get('reviewRating', {}).get('ratingValue')
                                if rating:
                                    try:
                                        rating = float(rating)
                                    except:
                                        rating = self._guess_rating_from_text(cleaned_text)
                                else:
                                    rating = self._guess_rating_from_text(cleaned_text)
                                
                                reviews.append({
                                    'text': cleaned_text[:400],
                                    'rating': rating,
                                    'platform': 'JSON-LD',
                                    'source': 'json_ld',
                                    'raw_text': text[:500]
                                })
            except:
                continue
        
        return reviews
    
    def _extract_text_pattern_reviews(self, soup):
        """Extract reviews using text patterns and clean them"""
        reviews = []
        text = soup.get_text(separator=' ', strip=True)
        
        # More specific patterns for reviews
        patterns = [
            r'(?:"|'')([^"''\.]{50,400}?[\.\!\?])(?:"|'')',  # Text in quotes
            r'(?:I (?:recently )?(?:purchased|bought|received|got|tried|used).{50,300}?[\.\!\?])',
            r'(?:This (?:product|item|thing).{50,300}?[\.\!\?])',
            r'(?:The (?:quality|product|item).{50,300}?[\.\!\?])',
            r'(?:Overall,.{30,200}?[\.\!\?])',
            r'(?:In (?:summary|conclusion),.{30,200}?[\.\!\?])',
            r'(?:Pros:.{30,200}?[\.\!\?])',
            r'(?:Cons:.{30,200}?[\.\!\?])',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                chunk = match.group(0).strip()
                if 50 < len(chunk) < 800:
                    cleaned_text = self._clean_review_text(chunk)
                    if not cleaned_text or len(cleaned_text) < 20:
                        continue
                    
                    rating = self._parse_rating_text(chunk)
                    if rating == 0:
                        # Look for rating in surrounding text
                        start = max(0, match.start() - 100)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end]
                        rating = self._parse_rating_text(context)
                    
                    if rating == 0:
                        rating = self._guess_rating_from_text(cleaned_text)
                    
                    reviews.append({
                        'text': cleaned_text[:400],
                        'rating': rating,
                        'platform': 'Text Pattern',
                        'source': 'text_pattern',
                        'raw_text': chunk[:500]
                    })
        
        return reviews
    
    def _extract_review_sections(self, soup):
        """Extract reviews from review sections"""
        reviews = []
        
        section_selectors = [
            'section[data-hook="reviews"]',
            'div.reviews-section',
            'div.customer-reviews',
            'div.product-reviews',
            'div[class*="review"][class*="section"]',
            'div[class*="comment"][class*="section"]',
            'div#reviews',
            'div#customerReviews',
            'div.review-list',
            'div.comment-list'
        ]
        
        for selector in section_selectors:
            sections = soup.select(selector)
            for section in sections:
                # Try to find individual review containers within section
                review_containers = section.select('div.review, div.comment, div.review-item, div.review-entry')
                
                if review_containers:
                    for container in review_containers:
                        try:
                            raw_text = container.get_text(strip=True, separator=' ')
                            if 40 < len(raw_text) < 600:
                                cleaned_text = self._clean_review_text(raw_text)
                                if not cleaned_text or len(cleaned_text) < 20:
                                    continue
                                
                                rating = self._parse_rating_from_element(container)
                                if rating == 0:
                                    rating = self._guess_rating_from_text(cleaned_text)
                                
                                reviews.append({
                                    'text': cleaned_text[:400],
                                    'rating': rating,
                                    'platform': 'Section',
                                    'source': 'section',
                                    'raw_text': raw_text[:500]
                                })
                        except:
                            continue
                else:
                    # Fallback: split section text
                    section_text = section.get_text(strip=True, separator=' ')
                    # Split by common separators
                    review_chunks = re.split(r'\n\s*\n|\t{2,}|•|◦|■|□|\d+\.\s+|\-{3,}', section_text)
                    
                    for chunk in review_chunks:
                        if 50 < len(chunk) < 600:
                            cleaned_text = self._clean_review_text(chunk)
                            if not cleaned_text or len(cleaned_text) < 20:
                                continue
                            
                            rating = self._parse_rating_text(chunk)
                            if rating == 0:
                                rating = self._guess_rating_from_text(cleaned_text)
                            
                            reviews.append({
                                'text': cleaned_text[:400],
                                'rating': rating,
                                'platform': 'Section',
                                'source': 'section',
                                'raw_text': chunk[:500]
                            })
        
        return reviews
    
    def _parse_rating_from_element(self, element):
        """Parse rating from HTML element"""
        text = element.get_text(strip=True, separator=' ')
        return self._parse_rating_text(text)
    
    def _parse_rating_text(self, text):
        """Parse rating from text string"""
        patterns = [
            r'(\d+(\.\d+)?)\s*out of\s*5',
            r'(\d+(\.\d+)?)\s*[/]\s*5',
            r'(\d+(\.\d+)?)\s*stars?',
            r'(\d+(\.\d+)?)\s*★',
            r'\b(\d+(\.\d+)?)/5\b',
            r'Rating[:\s]*(\d+(\.\d+)?)',
            r'Rated\s*(\d+(\.\d+)?)',
            r'\b(\d)\.(\d)\b',  # 4.5, 3.5 etc
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    rating = float(match.group(1))
                    if 1 <= rating <= 5:
                        return rating
                except:
                    continue
        
        # Count stars
        star_count = text.count('★') + text.count('⭐') + text.count('✰')
        if 1 <= star_count <= 5:
            return float(star_count)
        
        # Look for star ratings like "★★★★★"
        star_pattern = r'[★⭐✰]{1,5}'
        star_match = re.search(star_pattern, text)
        if star_match:
            stars = star_match.group(0)
            star_count = len(stars)
            if 1 <= star_count <= 5:
                return float(star_count)
        
        return 0
    
    def _guess_rating_from_text(self, text):
        """Guess rating based on text sentiment"""
        text_lower = text.lower()
        
        positive_words = ['excellent', 'perfect', 'best', 'awesome', 'good', 'great', 'love', 
                         'recommend', 'amazing', 'fantastic', 'outstanding', 'superb', 'wonderful',
                         'satisfied', 'happy', 'pleased', 'impressed', 'worth', 'value']
        
        negative_words = NEGATIVE_KEYWORDS + ['not good', "don't buy", 'avoid', 'return', 'refund',
                                             'complaint', 'issue', 'problem', 'faulty', 'defect',
                                             'broken', 'damaged', 'poor quality', 'waste of money']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if neg_count > 0:
            return max(1.0, 5.0 - (neg_count * 0.5))
        elif pos_count > 0:
            return min(5.0, 3.0 + (pos_count * 0.3))
        else:
            return 3.0
    
    def _extract_product_title(self, soup):
        """Extract product title"""
        title_selectors = [
            '#productTitle', 'h1.product-title', 'h1._2Kn22P', 
            'h1.sc-fznKkj', 'h1.pdp-title', 'h1.title',
            'h1.product-name', 'h1.productTitle', 'h1.product_name',
            'meta[property="og:title"]', 'meta[name="title"]',
            'title'
        ]
        
        for selector in title_selectors:
            if 'meta' in selector:
                meta = soup.select_one(selector)
                if meta and meta.get('content'):
                    title = meta.get('content')
                    # Remove common suffixes
                    title = re.sub(r'\s*-\s*(Amazon|Flipkart|Meesho|Shop).*$', '', title, flags=re.IGNORECASE)
                    return title[:150]
            else:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text(strip=True)
                    if title and len(title) > 3:
                        # Remove common suffixes
                        title = re.sub(r'\s*-\s*(Amazon|Flipkart|Meesho|Shop).*$', '', title, flags=re.IGNORECASE)
                        return title[:150]
        
        return "Product"
    
    def _generate_page_url(self, base_url, page):
        """Generate paginated URL"""
        base_url = base_url.split('?')[0]  # Remove existing query params
        
        if 'amazon' in base_url:
            if '/product-reviews/' in base_url:
                return f"{base_url}/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}"
            else:
                return f"{base_url}?reviewPageNumber={page}"
        elif 'flipkart' in base_url:
            return f"{base_url}&page={page}"
        elif 'meesho' in base_url:
            return f"{base_url}?page={page}"
        else:
            return f"{base_url}?page={page}"
    
    def _deduplicate_reviews(self, reviews):
        """Remove duplicate reviews using hash"""
        unique_reviews = []
        seen_hashes = set()
        
        for review in reviews:
            text_for_hash = review['text'].strip().lower()
            # Take first 100 chars for hash
            text_hash = hashlib.md5(text_for_hash[:100].encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_reviews.append(review)
        
        return unique_reviews

class AnalysisAgent:
    """Agent responsible for sentiment analysis and insights"""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if self.api_key:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
    
    def analyze_reviews(self, reviews: List[Dict]) -> str:
        """Agent task: Analyze sentiment of reviews"""
        if not reviews:
            return json.dumps([])
        
        analyzed_reviews = []
        
        # Process in batches
        batch_size = 5  # Smaller batch for better accuracy
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            if hasattr(self, 'llm'):
                analyzed_reviews.extend(self._ai_analyze_batch(batch))
            else:
                analyzed_reviews.extend(self._basic_analyze_batch(batch))
            if self.api_key:
                time.sleep(0.5)
        return json.dumps({
            "analyzed_reviews": analyzed_reviews
        })
            
    def _ai_analyze_batch(self, batch):
        try:
            batch_text = []
            for idx, r in enumerate(batch):
                batch_text.append(f"{idx+1}. {r['text'][:150]}...")
            
            prompt = f"""Analyze these product reviews. For EACH review, provide ONLY a JSON array with objects containing:
            1. sentiment: "Positive", "Negative", or "Neutral"
            2. confidence: number between 0.0 and 1.0
            3. emotion: "satisfied", "disappointed", "angry", "happy", or "neutral"
            4. is_critical: true or false (true if review mentions serious issues or defects)
            
            Reviews:
            {chr(10).join(batch_text)}
            
            Example format: [{{"sentiment": "Positive", "confidence": 0.9, "emotion": "happy", "is_critical": false}}]
            
            Return ONLY the JSON array, nothing else."""
            
            response = self.llm.invoke(prompt)
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            
            if json_match:
                analysis_data = json.loads(json_match.group())
                result = []
                for i in range(len(batch)):
                    if i < len(analysis_data):
                        combined = {**batch[i], **analysis_data[i]}
                        result.append(combined)
                    else:
                        # Fallback if AI returns fewer items
                        result.append(self._basic_analyze_review(batch[i]))
                return result
        except Exception as e:
            pass
        
        return self._basic_analyze_batch(batch)
    
    def _basic_analyze_review(self, review):
        """Basic sentiment analysis for a single review"""
        rating = review.get('rating', 0)
        text = review.get('text', '').lower()
        
        negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text)
        positive_words = ['excellent', 'perfect', 'best', 'awesome', 'good', 'great', 'love', 
                         'recommend', 'amazing', 'fantastic', 'satisfied', 'happy', 'pleased']
        positive_count = sum(1 for word in positive_words if word in text)
        
        if rating <= 2 or negative_count > positive_count:
            sentiment = "Negative"
            emotion = "angry" if rating == 1 else "disappointed"
            confidence = 0.9 if rating <= 2 else 0.7
            is_critical = True if negative_count >= 2 else False
        elif rating >= 4 and positive_count > negative_count:
            sentiment = "Positive"
            emotion = "happy" if rating >= 4.5 else "satisfied"
            confidence = 0.8
            is_critical = False
        else:
            sentiment = "Neutral"
            emotion = "neutral"
            confidence = 0.6
            is_critical = False
        
        return {
            **review,
            'sentiment': sentiment,
            'confidence': confidence,
            'emotion': emotion,
            'is_critical': is_critical
        }
    
    def _basic_analyze_batch(self, batch):
        """Basic sentiment analysis for a batch"""
        return [self._basic_analyze_review(review) for review in batch]
    
    def generate_insights(self, reviews: List[Dict], product_title: str) -> str:
        """Agent task: Generate insights and summary"""
        if not reviews:
            return json.dumps({'product_title': product_title, 'total_reviews': 0})
        
        total = len(reviews)
        ratings = [r.get('rating', 0) for r in reviews]
        sentiments = [r.get('sentiment', 'Neutral') for r in reviews]
        
        positive = sum(1 for s in sentiments if s == 'Positive')
        negative = sum(1 for s in sentiments if s == 'Negative')
        neutral = total - positive - negative
        
        avg_rating = sum(ratings) / total if total > 0 else 0
        
        rating_dist = {}
        for i in range(1, 6):
            rating_dist[str(i)] = sum(1 for r in ratings if i - 0.5 <= r <= i + 0.5)
        
        summary = {
            'product_title': product_title,
            'total_reviews': total,
            'avg_rating': round(avg_rating, 2),
            'positive_pct': round(positive / total * 100, 1) if total > 0 else 0,
            'negative_pct': round(negative / total * 100, 1) if total > 0 else 0,
            'neutral_pct': round(neutral / total * 100, 1) if total > 0 else 0,
            'sentiment': 'Positive' if avg_rating >= 3.5 else 'Negative' if avg_rating <= 2.5 else 'Neutral',
            'rating_distribution': rating_dist,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate AI insights if API key available and enough reviews
        if self.api_key and total >= 5:
            try:
                # Get sample of reviews for analysis
                sample_size = min(15, total)
                sample_indices = list(range(0, total, max(1, total // sample_size)))
                sample_reviews = [reviews[i] for i in sample_indices[:sample_size]]
                
                review_samples = []
                for i, r in enumerate(sample_reviews):
                    review_samples.append(f"{i+1}. Rating: {r['rating']}/5 - {r['text'][:120]}...")
                
                prompt = f"""Product: {product_title}
                Total Reviews: {total} | Average Rating: {avg_rating:.1f}/5 
                Positive: {positive/total*100:.1f}% | Negative: {negative/total*100:.1f}% | Neutral: {neutral/total*100:.1f}%
                
                Sample Reviews:
                {chr(10).join(review_samples)}
                
                Provide a JSON object with these keys:
                1. main_strengths: array of 2-3 main positive points mentioned
                2. main_issues: array of 2-3 main problems mentioned
                3. recommendation: brief 1-2 sentence recommendation for potential buyers
                4. overall_summary: brief 2-3 sentence summary of what customers are saying
                
                Be specific and reference the actual review content."""
                
                response = self.llm.invoke(prompt)
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    insights = json.loads(json_match.group())
                    summary.update(insights)
            except Exception as e:
                # Add fallback insights
                positive_reviews = [r for r in reviews if r['sentiment'] == 'Positive']
                negative_reviews = [r for r in reviews if r['sentiment'] == 'Negative']
                
                common_positives = []
                common_negatives = []
                
                if positive_reviews:
                    pos_texts = ' '.join([r['text'].lower() for r in positive_reviews[:10]])
                    if 'good' in pos_texts or 'great' in pos_texts:
                        common_positives.append("Good quality")
                    if 'value' in pos_texts or 'worth' in pos_texts:
                        common_positives.append("Good value for money")
                    if 'love' in pos_texts or 'like' in pos_texts:
                        common_positives.append("Customers like the product")
                
                if negative_reviews:
                    neg_texts = ' '.join([r['text'].lower() for r in negative_reviews[:10]])
                    if any(word in neg_texts for word in ['broken', 'defective', 'damaged']):
                        common_negatives.append("Quality/defect issues")
                    if any(word in neg_texts for word in ['waste', 'not worth', 'expensive']):
                        common_negatives.append("Not value for money")
                    if 'return' in neg_texts or 'refund' in neg_texts:
                        common_negatives.append("Customers want to return")
                
                summary.update({
                    'main_strengths': common_positives[:3] or ["Positive feedback received"],
                    'main_issues': common_negatives[:3] or ["Some issues reported"],
                    'recommendation': f"Based on {total} reviews with {avg_rating:.1f}/5 average rating",
                    'overall_summary': f"Customers generally {'like' if avg_rating >= 3 else 'have mixed feelings about'} this product"
                })
        
        return json.dumps({
            "analysis_summary": summary
        })


# ==================== TOOL WRAPPERS (REQUIRED BY CREWAI) ====================

@tool("scrape_reviews")
def scrape_reviews_tool(url: str) -> str:
    """
    Scrape and clean product reviews from a product URL.
    """
    return scraper_tool.scrape_reviews(url)


@tool("analyze_reviews")
def analyze_reviews_tool(reviews_json: str) -> str:
    """
    Analyze sentiment and emotions from scraped reviews.
    """
    reviews = json.loads(reviews_json)
    return analysis_tool.analyze_reviews(reviews)


@tool("generate_insights")
def generate_insights_tool(reviews_json: str, product_title: str) -> str:
    """
    Generate business insights and recommendations from analyzed reviews.
    """
    reviews = json.loads(reviews_json)
    return analysis_tool.generate_insights(reviews, product_title)



# ==================== CREWAI AGENTS ====================

scraper_tool = ScrapingAgent()
analysis_tool = AnalysisAgent()

manager_agent = Agent(
    role="Review Analysis Manager",
    goal="Decide which analysis steps are required and coordinate agents to analyze product reviews.",
    backstory="You are a senior AI manager who plans review analysis logically.",
    llm=llm,
    allow_delegation=False,
    verbose=False
)

scraper_agent = Agent(
    role="Web Scraping Specialist",
    goal="Scrape and clean product reviews from e-commerce websites.",
    backstory="Expert in extracting clean customer reviews.",
    llm=llm,
    tools=[scrape_reviews_tool],
    verbose=False
)

sentiment_agent = Agent(
    role="Sentiment Analyst",
    goal="Analyze customer sentiment and emotions from reviews.",
    backstory="Expert in sentiment analysis and customer emotions.",
    llm=llm,
    tools=[analyze_reviews_tool],
    verbose=False
)

insight_agent = Agent(
    role="Business Insight Analyst",
    goal="Generate insights and recommendations from analyzed reviews.",
    backstory="Expert in summarizing customer feedback into business insights.",
    llm=llm,
    tools=[generate_insights_tool],
    verbose=False
)


# ==================== CREWAI TASKS ====================

scrape_task = Task(
    description="Scrape and clean reviews from the given product URL.",
    expected_output="JSON with raw_reviews, product_title, and error",
    agent=scraper_agent
)

sentiment_task = Task(
    description="Analyze sentiment and emotions from the scraped reviews.",
    expected_output="JSON with analyzed_reviews including sentiment, confidence, emotion",
    agent=sentiment_agent
)

insight_task = Task(
    description="Generate insights and recommendations from analyzed reviews.",
    expected_output="JSON with analysis_summary including metrics and insights",
    agent=insight_agent
)

crew = Crew(
    agents=[scraper_agent, sentiment_agent, insight_agent],
    tasks=[scrape_task, sentiment_task, insight_task],
    process=Process.sequential,
    verbose=False,
    embedder=None
)

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(page_title=" Review Analyzer", page_icon="📊", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {text-align: center; color: #1E3A8A; font-size: 2.8rem; margin-bottom: 1rem;}
    .metric-card {background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 5px;}
    .positive-box {background: #F0FDF4; border-left: 4px solid #10B981; padding: 1rem; border-radius: 8px; margin: 10px 0;}
    .negative-box {background: #FEF2F2; border-left: 4px solid #EF4444; padding: 1rem; border-radius: 8px; margin: 10px 0;}
    .neutral-box {background: #FEFCE8; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 8px; margin: 10px 0;}
    .badge {background: linear-gradient(90deg, #667eea, #764ba2); color: white; padding: 5px 15px; 
            border-radius: 20px; font-size: 0.9rem; font-weight: bold;}
    .agent-box {background: linear-gradient(90deg, #E3F2FD, #F3E5F5); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #3B82F6;}
    .workflow-step {background: #F8FAFC; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #3B82F6;}
    .review-text {font-size: 0.95rem; line-height: 1.5; color: #374151;}
    .stars {color: #F59E0B; font-size: 1.2rem;}
    </style>
    <h1 class="main-title">🤖 Agentic  Review Analyzer</h1>
    """, unsafe_allow_html=True)
    
    # Main input
    url = st.text_input(
        "🔗 **Enter Product URL for Review Analysis:**",
        placeholder="https://www.amazon.in/dp/... or any e-commerce product page",
        key="url_input"
    )
    
    if st.button("🚀 **Run Agentic Analysis**", type="primary", width='stretch', disabled=not url):
        
        
        with st.spinner("🤖 Agents are analyzing reviews..."):
            try:
                raw = json.loads(scraper_tool.scrape_reviews(url))

                analyzed = json.loads(
                    analysis_tool.analyze_reviews(raw["raw_reviews"])
                )

                final = json.loads(
                    analysis_tool.generate_insights(
                        analyzed["analyzed_reviews"],
                        raw["product_title"]
                    )
                )

                # ✅ Store clean results
                st.session_state.reviews = analyzed["analyzed_reviews"]
                st.session_state.summary = final["analysis_summary"]

                st.success("✅ Agentic analysis completed!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Agent execution failed: {e}")

    # Display results from LangGraph workflow
    if 'reviews' in st.session_state and 'summary' in st.session_state:
        reviews = st.session_state.reviews
        summary = st.session_state.summary
        total_reviews = summary['total_reviews']
        
        # Product header
        st.markdown(f"""
        <div style="text-align: center; color: #4B5563; font-size: 1.2rem; margin: 1rem 0; padding: 12px; 
                    background: linear-gradient(90deg, #E3FDF5, #E3F2FD); border-radius: 10px;">
            🤖 <strong>Agent Analysis Complete!</strong> | 
            📦 <strong>Product:</strong> {summary['product_title']} | 
            <strong>Reviews Analyzed:</strong> {total_reviews}
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        st.subheader("📊 **Analysis Dashboard**")
        cols = st.columns(4)
        metrics = [
            ("Total Reviews", f"{total_reviews}", "#3B82F6"),
            ("Avg Rating", f"{summary['avg_rating']}/5", 
             "#10B981" if summary['avg_rating'] >= 3.5 else "#EF4444" if summary['avg_rating'] <= 2.5 else "#F59E0B"),
            ("Positive", f"{summary['positive_pct']}%", "#10B981"),
            ("Negative", f"{summary['negative_pct']}%", "#EF4444")
        ]
        
        for col, (label, value, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div style="color: {color}; font-size: 1.8rem; font-weight: bold;">{value}</div>
                    <div style="color: #6B7280; font-size: 0.9rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # AI Insights if available
        if 'main_strengths' in summary or 'main_issues' in summary or 'overall_summary' in summary:
            st.subheader("💡 **AI Insights**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'main_strengths' in summary:
                    st.markdown("""
                    <div class="positive-box">
                    <strong>✅ Main Strengths:</strong>
                    """, unsafe_allow_html=True)
                    if isinstance(summary['main_strengths'], list):
                        for strength in summary['main_strengths'][:3]:
                            st.markdown(f"• {strength}")
                    else:
                        st.markdown(f"• {summary['main_strengths']}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                if 'main_issues' in summary:
                    st.markdown("""
                    <div class="negative-box">
                    <strong>⚠️ Main Issues:</strong>
                    """, unsafe_allow_html=True)
                    if isinstance(summary['main_issues'], list):
                        for issue in summary['main_issues'][:3]:
                            st.markdown(f"• {issue}")
                    else:
                        st.markdown(f"• {summary['main_issues']}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            if 'overall_summary' in summary:
                st.markdown(f"""
                <div class="neutral-box">
                <strong>📋 Overall Summary:</strong> {summary['overall_summary']}
                </div>
                """, unsafe_allow_html=True)
            
            if 'recommendation' in summary:
                st.markdown(f"""
                <div class="positive-box" style="margin-top: 10px;">
                <strong>🎯 Recommendation:</strong> {summary['recommendation']}
                </div>
                """, unsafe_allow_html=True)
        
        # Visualizations
        st.subheader("📈 **Review Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Pie Chart
            sentiments = [r['sentiment'] for r in reviews]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            fig1 = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=['#10B981', '#EF4444', '#F59E0B'],
                textinfo='label+percent',
                hoverinfo='label+value+percent'
            )])
            fig1.update_layout(
                title=f"Sentiment Distribution ({total_reviews} reviews)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Rating Distribution
            ratings = [r['rating'] for r in reviews]
            
            fig2 = px.histogram(
                x=ratings,
                nbins=10,
                title=f"Rating Distribution ({total_reviews} reviews)",
                labels={'x': 'Rating', 'y': 'Number of Reviews'},
                color_discrete_sequence=['#3B82F6']
            )
            
            fig2.update_layout(
                height=400,
                xaxis_title="Rating (out of 5)",
                yaxis_title="Count",
                bargap=0.1
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Review Samples - Now showing CLEAN reviews only
        st.subheader("⭐ **Review**")
        
        tab1, tab2 = st.tabs(["👍 Positive Reviews", "👎 Negative Reviews"])
        
        with tab1:
            positive_reviews = [r for r in reviews if r['sentiment'] == 'Positive']
            if positive_reviews:
                for i, review in enumerate(positive_reviews[:10]):
                    # Create star rating display
                    rating = review['rating']
                    full_stars = int(rating)
                    has_half = rating - full_stars >= 0.5
                    stars = "★" * full_stars + "½" * has_half + "☆" * (5 - full_stars - has_half)
                    
                    st.markdown(f"""
                    <div class="positive-box">
                        <div class="stars">{stars} ({rating:.1f}/5)</div>
                        <div class="review-text">"{review['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No positive reviews found.")
        
        with tab2:
            negative_reviews = [r for r in reviews if r['sentiment'] == 'Negative']
            if negative_reviews:
                for i, review in enumerate(negative_reviews[:10]):
                    # Create star rating display
                    rating = review['rating']
                    full_stars = int(rating)
                    has_half = rating - full_stars >= 0.5
                    stars = "★" * full_stars + "½" * has_half + "☆" * (5 - full_stars - has_half)
                    
                    st.markdown(f"""
                    <div class="negative-box">
                        <div class="stars">{stars} ({rating:.1f}/5)</div>
                        <div class="review-text">"{review['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No negative reviews found.")
        
if __name__ == "__main__":
    main()

    

