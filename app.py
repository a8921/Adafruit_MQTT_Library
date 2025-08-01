from flask import Flask, render_template, request, jsonify, session
import requests
import os
import sqlite3
import re
import json
import hashlib
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import uuid
import traceback
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lysis_ai.log'),
        logging.StreamHandler()
    ]
)


def log_error(function_name: str, error: Exception, context: Dict = None):
    """Centralized error logging"""
    logging.error(f"Error in {function_name}: {str(error)}")
    if context:
        logging.error(f"Context: {context}")
    logging.error(f"Traceback: {traceback.format_exc()}")


# Basic imports that should always work
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ NumPy available")
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy not available")

# ===== NUMERICAL SOLVER IMPORTS =====
try:
    import sympy as sp
    from sympy import symbols, solve, simplify, diff, integrate, limit, expand, factor
    from sympy.parsing.sympy_parser import parse_expr
    import math
    SYMPY_AVAILABLE = True
    print("✅ SymPy available for numerical solving")
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️ SymPy not available - install with: pip install sympy")

# ML imports with better fallback handling - wrapped in individual try/catch blocks
SENTENCE_TRANSFORMERS_AVAILABLE = False
SKLEARN_AVAILABLE = False
CLUSTERING_AVAILABLE = False
KEYBERT_AVAILABLE = False
SPACY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

print("🔄 Checking ML library availability...")

try:
    # Test numpy compatibility first
    if NUMPY_AVAILABLE:
        test_array = np.array([1, 2, 3])
        print("✅ NumPy compatibility verified")
except Exception as e:
    print(f"⚠️ NumPy compatibility issue: {e}")
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    if NUMPY_AVAILABLE:
        from sklearn.metrics.pairwise import cosine_similarity
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        print("✅ sentence-transformers available")
    else:
        print("⚠️ sentence-transformers requires NumPy - skipping")
except ImportError as e:
    print(f"⚠️ sentence-transformers not available: {str(e)[:100]}...")
except Exception as e:
    print(f"⚠️ sentence-transformers error: {str(e)[:100]}...")

try:
    import umap.umap_ as umap
    import hdbscan
    CLUSTERING_AVAILABLE = True
    print("✅ clustering libraries available")
except ImportError as e:
    print(f"⚠️ umap/hdbscan not available: {str(e)[:50]}...")
except Exception as e:
    print(f"⚠️ clustering libraries error: {str(e)[:50]}...")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
    print("✅ KeyBERT available")
except ImportError as e:
    print(f"⚠️ KeyBERT not available: {str(e)[:50]}...")
except Exception as e:
    print(f"⚠️ KeyBERT error: {str(e)[:50]}...")

try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ spaCy available")
except ImportError as e:
    print(f"⚠️ spaCy not available: {str(e)[:50]}...")
except Exception as e:
    print(f"⚠️ spaCy error: {str(e)[:50]}...")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("✅ transformers available")
except ImportError as e:
    print(f"⚠️ transformers not available: {str(e)[:50]}...")
except Exception as e:
    print(f"⚠️ transformers error: {str(e)[:50]}...")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn available")
except ImportError as e:
    print(f"⚠️ scikit-learn not available: {str(e)[:50]}...")
except Exception as e:
    print(f"⚠️ scikit-learn error: {str(e)[:50]}...")

print(f"""
📊 ML Libraries Status:
   NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}
   SymPy: {'✅' if SYMPY_AVAILABLE else '❌'}
   SentenceTransformers: {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}
   scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}
   Clustering (UMAP/HDBSCAN): {'✅' if CLUSTERING_AVAILABLE else '❌'}
   KeyBERT: {'✅' if KEYBERT_AVAILABLE else '❌'}
   spaCy: {'✅' if SPACY_AVAILABLE else '❌'}
   Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}
""")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Global variables for caching and rate limiting
conversation_cache = {}
rate_limit_cache = defaultdict(deque)
analysis_cache = {}

# Initialize ML models if available
sentence_model = None
keybert_model = None
nlp = None

if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded")
    except Exception as e:
        print(f"⚠️ Failed to load sentence transformer: {e}")
        SENTENCE_TRANSFORMERS_AVAILABLE = False

if KEYBERT_AVAILABLE:
    try:
        keybert_model = KeyBERT()
        print("✅ KeyBERT model loaded")
    except Exception as e:
        print(f"⚠️ Failed to load KeyBERT: {e}")
        KEYBERT_AVAILABLE = False

if SPACY_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded")
    except Exception as e:
        print(f"⚠️ Failed to load spaCy model: {e}")
        try:
            # Try alternative model names
            nlp = spacy.load("en")
            print("✅ spaCy basic model loaded")
        except:
            SPACY_AVAILABLE = False
            print("⚠️ spaCy models not available")


@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    analysis_type: str
    content: str
    result: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path='lysis_ai.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        message TEXT,
                        response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        analysis_type TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Analysis results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id TEXT PRIMARY KEY,
                        analysis_type TEXT,
                        content TEXT,
                        result TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processing_time REAL,
                        confidence REAL,
                        metadata TEXT
                    )
                ''')
                
                # User sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_data TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                print("✅ Database initialized successfully")
                
        except Exception as e:
            log_error("init_database", e)
    
    def save_conversation(self, user_id: str, message: str, response: str, 
                         analysis_type: str = None, metadata: Dict = None):
        """Save conversation to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                conversation_id = str(uuid.uuid4())
                
                cursor.execute('''
                    INSERT INTO conversations 
                    (id, user_id, message, response, analysis_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (conversation_id, user_id, message, response, 
                      analysis_type, json.dumps(metadata) if metadata else None))
                
                conn.commit()
                return conversation_id
                
        except Exception as e:
            log_error("save_conversation", e)
            return None
    
    def save_analysis_result(self, result: AnalysisResult):
        """Save analysis result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                result_id = str(uuid.uuid4())
                
                cursor.execute('''
                    INSERT INTO analysis_results 
                    (id, analysis_type, content, result, processing_time, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (result_id, result.analysis_type, result.content,
                      json.dumps(result.result), result.processing_time,
                      result.confidence, json.dumps(result.metadata) if result.metadata else None))
                
                conn.commit()
                return result_id
                
        except Exception as e:
            log_error("save_analysis_result", e)
            return None
    
    def get_conversation_history(self, user_id: str, limit: int = 10):
        """Get conversation history for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message, response, timestamp, analysis_type, metadata
                    FROM conversations 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
                
                return cursor.fetchall()
                
        except Exception as e:
            log_error("get_conversation_history", e)
            return []


# Initialize database manager
db_manager = DatabaseManager()


def rate_limit_check(user_id: str, max_requests: int = 60, window_minutes: int = 1):
    """Check if user has exceeded rate limit"""
    try:
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Clean old requests
        user_requests = rate_limit_cache[user_id]
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
        
        # Check if limit exceeded
        if len(user_requests) >= max_requests:
            return False, f"Rate limit exceeded. Max {max_requests} requests per {window_minutes} minute(s)."
        
        # Add current request
        user_requests.append(current_time)
        return True, None
        
    except Exception as e:
        log_error("rate_limit_check", e)
        return True, None  # Allow request on error


def get_user_id():
    """Get or create user ID from session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Main analysis endpoint"""
    start_time = time.time()
    user_id = get_user_id()
    
    try:
        # Rate limiting
        allowed, error_msg = rate_limit_check(user_id)
        if not allowed:
            return jsonify({'error': error_msg}), 429
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        analysis_type = data.get('analysis_type', 'auto')
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 50000:  # Reasonable limit
            return jsonify({'error': 'Text too long (max 50,000 characters)'}), 400
        
        # Check cache
        cache_key = hashlib.md5(f"{text}_{analysis_type}".encode()).hexdigest()
        if cache_key in analysis_cache:
            cached_result = analysis_cache[cache_key]
            if time.time() - cached_result['timestamp'] < 3600:  # 1 hour cache
                return jsonify(cached_result['result'])
        
        # Perform analysis
        result = perform_analysis(text, analysis_type)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create analysis result object
        analysis_result = AnalysisResult(
            analysis_type=analysis_type,
            content=text[:1000],  # Store first 1000 chars
            result=result,
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
        # Save to database
        db_manager.save_analysis_result(analysis_result)
        
        # Cache result
        analysis_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Clean cache if too large
        if len(analysis_cache) > 1000:
            oldest_key = min(analysis_cache.keys(), 
                           key=lambda k: analysis_cache[k]['timestamp'])
            del analysis_cache[oldest_key]
        
        return jsonify(result)
        
    except Exception as e:
        log_error("analyze_text", e, {'user_id': user_id, 'text_length': len(text) if 'text' in locals() else 0})
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500


def perform_analysis(text: str, analysis_type: str = 'auto') -> Dict[str, Any]:
    """Perform the requested analysis on the text"""
    try:
        result = {
            'analysis_type': analysis_type,
            'basic_stats': get_basic_stats(text),
            'available_features': get_available_features()
        }
        
        # Auto-detect analysis type if not specified
        if analysis_type == 'auto':
            analysis_type = detect_analysis_type(text)
            result['detected_type'] = analysis_type
        
        # Perform specific analysis based on type
        if analysis_type == 'mathematical' and SYMPY_AVAILABLE:
            result['mathematical'] = analyze_mathematical_content(text)
        
        elif analysis_type == 'sentiment' and (TRANSFORMERS_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE):
            result['sentiment'] = analyze_sentiment(text)
        
        elif analysis_type == 'keywords' and (KEYBERT_AVAILABLE or SKLEARN_AVAILABLE):
            result['keywords'] = extract_keywords(text)
        
        elif analysis_type == 'summary' and TRANSFORMERS_AVAILABLE:
            result['summary'] = generate_summary(text)
        
        elif analysis_type == 'entities' and SPACY_AVAILABLE:
            result['entities'] = extract_entities(text)
        
        elif analysis_type == 'similarity' and SENTENCE_TRANSFORMERS_AVAILABLE:
            result['similarity'] = analyze_text_similarity(text)
        
        elif analysis_type == 'comprehensive':
            # Perform all available analyses
            if SYMPY_AVAILABLE:
                result['mathematical'] = analyze_mathematical_content(text)
            if TRANSFORMERS_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE:
                result['sentiment'] = analyze_sentiment(text)
            if KEYBERT_AVAILABLE or SKLEARN_AVAILABLE:
                result['keywords'] = extract_keywords(text)
            if TRANSFORMERS_AVAILABLE:
                result['summary'] = generate_summary(text)
            if SPACY_AVAILABLE:
                result['entities'] = extract_entities(text)
        
        else:
            # Fallback to basic analysis
            result['basic_analysis'] = perform_basic_analysis(text)
        
        return result
        
    except Exception as e:
        log_error("perform_analysis", e, {'analysis_type': analysis_type})
        return {
            'error': 'Analysis failed',
            'basic_stats': get_basic_stats(text),
            'available_features': get_available_features()
        }


def get_basic_stats(text: str) -> Dict[str, Any]:
    """Get basic statistics about the text"""
    try:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'average_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0
        }
    except Exception as e:
        log_error("get_basic_stats", e)
        return {'error': 'Failed to calculate basic stats'}


def get_available_features() -> Dict[str, bool]:
    """Return which features are available based on installed libraries"""
    return {
        'mathematical_analysis': SYMPY_AVAILABLE,
        'sentiment_analysis': TRANSFORMERS_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE,
        'keyword_extraction': KEYBERT_AVAILABLE or SKLEARN_AVAILABLE,
        'text_summarization': TRANSFORMERS_AVAILABLE,
        'entity_extraction': SPACY_AVAILABLE,
        'similarity_analysis': SENTENCE_TRANSFORMERS_AVAILABLE,
        'clustering': CLUSTERING_AVAILABLE,
        'basic_analysis': True
    }


def detect_analysis_type(text: str) -> str:
    """Auto-detect the most appropriate analysis type for the text"""
    try:
        # Check for mathematical content
        math_patterns = [
            r'\d+\s*[+\-*/=]\s*\d+',  # Basic arithmetic
            r'[xy]\s*[+\-*/=]',       # Variables
            r'sin|cos|tan|log|exp',    # Functions
            r'\b\d+x\b',              # Coefficients
            r'\^|\*\*',               # Exponents
            r'∫|∑|∏|∂',               # Mathematical symbols
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in math_patterns):
            return 'mathematical'
        
        # Check for long text that might benefit from summarization
        if len(text.split()) > 200:
            return 'summary'
        
        # Check for text that might need sentiment analysis
        sentiment_indicators = ['feel', 'think', 'opinion', 'good', 'bad', 'love', 'hate']
        if any(word in text.lower() for word in sentiment_indicators):
            return 'sentiment'
        
        # Default to keyword extraction for medium-length texts
        if len(text.split()) > 50:
            return 'keywords'
        
        # For short texts, do basic analysis
        return 'basic'
        
    except Exception as e:
        log_error("detect_analysis_type", e)
        return 'basic'


def analyze_mathematical_content(text: str) -> Dict[str, Any]:
    """Analyze mathematical expressions in the text"""
    if not SYMPY_AVAILABLE:
        return {'error': 'SymPy not available'}
    
    try:
        result = {
            'equations_found': [],
            'solutions': [],
            'derivatives': [],
            'integrals': [],
            'simplified_expressions': []
        }
        
        # Find mathematical expressions
        math_patterns = [
            r'[xy]\s*[+\-*/=][^.!?]*',
            r'\d+\s*[+\-*/=][^.!?]*',
            r'sin\([^)]+\)|cos\([^)]+\)|tan\([^)]+\)',
            r'log\([^)]+\)|exp\([^)]+\)',
        ]
        
        expressions = []
        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend(matches)
        
        for expr_str in expressions[:10]:  # Limit to 10 expressions
            try:
                # Clean the expression
                cleaned_expr = re.sub(r'[^\w\s+\-*/=().,^]', '', expr_str)
                cleaned_expr = cleaned_expr.replace('^', '**')
                
                if '=' in cleaned_expr:
                    # It's an equation
                    left, right = cleaned_expr.split('=', 1)
                    try:
                        left_expr = parse_expr(left.strip())
                        right_expr = parse_expr(right.strip())
                        equation = sp.Eq(left_expr, right_expr)
                        result['equations_found'].append(str(equation))
                        
                        # Try to solve
                        solutions = solve(equation)
                        if solutions:
                            result['solutions'].append({
                                'equation': str(equation),
                                'solutions': [str(sol) for sol in solutions]
                            })
                    except:
                        pass
                else:
                    # It's an expression
                    try:
                        expr = parse_expr(cleaned_expr)
                        result['simplified_expressions'].append({
                            'original': cleaned_expr,
                            'simplified': str(simplify(expr))
                        })
                        
                        # Try to find derivatives
                        free_symbols = expr.free_symbols
                        if free_symbols:
                            symbol = list(free_symbols)[0]
                            derivative = diff(expr, symbol)
                            result['derivatives'].append({
                                'expression': str(expr),
                                'derivative': str(derivative),
                                'variable': str(symbol)
                            })
                    except:
                        pass
                        
            except Exception as e:
                continue
        
        return result
        
    except Exception as e:
        log_error("analyze_mathematical_content", e)
        return {'error': 'Mathematical analysis failed'}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of the text"""
    try:
        if TRANSFORMERS_AVAILABLE:
            try:
                sentiment_pipeline = pipeline("sentiment-analysis")
                result = sentiment_pipeline(text[:512])  # Limit text length
                return {
                    'method': 'transformers',
                    'sentiment': result[0]['label'].lower(),
                    'confidence': result[0]['score'],
                    'details': result
                }
            except Exception as e:
                print(f"Transformers sentiment analysis failed: {e}")
        
        # Fallback to simple lexicon-based approach
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'disappointed']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = 'negative' 
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'method': 'lexicon-based',
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words_found': positive_count,
            'negative_words_found': negative_count
        }
        
    except Exception as e:
        log_error("analyze_sentiment", e)
        return {'error': 'Sentiment analysis failed'}


def extract_keywords(text: str) -> Dict[str, Any]:
    """Extract keywords from the text"""
    try:
        if KEYBERT_AVAILABLE and keybert_model:
            try:
                keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_k=10)
                return {
                    'method': 'KeyBERT',
                    'keywords': [{'keyword': kw[0], 'score': kw[1]} for kw in keywords]
                }
            except Exception as e:
                print(f"KeyBERT failed: {e}")
        
        if SKLEARN_AVAILABLE:
            try:
                # Use TF-IDF for keyword extraction
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Split into sentences for better analysis
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) < 2:
                    sentences = [text]  # Use full text if too few sentences
                
                vectorizer = TfidfVectorizer(
                    max_features=20,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1
                
                # Get top keywords
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                return {
                    'method': 'TF-IDF',
                    'keywords': [{'keyword': kw[0], 'score': float(kw[1])} for kw in keyword_scores[:10]]
                }
            except Exception as e:
                print(f"TF-IDF failed: {e}")
        
        # Fallback to simple frequency analysis
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = defaultdict(int)
        for word in filtered_words:
            word_freq[word] += 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'method': 'frequency-based',
            'keywords': [{'keyword': word, 'score': count} for word, count in sorted_words[:10]]
        }
        
    except Exception as e:
        log_error("extract_keywords", e)
        return {'error': 'Keyword extraction failed'}


def generate_summary(text: str) -> Dict[str, Any]:
    """Generate a summary of the text"""
    try:
        if TRANSFORMERS_AVAILABLE:
            try:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                
                # Split long text into chunks
                max_chunk_length = 1024
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                
                summaries = []
                for chunk in chunks[:3]:  # Limit to 3 chunks
                    if len(chunk.split()) > 30:  # Only summarize if chunk is substantial
                        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                        summaries.append(summary[0]['summary_text'])
                
                combined_summary = ' '.join(summaries)
                
                return {
                    'method': 'BART',
                    'summary': combined_summary,
                    'original_length': len(text.split()),
                    'summary_length': len(combined_summary.split()),
                    'compression_ratio': len(combined_summary.split()) / len(text.split()) if text.split() else 0
                }
            except Exception as e:
                print(f"BART summarization failed: {e}")
        
        # Fallback to extractive summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) <= 3:
            return {
                'method': 'extractive',
                'summary': text,
                'note': 'Text too short to summarize'
            }
        
        # Simple extractive summarization - pick first, middle, and last sentences
        summary_sentences = []
        summary_sentences.append(sentences[0])
        if len(sentences) > 2:
            summary_sentences.append(sentences[len(sentences)//2])
        summary_sentences.append(sentences[-1])
        
        summary = '. '.join(summary_sentences) + '.'
        
        return {
            'method': 'extractive',
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()),
            'compression_ratio': len(summary.split()) / len(text.split())
        }
        
    except Exception as e:
        log_error("generate_summary", e)
        return {'error': 'Summarization failed'}


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract named entities from the text"""
    if not SPACY_AVAILABLE or not nlp:
        return {'error': 'spaCy not available'}
    
    try:
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for ent in entities:
            entities_by_type[ent['label']].append(ent['text'])
        
        return {
            'method': 'spaCy',
            'entities': entities,
            'entities_by_type': dict(entities_by_type),
            'total_entities': len(entities)
        }
        
    except Exception as e:
        log_error("extract_entities", e)
        return {'error': 'Entity extraction failed'}


def analyze_text_similarity(text: str) -> Dict[str, Any]:
    """Analyze text similarity (placeholder for comparison with other texts)"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not sentence_model:
        return {'error': 'Sentence transformers not available'}
    
    try:
        # For now, just return the embedding info
        # In a real application, you'd compare with other texts
        embedding = sentence_model.encode([text])
        
        return {
            'method': 'sentence-transformers',
            'embedding_dimension': embedding.shape[1],
            'note': 'Similarity analysis requires comparison texts'
        }
        
    except Exception as e:
        log_error("analyze_text_similarity", e)
        return {'error': 'Similarity analysis failed'}


def perform_basic_analysis(text: str) -> Dict[str, Any]:
    """Perform basic text analysis without ML libraries"""
    try:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Word frequency
        word_freq = defaultdict(int)
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 2:
                word_freq[clean_word] += 1
        
        most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Reading time estimation (average 200 words per minute)
        reading_time = len(words) / 200
        
        return {
            'word_frequency': dict(most_common),
            'estimated_reading_time_minutes': round(reading_time, 1),
            'unique_words': len(word_freq),
            'lexical_diversity': len(word_freq) / len(words) if words else 0
        }
        
    except Exception as e:
        log_error("perform_basic_analysis", e)
        return {'error': 'Basic analysis failed'}


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for conversational AI"""
    user_id = get_user_id()
    
    try:
        # Rate limiting
        allowed, error_msg = rate_limit_check(user_id, max_requests=30)
        if not allowed:
            return jsonify({'error': error_msg}), 429
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Simple response generation (placeholder)
        response = generate_chat_response(message, user_id)
        
        # Save conversation
        db_manager.save_conversation(user_id, message, response, 'chat')
        
        return jsonify({'response': response})
        
    except Exception as e:
        log_error("chat", e, {'user_id': user_id})
        return jsonify({'error': 'Chat failed. Please try again.'}), 500


def generate_chat_response(message: str, user_id: str) -> str:
    """Generate a response to the user's message"""
    try:
        message_lower = message.lower()
        
        # Simple pattern matching for demo
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm Lysis AI, your text analysis assistant. How can I help you today?"
        
        elif any(word in message_lower for word in ['help', 'what can you do']):
            features = get_available_features()
            available = [k.replace('_', ' ').title() for k, v in features.items() if v]
            return f"I can help you with: {', '.join(available)}. Just paste your text and I'll analyze it!"
        
        elif any(word in message_lower for word in ['math', 'equation', 'solve']):
            if SYMPY_AVAILABLE:
                return "I can help you solve mathematical equations! Just paste your math problem and I'll analyze it."
            else:
                return "Mathematical analysis is not available. Please install SymPy to enable this feature."
        
        elif any(word in message_lower for word in ['sentiment', 'emotion', 'feeling']):
            if TRANSFORMERS_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE:
                return "I can analyze the sentiment of your text! Paste your text and I'll tell you if it's positive, negative, or neutral."
            else:
                return "Sentiment analysis is available with basic functionality. For advanced analysis, install transformers library."
        
        else:
            return "I'm here to help with text analysis! You can ask me to analyze text for sentiment, extract keywords, solve math problems, or just paste any text you'd like me to examine."
    
    except Exception as e:
        log_error("generate_chat_response", e)
        return "I'm sorry, I encountered an error. Please try again."


@app.route('/api/history')
def get_history():
    """Get conversation history for the current user"""
    user_id = get_user_id()
    
    try:
        history = db_manager.get_conversation_history(user_id, limit=20)
        formatted_history = []
        
        for row in history:
            formatted_history.append({
                'message': row[0],
                'response': row[1],
                'timestamp': row[2],
                'analysis_type': row[3],
                'metadata': json.loads(row[4]) if row[4] else None
            })
        
        return jsonify({'history': formatted_history})
        
    except Exception as e:
        log_error("get_history", e, {'user_id': user_id})
        return jsonify({'error': 'Failed to retrieve history'}), 500


@app.route('/api/status')
def get_status():
    """Get system status and available features"""
    return jsonify({
        'status': 'online',
        'features': get_available_features(),
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting Lysis AI server...")
    print(f"📊 Features available: {sum(get_available_features().values())}/{len(get_available_features())}")
    app.run(debug=True, host='0.0.0.0', port=5000)