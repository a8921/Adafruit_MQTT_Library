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
import math
import sympy
from sympy import symbols, solve, Eq

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
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy not available")

# ML imports with better fallback handling
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ sentence-transformers available")
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ sentence-transformers not available: {e}")

try:
    import umap.umap_ as umap
    import hdbscan
    CLUSTERING_AVAILABLE = True
    print("✅ clustering libraries available")
except ImportError as e:
    CLUSTERING_AVAILABLE = False
    print(f"⚠️ umap/hdbscan not available: {e}")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
    print("✅ KeyBERT available")
except ImportError as e:
    KEYBERT_AVAILABLE = False
    print(f"⚠️ KeyBERT not available: {e}")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist, squareform
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn available")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"⚠️ scikit-learn not available: {e}")

try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ spaCy available")
except ImportError as e:
    SPACY_AVAILABLE = False
    print(f"⚠️ spaCy not available: {e}")

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ transformers available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ transformers not available: {e}")

# Configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b-instruct-q4_k_m"

# Database paths
DATABASE_PATH = "output/papers.db"
CONVERSATIONS_DB = "output/conversations.db"
CONTEXT_DB = "output/context.db"

# Use simpler embedding model if advanced ones fail
if SENTENCE_TRANSFORMERS_AVAILABLE:
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
else:
    EMBEDDING_MODEL = None

MAX_CONTEXT_MESSAGES = 10
CONVERSATION_TIMEOUT = 3600
THINKING_STEPS_COUNT = 8
INTELLIGENCE_LEVEL = "EXPERT"

# Global ML models
embedding_model = None
keyword_extractor = None
gap_analyzer = None
summarizer = None
nlp = None
t5_model = None
t5_tokenizer = None

# ===== UNIVERSAL INTENT DETECTION =====
class UniversalIntentDetector:
    def __init__(self):
        self.intent_patterns = {
            'what_is': [
                'what is', 'define', 'explain', 'describe', 'tell me about',
                'what are', 'what does', 'meaning of', 'definition of',
                'concept of', 'understanding of', 'overview of'
            ],
            'how_to_make': [
                'how to make', 'manufacture', 'synthesize', 'create', 'build',
                'produce', 'develop', 'construct', 'fabricate', 'assemble',
                'how do you make', 'process to create', 'steps to make',
                'method to produce', 'technique to build'
            ],
            'comparison': [
                'compare', 'vs', 'versus', 'difference', 'differences',
                'better', 'best', 'worse', 'worst', 'contrast', 'alternative',
                'which is better', 'pros and cons', 'advantages disadvantages',
                'trade off', 'trade-off', 'relative to', 'compared to'
            ],
            'troubleshoot': [
                'not working', 'fix', 'problem', 'issue', 'error', 'bug',
                'broken', 'failed', 'failure', 'trouble', 'difficulty',
                'how to fix', 'solve problem', 'resolve issue', 'debug',
                'what went wrong', 'why is it not working', 'error message'
            ],
            'optimize': [
                'improve', 'optimize', 'enhance', 'better performance',
                'increase efficiency', 'maximize', 'minimize', 'best practice',
                'optimization', 'improvement', 'enhancement', 'upgrade',
                'boost', 'accelerate', 'streamline', 'refine'
            ],
            'applications': [
                'uses', 'applications', 'use cases', 'where used',
                'practical uses', 'real world', 'industry', 'commercial',
                'implementation', 'deployment', 'usage', 'purpose',
                'what can it do', 'capabilities', 'functions'
            ],
            'future_trends': [
                'future', 'trends', 'next', 'upcoming', 'emerging',
                'development', 'evolution', 'advancement', 'innovation',
                'what\'s next', 'future of', 'trending', 'growing',
                'emerging technology', 'next generation', 'upcoming years'
            ],
            'research_gaps': [
                'gaps', 'unsolved', 'open problems', 'challenges',
                'limitations', 'weaknesses', 'deficiencies', 'missing',
                'unresolved', 'unknown', 'uncertain', 'unclear',
                'research needs', 'future work', 'opportunities'
            ],
            'numerical': [
                'solve', 'calculate', 'compute', 'evaluate', 'find',
                'determine', 'measure', 'quantify', 'estimate'
            ]
        }
        
        # Mathematical symbols and patterns for numerical detection
        self.math_symbols = ['+', '-', '*', '/', '^', '=', '(', ')', '[', ']', '{', '}', 
                            'sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'pi', 'e']
        self.math_keywords = ['equation', 'formula', 'solve', 'calculate', 'compute', 
                             'evaluate', 'integral', 'derivative', 'function', 'variable']

    def detect_intent(self, message: str, conversation_history: List[dict] = None) -> Dict[str, Any]:
        """Enhanced universal intent detection with numerical expression support"""
        try:
            message_lower = message.lower().strip()
            
            # ===== NUMERICAL EXPRESSION DETECTION =====
            if self._is_numerical_expression(message):
                return {
                    'primary_intent': 'numerical',
                    'all_intents': ['numerical'],
                    'confidence_scores': {'numerical': 1.0},
                    'is_followup': False,
                    'is_creation_request': False,
                    'context_relevance': 0.0,
                    'complexity_level': 'expert',
                    'requires_thinking': False,
                    'is_casual': False,
                    'expression_type': self._classify_expression(message),
                    'has_math_symbols': True
                }
            
            # ===== UNIVERSAL INTENT DETECTION =====
            intent_scores = {}
            
            for intent, patterns in self.intent_patterns.items():
                score = 0
                for pattern in patterns:
                    if pattern in message_lower:
                        # Weight exact matches higher
                        if f" {pattern} " in f" {message_lower} ":
                            score += 2
                        else:
                            score += 1
                if score > 0:
                    intent_scores[intent] = score
            
            # Determine primary intent
            if intent_scores:
                primary_intent = max(intent_scores, key=intent_scores.get)
                confidence = min(intent_scores[primary_intent] / 3.0, 1.0)
            else:
                primary_intent = 'general_question'
                confidence = 0.5
            
            # Detect follow-up patterns
            followup_indicators = ['also', 'and', 'furthermore', 'additionally',
                                   'moreover', 'besides', 'what about', 'how about', 
                                   'similarly', 'likewise', 'in addition', 'as well']
            is_followup = any(word in message_lower for word in followup_indicators)
            
            # Determine complexity level
            complexity_level = self._determine_complexity(message_lower, primary_intent)
            
            return {
                'primary_intent': primary_intent,
                'all_intents': list(intent_scores.keys()) if intent_scores else [primary_intent],
                'confidence_scores': {intent: min(score / 3.0, 1.0) for intent, score in intent_scores.items()},
                'is_followup': is_followup,
                'is_creation_request': primary_intent in ['how_to_make', 'creation'],
                'context_relevance': 0.9 if primary_intent in ['comparison', 'optimize', 'troubleshoot'] else 0.6,
                'complexity_level': complexity_level,
                'requires_thinking': primary_intent not in ['what_is', 'casual_conversation'],
                'is_casual': False,
                'previous_intent': conversation_history[-1]['intent'] if conversation_history else None,
                'expression_type': None,
                'has_math_symbols': False
            }
            
        except Exception as e:
            log_error("detect_intent", e, {"message": message})
            return {
                'primary_intent': 'general_question',
                'complexity_level': 'medium',
                'is_followup': False,
                'is_creation_request': False,
                'requires_thinking': True,
                'confidence_scores': {'general_question': 0.5}
            }

    def _is_numerical_expression(self, message: str) -> bool:
        """Detect if message contains numerical expressions or mathematical operations"""
        message_lower = message.lower()
        
        # Check for mathematical symbols
        has_symbols = any(symbol in message for symbol in self.math_symbols)
        
        # Check for mathematical keywords
        has_keywords = any(keyword in message_lower for keyword in self.math_keywords)
        
        # Check for equation patterns (e.g., "solve x^2 + 2x - 3 = 0")
        has_equation_pattern = re.search(r'solve\s+[a-zA-Z][^=]*=', message_lower)
        
        # Check for calculation patterns (e.g., "calculate 3+2*5")
        has_calc_pattern = re.search(r'calculate\s+[\d\+\-\*\/\^\(\)\s]+', message_lower)
        
        # Check for variable assignments (e.g., "x = 5 + 3")
        has_assignment = re.search(r'[a-zA-Z]\s*=\s*[\d\+\-\*\/\^\(\)\s]+', message_lower)
        
        return has_symbols or has_keywords or has_equation_pattern or has_calc_pattern or has_assignment

    def _classify_expression(self, message: str) -> str:
        """Classify the type of mathematical expression"""
        message_lower = message.lower()
        
        if 'solve' in message_lower and '=' in message:
            return 'equation_solving'
        elif 'calculate' in message_lower or 'compute' in message_lower:
            return 'arithmetic'
        elif any(func in message_lower for func in ['sin', 'cos', 'tan', 'log', 'ln', 'sqrt']):
            return 'mathematical_function'
        elif 'integral' in message_lower or '∫' in message:
            return 'integration'
        elif 'derivative' in message_lower or 'd/dx' in message:
            return 'differentiation'
        else:
            return 'general_expression'

    def _determine_complexity(self, message: str, intent: str) -> str:
        """Determine the complexity level of the question"""
        complexity_indicators = {
            'expert': ['advanced', 'complex', 'sophisticated', 'detailed analysis', 
                      'comprehensive', 'in-depth', 'technical', 'professional', 
                      'research', 'implementation', 'optimization', 'calculate', 
                      'formula', 'equation', 'methodology', 'quantitative', 
                      'statistical', 'experimental', 'algorithm', 'protocol'],
            'beginner': ['basic', 'simple', 'introduction', 'beginner', 'explain', 
                        'what is', 'how does', 'overview', 'fundamentals', 'basics', 
                        'elementary', 'primer', 'start', 'first time']
        }
        
        message_lower = message.lower()
        expert_score = sum(1 for indicator in complexity_indicators['expert'] if indicator in message_lower)
        beginner_score = sum(1 for indicator in complexity_indicators['beginner'] if indicator in message_lower)
        
        if expert_score > beginner_score:
            return 'expert'
        elif beginner_score > 0:
            return 'beginner'
        else:
            return 'intermediate'

# ===== NUMERICAL EXPRESSION SOLVER =====
class NumericalExpressionSolver:
    def __init__(self):
        self.supported_operations = {
            'arithmetic': ['+', '-', '*', '/', '^', '**'],
            'functions': ['sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'abs'],
            'constants': ['pi', 'e']
        }
    
    def solve_expression(self, expression: str) -> Dict[str, Any]:
        """Solve numerical expressions and equations"""
        try:
            expression = expression.strip()
            
            # Clean and prepare expression
            clean_expr = self._clean_expression(expression)
            
            # Determine expression type
            expr_type = self._classify_expression(clean_expr)
            
            if expr_type == 'equation':
                return self._solve_equation(clean_expr)
            elif expr_type == 'arithmetic':
                return self._evaluate_arithmetic(clean_expr)
            elif expr_type == 'function':
                return self._evaluate_function(clean_expr)
            else:
                return self._evaluate_general(clean_expr)
                
        except Exception as e:
            log_error("solve_expression", e, {"expression": expression})
            return {
                'success': False,
                'error': f"Could not solve expression: {str(e)}",
                'expression': expression,
                'type': 'error'
            }
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and standardize mathematical expression"""
        # Remove common prefixes
        prefixes_to_remove = ['solve', 'calculate', 'compute', 'evaluate', 'find']
        for prefix in prefixes_to_remove:
            if expression.lower().startswith(prefix):
                expression = expression[len(prefix):].strip()
        
        # Replace common mathematical notations
        replacements = {
            '^': '**',  # Power operator
            '×': '*',   # Multiplication symbol
            '÷': '/',   # Division symbol
            'π': 'pi',  # Pi constant
            '∞': 'oo',  # Infinity
            '√': 'sqrt', # Square root
            '∫': 'integrate', # Integral
            '∂': 'diff'  # Partial derivative
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression.strip()
    
    def _classify_expression(self, expression: str) -> str:
        """Classify the type of mathematical expression"""
        if '=' in expression:
            return 'equation'
        elif any(func in expression.lower() for func in ['sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'exp']):
            return 'function'
        elif any(op in expression for op in ['+', '-', '*', '/', '**']):
            return 'arithmetic'
        else:
            return 'general'
    
    def _solve_equation(self, equation: str) -> Dict[str, Any]:
        """Solve mathematical equations"""
        try:
            # Split equation into left and right sides
            if '=' in equation:
                left_side, right_side = equation.split('=', 1)
                left_side = left_side.strip()
                right_side = right_side.strip()
                
                # Move everything to left side
                if right_side != '0':
                    equation = f"({left_side}) - ({right_side})"
                else:
                    equation = left_side
            else:
                equation = equation.strip()
            
            # Extract variables
            variables = self._extract_variables(equation)
            
            if not variables:
                return {
                    'success': False,
                    'error': 'No variables found in equation',
                    'expression': equation,
                    'type': 'equation'
                }
            
            # Create symbolic equation
            x = symbols('x')
            if len(variables) == 1 and 'x' in variables:
                sympy_expr = sympy.sympify(equation)
                solutions = solve(sympy_expr, x)
            else:
                # Handle multiple variables
                sympy_expr = sympy.sympify(equation)
                solutions = solve(sympy_expr)
            
            # Convert solutions to readable format
            readable_solutions = []
            for sol in solutions:
                if hasattr(sol, 'evalf'):
                    readable_solutions.append(str(sol.evalf()))
                else:
                    readable_solutions.append(str(sol))
            
            return {
                'success': True,
                'result': readable_solutions,
                'expression': equation,
                'type': 'equation',
                'variables': variables,
                'solution_count': len(readable_solutions),
                'method': 'symbolic_solving'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Equation solving error: {str(e)}",
                'expression': equation,
                'type': 'equation'
            }
    
    def _evaluate_arithmetic(self, expression: str) -> Dict[str, Any]:
        """Evaluate arithmetic expressions"""
        try:
            # Replace mathematical functions with safe equivalents
            safe_expr = self._make_safe_expression(expression)
            
            # Evaluate using sympy for safety
            result = sympy.sympify(safe_expr).evalf()
            
            return {
                'success': True,
                'result': str(result),
                'expression': expression,
                'type': 'arithmetic',
                'method': 'arithmetic_evaluation',
                'numerical_result': float(result) if result.is_number else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Arithmetic evaluation error: {str(e)}",
                'expression': expression,
                'type': 'arithmetic'
            }
    
    def _evaluate_function(self, expression: str) -> Dict[str, Any]:
        """Evaluate mathematical functions"""
        try:
            # Create safe expression with proper function calls
            safe_expr = self._make_safe_expression(expression)
            
            # Evaluate using sympy
            result = sympy.sympify(safe_expr).evalf()
            
            return {
                'success': True,
                'result': str(result),
                'expression': expression,
                'type': 'function',
                'method': 'function_evaluation',
                'numerical_result': float(result) if result.is_number else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Function evaluation error: {str(e)}",
                'expression': expression,
                'type': 'function'
            }
    
    def _evaluate_general(self, expression: str) -> Dict[str, Any]:
        """Evaluate general mathematical expressions"""
        try:
            safe_expr = self._make_safe_expression(expression)
            result = sympy.sympify(safe_expr).evalf()
            
            return {
                'success': True,
                'result': str(result),
                'expression': expression,
                'type': 'general',
                'method': 'general_evaluation',
                'numerical_result': float(result) if result.is_number else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"General evaluation error: {str(e)}",
                'expression': expression,
                'type': 'general'
            }
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variables from mathematical expression"""
        variables = set()
        
        # Find single letter variables (x, y, z, etc.)
        for char in expression:
            if char.isalpha() and char not in ['e', 'i', 'pi', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt']:
                variables.add(char)
        
        return list(variables)
    
    def _make_safe_expression(self, expression: str) -> str:
        """Make expression safe for evaluation"""
        # Replace common mathematical functions
        replacements = {
            'sqrt': 'sqrt',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'log': 'log',
            'ln': 'log',  # Natural log
            'exp': 'exp',
            'abs': 'Abs'
        }
        
        safe_expr = expression
        for old, new in replacements.items():
            safe_expr = safe_expr.replace(old, new)
        
        return safe_expr

# ===== UNIVERSAL R&D PROMPT GENERATOR =====
class UniversalRDPromptGenerator:
    def __init__(self):
        self.domain_detector = DomainDetector()
        self.data_generator = ResearchDataGenerator()

    def generate_universal_prompt(self, question: str, conversation_history: List[dict],
                                  relevant_papers: List[dict], intent_data: Dict[str, Any]) -> str:
        """Generate universal R&D prompt for any technical/scientific/engineering question"""

        detected_domain = self.domain_detector.detect_domain(question)
        complexity_level = self.domain_detector.detect_complexity_level(question)
        primary_intent = intent_data.get('primary_intent', 'question')

        # Extract topic from question
        topic = self._extract_topic(question)

        # Get relevant research data
        domain_data = self.data_generator.get_domain_data(detected_domain, topic)

        # Build conversation context
        history_context = ""
        if conversation_history:
            history_context = "\n".join([
                f"Previous Q: {msg['message'][:100]}...\nPrevious A: {msg['response'][:100]}..."
                for msg in conversation_history[-2:]
            ])

        # Build paper context
        paper_context = ""
        paper_count = len(relevant_papers)
        if relevant_papers:
            paper_context = f"## Relevant Research Papers ({paper_count} papers found)\n\n"
            for i, paper in enumerate(relevant_papers, 1):
                paper_context += f"**Paper {i}**: {paper['title']}\n"
                paper_context += f"**Domain**: {paper.get('domain', 'Unknown')}\n"
                paper_context += f"**Abstract**: {paper.get('abstract', '')[:200]}...\n\n"

        # Generate universal R&D prompt
        prompt = f"""You are the world's most advanced R&D assistant. The user asked:
"{question}"

Intent: {primary_intent}
Relevant Papers:
{paper_context}

Respond in this structure:

## Direct Answer
Provide a clear, concise answer to the user's question with key insights and findings.

## Academic Research Foundation
- Summarize relevant academic research and theoretical background
- Cite key studies and methodologies
- Explain fundamental principles and concepts

## Real-World Implementation
- Describe practical applications and use cases
- Provide implementation guidelines and best practices
- Include technical specifications and requirements

## Innovation Insights
- Identify current trends and emerging technologies
- Highlight breakthrough developments and innovations
- Discuss future directions and opportunities

## Actionable Guidance
- Provide specific recommendations and next steps
- Include risk assessment and mitigation strategies
- Offer optimization and improvement suggestions

## References
- List key research papers and sources
- Include industry standards and guidelines
- Provide additional reading materials

**CRITICAL**: Ensure comprehensive, research-grade quality with specific data, technical details, and actionable insights."""

        return prompt

    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question"""
        question_lower = question.lower()

        # Common technical topics
        topics = [
            'solar cell', 'battery', 'photovoltaic', 'fuel cell', 'wind energy',
            'machine learning', 'artificial intelligence', 'quantum computing',
            'drug delivery', 'medical device', 'biotechnology', 'gene therapy',
            'renewable energy', 'energy storage', 'smart grid', 'carbon capture',
            'market analysis', 'investment strategy', 'business model', 'supply chain',
            'psychology', 'education', 'social media', 'public policy',
            'literature', 'philosophy', 'history', 'art', 'music'
        ]

        for topic in topics:
            if topic in question_lower:
                return topic

        # Extract meaningful nouns
        words = question_lower.split()
        meaningful_words = [w for w in words if len(w) > 3 and w not in [
            'what', 'how', 'why', 'when', 'where', 'which', 'explain', 'describe', 'analyze'
        ]]

        return meaningful_words[0] if meaningful_words else 'technology'

# Database Helper Functions
def init_databases():
    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CONVERSATIONS_DB), exist_ok=True)

        print(f"📁 Initializing database system...")
        print(f"   Papers DB: {DATABASE_PATH}")
        print(f"   Conversations DB: {CONVERSATIONS_DB}")

        # Initialize conversations database
        conn = sqlite3.connect(CONVERSATIONS_DB)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, 
            session_id TEXT, 
            message TEXT, 
            response TEXT, 
            timestamp DATETIME, 
            intent TEXT, 
            context_used TEXT, 
            followup_detected BOOLEAN, 
            thinking_steps TEXT
        )''')
        conn.commit()
        conn.close()
        print("✅ Conversations database initialized")

        # Check/initialize papers database
        if os.path.exists(DATABASE_PATH):
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            conn.close()
            print(f"✅ Papers database loaded: {paper_count} papers available")
        else:
            print(f"⚠️ Papers database not found at: {DATABASE_PATH}")
            print("💡 Creating sample database for demonstration...")

            # Create sample database
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                domain TEXT,
                subdomain TEXT,
                publication_date TEXT,
                doi TEXT,
                keywords TEXT,
                relevance_score REAL,
                quality_score REAL,
                file_path TEXT,
                processing_date TEXT,
                properties TEXT,
                equations TEXT
            )''')

            # Add sample papers
            sample_papers = [
                ('sample_tech_1', 'Advanced Technology Research',
                 'Comprehensive analysis of emerging technologies and their applications.',
                 '["Dr. Research", "Prof. Technology"]', 'Science & Technology', 'General', '2024-01-15',
                 '10.1000/tech2024', '["technology", "research", "innovation"]', 0.95, 8.8,
                 'tech_research.pdf', '2024-01-15T10:00:00',
                 '{"efficiency": "85%", "cost": "$1000", "performance": "excellent"}',
                 '["E = mc²", "F = ma"]')
            ]

            cursor.executemany('''INSERT INTO papers 
                (id, title, abstract, authors, domain, subdomain, publication_date, doi, 
                 keywords, relevance_score, quality_score, file_path, processing_date, properties, equations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', sample_papers)
            conn.commit()
            conn.close()
            print("✅ Sample database created")

        print("✅ Database system initialization complete")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        traceback.print_exc()

def get_conversation_history(session_id: str, limit: int = MAX_CONTEXT_MESSAGES) -> List[Dict]:
    try:
        conn = sqlite3.connect(CONVERSATIONS_DB)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT message, response, intent, followup_detected, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?',
            (session_id, limit)
        )
        results = cursor.fetchall()
        conn.close()
        history = [{'message': r[0], 'response': r[1], 'intent': r[2], 'followup_detected': bool(
            r[3]), 'timestamp': r[4]} for r in reversed(results)]
        return history
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def save_conversation(conv_msg: ConversationMessage):
    try:
        conn = sqlite3.connect(CONVERSATIONS_DB)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO conversations (id, session_id, message, response, timestamp, intent, context_used, followup_detected, thinking_steps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (conv_msg.id, conv_msg.session_id, conv_msg.message, conv_msg.response, conv_msg.timestamp,
             conv_msg.intent, json.dumps(conv_msg.context_used), conv_msg.followup_detected,
             json.dumps(conv_msg.thinking_steps))
        )
        conn.commit()
        conn.close()
        print(f"💾 Conversation saved: {conv_msg.id}")
    except Exception as e:
        print(f"Error saving conversation: {e}")

def search_papers(query: str, limit: int = 5) -> List[Dict]:
    try:
        print(f"🔍 Searching papers database with query: '{query}'")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]

        if not query_terms:
            cursor.execute("""
                SELECT id, title, abstract, authors, domain, subdomain, 
                       keywords, quality_score, file_path, properties, equations 
                FROM papers ORDER BY quality_score DESC LIMIT ?
            """, (limit,))
        else:
            conditions = []
            params = []
            for term in query_terms:
                like_term = f"%{term}%"
                conditions.append("""(
                    title LIKE ? OR 
                    abstract LIKE ? OR 
                    domain LIKE ? OR 
                    keywords LIKE ?
                )""")
                params.extend([like_term] * 4)

            where_clause = " OR ".join(conditions)
            params.append(limit)

            query_sql = f"""
                SELECT id, title, abstract, authors, domain, subdomain, 
                       keywords, quality_score, file_path, properties, equations
                FROM papers 
                WHERE {where_clause}
                ORDER BY quality_score DESC
                LIMIT ?
            """
            cursor.execute(query_sql, params)

        results = cursor.fetchall()
        conn.close()

        papers = []
        for r in results:
            try:
                keywords = json.loads(r[6]) if r[6] else []
            except:
                keywords = r[6].split(',') if r[6] else []

            try:
                authors = json.loads(r[3]) if r[3] else []
            except:
                authors = [r[3]] if r[3] else []

            paper = {
                'id': r[0], 'title': r[1], 'abstract': r[2] or '',
                'authors': authors, 'domain': r[4] or 'Unknown',
                'subdomain': r[5] or 'General', 'keywords': keywords,
                'quality_score': r[7], 'file_path': r[8],
                'content': r[2] or '', 'categories': r[4] or 'Unknown'
            }
            papers.append(paper)

        print(f"📄 Found {len(papers)} relevant papers")
        return papers
    except Exception as e:
        print(f"Error searching papers: {e}")
        return []

def search_papers_semantic(query: str, limit: int = 5) -> List[Dict]:
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not NUMPY_AVAILABLE:
        return search_papers(query, limit)

    try:
        print(f"🧠 Using semantic search for: '{query}'")
        global embedding_model

        if embedding_model is None:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, abstract, authors, domain, subdomain, 
                   keywords, quality_score, file_path, properties, equations 
            FROM papers
        """)

        papers = []
        for r in cursor.fetchall():
            try:
                keywords = json.loads(r[6]) if r[6] else []
            except:
                keywords = r[6].split(',') if r[6] else []

            papers.append({
                'id': r[0], 'title': r[1], 'abstract': r[2] or '',
                'authors': json.loads(r[3]) if r[3] else [], 'domain': r[4] or 'Unknown',
                'subdomain': r[5] or 'General', 'keywords': keywords,
                'quality_score': r[7], 'file_path': r[8],
                'content': f"{r[1]} {r[2] or ''} {' '.join(keywords)}"
            })
        conn.close()

        if not papers:
            return []

        query_embedding = embedding_model.encode([query])[0]
        paper_texts = [paper['content'] for paper in papers]
        paper_embeddings = embedding_model.encode(paper_texts)

        similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:limit]

        relevant_papers = []
        for i in top_indices:
            if similarities[i] > 0.1:
                paper = papers[i].copy()
                paper['similarity_score'] = float(similarities[i])
                relevant_papers.append(paper)

        return relevant_papers if relevant_papers else search_papers(query, limit)
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return search_papers(query, limit)

# Data Classes
@dataclass
class ConversationMessage:
    id: str
    session_id: str
    message: str
    response: str
    timestamp: datetime
    intent: str
    context_used: List[str]
    followup_detected: bool
    thinking_steps: List[str]

# Legacy classes for compatibility
class DomainDetector:
    def __init__(self):
        self.domain_keywords = {
            'Science & Technology': ['physics', 'chemistry', 'biology', 'mathematics', 'computer science'],
            'Medicine & Health': ['medical', 'healthcare', 'pharmaceutical', 'clinical', 'biotechnology'],
            'Energy & Environment': ['renewable', 'battery', 'solar', 'wind', 'climate', 'environmental'],
            'Business & Economics': ['finance', 'marketing', 'management', 'economics', 'business'],
            'Social Sciences': ['psychology', 'sociology', 'anthropology', 'political', 'education'],
            'Arts & Humanities': ['literature', 'philosophy', 'history', 'art', 'music'],
            'Emerging Fields': ['artificial intelligence', 'machine learning', 'robotics', 'blockchain']
        }

    def detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'Science & Technology'

    def detect_complexity_level(self, text: str) -> str:
        text_lower = text.lower()
        expert_indicators = ['advanced', 'complex', 'sophisticated', 'detailed analysis']
        beginner_indicators = ['basic', 'simple', 'introduction', 'beginner']
        
        expert_score = sum(1 for indicator in expert_indicators if indicator in text_lower)
        beginner_score = sum(1 for indicator in beginner_indicators if indicator in text_lower)
        
        if expert_score > beginner_score:
            return 'expert'
        elif beginner_score > 0:
            return 'beginner'
        else:
            return 'intermediate'

class ResearchDataGenerator:
    def __init__(self):
        self.domain_data = {
            'Science & Technology': {
                'solar_cell': {
                    'materials': [
                        {'Material': 'Crystalline Silicon', 'Bandgap (eV)': '1.12', 'Efficiency (Lab, %)': '26.7'},
                        {'Material': 'Perovskite', 'Bandgap (eV)': '1.5-2.3', 'Efficiency (Lab, %)': '25.2'}
                    ]
                }
            }
        }

    def get_domain_data(self, domain: str, topic: str = None) -> Dict[str, Any]:
        if domain in self.domain_data:
            return self.domain_data[domain]
        return {}

# ===== ENHANCED FLASK ROUTE LOGIC =====
@app.route('/ask', methods=['POST'])
def ask():
    try:
        print("\n" + "="*80)
        print("🔬 UNIVERSAL R&D AI ASSISTANT REQUEST")
        print("="*80)

        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': 'No question provided'}), 400

        question = data['question'].strip()
        if not question:
            return jsonify({'success': False, 'error': 'Empty question provided'}), 400

        print(f"💬 Question: {question}")

        # Session management
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        print(f"🔑 Session ID: {session_id}")

        # Get conversation context
        conversation_history = get_conversation_history(session_id)
        print(f"📚 Conversation history: {len(conversation_history)} messages")

        # ===== UNIVERSAL INTENT DETECTION =====
        intent_detector = UniversalIntentDetector()
        intent_data = intent_detector.detect_intent(question, conversation_history)
        print(f"🎯 Intent detected: {intent_data.get('primary_intent', 'unknown')}")

        # ===== NUMERICAL EXPRESSION SOLVING =====
        if intent_data.get('primary_intent') == 'numerical':
            print("🧮 Processing numerical expression...")
            numerical_solver = NumericalExpressionSolver()
            result = numerical_solver.solve_expression(question)
            
            if result['success']:
                answer = f"""## Numerical Solution

**Expression**: {result['expression']}
**Type**: {result['type']}
**Method**: {result['method']}

### Result
{result['result']}

### Analysis
- **Expression Type**: {result['type']}
- **Solution Method**: {result['method']}
- **Numerical Value**: {result.get('numerical_result', 'N/A')}
- **Variables**: {result.get('variables', [])}
- **Solution Count**: {result.get('solution_count', 1)}

### Technical Details
This solution was computed using advanced mathematical libraries (SymPy) for accuracy and reliability."""
            else:
                answer = f"""## Numerical Expression Error

**Expression**: {result['expression']}
**Error**: {result['error']}

### Suggestions
- Check for syntax errors in mathematical notation
- Ensure all variables are properly defined
- Verify that the expression is mathematically valid
- Try simplifying complex expressions"""
            
            # Save conversation
            conv_msg = ConversationMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                message=question,
                response=answer,
                timestamp=datetime.now(),
                intent='numerical',
                context_used=[],
                followup_detected=False,
                thinking_steps=['detected_numerical_intent', 'parsed_expression', 'solved_mathematically', 'formatted_result']
            )
            save_conversation(conv_msg)

            return jsonify({
                'success': True,
                'question': question,
                'answer': answer,
                'intent': 'numerical',
                'thinking_steps': ['detected_numerical_intent', 'parsed_expression', 'solved_mathematically', 'formatted_result'],
                'debug': {
                    'intent_detected': 'numerical',
                    'expression_type': result.get('type'),
                    'solution_method': result.get('method'),
                    'papers_used': 0,
                    'domain_detected': 'mathematics'
                },
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            })

        # ===== STANDARD R&D PROCESSING =====
        print("🔬 Processing R&D question with universal AI...")
        
        # Generate thinking steps
        thinking_steps = [
            'detected_intent',
            'searched_research_database',
            'generated_universal_prompt',
            'sent_to_llm',
            'formatted_response'
        ]

        # Search relevant papers
        print("🔍 Searching research papers database...")
        try:
            relevant_papers = search_papers_semantic(question) if SENTENCE_TRANSFORMERS_AVAILABLE else search_papers(question)
            print(f"📄 Found {len(relevant_papers)} relevant papers")
        except Exception as search_error:
            print(f"❌ Paper search error: {search_error}")
            relevant_papers = []

        # Generate universal R&D prompt
        prompt_generator = UniversalRDPromptGenerator()
        universal_prompt = prompt_generator.generate_universal_prompt(
            question, conversation_history, relevant_papers, intent_data
        )

        print(f"🚀 Sending universal R&D request to Ollama...")
        print(f"📋 Papers found: {len(relevant_papers)}")
        print(f"🎯 Intent: {intent_data.get('primary_intent')}")

        # Send to Ollama
        payload = {
            "model": MODEL_NAME,
            "prompt": universal_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4000,
                "top_p": 0.8,
                "top_k": 25
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=300)

        if response.status_code == 200:
            response_data = response.json()
            answer = response_data.get('response', 'No response generated')

            # Enhanced post-processing
            if relevant_papers:
                answer += f"\n\n## Research Sources\n"
                answer += f"Based on analysis of {len(relevant_papers)} research papers:\n"
                for i, paper in enumerate(relevant_papers[:3], 1):
                    answer += f"{i}. {paper['title']} ({paper.get('domain', 'Unknown')})\n"

            print("✅ Universal R&D response generated successfully")
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get('error', error_msg)
            except:
                pass
            answer = f"Error generating response: {error_msg}"

        # Save conversation
        conv_msg = ConversationMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            message=question,
            response=answer,
            timestamp=datetime.now(),
            intent=intent_data.get('primary_intent'),
            context_used=[p['id'] for p in relevant_papers],
            followup_detected=intent_data.get('is_followup', False),
            thinking_steps=thinking_steps
        )
        save_conversation(conv_msg)

        print("✅ Universal R&D AI request completed successfully")
        print("="*80)

        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'thinking_steps': thinking_steps,
            'debug': {
                'intent_detected': intent_data.get('primary_intent'),
                'domain_detected': prompt_generator.domain_detector.detect_domain(question),
                'papers_used': len(relevant_papers),
                'complexity_level': intent_data.get('complexity_level'),
                'confidence_scores': intent_data.get('confidence_scores', {})
            },
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        })

    except Exception as e:
        print(f"❌ Critical error in universal R&D endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

# API Endpoints
@app.route('/api/status')
def api_status():
    try:
        ollama_connected = False
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            ollama_connected = response.status_code == 200
        except:
            pass

        database_size = 0
        try:
            if os.path.exists(DATABASE_PATH):
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                database_size = cursor.fetchone()[0]
                conn.close()
        except:
            pass

        return jsonify({
            'success': True,
            'system_status': 'online' if ollama_connected else 'partial',
            'ollama_connected': ollama_connected,
            'database_size': database_size,
            'universal_features': {
                'intent_detection': True,
                'numerical_solving': True,
                'semantic_search': SENTENCE_TRANSFORMERS_AVAILABLE,
                'research_grade_responses': True
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

# Main Execution
if __name__ == '__main__':
    print("🚀 Starting UNIVERSAL R&D AI ASSISTANT")
    print("=" * 90)

    # Initialize databases
    print("📂 Initializing database system...")
    init_databases()

    # Check Ollama status
    print("🔗 Checking Ollama AI service...")
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            models = test_response.json().get('models', [])
            model_names = [m.get('name', 'unknown') for m in models]
            print(f"✅ Ollama connected. Available models: {model_names}")
            if MODEL_NAME in model_names:
                print(f"✅ Target model {MODEL_NAME} is available")
            else:
                print(f"⚠️ Target model {MODEL_NAME} not found. Run: ollama pull {MODEL_NAME}")
        else:
            print(f"⚠️ Ollama responded with status {test_response.status_code}")
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        print("💡 Start Ollama with: ollama serve")

    # Display Universal R&D AI features
    print("\n🎯 UNIVERSAL R&D AI FEATURES ACTIVE:")
    print("   ✅ Universal Intent Detection (what_is, how_to_make, comparison, troubleshoot, optimize, applications, future_trends, research_gaps, numerical)")
    print("   ✅ Numerical Expression Solver with SymPy integration")
    print("   ✅ Universal R&D Prompt Generator")
    print("   ✅ Enhanced Flask Route Logic with Intent-Based Processing")
    print("   ✅ Research-Grade Response Generation")
    print("   ✅ Multi-Domain Support")
    print("   ✅ Semantic Search with Embeddings")
    print("   ✅ Comprehensive Debug Information")
    print("   ✅ Thinking Steps Tracking")
    print("   ✅ Quality Assessment and Reliability Scoring")

    print("\n🌐 Starting Flask server...")
    print("📋 Available API endpoints:")
    print("   • / - Main Universal R&D AI interface")
    print("   • /ask - Universal R&D AI endpoint with intent detection")
    print("   • /api/status - Comprehensive system status")

    print(f"\n🔗 Access Universal R&D AI at: http://127.0.0.1:5000")
    print("=" * 90)

    app.run(host='0.0.0.0', port=5000, debug=False)