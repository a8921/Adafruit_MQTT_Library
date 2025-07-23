# main.py - Research-to-Product Intelligence Platform with Enhanced Implementation Features
# Integrated with updated process_papers_fixed.py for comprehensive implementation guidance

from flask import Flask, render_template, request, jsonify, session
import requests
import os
import sqlite3
import re
import json
import time
import threading
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple, Union
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b-instruct-q4_k_m"
DATABASE_PATH = "data/database/papers.db"
CONVERSATIONS_DB = "data/database/conversations.db"

# System Configuration
MAX_CONTEXT_MESSAGES = 10
PERFORMANCE_THRESHOLD = 5.0
OPTIMIZATION_GROWTH_THRESHOLD = 0.1

@dataclass
class ResearchQuery:
    """Data class for research query tracking with implementation focus"""
    id: str
    session_id: str
    question: str
    intent: str
    domain: str
    timestamp: datetime
    papers_found: int
    search_time: float
    response: str
    implementation_guidance_count: int = 0
    supplier_suggestions_count: int = 0
    cost_recommendations_count: int = 0

class DynamicDatabaseManager:
    """Enhanced database manager with implementation intelligence support"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.current_size = 0
        self.last_optimization = None
        self.optimization_history = []
        self.performance_stats = {
            'avg_search_time': 0,
            'total_searches': 0,
            'optimization_count': 0
        }
        self.initialize_database()

    def initialize_database(self):
        """Initialize database with enhanced schema for implementation data"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.current_size = self.get_current_size()
            self.create_enhanced_schema()
            self.create_dynamic_indexes()
            self.create_performance_table()
            print(f"📊 Enhanced Database Manager initialized: {self.current_size:,} papers")
        except Exception as e:
            print(f"⚠️ Database initialization error: {e}")
            self.current_size = 0

    def create_enhanced_schema(self):
        """Create enhanced database schema with implementation fields"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Enhanced papers table with all implementation fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    domain TEXT,
                    subdomain TEXT,
                    keywords TEXT,
                    materials TEXT,
                    processes TEXT,
                    experimental_params TEXT,
                    performance_metrics TEXT,
                    limitations TEXT,
                    future_scope TEXT,
                    suppliers TEXT,
                    implementation_guidance TEXT,
                    supplier_suggestions TEXT,
                    cost_recommendations TEXT,
                    content_preview TEXT,
                    full_content_path TEXT,
                    technical_info_path TEXT,
                    dynamic_category TEXT,
                    file_path TEXT,
                    processed_date TEXT,
                    journal_type TEXT,
                    word_count INTEGER,
                    file_size INTEGER,
                    content_hash TEXT,
                    topic_keywords TEXT,
                    category_confidence REAL,
                    processing_time_ms INTEGER,
                    extraction_quality REAL,
                    technical_depth REAL,
                    llama_processing_time REAL
                )
            ''')

            conn.commit()
            conn.close()
            print("✅ Enhanced schema created/updated with implementation fields")
        except Exception as e:
            print(f"Error creating enhanced schema: {e}")

    def get_current_size(self) -> int:
        """Get current database size"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            size = cursor.fetchone()[0] if cursor.fetchone() else 0
            conn.close()
            return size
        except Exception as e:
            print(f"Error getting database size: {e}")
            return 0

    def create_dynamic_indexes(self):
        """Create optimized indexes for enhanced search performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Core search indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_title ON papers(title)",
                "CREATE INDEX IF NOT EXISTS idx_domain ON papers(domain)",
                "CREATE INDEX IF NOT EXISTS idx_category ON papers(dynamic_category)",
                "CREATE INDEX IF NOT EXISTS idx_content ON papers(content_preview)",
                "CREATE INDEX IF NOT EXISTS idx_implementation ON papers(implementation_guidance)",
                "CREATE INDEX IF NOT EXISTS idx_suppliers ON papers(supplier_suggestions)",
                "CREATE INDEX IF NOT EXISTS idx_materials ON papers(materials)",
                "CREATE INDEX IF NOT EXISTS idx_costs ON papers(cost_recommendations)"
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            conn.commit()
            conn.close()
            print("✅ Dynamic indexes created/updated")
        except Exception as e:
            print(f"Error creating indexes: {e}")

    def create_performance_table(self):
        """Create table to track performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    database_size INTEGER,
                    search_time REAL,
                    papers_found INTEGER,
                    query_type TEXT,
                    optimization_level TEXT
                )
            ''')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error creating performance table: {e}")

    def check_and_optimize(self) -> bool:
        """Check if optimization is needed and perform it"""
        try:
            new_size = self.get_current_size()
            growth_ratio = (new_size - self.current_size) / max(self.current_size, 1)

            if growth_ratio > OPTIMIZATION_GROWTH_THRESHOLD:
                print(f"📈 Database growth detected: {self.current_size:,} → {new_size:,} ({growth_ratio:.1%})")
                self.optimize_for_size(new_size)
                self.current_size = new_size
                return True

            return False

        except Exception as e:
            print(f"Error in optimization check: {e}")
            return False

    def optimize_for_size(self, new_size: int):
        """Optimize database for current size"""
        try:
            print(f"🔧 Optimizing database for {new_size:,} papers...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("REINDEX")
            cursor.execute("ANALYZE")
            cursor.execute("PRAGMA optimize")
            
            conn.commit()
            conn.close()

            self.create_dynamic_indexes()
            
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'old_size': self.current_size,
                'new_size': new_size,
                'optimization_type': 'growth_triggered'
            })

            self.performance_stats['optimization_count'] += 1
            print("✅ Database optimization completed")

        except Exception as e:
            print(f"Error during optimization: {e}")

    def log_search_performance(self, search_time: float, papers_found: int, query_type: str):
        """Log search performance for monitoring"""
        try:
            self.performance_stats['total_searches'] += 1
            
            total = self.performance_stats['total_searches']
            current_avg = self.performance_stats['avg_search_time']
            self.performance_stats['avg_search_time'] = (current_avg * (total - 1) + search_time) / total

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, database_size, search_time, papers_found, query_type, optimization_level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                self.current_size,
                search_time,
                papers_found,
                query_type,
                self.get_optimization_level()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error logging performance: {e}")

    def get_optimization_level(self) -> str:
        """Determine current optimization level based on database size"""
        if self.current_size < 100000:
            return "basic"
        elif self.current_size < 500000:
            return "enhanced"
        else:
            return "advanced"

class AdaptiveSearchEngine:
    """Enhanced search engine with implementation intelligence"""

    def __init__(self, db_manager: DynamicDatabaseManager):
        self.db_manager = db_manager
        self.search_cache = {}

    def smart_search_with_implementation(self, query: str, query_analysis: Dict) -> List[Dict]:
        """Enhanced search that prioritizes implementation data"""
        start_time = time.time()

        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            search_terms = self._extract_search_terms(query)
            
            # Enhanced search query prioritizing implementation fields
            search_query = """
                SELECT id, title, domain, subdomain, content_preview, materials, processes,
                       implementation_guidance, supplier_suggestions, cost_recommendations,
                       experimental_params, performance_metrics, limitations, future_scope,
                       word_count, extraction_quality,
                       (CASE 
                        WHEN LOWER(title) LIKE ? THEN 20
                        WHEN LOWER(implementation_guidance) LIKE ? THEN 15
                        WHEN LOWER(supplier_suggestions) LIKE ? THEN 12
                        WHEN LOWER(content_preview) LIKE ? THEN 10
                        WHEN LOWER(materials) LIKE ? THEN 8
                        WHEN LOWER(processes) LIKE ? THEN 6
                        ELSE 1 END) as relevance_score
                FROM papers 
                WHERE LOWER(title) LIKE ? OR LOWER(content_preview) LIKE ? 
                   OR LOWER(implementation_guidance) LIKE ? OR LOWER(materials) LIKE ?
                   OR LOWER(supplier_suggestions) LIKE ? OR LOWER(processes) LIKE ?
                ORDER BY relevance_score DESC, extraction_quality DESC
                LIMIT 50
            """

            main_term = f"%{search_terms[0]}%" if search_terms else f"%{query.lower()}%"
            cursor.execute(search_query, [main_term] * 12)

            results = self._process_enhanced_results(cursor.fetchall())
            conn.close()

            search_time = time.time() - start_time
            self.db_manager.log_search_performance(
                search_time,
                len(results),
                query_analysis.get('intent', 'unknown')
            )

            print(f"🔍 Enhanced search completed: {len(results)} papers found in {search_time:.2f}s")
            return results

        except Exception as e:
            print(f"Enhanced search error: {e}")
            return []

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract relevant search terms from query"""
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords[:5]

    def _process_enhanced_results(self, raw_results: List[Tuple]) -> List[Dict]:
        """Process raw database results with implementation data"""
        processed = []

        for row in raw_results:
            try:
                paper = {
                    'id': row[0],
                    'title': row[1] or 'Unknown Title',
                    'domain': row[2] or 'Unknown',
                    'subdomain': row[3] or 'Unknown',
                    'content': row[4][:2000] if row[4] else '',
                    'materials': self._parse_json_field(row[5]) if row[5] else [],
                    'processes': self._parse_json_field(row[6]) if row[6] else [],
                    'implementation_guidance': row[7] or '',
                    'supplier_suggestions': self._parse_json_field(row[8]) if row[8] else [],
                    'cost_recommendations': self._parse_json_field(row[9]) if row[9] else [],
                    'experimental_params': self._parse_json_field(row[10]) if row[10] else [],
                    'performance_metrics': self._parse_json_field(row[11]) if row[11] else [],
                    'limitations': row[12] or '',
                    'future_scope': row[13] or '',
                    'word_count': row[14] or 0,
                    'extraction_quality': row[15] or 0,
                    'relevance_score': row[16] if len(row) > 16 else 1
                }
                processed.append(paper)
            except Exception as e:
                print(f"Error processing result: {e}")
                continue

        return processed

    def _parse_json_field(self, field_value: str) -> List:
        """Safely parse JSON field values"""
        try:
            if field_value and field_value.strip():
                return json.loads(field_value)
        except json.JSONDecodeError:
            pass
        return []

class DomainDiscoverySystem:
    """Automatically discovers research domains with implementation focus"""

    def __init__(self, db_manager: DynamicDatabaseManager):
        self.db_manager = db_manager
        self.known_domains = set()
        self.domain_keywords = {}

    def discover_new_domains(self) -> List[str]:
        """Discover new research domains in the database"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT DISTINCT domain FROM papers WHERE domain IS NOT NULL AND domain != ''")
            current_domains = {row[0] for row in cursor.fetchall()}

            new_domains = current_domains - self.known_domains

            if new_domains:
                print(f"🆕 New research domains discovered: {new_domains}")
                for domain in new_domains:
                    self.analyze_domain_characteristics(domain)
                    self.known_domains.add(domain)

            conn.close()
            return list(new_domains)

        except Exception as e:
            print(f"Error in domain discovery: {e}")
            return []

    def analyze_domain_characteristics(self, domain: str):
        """Analyze characteristics of a new domain"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT title, content_preview, implementation_guidance FROM papers 
                WHERE domain = ? 
                LIMIT 10
            """, (domain,))

            samples = cursor.fetchall()
            
            all_text = ' '.join([
                f"{title} {content[:500]} {guidance[:200]}"
                for title, content, guidance in samples
                if title and content
            ])
            keywords = self._extract_domain_keywords(all_text)

            self.domain_keywords[domain] = {
                'keywords': keywords,
                'sample_count': len(samples),
                'discovered_at': datetime.now()
            }

            print(f"📊 Domain '{domain}' analyzed: {len(keywords)} key terms identified")
            conn.close()

        except Exception as e:
            print(f"Error analyzing domain {domain}: {e}")

    def _extract_domain_keywords(self, text: str) -> List[str]:
        """Extract key terms from domain text"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were'
        }
        
        relevant_words = [
            word for word, count in word_counts.most_common(50)
            if len(word) > 3 and word not in stop_words and count > 2
        ]
        
        return relevant_words[:20]

class LLaMAIntegration:
    """Enhanced LLaMA integration for implementation intelligence"""

    def __init__(self):
        self.model_name = MODEL_NAME
        self.ollama_url = OLLAMA_URL

    def understand_query(self, user_query: str) -> Dict[str, Any]:
        """Use LLaMA to understand user query intent with implementation focus"""
        try:
            prompt = f"""
            Analyze this research query for implementation intelligence:
            
            Query: "{user_query}"
            
            Extract structured information focusing on practical implementation:
            1. Research domain/field
            2. Intent (research, implementation, comparison, suppliers, costs, analysis)
            3. Key search terms (5-10 important keywords)
            4. Implementation focus (yes/no - does user want practical guidance?)
            5. Commercial interest (yes/no - looking for suppliers/costs?)
            6. Output format preference
            
            Respond with ONLY valid JSON:
            {{
                "domain": "research domain",
                "intent": "primary intent",
                "search_terms": ["term1", "term2", "term3"],
                "implementation_focus": true,
                "commercial_interest": false,
                "output_format": "preferred format",
                "complexity": "basic"
            }}
            """

            response = self._call_ollama(prompt)
            return self._parse_llama_response(response, self._fallback_query_analysis(user_query))

        except Exception as e:
            print(f"Error in query understanding: {e}")
            return self._fallback_query_analysis(user_query)

    def generate_unified_implementation_plan(self, papers: List[Dict], query_analysis: Dict) -> Dict[str, Any]:
        """Generate unified implementation plan from multiple papers using LLaMA"""
        try:
            if not papers:
                return {"error": "No papers to analyze"}

            papers_content = self._prepare_papers_for_llama(papers[:10])

            prompt = f"""
            Create a unified implementation plan from these research papers:
            
            Research Intent: {query_analysis.get('intent', 'analysis')}
            Domain: {query_analysis.get('domain', 'general')}
            Implementation Focus: {query_analysis.get('implementation_focus', False)}
            Commercial Interest: {query_analysis.get('commercial_interest', False)}
            
            Papers with implementation data:
            {papers_content}
            
            Generate a comprehensive implementation plan combining insights from all papers:
            
            Respond with structured JSON:
            {{
                "executive_summary": "brief overview of implementation opportunity",
                "combined_materials": ["material1", "material2", "material3"],
                "unified_processes": ["process1", "process2", "process3"],
                "implementation_steps": ["step1", "step2", "step3"],
                "supplier_recommendations": ["supplier1", "supplier2", "supplier3"],
                "cost_analysis": ["cost1", "cost2", "cost3"],
                "performance_expectations": ["metric1", "metric2", "metric3"],
                "risk_mitigation": ["risk1", "risk2", "risk3"],
                "commercial_viability": "assessment of market potential",
                "timeline_estimate": "estimated implementation timeline",
                "next_steps": ["action1", "action2", "action3"],
                "papers_analyzed": {len(papers)}
            }}
            """

            response = self._call_ollama(prompt)
            return self._parse_llama_response(response, self._fallback_implementation_plan(papers, query_analysis))

        except Exception as e:
            print(f"Error generating unified implementation plan: {e}")
            return self._fallback_implementation_plan(papers, query_analysis)

    def generate_research_report(self, analysis_data: Dict, query_analysis: Dict, papers: List[Dict]) -> str:
        """Generate comprehensive research report with implementation intelligence"""
        try:
            implementation_data = self._extract_implementation_summary(papers)

            prompt = f"""
            Generate a comprehensive research-to-product implementation report:
            
            Query Context:
            - Intent: {query_analysis.get('intent', 'analysis')}
            - Domain: {query_analysis.get('domain', 'general')}
            - Implementation Focus: {query_analysis.get('implementation_focus', False)}
            - Commercial Interest: {query_analysis.get('commercial_interest', False)}
            
            Analysis Results:
            {json.dumps(analysis_data, indent=2)}
            
            Implementation Summary:
            {json.dumps(implementation_data, indent=2)}
            
            Papers Analyzed: {len(papers)}
            
            Generate a professional report with these sections:
            
            # Research-to-Product Implementation Report
            
            ## Executive Summary
            [Brief overview with key implementation insights]
            
            ## Research Findings
            [Key technical findings from papers]
            
            ## Implementation Roadmap
            [Step-by-step practical implementation guidance]
            
            ## Materials & Processes
            [Required materials and manufacturing processes]
            
            ## Supplier Ecosystem
            [Recommended suppliers and equipment manufacturers]
            
            ## Cost Analysis & Budget Planning
            [Cost considerations and budget recommendations]
            
            ## Performance Expectations
            [Expected outcomes and metrics]
            
            ## Risk Assessment & Mitigation
            [Potential challenges and solutions]
            
            ## Commercial Viability
            [Market readiness and commercialization potential]
            
            ## Timeline & Next Steps
            [Implementation timeline and actionable next steps]
            
            ## Supporting Research Papers
            [Citations and key paper references]
            
            Format as professional markdown with clear structure and actionable recommendations.
            """

            response = self._call_ollama(prompt)

            # Add metadata footer
            footer = f"""
            
            ---
            ## Report Metadata
            
            | Metric | Value |
            |--------|-------|
            | Papers Analyzed | {len(papers)} |
            | Implementation Guidance Available | {len([p for p in papers if p.get('implementation_guidance')])} |
            | Suppliers Identified | {len(set(s for p in papers for s in p.get('supplier_suggestions', [])))} |
            | Domains Covered | {len(set(p.get('domain', 'Unknown') for p in papers))} |
            | Analysis Date | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
            | Report Type | Research-to-Product Intelligence |
            
            **Confidence Level:** {self._calculate_report_confidence(papers, analysis_data)}/5
            """

            return response + footer

        except Exception as e:
            print(f"Error generating research report: {e}")
            return self._fallback_research_report(analysis_data, query_analysis, papers)

    def _prepare_papers_for_llama(self, papers: List[Dict]) -> str:
        """Prepare paper content with implementation data for LLaMA analysis"""
        content = ""
        for i, paper in enumerate(papers, 1):
            content += f"""
Paper {i}:
Title: {paper.get('title', 'Unknown')}
Domain: {paper.get('domain', 'Unknown')}
Materials: {', '.join(paper.get('materials', [])[:5])}
Processes: {', '.join(paper.get('processes', [])[:3])}
Implementation Guidance: {paper.get('implementation_guidance', 'Not available')[:300]}
Supplier Suggestions: {', '.join(paper.get('supplier_suggestions', [])[:5])}
Cost Recommendations: {', '.join(paper.get('cost_recommendations', [])[:3])}
Performance Metrics: {', '.join(paper.get('performance_metrics', [])[:3])}
Limitations: {paper.get('limitations', 'Not specified')[:200]}
Future Scope: {paper.get('future_scope', 'Not specified')[:200]}
Content: {paper.get('content', '')[:600]}...
---
"""
        return content

    def _extract_implementation_summary(self, papers: List[Dict]) -> Dict[str, Any]:
        """Extract and summarize implementation data from papers"""
        implementation_summary = {
            'total_papers_with_guidance': 0,
            'unique_materials': set(),
            'unique_suppliers': set(),
            'cost_ranges': [],
            'implementation_approaches': []
        }

        for paper in papers:
            if paper.get('implementation_guidance'):
                implementation_summary['total_papers_with_guidance'] += 1
                implementation_summary['implementation_approaches'].append(
                    paper.get('implementation_guidance', '')[:200]
                )

            if paper.get('materials'):
                implementation_summary['unique_materials'].update(paper['materials'])

            if paper.get('supplier_suggestions'):
                implementation_summary['unique_suppliers'].update(paper['supplier_suggestions'])

            if paper.get('cost_recommendations'):
                implementation_summary['cost_ranges'].extend(paper['cost_recommendations'])

        # Convert sets to lists for JSON serialization
        implementation_summary['unique_materials'] = list(implementation_summary['unique_materials'])[:10]
        implementation_summary['unique_suppliers'] = list(implementation_summary['unique_suppliers'])[:10]
        implementation_summary['cost_ranges'] = implementation_summary['cost_ranges'][:10]
        implementation_summary['implementation_approaches'] = implementation_summary['implementation_approaches'][:5]

        return implementation_summary

    def _calculate_report_confidence(self, papers: List[Dict], analysis_data: Dict) -> int:
        """Calculate confidence level for the report (1-5 scale)"""
        confidence_factors = 0

        # Factor 1: Number of papers
        if len(papers) >= 10:
            confidence_factors += 1
        elif len(papers) >= 5:
            confidence_factors += 0.5

        # Factor 2: Implementation guidance availability
        papers_with_guidance = len([p for p in papers if p.get('implementation_guidance')])
        if papers_with_guidance >= 5:
            confidence_factors += 1
        elif papers_with_guidance >= 2:
            confidence_factors += 0.5

        # Factor 3: Supplier information
        total_suppliers = len(set(s for p in papers for s in p.get('supplier_suggestions', [])))
        if total_suppliers >= 5:
            confidence_factors += 1
        elif total_suppliers >= 2:
            confidence_factors += 0.5

        # Factor 4: Cost information
        papers_with_costs = len([p for p in papers if p.get('cost_recommendations')])
        if papers_with_costs >= 3:
            confidence_factors += 1
        elif papers_with_costs >= 1:
            confidence_factors += 0.5

        # Factor 5: Domain consistency
        domains = set(p.get('domain', 'Unknown') for p in papers)
        if len(domains) <= 2:
            confidence_factors += 1
        elif len(domains) <= 5:
            confidence_factors += 0.5

        return min(int(confidence_factors), 5)

    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama with enhanced error handling"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 3000,
                    "top_p": 0.9
                }
            }

            response = requests.post(self.ollama_url, json=payload, timeout=150)

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Ollama API error: {response.status_code}")
                return ""

        except Exception as e:
            print(f"Ollama connection error: {e}")
            return ""

    def _parse_llama_response(self, response: str, fallback: Dict) -> Dict[str, Any]:
        """Parse LLaMA JSON response with fallback"""
        try:
            clean_response = response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith('```'):
                clean_response = clean_response[3:-3].strip()

            result = json.loads(clean_response)
            return result
        except json.JSONDecodeError:
            return fallback

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis when LLaMA is unavailable"""
        query_lower = query.lower()

        # Determine intent with implementation focus
        implementation_focus = False
        commercial_interest = False

        if any(word in query_lower for word in ['implement', 'build', 'manufacture', 'produce', 'scale']):
            intent = 'implementation'
            implementation_focus = True
        elif any(word in query_lower for word in ['supplier', 'vendor', 'manufacturer', 'equipment']):
            intent = 'suppliers'
            commercial_interest = True
            implementation_focus = True
        elif any(word in query_lower for word in ['cost', 'price', 'budget', 'economic', 'investment']):
            intent = 'costs'
            commercial_interest = True
            implementation_focus = True
        elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
            intent = 'comparison'
        else:
            intent = 'analysis'

        # Extract keywords
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [word for word in words if len(word) > 3][:5]

        return {
            'domain': 'general',
            'intent': intent,
            'search_terms': keywords,
            'implementation_focus': implementation_focus,
            'commercial_interest': commercial_interest,
            'output_format': 'comprehensive_report',
            'complexity': 'intermediate'
        }

    def _fallback_implementation_plan(self, papers: List[Dict], query_analysis: Dict) -> Dict[str, Any]:
        """Fallback implementation plan when LLaMA is unavailable"""
        # Extract implementation data from papers
        materials = set()
        suppliers = set()
        costs = []
        processes = set()

        for paper in papers:
            if paper.get('materials'):
                materials.update(paper['materials'][:5])
            if paper.get('supplier_suggestions'):
                suppliers.update(paper['supplier_suggestions'][:3])
            if paper.get('cost_recommendations'):
                costs.extend(paper['cost_recommendations'][:2])
            if paper.get('processes'):
                processes.update(paper['processes'][:3])

        return {
            'executive_summary': f'Implementation plan for {query_analysis.get("domain", "general research")} based on {len(papers)} papers',
            'combined_materials': list(materials)[:10],
            'unified_processes': list(processes)[:10],
            'implementation_steps': ['Review research findings', 'Analyze technical requirements', 'Develop implementation plan', 'Execute pilot project'],
            'supplier_recommendations': list(suppliers)[:10],
            'cost_analysis': costs[:10],
            'performance_expectations': ['Performance metrics from research papers', 'Expected outcomes based on literature'],
            'risk_mitigation': ['Technical risks identified', 'Market risks assessed', 'Implementation challenges noted'],
            'commercial_viability': 'Assessment based on available research data',
            'timeline_estimate': 'Timeline depends on complexity and resources',
            'next_steps': ['Detailed technical analysis', 'Supplier contact', 'Budget planning', 'Pilot implementation'],
            'papers_analyzed': len(papers)
        }

    def _fallback_research_report(self, analysis_data: Dict, query_analysis: Dict, papers: List[Dict]) -> str:
        """Fallback research report generation"""
        implementation_papers = len([p for p in papers if p.get('implementation_guidance')])

        return f"""# Research-to-Product Implementation Report

## Executive Summary
Comprehensive analysis of {len(papers)} research papers focusing on practical implementation guidance for {query_analysis.get('domain', 'general research')}.

## Research Findings
- Papers analyzed: {len(papers)}
- Papers with implementation guidance: {implementation_papers}
- Research focus: {query_analysis.get('intent', 'Analysis')}

## Implementation Roadmap
{chr(10).join([f"- {step}" for step in analysis_data.get('implementation_steps', [])[:5]])}

## Materials & Processes
**Key Materials Identified:**
{chr(10).join([f"- {material}" for material in analysis_data.get('combined_materials', [])[:10]])}

**Key Processes:**
{chr(10).join([f"- {process}" for process in analysis_data.get('unified_processes', [])[:10]])}

## Supplier Ecosystem
**Recommended Suppliers:**
{chr(10).join([f"- {supplier}" for supplier in analysis_data.get('supplier_recommendations', [])[:10]])}

## Cost Analysis
**Cost Considerations:**
{chr(10).join([f"- {cost}" for cost in analysis_data.get('cost_analysis', [])[:5]])}

## Commercial Viability
{analysis_data.get('commercial_viability', 'Assessment based on available research data')}

## Next Steps
{chr(10).join([f"- {step}" for step in analysis_data.get('next_steps', [])[:5]])}

## Supporting Research Papers
{chr(10).join([f"- {paper.get('title', 'Unknown title')} ({paper.get('domain', 'Unknown domain')})" for paper in papers[:10]])}

---
*Report generated from research database with implementation intelligence*
"""

# Initialize Global Components
db_manager = None
search_engine = None
domain_discovery = None
llama_integration = None

def initialize_system():
    """Initialize all system components"""
    global db_manager, search_engine, domain_discovery, llama_integration

    try:
        print("🚀 Initializing Research-to-Product Intelligence System...")

        # Initialize core components
        db_manager = DynamicDatabaseManager(DATABASE_PATH)
        search_engine = AdaptiveSearchEngine(db_manager)
        domain_discovery = DomainDiscoverySystem(db_manager)
        llama_integration = LLaMAIntegration()

        # Perform initial optimization
        db_manager.check_and_optimize()
        new_domains = domain_discovery.discover_new_domains()

        print(f"✅ System initialized successfully!")
        print(f"📊 Database: {db_manager.current_size:,} papers")
        print(f"🎯 Optimization level: {db_manager.get_optimization_level()}")
        print(f"🆕 Domains available: {len(domain_discovery.known_domains)}")

        return True

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

def test_ollama_connection():
    """Test connection to Ollama/LLaMA"""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": "Hello, this is a test.",
            "stream": False
        }, timeout=10)
        return response.status_code == 200
    except:
        return False

def init_conversation_database():
    """Initialize conversation tracking database"""
    try:
        os.makedirs(os.path.dirname(CONVERSATIONS_DB), exist_ok=True)
        conn = sqlite3.connect(CONVERSATIONS_DB)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_queries (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                question TEXT,
                intent TEXT,
                domain TEXT,
                timestamp DATETIME,
                papers_found INTEGER,
                search_time REAL,
                response TEXT,
                implementation_guidance_count INTEGER,
                supplier_suggestions_count INTEGER,
                cost_recommendations_count INTEGER
            )
        ''')

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error initializing conversation database: {e}")
        return False

def trigger_paper_processing():
    """Trigger processing of new papers using process_papers_fixed.py"""
    try:
        print("🔄 Triggering paper processing...")

        # Run the enhanced processor
        result = subprocess.run([
            sys.executable, 'process_papers_fixed.py'
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode == 0:
            print("✅ Paper processing completed successfully")

            # Trigger database optimization after processing
            if db_manager:
                db_manager.check_and_optimize()
                domain_discovery.discover_new_domains()

            return {
                'success': True,
                'message': 'Paper processing completed successfully',
                'output': result.stdout,
                'new_database_size': db_manager.get_current_size() if db_manager else 0
            }
        else:
            print(f"❌ Paper processing failed: {result.stderr}")
            return {
                'success': False,
                'message': 'Paper processing failed',
                'error': result.stderr,
                'output': result.stdout
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'message': 'Paper processing timed out (>1 hour)',
            'error': 'Processing timeout'
        }
    except Exception as e:
        print(f"Error triggering paper processing: {e}")
        return {
            'success': False,
            'message': 'Failed to trigger paper processing',
            'error': str(e)
        }

def save_research_query(research_query: ResearchQuery):
    """Save research query to conversation database"""
    try:
        conn = sqlite3.connect(CONVERSATIONS_DB)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO research_queries 
            (id, session_id, question, intent, domain, timestamp, papers_found, search_time, response,
             implementation_guidance_count, supplier_suggestions_count, cost_recommendations_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            research_query.id,
            research_query.session_id,
            research_query.question,
            research_query.intent,
            research_query.domain,
            research_query.timestamp,
            research_query.papers_found,
            research_query.search_time,
            research_query.response,
            research_query.implementation_guidance_count,
            research_query.supplier_suggestions_count,
            research_query.cost_recommendations_count
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error saving research query: {e}")

# Flask Routes

@app.route('/')
def index():
    """Main interface route"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/research', methods=['POST'])
def research():
    """Enhanced research endpoint with implementation intelligence"""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': 'No question provided'}), 400

        user_query = data['question'].strip()
        session_id = session.get('session_id', str(uuid.uuid4()))

        if not user_query:
            return jsonify({'success': False, 'error': 'Empty question'}), 400

        print(f"🔍 Processing research query: {user_query}")

        # Stage 1: Query understanding with implementation focus
        query_analysis = llama_integration.understand_query(user_query)
        print(f"🧠 Query analysis: {query_analysis.get('domain', 'unknown')} | {query_analysis.get('intent', 'unknown')}")

        # Stage 2: Enhanced search with implementation data
        relevant_papers = search_engine.smart_search_with_implementation(user_query, query_analysis)
        print(f"📚 Found {len(relevant_papers)} relevant papers")

        # Stage 3: Generate unified implementation plan
        implementation_plan = llama_integration.generate_unified_implementation_plan(relevant_papers, query_analysis)

        # Stage 4: Generate comprehensive research report
        research_report = llama_integration.generate_research_report(implementation_plan, query_analysis, relevant_papers)

        # Calculate performance metrics
        total_time = time.time() - start_time

        # Extract implementation metrics
        implementation_metrics = {
            'papers_with_guidance': len([p for p in relevant_papers if p.get('implementation_guidance')]),
            'unique_suppliers': len(set(s for p in relevant_papers for s in p.get('supplier_suggestions', []))),
            'cost_data_available': len([p for p in relevant_papers if p.get('cost_recommendations')]),
            'implementation_coverage': len([p for p in relevant_papers if p.get('implementation_guidance')]) / len(relevant_papers) * 100 if relevant_papers else 0
        }

        # Save research query for tracking
        research_query = ResearchQuery(
            id=str(uuid.uuid4()),
            session_id=session_id,
            question=user_query,
            intent=query_analysis.get('intent', 'unknown'),
            domain=query_analysis.get('domain', 'unknown'),
            timestamp=datetime.now(),
            papers_found=len(relevant_papers),
            search_time=total_time,
            response=research_report[:1000] + "..." if len(research_report) > 1000 else research_report,
            implementation_guidance_count=implementation_metrics['papers_with_guidance'],
            supplier_suggestions_count=implementation_metrics['unique_suppliers'],
            cost_recommendations_count=implementation_metrics['cost_data_available']
        )

        save_research_query(research_query)

        # Prepare comprehensive response with implementation data
        return jsonify({
            'success': True,
            'question': user_query,
            'research_report': research_report,
            'implementation_plan': implementation_plan,
            'implementation_metrics': implementation_metrics,
            'metadata': {
                'papers_found': len(relevant_papers),
                'search_time': total_time,
                'database_size': db_manager.current_size,
                'optimization_level': db_manager.get_optimization_level(),
                'domains_available': len(domain_discovery.known_domains),
                'query_analysis': query_analysis,
                'performance_stats': db_manager.performance_stats
            },
            'paper_sources': [
                {
                    'title': paper.get('title', 'Unknown'),
                    'domain': paper.get('domain', 'Unknown'),
                    'subdomain': paper.get('subdomain', 'Unknown'),
                    'relevance': paper.get('relevance_score', 1),
                    'word_count': paper.get('word_count', 0),
                    'has_implementation_guidance': bool(paper.get('implementation_guidance')),
                    'has_supplier_info': bool(paper.get('supplier_suggestions')),
                    'has_cost_info': bool(paper.get('cost_recommendations')),
                    'materials_count': len(paper.get('materials', [])),
                    'processes_count': len(paper.get('processes', [])),
                    'extraction_quality': paper.get('extraction_quality', 0)
                }
                for paper in relevant_papers[:15]
            ],
            'implementation_summary': {
                'total_materials': len(set(m for p in relevant_papers for m in p.get('materials', []))),
                'total_processes': len(set(pr for p in relevant_papers for pr in p.get('processes', []))),
                'total_suppliers': len(set(s for p in relevant_papers for s in p.get('supplier_suggestions', []))),
                'papers_with_costs': len([p for p in relevant_papers if p.get('cost_recommendations')]),
                'implementation_approaches': len([p for p in relevant_papers if p.get('implementation_guidance')])
            },
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        })

    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Research endpoint error: {e}")

        return jsonify({
            'success': False,
            'error': f'Research processing error: {str(e)}',
            'processing_time': error_time,
            'metadata': {
                'database_size': db_manager.current_size if db_manager else 0,
                'system_status': 'error'
            }
        }), 500

@app.route('/api/process', methods=['POST'])
def process_new_papers():
    """Endpoint to trigger processing of new papers on demand"""
    try:
        print("📄 Processing new papers endpoint called")

        # Get current database size before processing
        old_size = db_manager.current_size if db_manager else 0

        # Trigger paper processing
        processing_result = trigger_paper_processing()

        # Get new database size after processing
        new_size = db_manager.get_current_size() if db_manager else 0
        papers_added = new_size - old_size

        if processing_result['success']:
            # Refresh system components after processing
            if db_manager:
                db_manager.check_and_optimize()
            if domain_discovery:
                new_domains = domain_discovery.discover_new_domains()
            else:
                new_domains = []

            return jsonify({
                'success': True,
                'message': 'Paper processing completed successfully',
                'processing_details': {
                    'papers_before': old_size,
                    'papers_after': new_size,
                    'papers_added': papers_added,
                    'new_domains_discovered': new_domains,
                    'processing_output': processing_result.get('output', ''),
                    'optimization_performed': papers_added > 0
                },
                'system_status': {
                    'database_size': new_size,
                    'optimization_level': db_manager.get_optimization_level() if db_manager else 'unknown',
                    'domains_available': len(domain_discovery.known_domains) if domain_discovery else 0
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': processing_result['message'],
                'error': processing_result.get('error', 'Unknown error'),
                'processing_details': {
                    'papers_before': old_size,
                    'papers_after': new_size,
                    'processing_output': processing_result.get('output', '')
                },
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        print(f"Error in process endpoint: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to process papers',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/status')
def system_status():
    """System status with implementation intelligence metrics"""
    try:
        ollama_status = test_ollama_connection()

        # Get implementation intelligence metrics
        implementation_stats = {}
        if db_manager:
            try:
                conn = sqlite3.connect(db_manager.db_path)
                cursor = conn.cursor()

                # Get implementation data statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_papers,
                        COUNT(CASE WHEN implementation_guidance IS NOT NULL AND implementation_guidance != '' THEN 1 END) as papers_with_guidance,
                        COUNT(CASE WHEN supplier_suggestions IS NOT NULL AND supplier_suggestions != '[]' THEN 1 END) as papers_with_suppliers,
                        COUNT(CASE WHEN cost_recommendations IS NOT NULL AND cost_recommendations != '[]' THEN 1 END) as papers_with_costs
                    FROM papers
                """)

                stats = cursor.fetchone()
                if stats:
                    total = stats[0]
                    implementation_stats = {
                        'total_papers': total,
                        'papers_with_implementation_guidance': stats[1],
                        'papers_with_supplier_info': stats[2],
                        'papers_with_cost_info': stats[3],
                        'implementation_coverage': (stats[1] / total * 100) if total > 0 else 0,
                        'commercial_data_coverage': ((stats[2] + stats[3]) / (total * 2) * 100) if total > 0 else 0
                    }

                conn.close()
            except Exception as e:
                print(f"Error getting implementation stats: {e}")

        return jsonify({
            'system_status': 'online',
            'version': '2.0 - Research-to-Product Intelligence',
            'timestamp': datetime.now().isoformat(),
            'ollama_connected': ollama_status,
            'model': MODEL_NAME,
            'llama_status': 'available' if ollama_status else 'unavailable',
            'database': {
                'connected': db_manager is not None,
                'current_size': db_manager.current_size if db_manager else 0,
                'optimization_level': db_manager.get_optimization_level() if db_manager else 'unknown',
                'last_optimization': db_manager.last_optimization if db_manager else None,
                'optimization_count': db_manager.performance_stats.get('optimization_count', 0) if db_manager else 0
            },
            'implementation_intelligence': implementation_stats,
            'performance': {
                'avg_search_time': db_manager.performance_stats.get('avg_search_time', 0) if db_manager else 0,
                'total_searches': db_manager.performance_stats.get('total_searches', 0) if db_manager else 0,
                'performance_threshold': PERFORMANCE_THRESHOLD
            },
            'domains': {
                'total_discovered': len(domain_discovery.known_domains) if domain_discovery else 0,
                'known_domains': list(domain_discovery.known_domains) if domain_discovery else [],
                'discovery_active': domain_discovery is not None
            },
            'capabilities': {
                'implementation_guidance_extraction': True,
                'supplier_intelligence': True,
                'cost_analysis': True,
                'research_to_product_pipeline': True,
                'unified_implementation_plans': True,
                'dynamic_optimization': True,
                'on_demand_processing': True
            }
        })

    except Exception as e:
        return jsonify({
            'system_status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/domains')
def get_domains():
    """Get discovered domains with implementation metrics"""
    try:
        if not domain_discovery:
            return jsonify({'error': 'Domain discovery system not initialized'}), 500

        domains_info = {
            'total_domains': len(domain_discovery.known_domains),
            'domains': {}
        }

        # Get implementation metrics per domain
        if db_manager:
            try:
                conn = sqlite3.connect(db_manager.db_path)
                cursor = conn.cursor()

                for domain in domain_discovery.known_domains:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_papers,
                            COUNT(CASE WHEN implementation_guidance IS NOT NULL AND implementation_guidance != '' THEN 1 END) as with_guidance,
                            COUNT(CASE WHEN supplier_suggestions IS NOT NULL AND supplier_suggestions != '[]' THEN 1 END) as with_suppliers
                        FROM papers WHERE domain = ?
                    """, (domain,))

                    stats = cursor.fetchone()
                    domain_info = domain_discovery.domain_keywords.get(domain, {})

                    domains_info['domains'][domain] = {
                        'keywords': domain_info.get('keywords', [])[:10],
                        'sample_count': domain_info.get('sample_count', 0),
                        'discovered_at': domain_info.get('discovered_at').isoformat() if domain_info.get('discovered_at') else None,
                        'implementation_stats': {
                            'total_papers': stats[0] if stats else 0,
                            'papers_with_guidance': stats[1] if stats else 0,
                            'papers_with_suppliers': stats[2] if stats else 0,
                            'implementation_coverage': (stats[1] / stats[0] * 100) if stats and stats[0] > 0 else 0
                        }
                    }

                conn.close()

            except Exception as e:
                print(f"Error getting domain implementation stats: {e}")

        return jsonify(domains_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        if not db_manager:
            return jsonify({'error': 'Database manager not initialized'}), 500

        # Get performance history
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                AVG(search_time) as avg_time,
                COUNT(*) as search_count,
                AVG(papers_found) as avg_papers
            FROM performance_metrics 
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """)

        daily_stats = cursor.fetchall()
        conn.close()

        return jsonify({
            'current_stats': db_manager.performance_stats,
            'daily_performance': [
                {
                    'date': row[0],
                    'avg_search_time': round(row[1], 2) if row[1] else 0,
                    'search_count': row[2],
                    'avg_papers_found': round(row[3], 1) if row[3] else 0
                }
                for row in daily_stats
            ],
            'optimization_history': [
                {
                    'timestamp': opt['timestamp'].isoformat(),
                    'old_size': opt['old_size'],
                    'new_size': opt['new_size'],
                    'type': opt['optimization_type']
                }
                for opt in db_manager.optimization_history[-10:]
            ],
            'database_size': db_manager.current_size,
            'optimization_level': db_manager.get_optimization_level()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

# Background optimization task
def background_optimization():
    """Background task for continuous system optimization"""
    while True:
        try:
            time.sleep(300)  # Check every 5 minutes

            if db_manager:
                optimization_needed = db_manager.check_and_optimize()
                if optimization_needed:
                    print("🔧 Background optimization completed")

                if domain_discovery:
                    new_domains = domain_discovery.discover_new_domains()
                    if new_domains:
                        print(f"🆕 Background domain discovery: {new_domains}")

        except Exception as e:
            print(f"Background optimization error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    print("🚀 Starting Research-to-Product Intelligence Platform")
    print("=" * 80)
    print("🧠 IMPLEMENTATION INTELLIGENCE FEATURES:")
    print("   📊 Enhanced Database with Implementation Fields")
    print("   🔍 Implementation-Focused Search Engine")
    print("   🏭 Research-to-Product Pipeline")
    print("   🏢 Supplier Intelligence System")
    print("   💰 Cost Analysis & Budget Planning")
    print("   🆕 Automatic Domain Discovery")
    print("   ⚡ Performance Monitoring & Optimization")
    print("   🔧 On-Demand Paper Processing")
    print("=" * 80)
    print("🎯 API CAPABILITIES:")
    print("   💬 Natural Language Research Queries")
    print("   📚 Multi-Paper Implementation Plans")
    print("   🏭 Step-by-Step Implementation Guidance")
    print("   🔧 Equipment & Supplier Recommendations")
    print("   💰 Cost Analysis & Budget Planning")
    print("   📊 Commercial Viability Assessment")
    print("   📝 Professional Implementation Reports")
    print("=" * 80)

    # Initialize conversation database
    conv_db_init = init_conversation_database()
    if conv_db_init:
        print("✅ Conversation database initialized")
    else:
        print("⚠️ Conversation database initialization failed")

    # Initialize system
    system_init = initialize_system()

    if system_init:
        print("✅ System initialization successful")
        print("🏭 Research-to-Product Intelligence: ACTIVE")

        # Start background optimization thread
        optimization_thread = threading.Thread(target=background_optimization, daemon=True)
        optimization_thread.start()
        print("✅ Background optimization thread started")

    else:
        print("❌ System initialization failed - some features may not work")

    # Test Ollama connection
    ollama_available = test_ollama_connection()
    if ollama_available:
        print(f"✅ LLaMA 3.1 8B connected via Ollama")
        print("🧠 Advanced implementation intelligence available")
    else:
        print(f"⚠️ LLaMA 3.1 8B not available - system will work with reduced intelligence")
        print("   Start Ollama: ollama serve")
        print(f"   Pull model: ollama pull {MODEL_NAME}")

    print("🌐 Starting Research-to-Product Intelligence Platform at: http://localhost:5000")
    print("🔬 Ready for intelligent research-to-product queries!")
    print("🏭 Implementation guidance, suppliers, and costs included!")
    print("🔄 Use /api/process to add new papers anytime!")
    print("=" * 80)

    # API Endpoints Summary
    print("📡 AVAILABLE API ENDPOINTS:")
    print("   🔍 POST /api/research - Enhanced research with implementation intelligence")
    print("   🔄 POST /api/process - Process new papers on demand")
    print("   📊 GET /api/status - System status with implementation metrics")
    print("   🏷️ GET /api/domains - Domain discovery with implementation coverage")
    print("   📈 GET /api/performance - Performance metrics and optimization history")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5000, debug=True)