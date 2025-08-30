#!/usr/bin/env python3
"""
Comprehensive Benchmark System for All 4 Databases
Tests search, full-text search, and complex queries with detailed analysis
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import concurrent.futures
import random
import psycopg2
import duckdb

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveBenchmarkSystem:
    """Complete benchmarking system for all 4 databases."""
    
    def __init__(self):
        self.databases = {}
        self.benchmark_results = {}
        
        # Define comprehensive test queries
        self.test_queries = {
            'basic_search': {
                'name': 'Basic Search Queries',
                'tests': [
                    {'name': 'Count All', 'description': 'Total record count'},
                    {'name': 'Status Filter', 'description': 'Filter by status=Open'},
                    {'name': 'Department Filter', 'description': 'Filter by specific department'},
                    {'name': 'Date Range', 'description': 'Records from last 30 days'},
                    {'name': 'Ward Filter', 'description': 'Records from specific ward'}
                ]
            },
            'text_search': {
                'name': 'Full-Text Search Queries',
                'tests': [
                    {'name': 'Simple Text', 'description': 'Search for "Pothole"'},
                    {'name': 'Multi-word Search', 'description': 'Search for "Street Light"'},
                    {'name': 'Partial Match', 'description': 'Search for "Graffiti"'},
                    {'name': 'Complex Text', 'description': 'Search for "Water Leak"'},
                    {'name': 'Case Insensitive', 'description': 'Search for "TREE TRIM"'}
                ]
            },
            'complex_queries': {
                'name': 'Complex Analytical Queries',
                'tests': [
                    {'name': 'Geospatial Range', 'description': 'Geographic bounding box query'},
                    {'name': 'Multi-condition', 'description': 'Status + Date + Department'},
                    {'name': 'Aggregation', 'description': 'Group by department with counts'},
                    {'name': 'Top Wards', 'description': 'Top 10 wards by request count'},
                    {'name': 'Time Series', 'description': 'Requests by month/year'}
                ]
            }
        }
    
    def setup_database_connections(self):
        """Connect to all available databases."""
        print("🔄 Connecting to all databases for benchmarking...")
        
        # MongoDB
        try:
            mongo_handler = MongoDBHandler()
            count = mongo_handler.collection.count_documents({})
            if count > 0:
                self.databases['MongoDB'] = {
                    'handler': mongo_handler,
                    'count': count,
                    'type': 'document'
                }
                print(f"✅ MongoDB: {count:,} records")
        except Exception as e:
            print(f"❌ MongoDB: {e}")
        
        # Elasticsearch
        try:
            es_handler = ElasticsearchHandler()
            result = es_handler.es.count(index=es_handler.index_name)
            count = result['count']
            if count > 0:
                self.databases['Elasticsearch'] = {
                    'handler': es_handler,
                    'count': count,
                    'type': 'search_engine'
                }
                print(f"✅ Elasticsearch: {count:,} documents")
        except Exception as e:
            print(f"❌ Elasticsearch: {e}")
        
        # PostgreSQL
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="chicago_311_complete",
                user="postgres",
                password=""
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chicago_311_requests")
            count = cursor.fetchone()[0]
            cursor.close()
            
            if count > 0:
                self.databases['PostgreSQL'] = {
                    'handler': conn,
                    'count': count,
                    'type': 'relational'
                }
                print(f"✅ PostgreSQL: {count:,} records")
        except Exception as e:
            print(f"❌ PostgreSQL: {e}")
        
        # DuckDB
        try:
            conn = duckdb.connect("chicago_311_complete.duckdb")
            result = conn.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()
            count = result[0] if result else 0
            
            if count > 0:
                self.databases['DuckDB'] = {
                    'handler': conn,
                    'count': count,
                    'type': 'analytical'
                }
                print(f"✅ DuckDB: {count:,} records")
        except Exception as e:
            print(f"❌ DuckDB: {e}")
        
        if not self.databases:
            raise Exception("No databases with data available!")
        
        print(f"📊 Ready for benchmarking: {list(self.databases.keys())}")
        
    def execute_basic_search_queries(self, db_name: str) -> Dict:
        """Execute basic search queries."""
        results = {}
        handler_info = self.databases[db_name]
        handler = handler_info['handler']
        
        print(f"   🔍 Basic Search Tests for {db_name}...")
        
        try:
            if db_name == 'MongoDB':
                # Count All
                start_time = time.time()
                count = handler.collection.count_documents({})
                results['Count All'] = {'duration': time.time() - start_time, 'count': count}
                
                # Status Filter
                start_time = time.time()
                count = handler.collection.count_documents({'status': 'Open'})
                results['Status Filter'] = {'duration': time.time() - start_time, 'count': count}
                
                # Department Filter
                start_time = time.time()
                count = handler.collection.count_documents({'owner_department': 'Streets & Sanitation'})
                results['Department Filter'] = {'duration': time.time() - start_time, 'count': count}
                
                # Date Range
                start_time = time.time()
                thirty_days_ago = datetime.now() - timedelta(days=30)
                count = handler.collection.count_documents({'created_date': {'$gte': thirty_days_ago}})
                results['Date Range'] = {'duration': time.time() - start_time, 'count': count}
                
                # Ward Filter
                start_time = time.time()
                count = handler.collection.count_documents({'ward': 1})
                results['Ward Filter'] = {'duration': time.time() - start_time, 'count': count}
            
            elif db_name == 'Elasticsearch':
                # Count All
                start_time = time.time()
                result = handler.es.count(index=handler.index_name)
                results['Count All'] = {'duration': time.time() - start_time, 'count': result['count']}
                
                # Status Filter
                start_time = time.time()
                query = {"query": {"term": {"status": "Open"}}}
                result = handler.es.count(index=handler.index_name, body=query)
                results['Status Filter'] = {'duration': time.time() - start_time, 'count': result['count']}
                
                # Department Filter
                start_time = time.time()
                query = {"query": {"term": {"owner_department": "Streets & Sanitation"}}}
                result = handler.es.count(index=handler.index_name, body=query)
                results['Department Filter'] = {'duration': time.time() - start_time, 'count': result['count']}
                
                # Date Range
                start_time = time.time()
                query = {"query": {"range": {"created_date": {"gte": "now-30d"}}}}
                result = handler.es.count(index=handler.index_name, body=query)
                results['Date Range'] = {'duration': time.time() - start_time, 'count': result['count']}
                
                # Ward Filter
                start_time = time.time()
                query = {"query": {"term": {"ward": 1}}}
                result = handler.es.count(index=handler.index_name, body=query)
                results['Ward Filter'] = {'duration': time.time() - start_time, 'count': result['count']}
            
            elif db_name in ['PostgreSQL', 'DuckDB']:
                cursor = handler.cursor() if db_name == 'PostgreSQL' else None
                
                # Count All
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor.execute("SELECT COUNT(*) FROM chicago_311_requests")
                    count = cursor.fetchone()[0]
                else:  # DuckDB
                    count = handler.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()[0]
                results['Count All'] = {'duration': time.time() - start_time, 'count': count}
                
                # Status Filter
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE status = %s", ('Open',))
                    count = cursor.fetchone()[0]
                else:
                    count = handler.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE status = 'Open'").fetchone()[0]
                results['Status Filter'] = {'duration': time.time() - start_time, 'count': count}
                
                # Department Filter
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE owner_department = %s", ('Streets & Sanitation',))
                    count = cursor.fetchone()[0]
                else:
                    count = handler.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE owner_department = 'Streets & Sanitation'").fetchone()[0]
                results['Department Filter'] = {'duration': time.time() - start_time, 'count': count}
                
                # Date Range
                start_time = time.time()
                thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                if db_name == 'PostgreSQL':
                    cursor.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE created_date >= %s", (thirty_days_ago,))
                    count = cursor.fetchone()[0]
                else:
                    count = handler.execute(f"SELECT COUNT(*) FROM chicago_311_requests WHERE created_date >= '{thirty_days_ago}'").fetchone()[0]
                results['Date Range'] = {'duration': time.time() - start_time, 'count': count}
                
                # Ward Filter
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE ward = %s", (1,))
                    count = cursor.fetchone()[0]
                    cursor.close()
                else:
                    count = handler.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE ward = 1").fetchone()[0]
                results['Ward Filter'] = {'duration': time.time() - start_time, 'count': count}
            
        except Exception as e:
            logger.error(f"Basic search failed for {db_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def execute_text_search_queries(self, db_name: str) -> Dict:
        """Execute full-text search queries."""
        results = {}
        handler_info = self.databases[db_name]
        handler = handler_info['handler']
        
        print(f"   📝 Full-Text Search Tests for {db_name}...")
        
        search_terms = [
            ('Simple Text', 'Pothole'),
            ('Multi-word Search', 'Street Light'),
            ('Partial Match', 'Graffiti'),
            ('Complex Text', 'Water Leak'),
            ('Case Insensitive', 'TREE TRIM')
        ]
        
        try:
            for test_name, search_term in search_terms:
                start_time = time.time()
                
                if db_name == 'MongoDB':
                    # Use regex for text search
                    count = handler.collection.count_documents({
                        'sr_type': {'$regex': search_term, '$options': 'i'}
                    })
                
                elif db_name == 'Elasticsearch':
                    # Use match query for better text search
                    query = {
                        "query": {
                            "match": {
                                "sr_type": {
                                    "query": search_term,
                                    "operator": "and"
                                }
                            }
                        }
                    }
                    result = handler.es.count(index=handler.index_name, body=query)
                    count = result['count']
                
                elif db_name in ['PostgreSQL', 'DuckDB']:
                    if db_name == 'PostgreSQL':
                        cursor = handler.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM chicago_311_requests WHERE sr_type ILIKE %s",
                            (f'%{search_term}%',)
                        )
                        count = cursor.fetchone()[0]
                        cursor.close()
                    else:  # DuckDB
                        count = handler.execute(
                            f"SELECT COUNT(*) FROM chicago_311_requests WHERE sr_type ILIKE '%{search_term}%'"
                        ).fetchone()[0]
                
                results[test_name] = {
                    'duration': time.time() - start_time,
                    'count': count,
                    'search_term': search_term
                }
        
        except Exception as e:
            logger.error(f"Text search failed for {db_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def execute_complex_queries(self, db_name: str) -> Dict:
        """Execute complex analytical queries."""
        results = {}
        handler_info = self.databases[db_name]
        handler = handler_info['handler']
        
        print(f"   ⚙️ Complex Query Tests for {db_name}...")
        
        try:
            if db_name == 'MongoDB':
                # Geospatial Range
                start_time = time.time()
                count = handler.collection.count_documents({
                    'latitude': {'$gte': 41.8, '$lte': 41.9},
                    'longitude': {'$gte': -87.7, '$lte': -87.6}
                })
                results['Geospatial Range'] = {'duration': time.time() - start_time, 'count': count}
                
                # Multi-condition
                start_time = time.time()
                count = handler.collection.count_documents({
                    'status': 'Open',
                    'owner_department': 'Streets & Sanitation',
                    'created_date': {'$gte': datetime.now() - timedelta(days=90)}
                })
                results['Multi-condition'] = {'duration': time.time() - start_time, 'count': count}
                
                # Aggregation
                start_time = time.time()
                pipeline = [
                    {'$group': {'_id': '$owner_department', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                    {'$limit': 10}
                ]
                agg_results = list(handler.collection.aggregate(pipeline))
                results['Aggregation'] = {'duration': time.time() - start_time, 'count': len(agg_results)}
                
                # Top Wards
                start_time = time.time()
                pipeline = [
                    {'$match': {'ward': {'$ne': None}}},
                    {'$group': {'_id': '$ward', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                    {'$limit': 10}
                ]
                ward_results = list(handler.collection.aggregate(pipeline))
                results['Top Wards'] = {'duration': time.time() - start_time, 'count': len(ward_results)}
            
            elif db_name == 'Elasticsearch':
                # Geospatial Range
                start_time = time.time()
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"range": {"latitude": {"gte": 41.8, "lte": 41.9}}},
                                {"range": {"longitude": {"gte": -87.7, "lte": -87.6}}}
                            ]
                        }
                    }
                }
                result = handler.es.count(index=handler.index_name, body=query)
                results['Geospatial Range'] = {'duration': time.time() - start_time, 'count': result['count']}
                
                # Multi-condition
                start_time = time.time()
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"status": "Open"}},
                                {"term": {"owner_department": "Streets & Sanitation"}},
                                {"range": {"created_date": {"gte": "now-90d"}}}
                            ]
                        }
                    }
                }
                result = handler.es.count(index=handler.index_name, body=query)
                results['Multi-condition'] = {'duration': time.time() - start_time, 'count': result['count']}
                
                # Aggregation
                start_time = time.time()
                query = {
                    "size": 0,
                    "aggs": {
                        "departments": {
                            "terms": {"field": "owner_department", "size": 10}
                        }
                    }
                }
                result = handler.es.search(index=handler.index_name, body=query)
                agg_count = len(result['aggregations']['departments']['buckets'])
                results['Aggregation'] = {'duration': time.time() - start_time, 'count': agg_count}
                
                # Top Wards
                start_time = time.time()
                query = {
                    "size": 0,
                    "aggs": {
                        "top_wards": {
                            "terms": {"field": "ward", "size": 10}
                        }
                    }
                }
                result = handler.es.search(index=handler.index_name, body=query)
                ward_count = len(result['aggregations']['top_wards']['buckets'])
                results['Top Wards'] = {'duration': time.time() - start_time, 'count': ward_count}
            
            elif db_name in ['PostgreSQL', 'DuckDB']:
                # Geospatial Range
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor = handler.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM chicago_311_requests 
                        WHERE latitude BETWEEN %s AND %s 
                        AND longitude BETWEEN %s AND %s
                    """, (41.8, 41.9, -87.7, -87.6))
                    count = cursor.fetchone()[0]
                    cursor.close()
                else:
                    count = handler.execute("""
                        SELECT COUNT(*) FROM chicago_311_requests 
                        WHERE latitude BETWEEN 41.8 AND 41.9 
                        AND longitude BETWEEN -87.7 AND -87.6
                    """).fetchone()[0]
                results['Geospatial Range'] = {'duration': time.time() - start_time, 'count': count}
                
                # Multi-condition
                start_time = time.time()
                ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                if db_name == 'PostgreSQL':
                    cursor = handler.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM chicago_311_requests 
                        WHERE status = %s AND owner_department = %s 
                        AND created_date >= %s
                    """, ('Open', 'Streets & Sanitation', ninety_days_ago))
                    count = cursor.fetchone()[0]
                    cursor.close()
                else:
                    count = handler.execute(f"""
                        SELECT COUNT(*) FROM chicago_311_requests 
                        WHERE status = 'Open' AND owner_department = 'Streets & Sanitation' 
                        AND created_date >= '{ninety_days_ago}'
                    """).fetchone()[0]
                results['Multi-condition'] = {'duration': time.time() - start_time, 'count': count}
                
                # Aggregation
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor = handler.cursor()
                    cursor.execute("""
                        SELECT owner_department, COUNT(*) as count 
                        FROM chicago_311_requests 
                        WHERE owner_department IS NOT NULL
                        GROUP BY owner_department 
                        ORDER BY count DESC LIMIT 10
                    """)
                    agg_results = cursor.fetchall()
                    cursor.close()
                else:
                    agg_results = handler.execute("""
                        SELECT owner_department, COUNT(*) as count 
                        FROM chicago_311_requests 
                        WHERE owner_department IS NOT NULL
                        GROUP BY owner_department 
                        ORDER BY count DESC LIMIT 10
                    """).fetchall()
                results['Aggregation'] = {'duration': time.time() - start_time, 'count': len(agg_results)}
                
                # Top Wards
                start_time = time.time()
                if db_name == 'PostgreSQL':
                    cursor = handler.cursor()
                    cursor.execute("""
                        SELECT ward, COUNT(*) as count 
                        FROM chicago_311_requests 
                        WHERE ward IS NOT NULL
                        GROUP BY ward 
                        ORDER BY count DESC LIMIT 10
                    """)
                    ward_results = cursor.fetchall()
                    cursor.close()
                else:
                    ward_results = handler.execute("""
                        SELECT ward, COUNT(*) as count 
                        FROM chicago_311_requests 
                        WHERE ward IS NOT NULL
                        GROUP BY ward 
                        ORDER BY count DESC LIMIT 10
                    """).fetchall()
                results['Top Wards'] = {'duration': time.time() - start_time, 'count': len(ward_results)}
        
        except Exception as e:
            logger.error(f"Complex queries failed for {db_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_comprehensive_benchmarks(self):
        """Run all benchmark tests."""
        print("\n🎯 COMPREHENSIVE BENCHMARKING SYSTEM")
        print("=" * 80)
        
        for db_name, db_info in self.databases.items():
            print(f"\n📊 Benchmarking {db_name} ({db_info['count']:,} records)")
            print("-" * 60)
            
            # Run all test categories
            basic_results = self.execute_basic_search_queries(db_name)
            text_results = self.execute_text_search_queries(db_name)
            complex_results = self.execute_complex_queries(db_name)
            
            # Store results
            self.benchmark_results[db_name] = {
                'record_count': db_info['count'],
                'database_type': db_info['type'],
                'basic_search': basic_results,
                'text_search': text_results,
                'complex_queries': complex_results
            }
            
            # Display summary
            self.display_database_summary(db_name, basic_results, text_results, complex_results)
    
    def display_database_summary(self, db_name: str, basic: Dict, text: Dict, complex: Dict):
        """Display summary for a database."""
        print(f"   📋 {db_name} Results Summary:")
        
        # Basic Search
        if 'error' not in basic:
            avg_basic = np.mean([r['duration'] for r in basic.values() if isinstance(r, dict)])
            print(f"   • Basic Search Avg: {avg_basic:.4f}s")
        
        # Text Search
        if 'error' not in text:
            avg_text = np.mean([r['duration'] for r in text.values() if isinstance(r, dict)])
            print(f"   • Text Search Avg: {avg_text:.4f}s")
        
        # Complex Queries
        if 'error' not in complex:
            avg_complex = np.mean([r['duration'] for r in complex.values() if isinstance(r, dict)])
            print(f"   • Complex Query Avg: {avg_complex:.4f}s")
    
    def generate_benchmark_charts(self):
        """Generate comprehensive benchmark visualization charts."""
        print("\n📊 Generating Benchmark Charts...")
        
        # Set up the plotting environment
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Chart 1: Basic Search Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_basic_search_chart(ax1)
        
        # Chart 2: Text Search Performance Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_text_search_chart(ax2)
        
        # Chart 3: Complex Query Performance Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_complex_query_chart(ax3)
        
        # Chart 4: Overall Performance Heatmap
        ax4 = fig.add_subplot(gs[1, :])
        self._create_performance_heatmap(ax4)
        
        # Chart 5: Database Type Comparison
        ax5 = fig.add_subplot(gs[2, 0])
        self._create_database_type_comparison(ax5)
        
        # Chart 6: Query Category Winners
        ax6 = fig.add_subplot(gs[2, 1])
        self._create_category_winners_chart(ax6)
        
        # Chart 7: Performance Distribution
        ax7 = fig.add_subplot(gs[2, 2])
        self._create_performance_distribution(ax7)
        
        plt.suptitle('Comprehensive Database Benchmark Results - Chicago 311 Data', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('comprehensive_benchmark_charts.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Benchmark charts saved as 'comprehensive_benchmark_charts.png'")
    
    def _create_basic_search_chart(self, ax):
        """Create basic search performance chart."""
        data = []
        for db_name, results in self.benchmark_results.items():
            if 'basic_search' in results and 'error' not in results['basic_search']:
                for query_name, metrics in results['basic_search'].items():
                    if isinstance(metrics, dict) and 'duration' in metrics:
                        data.append({
                            'Database': db_name,
                            'Query': query_name,
                            'Duration': metrics['duration']
                        })
        
        if data:
            df = pd.DataFrame(data)
            df_pivot = df.pivot(index='Query', columns='Database', values='Duration')
            df_pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Basic Search Query Performance', fontweight='bold')
            ax.set_ylabel('Duration (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _create_text_search_chart(self, ax):
        """Create text search performance chart."""
        data = []
        for db_name, results in self.benchmark_results.items():
            if 'text_search' in results and 'error' not in results['text_search']:
                for query_name, metrics in results['text_search'].items():
                    if isinstance(metrics, dict) and 'duration' in metrics:
                        data.append({
                            'Database': db_name,
                            'Query': query_name,
                            'Duration': metrics['duration']
                        })
        
        if data:
            df = pd.DataFrame(data)
            df_pivot = df.pivot(index='Query', columns='Database', values='Duration')
            df_pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Full-Text Search Performance', fontweight='bold')
            ax.set_ylabel('Duration (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _create_complex_query_chart(self, ax):
        """Create complex query performance chart."""
        data = []
        for db_name, results in self.benchmark_results.items():
            if 'complex_queries' in results and 'error' not in results['complex_queries']:
                for query_name, metrics in results['complex_queries'].items():
                    if isinstance(metrics, dict) and 'duration' in metrics:
                        data.append({
                            'Database': db_name,
                            'Query': query_name,
                            'Duration': metrics['duration']
                        })
        
        if data:
            df = pd.DataFrame(data)
            df_pivot = df.pivot(index='Query', columns='Database', values='Duration')
            df_pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Complex Analytical Query Performance', fontweight='bold')
            ax.set_ylabel('Duration (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _create_performance_heatmap(self, ax):
        """Create overall performance heatmap."""
        # Collect all query performances
        heatmap_data = []
        databases = list(self.benchmark_results.keys())
        queries = set()
        
        # Collect all query names
        for db_name, results in self.benchmark_results.items():
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in results and 'error' not in results[category]:
                    queries.update(results[category].keys())
        
        queries = list(queries)
        
        # Build heatmap matrix
        matrix = []
        for query in queries:
            row = []
            for db_name in databases:
                found = False
                for category in ['basic_search', 'text_search', 'complex_queries']:
                    if (category in self.benchmark_results[db_name] and 
                        query in self.benchmark_results[db_name][category] and
                        isinstance(self.benchmark_results[db_name][category][query], dict)):
                        duration = self.benchmark_results[db_name][category][query].get('duration', np.nan)
                        row.append(duration)
                        found = True
                        break
                if not found:
                    row.append(np.nan)
            matrix.append(row)
        
        if matrix:
            heatmap_df = pd.DataFrame(matrix, index=queries, columns=databases)
            sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': 'Duration (seconds)'})
            ax.set_title('Overall Performance Heatmap', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
    
    def _create_database_type_comparison(self, ax):
        """Create database type performance comparison."""
        type_performance = {}
        
        for db_name, results in self.benchmark_results.items():
            db_type = results['database_type']
            if db_type not in type_performance:
                type_performance[db_type] = []
            
            # Collect all durations
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in results and 'error' not in results[category]:
                    for metrics in results[category].values():
                        if isinstance(metrics, dict) and 'duration' in metrics:
                            type_performance[db_type].append(metrics['duration'])
        
        if type_performance:
            types = list(type_performance.keys())
            avg_performances = [np.mean(type_performance[t]) for t in types]
            
            bars = ax.bar(types, avg_performances, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(types)])
            ax.set_title('Average Performance by Database Type', fontweight='bold')
            ax.set_ylabel('Average Duration (seconds)')
            
            # Add value labels
            for bar, value in zip(bars, avg_performances):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.4f}', ha='center', va='bottom')
    
    def _create_category_winners_chart(self, ax):
        """Create category winners chart."""
        winners = {}
        categories = ['basic_search', 'text_search', 'complex_queries']
        
        for category in categories:
            category_performance = {}
            for db_name, results in self.benchmark_results.items():
                if category in results and 'error' not in results[category]:
                    durations = [m['duration'] for m in results[category].values() 
                                if isinstance(m, dict) and 'duration' in m]
                    if durations:
                        category_performance[db_name] = np.mean(durations)
            
            if category_performance:
                winner = min(category_performance.keys(), key=lambda x: category_performance[x])
                winners[category.replace('_', ' ').title()] = winner
        
        if winners:
            categories = list(winners.keys())
            databases = list(winners.values())
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            db_colors = {db: colors[i % len(colors)] for i, db in enumerate(set(databases))}
            bar_colors = [db_colors[db] for db in databases]
            
            ax.bar(categories, [1]*len(categories), color=bar_colors)
            ax.set_title('Category Winners', fontweight='bold')
            ax.set_ylabel('Winner')
            ax.tick_params(axis='x', rotation=45)
            
            # Add database names as labels
            for i, db in enumerate(databases):
                ax.text(i, 0.5, db, ha='center', va='center', fontweight='bold', color='white')
    
    def _create_performance_distribution(self, ax):
        """Create performance distribution chart."""
        all_durations = {}
        
        for db_name, results in self.benchmark_results.items():
            durations = []
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in results and 'error' not in results[category]:
                    durations.extend([m['duration'] for m in results[category].values() 
                                     if isinstance(m, dict) and 'duration' in m])
            all_durations[db_name] = durations
        
        if all_durations:
            # Create violin plot
            data_for_plot = []
            labels = []
            for db_name, durations in all_durations.items():
                if durations:
                    data_for_plot.append(durations)
                    labels.append(db_name)
            
            if data_for_plot:
                ax.violinplot(data_for_plot, positions=range(len(labels)), showmeans=True)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                ax.set_title('Performance Distribution', fontweight='bold')
                ax.set_ylabel('Duration (seconds)')
    
    def save_benchmark_results(self):
        """Save comprehensive benchmark results."""
        # Clean results for JSON serialization
        clean_results = {}
        for db_name, results in self.benchmark_results.items():
            clean_results[db_name] = results.copy()
            # Remove handler references
            if 'handler' in clean_results[db_name]:
                del clean_results[db_name]['handler']
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_type': 'comprehensive_performance',
            'test_categories': self.test_queries,
            'results': clean_results
        }
        
        with open('comprehensive_benchmark_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print("✅ Results saved to 'comprehensive_benchmark_results.json'")
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DATABASE BENCHMARK REPORT")
        report.append("Chicago 311 Service Requests - Performance Analysis")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        total_records = sum(db['record_count'] for db in self.benchmark_results.values())
        report.append(f"• Total Records Analyzed: {total_records:,}")
        report.append(f"• Databases Tested: {len(self.benchmark_results)}")
        report.append(f"• Test Categories: 3 (Basic Search, Text Search, Complex Queries)")
        report.append("")
        
        # Performance Rankings
        report.append("PERFORMANCE RANKINGS BY CATEGORY")
        report.append("-" * 40)
        
        for category in ['basic_search', 'text_search', 'complex_queries']:
            category_name = category.replace('_', ' ').title()
            report.append(f"\n🏆 {category_name}:")
            
            # Find best performer in this category
            category_performance = {}
            for db_name, results in self.benchmark_results.items():
                if category in results and 'error' not in results[category]:
                    durations = [m['duration'] for m in results[category].values() 
                                if isinstance(m, dict) and 'duration' in m]
                    if durations:
                        category_performance[db_name] = np.mean(durations)
            
            if category_performance:
                sorted_dbs = sorted(category_performance.items(), key=lambda x: x[1])
                for i, (db_name, avg_time) in enumerate(sorted_dbs, 1):
                    report.append(f"   {i}. {db_name}: {avg_time:.4f}s average")
        
        # Detailed Results
        report.append(f"\n\nDETAILED RESULTS BY DATABASE")
        report.append("-" * 40)
        
        for db_name, results in self.benchmark_results.items():
            report.append(f"\n{db_name.upper()} ({results['database_type']})")
            report.append(f"Records: {results['record_count']:,}")
            report.append("=" * 20)
            
            for category in ['basic_search', 'text_search', 'complex_queries']:
                category_name = category.replace('_', ' ').title()
                report.append(f"\n{category_name}:")
                
                if category in results and 'error' not in results[category]:
                    for query_name, metrics in results[category].items():
                        if isinstance(metrics, dict) and 'duration' in metrics:
                            duration = metrics['duration']
                            count = metrics.get('count', 'N/A')
                            report.append(f"  • {query_name}: {duration:.4f}s ({count} results)")
                elif 'error' in results.get(category, {}):
                    report.append(f"  ❌ Error: {results[category]['error']}")
        
        # Recommendations
        report.append(f"\n\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find overall best performer
        overall_performance = {}
        for db_name, results in self.benchmark_results.items():
            all_durations = []
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in results and 'error' not in results[category]:
                    all_durations.extend([m['duration'] for m in results[category].values() 
                                         if isinstance(m, dict) and 'duration' in m])
            if all_durations:
                overall_performance[db_name] = np.mean(all_durations)
        
        if overall_performance:
            best_overall = min(overall_performance.keys(), key=lambda x: overall_performance[x])
            report.append(f"🎯 Overall Best Performer: {best_overall}")
            
            # Specific recommendations based on database types
            for db_name, results in self.benchmark_results.items():
                db_type = results['database_type']
                if db_type == 'search_engine':
                    report.append(f"   • {db_name}: Excellent for full-text search and analytics")
                elif db_type == 'document':
                    report.append(f"   • {db_name}: Best for flexible document operations")
                elif db_type == 'relational':
                    report.append(f"   • {db_name}: Ideal for complex relational queries")
                elif db_type == 'analytical':
                    report.append(f"   • {db_name}: Optimized for analytical workloads")
        
        report.append("\n💡 General Recommendations:")
        report.append("   • Use Elasticsearch for search-heavy applications")
        report.append("   • Use PostgreSQL for complex relational operations")
        report.append("   • Use MongoDB for flexible document storage")
        report.append("   • Use DuckDB for analytical and data science workloads")
        report.append("   • Consider data size and query patterns when choosing")
        
        # Save report
        with open('comprehensive_benchmark_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("✅ Benchmark report saved as 'comprehensive_benchmark_report.txt'")
        return '\n'.join(report)
    
    def close_connections(self):
        """Close all database connections."""
        for db_info in self.databases.values():
            try:
                handler = db_info['handler']
                if hasattr(handler, 'close'):
                    handler.close()
                elif hasattr(handler, 'connection'):
                    handler.connection.close()
            except:
                pass

def main():
    """Run comprehensive benchmarking system."""
    benchmark_system = ComprehensiveBenchmarkSystem()
    
    try:
        benchmark_system.setup_database_connections()
        benchmark_system.run_comprehensive_benchmarks()
        benchmark_system.generate_benchmark_charts()
        benchmark_system.save_benchmark_results()
        report = benchmark_system.generate_benchmark_report()
        
        print(f"\n{'='*80}")
        print("🎉 COMPREHENSIVE BENCHMARKING COMPLETED!")
        print(f"{'='*80}")
        print("📁 Generated Files:")
        print("   • comprehensive_benchmark_charts.png")
        print("   • comprehensive_benchmark_results.json") 
        print("   • comprehensive_benchmark_report.txt")
        
        # Show quick summary
        print(f"\n📋 Quick Summary:")
        print(report.split("DETAILED RESULTS")[0])
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        benchmark_system.close_connections()

if __name__ == "__main__":
    main()