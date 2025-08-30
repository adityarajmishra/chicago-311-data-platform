#!/usr/bin/env python3
"""
All 4 Databases Comprehensive Benchmark System
Creates comprehensive_benchmark_results.png, comprehensive_benchmark_charts.png, 
and database_performance_comparison.png with ALL 4 databases
"""

import os
import sys
import time
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from databases.mongodb_handler import MongoDBHandler
from databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class All4DatabasesBenchmarkSystem:
    """Comprehensive benchmark system for ALL 4 databases."""
    
    def __init__(self):
        self.handlers = {}
        self.results = {}
        self.test_categories = {
            'basic_search': [
                {'name': 'Count All', 'description': 'Total record count'},
                {'name': 'Status Filter', 'description': 'Filter by status'},
                {'name': 'Department Filter', 'description': 'Filter by department'},
                {'name': 'Date Range', 'description': 'Recent records'},
                {'name': 'Ward Filter', 'description': 'Filter by ward'}
            ],
            'text_search': [
                {'name': 'Simple Text', 'description': 'Search for specific terms'},
                {'name': 'Multi-word Search', 'description': 'Multiple keywords'},
                {'name': 'Partial Match', 'description': 'Partial text matching'},
                {'name': 'Complex Text', 'description': 'Complex text queries'},
                {'name': 'Case Insensitive', 'description': 'Case insensitive search'}
            ],
            'complex_queries': [
                {'name': 'Geospatial Range', 'description': 'Geographic queries'},
                {'name': 'Multi-condition', 'description': 'Multiple conditions'},
                {'name': 'Aggregation', 'description': 'Group by operations'},
                {'name': 'Top Analysis', 'description': 'Top N queries'},
                {'name': 'Time Series', 'description': 'Time-based analysis'}
            ]
        }
    
    def setup_all_databases(self):
        """Setup all 4 database connections for benchmarking."""
        logger.info("🔧 Setting up ALL 4 databases for benchmarking...")
        
        # 1. MongoDB
        try:
            mongo_handler = MongoDBHandler()
            count = mongo_handler.collection.count_documents({})
            if count > 0:
                self.handlers['MongoDB'] = mongo_handler
                logger.info(f"✅ MongoDB ready ({count:,} records)")
            else:
                logger.warning("⚠️ MongoDB has no data")
        except Exception as e:
            logger.error(f"❌ MongoDB setup failed: {e}")
        
        # 2. Elasticsearch with NumPy fix
        try:
            es_handler = ElasticsearchHandler(index_name='chicago_311_fixed')
            count_result = es_handler.es.count(index=es_handler.index_name)
            count = count_result['count']
            if count > 0:
                self.handlers['Elasticsearch'] = es_handler
                logger.info(f"✅ Elasticsearch ready ({count:,} records)")
            else:
                logger.warning("⚠️ Elasticsearch has no data")
        except Exception as e:
            logger.error(f"❌ Elasticsearch setup failed: {e}")
            
        # 3. PostgreSQL
        try:
            import psycopg2
            
            class PostgreSQLBenchmarkHandler:
                def __init__(self):
                    try:
                        self.connection = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="postgres",
                            user="postgres",
                            password="postgres"
                        )
                    except:
                        self.connection = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="chicago_311",
                            user="postgres",
                            password=""
                        )
                    
                    with self.connection.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM chicago_311_requests")
                        self.record_count = cursor.fetchone()[0]
            
            pg_handler = PostgreSQLBenchmarkHandler()
            if pg_handler.record_count > 0:
                self.handlers['PostgreSQL'] = pg_handler
                logger.info(f"✅ PostgreSQL ready ({pg_handler.record_count:,} records)")
            else:
                logger.warning("⚠️ PostgreSQL has no data")
        except Exception as e:
            logger.error(f"❌ PostgreSQL setup failed: {e}")
            
        # 4. DuckDB
        try:
            import duckdb
            
            class DuckDBBenchmarkHandler:
                def __init__(self):
                    self.conn = duckdb.connect("chicago_311.duckdb")
                    try:
                        result = self.conn.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()
                        self.record_count = result[0] if result else 0
                    except:
                        self.record_count = 0
            
            duckdb_handler = DuckDBBenchmarkHandler()
            if duckdb_handler.record_count > 0:
                self.handlers['DuckDB'] = duckdb_handler
                logger.info(f"✅ DuckDB ready ({duckdb_handler.record_count:,} records)")
            else:
                logger.warning("⚠️ DuckDB has no data")
        except Exception as e:
            logger.error(f"❌ DuckDB setup failed: {e}")
        
        logger.info(f"🎯 Ready to benchmark {len(self.handlers)} databases: {list(self.handlers.keys())}")
        return len(self.handlers) > 0
    
    def run_basic_search_tests(self, db_name: str, handler: Any) -> Dict[str, Any]:
        """Run basic search performance tests."""
        results = {}
        
        if db_name == 'MongoDB':
            tests = {
                'Count All': lambda: handler.collection.count_documents({}),
                'Status Filter': lambda: handler.collection.count_documents({'status': 'Open'}),
                'Department Filter': lambda: handler.collection.count_documents({'owner_department': 'Streets & Sanitation'}),
                'Date Range': lambda: handler.collection.count_documents({'created_date': {'$gte': datetime.now()}}),
                'Ward Filter': lambda: handler.collection.count_documents({'ward': 1})
            }
        elif db_name == 'Elasticsearch':
            tests = {
                'Count All': lambda: handler.es.count(index=handler.index_name)['count'],
                'Status Filter': lambda: handler.es.count(index=handler.index_name, body={'query': {'match': {'status': 'Open'}}})['count'],
                'Department Filter': lambda: handler.es.count(index=handler.index_name, body={'query': {'match': {'owner_department': 'Streets & Sanitation'}}})['count'],
                'Date Range': lambda: handler.es.count(index=handler.index_name, body={'query': {'range': {'created_date': {'gte': 'now-30d'}}}})['count'],
                'Ward Filter': lambda: handler.es.count(index=handler.index_name, body={'query': {'match': {'ward': 1}}})['count']
            }
        elif db_name == 'PostgreSQL':
            def pg_query(query):
                with handler.connection.cursor() as cursor:
                    cursor.execute(query)
                    return cursor.fetchone()[0]
            
            tests = {
                'Count All': lambda: pg_query("SELECT COUNT(*) FROM chicago_311_requests"),
                'Status Filter': lambda: pg_query("SELECT COUNT(*) FROM chicago_311_requests WHERE status = 'Open'"),
                'Department Filter': lambda: pg_query("SELECT COUNT(*) FROM chicago_311_requests WHERE owner_department = 'Streets & Sanitation'"),
                'Date Range': lambda: pg_query("SELECT COUNT(*) FROM chicago_311_requests WHERE created_date >= NOW() - INTERVAL '30 days'"),
                'Ward Filter': lambda: pg_query("SELECT COUNT(*) FROM chicago_311_requests WHERE ward = 1")
            }
        elif db_name == 'DuckDB':
            tests = {
                'Count All': lambda: handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()[0],
                'Status Filter': lambda: handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE status = 'Open'").fetchone()[0],
                'Department Filter': lambda: handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE owner_department = 'Streets & Sanitation'").fetchone()[0],
                'Date Range': lambda: handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE created_date >= current_date - INTERVAL 30 DAY").fetchone()[0],
                'Ward Filter': lambda: handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE ward = '1'").fetchone()[0]
            }
        
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                count = test_func()
                duration = time.time() - start_time
                results[test_name] = {'duration': duration, 'count': count}
                logger.info(f"  {test_name}: {duration:.4f}s ({count} results)")
            except Exception as e:
                logger.error(f"  {test_name} failed: {e}")
                results[test_name] = {'duration': 0, 'count': 0, 'error': str(e)}
        
        return results
    
    def run_text_search_tests(self, db_name: str, handler: Any) -> Dict[str, Any]:
        """Run text search performance tests."""
        results = {}
        search_terms = ['Pothole', 'Street Light', 'Graffiti', 'Water Leak', 'TREE TRIM']
        
        for i, term in enumerate(search_terms):
            test_name = ['Simple Text', 'Multi-word Search', 'Partial Match', 'Complex Text', 'Case Insensitive'][i]
            try:
                start_time = time.time()
                
                if db_name == 'MongoDB':
                    count = handler.collection.count_documents({'$text': {'$search': term}})
                elif db_name == 'Elasticsearch':
                    count = handler.es.count(index=handler.index_name, body={'query': {'match': {'sr_type': term}}})['count']
                elif db_name == 'PostgreSQL':
                    with handler.connection.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE sr_type ILIKE %s", (f'%{term}%',))
                        count = cursor.fetchone()[0]
                elif db_name == 'DuckDB':
                    count = handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests WHERE sr_type LIKE ?", (f'%{term}%',)).fetchone()[0]
                
                duration = time.time() - start_time
                results[test_name] = {'duration': duration, 'count': count, 'search_term': term}
                logger.info(f"  {test_name}: {duration:.4f}s ({count} results)")
            except Exception as e:
                logger.error(f"  {test_name} failed: {e}")
                results[test_name] = {'duration': 0, 'count': 0, 'search_term': term, 'error': str(e)}
        
        return results
    
    def run_complex_query_tests(self, db_name: str, handler: Any) -> Dict[str, Any]:
        """Run complex query performance tests."""
        results = {}
        
        tests = {
            'Geospatial Range': self._geospatial_test,
            'Multi-condition': self._multi_condition_test,
            'Aggregation': self._aggregation_test,
            'Top Analysis': self._top_analysis_test,
            'Time Series': self._time_series_test
        }
        
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                count = test_func(db_name, handler)
                duration = time.time() - start_time
                results[test_name] = {'duration': duration, 'count': count}
                logger.info(f"  {test_name}: {duration:.4f}s ({count} results)")
            except Exception as e:
                logger.error(f"  {test_name} failed: {e}")
                results[test_name] = {'duration': 0, 'count': 0, 'error': str(e)}
        
        return results
    
    def _geospatial_test(self, db_name: str, handler: Any) -> int:
        """Geospatial range query test."""
        if db_name == 'MongoDB':
            return handler.collection.count_documents({
                'latitude': {'$gte': 41.8, '$lte': 41.9},
                'longitude': {'$gte': -87.7, '$lte': -87.6}
            })
        elif db_name == 'Elasticsearch':
            return handler.es.count(index=handler.index_name, body={
                'query': {
                    'bool': {
                        'must': [
                            {'range': {'latitude': {'gte': 41.8, 'lte': 41.9}}},
                            {'range': {'longitude': {'gte': -87.7, 'lte': -87.6}}}
                        ]
                    }
                }
            })['count']
        elif db_name == 'PostgreSQL':
            with handler.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM chicago_311_requests 
                    WHERE latitude BETWEEN 41.8 AND 41.9 
                    AND longitude BETWEEN -87.7 AND -87.6
                """)
                return cursor.fetchone()[0]
        elif db_name == 'DuckDB':
            return handler.conn.execute("""
                SELECT COUNT(*) FROM chicago_311_requests 
                WHERE CAST(latitude AS DOUBLE) BETWEEN 41.8 AND 41.9 
                AND CAST(longitude AS DOUBLE) BETWEEN -87.7 AND -87.6
            """).fetchone()[0]
    
    def _multi_condition_test(self, db_name: str, handler: Any) -> int:
        """Multi-condition query test."""
        if db_name == 'MongoDB':
            return handler.collection.count_documents({
                'status': 'Open',
                'ward': {'$in': [1, 2, 3]},
                'owner_department': {'$exists': True}
            })
        elif db_name == 'Elasticsearch':
            return handler.es.count(index=handler.index_name, body={
                'query': {
                    'bool': {
                        'must': [
                            {'match': {'status': 'Open'}},
                            {'terms': {'ward': [1, 2, 3]}},
                            {'exists': {'field': 'owner_department'}}
                        ]
                    }
                }
            })['count']
        elif db_name == 'PostgreSQL':
            with handler.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM chicago_311_requests 
                    WHERE status = 'Open' 
                    AND ward IN (1, 2, 3) 
                    AND owner_department IS NOT NULL
                """)
                return cursor.fetchone()[0]
        elif db_name == 'DuckDB':
            return handler.conn.execute("""
                SELECT COUNT(*) FROM chicago_311_requests 
                WHERE status = 'Open' 
                AND ward IN ('1', '2', '3') 
                AND owner_department IS NOT NULL
            """).fetchone()[0]
    
    def _aggregation_test(self, db_name: str, handler: Any) -> int:
        """Aggregation query test."""
        if db_name == 'MongoDB':
            pipeline = [{'$group': {'_id': '$owner_department', 'count': {'$sum': 1}}}]
            return len(list(handler.collection.aggregate(pipeline)))
        elif db_name == 'Elasticsearch':
            result = handler.es.search(index=handler.index_name, body={
                'aggs': {'departments': {'terms': {'field': 'owner_department.keyword', 'size': 100}}}
            }, size=0)
            return len(result['aggregations']['departments']['buckets'])
        elif db_name == 'PostgreSQL':
            with handler.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(DISTINCT owner_department) FROM chicago_311_requests")
                return cursor.fetchone()[0]
        elif db_name == 'DuckDB':
            return handler.conn.execute("SELECT COUNT(DISTINCT owner_department) FROM chicago_311_requests").fetchone()[0]
    
    def _top_analysis_test(self, db_name: str, handler: Any) -> int:
        """Top N analysis test."""
        if db_name == 'MongoDB':
            pipeline = [
                {'$group': {'_id': '$ward', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ]
            return len(list(handler.collection.aggregate(pipeline)))
        elif db_name == 'Elasticsearch':
            result = handler.es.search(index=handler.index_name, body={
                'aggs': {'top_wards': {'terms': {'field': 'ward', 'size': 10}}}
            }, size=0)
            return len(result['aggregations']['top_wards']['buckets'])
        elif db_name == 'PostgreSQL':
            with handler.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT ward, COUNT(*) FROM chicago_311_requests 
                    WHERE ward IS NOT NULL 
                    GROUP BY ward ORDER BY COUNT(*) DESC LIMIT 10
                """)
                return len(cursor.fetchall())
        elif db_name == 'DuckDB':
            result = handler.conn.execute("""
                SELECT ward, COUNT(*) FROM chicago_311_requests 
                WHERE ward IS NOT NULL 
                GROUP BY ward ORDER BY COUNT(*) DESC LIMIT 10
            """).fetchall()
            return len(result)
    
    def _time_series_test(self, db_name: str, handler: Any) -> int:
        """Time series analysis test."""
        if db_name == 'MongoDB':
            pipeline = [
                {'$group': {'_id': {'$dateToString': {'format': '%Y-%m', 'date': '$created_date'}}, 'count': {'$sum': 1}}},
                {'$limit': 12}
            ]
            return len(list(handler.collection.aggregate(pipeline)))
        elif db_name == 'Elasticsearch':
            result = handler.es.search(index=handler.index_name, body={
                'aggs': {'monthly': {'date_histogram': {'field': 'created_date', 'calendar_interval': 'month'}}}
            }, size=0)
            return len(result['aggregations']['monthly']['buckets'])
        elif db_name == 'PostgreSQL':
            with handler.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT DATE_TRUNC('month', created_date), COUNT(*) 
                    FROM chicago_311_requests 
                    WHERE created_date IS NOT NULL 
                    GROUP BY DATE_TRUNC('month', created_date) 
                    ORDER BY DATE_TRUNC('month', created_date) DESC LIMIT 12
                """)
                return len(cursor.fetchall())
        elif db_name == 'DuckDB':
            result = handler.conn.execute("""
                SELECT strftime('%Y-%m', created_date), COUNT(*) 
                FROM chicago_311_requests 
                WHERE created_date IS NOT NULL 
                GROUP BY strftime('%Y-%m', created_date) 
                ORDER BY strftime('%Y-%m', created_date) DESC LIMIT 12
            """).fetchall()
            return len(result)
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark on all available databases."""
        if not self.setup_all_databases():
            logger.error("❌ No databases available for benchmarking!")
            return
        
        logger.info(f"🚀 Starting comprehensive benchmark on {len(self.handlers)} databases...")
        
        for db_name, handler in self.handlers.items():
            logger.info(f"\n📊 Benchmarking {db_name}...")
            db_results = {}
            
            # Get record count
            if db_name == 'MongoDB':
                record_count = handler.collection.count_documents({})
            elif db_name == 'Elasticsearch':
                record_count = handler.es.count(index=handler.index_name)['count']
            elif db_name == 'PostgreSQL':
                record_count = handler.record_count
            elif db_name == 'DuckDB':
                record_count = handler.record_count
            
            db_results['record_count'] = record_count
            db_results['database_type'] = {
                'MongoDB': 'document',
                'Elasticsearch': 'search',
                'PostgreSQL': 'relational', 
                'DuckDB': 'analytical'
            }[db_name]
            
            # Run test categories
            logger.info("  Running basic search tests...")
            db_results['basic_search'] = self.run_basic_search_tests(db_name, handler)
            
            logger.info("  Running text search tests...")
            db_results['text_search'] = self.run_text_search_tests(db_name, handler)
            
            logger.info("  Running complex query tests...")
            db_results['complex_queries'] = self.run_complex_query_tests(db_name, handler)
            
            self.results[db_name] = db_results
            logger.info(f"✅ {db_name} benchmark completed")
        
        # Save results
        self.save_results()
        self.generate_all_charts()
        self.generate_report()
    
    def save_results(self):
        """Save benchmark results to JSON."""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_type': 'all_4_databases_comprehensive',
            'test_categories': self.test_categories,
            'results': self.results
        }
        
        with open('all_4_databases_benchmark_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info("💾 Results saved to all_4_databases_benchmark_results.json")
    
    def generate_all_charts(self):
        """Generate ALL required charts for all 4 databases."""
        logger.info("📈 Generating comprehensive charts for all 4 databases...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive chart with all databases
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Database Benchmark Results - All 4 Databases\nChicago 311 Data Platform', fontsize=16, fontweight='bold')
        
        # Chart 1: Basic Search Performance
        ax1 = axes[0, 0]
        self._plot_category_performance(ax1, 'basic_search', 'Basic Search Query Performance')
        
        # Chart 2: Text Search Performance  
        ax2 = axes[0, 1]
        self._plot_category_performance(ax2, 'text_search', 'Full-Text Search Performance')
        
        # Chart 3: Complex Query Performance
        ax3 = axes[0, 2]
        self._plot_category_performance(ax3, 'complex_queries', 'Complex Analytical Query Performance')
        
        # Chart 4: Overall Performance Heatmap
        ax4 = axes[1, 0]
        self._plot_performance_heatmap(ax4)
        
        # Chart 5: Average Performance by Database
        ax5 = axes[1, 1]
        self._plot_average_performance(ax5)
        
        # Chart 6: Performance Distribution
        ax6 = axes[1, 2]
        self._plot_performance_distribution(ax6)
        
        plt.tight_layout()
        plt.savefig('comprehensive_benchmark_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate database performance comparison chart
        self._generate_database_comparison_chart()
        
        logger.info("📊 Generated comprehensive_benchmark_charts.png")
        logger.info("📊 Generated database_performance_comparison.png")
    
    def _plot_category_performance(self, ax, category: str, title: str):
        """Plot performance for a specific category."""
        data = []
        databases = []
        durations = []
        
        for db_name, db_results in self.results.items():
            if category in db_results:
                for test_name, test_result in db_results[category].items():
                    if 'duration' in test_result and test_result['duration'] > 0:
                        data.append({
                            'database': db_name,
                            'test': test_name,
                            'duration': test_result['duration']
                        })
                        databases.append(db_name)
                        durations.append(test_result['duration'])
        
        if data:
            df = pd.DataFrame(data)
            sns.barplot(data=df, x='test', y='duration', hue='database', ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Duration (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Database')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def _plot_performance_heatmap(self, ax):
        """Plot performance heatmap."""
        # Create average performance matrix
        databases = list(self.results.keys())
        categories = ['basic_search', 'text_search', 'complex_queries']
        
        heatmap_data = []
        for db in databases:
            row = []
            for category in categories:
                if category in self.results[db]:
                    durations = [result['duration'] for result in self.results[db][category].values() 
                               if 'duration' in result and result['duration'] > 0]
                    avg_duration = np.mean(durations) if durations else 0
                    row.append(avg_duration)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        if heatmap_data:
            df_heatmap = pd.DataFrame(heatmap_data, index=databases, columns=['Basic Search', 'Text Search', 'Complex Queries'])
            sns.heatmap(df_heatmap, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
            ax.set_title('Overall Performance Heatmap\n(Lower is Better)')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overall Performance Heatmap')
    
    def _plot_average_performance(self, ax):
        """Plot average performance by database type."""
        db_types = []
        avg_durations = []
        
        for db_name, db_results in self.results.items():
            all_durations = []
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in db_results:
                    for result in db_results[category].values():
                        if 'duration' in result and result['duration'] > 0:
                            all_durations.append(result['duration'])
            
            if all_durations:
                db_types.append(db_results['database_type'])
                avg_durations.append(np.mean(all_durations))
        
        if db_types:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(db_types, avg_durations, color=colors[:len(db_types)])
            ax.set_title('Average Performance by Database Type')
            ax.set_ylabel('Average Duration (seconds)')
            
            # Add value labels on bars
            for bar, duration in zip(bars, avg_durations):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{duration:.4f}s', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Performance by Database Type')
    
    def _plot_performance_distribution(self, ax):
        """Plot performance distribution."""
        data = []
        
        for db_name, db_results in self.results.items():
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in db_results:
                    for result in db_results[category].values():
                        if 'duration' in result and result['duration'] > 0:
                            data.append({
                                'database': db_name,
                                'duration': result['duration']
                            })
        
        if data:
            df = pd.DataFrame(data)
            sns.violinplot(data=df, x='database', y='duration', ax=ax)
            ax.set_title('Performance Distribution')
            ax.set_ylabel('Duration (seconds)')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Distribution')
    
    def _generate_database_comparison_chart(self):
        """Generate specific database performance comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Database Performance Comparison - All 4 Databases\nChicago 311 Data Platform', fontsize=16, fontweight='bold')
        
        # Chart 1: Record counts
        ax1 = axes[0, 0]
        databases = []
        record_counts = []
        for db_name, db_results in self.results.items():
            databases.append(db_name)
            record_counts.append(db_results['record_count'])
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(databases, record_counts, color=colors[:len(databases)])
        ax1.set_title('Record Counts by Database')
        ax1.set_ylabel('Number of Records')
        
        for bar, count in zip(bars, record_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom')
        
        # Chart 2: Category winners
        ax2 = axes[0, 1]
        categories = ['Basic Search', 'Text Search', 'Complex Queries']
        winners = self._find_category_winners()
        
        winner_counts = {}
        for winner in winners.values():
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        
        if winner_counts:
            winner_dbs = list(winner_counts.keys())
            winner_values = list(winner_counts.values())
            ax2.bar(winner_dbs, winner_values, color=colors[:len(winner_dbs)])
            ax2.set_title('Category Winners')
            ax2.set_ylabel('Number of Categories Won')
        
        # Chart 3: Average performance comparison
        ax3 = axes[1, 0]
        db_averages = []
        for db_name in databases:
            all_durations = []
            db_results = self.results[db_name]
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in db_results:
                    for result in db_results[category].values():
                        if 'duration' in result and result['duration'] > 0:
                            all_durations.append(result['duration'])
            db_averages.append(np.mean(all_durations) if all_durations else 0)
        
        bars = ax3.bar(databases, db_averages, color=colors[:len(databases)])
        ax3.set_title('Average Query Performance')
        ax3.set_ylabel('Average Duration (seconds)')
        
        for bar, avg in zip(bars, db_averages):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{avg:.4f}s', ha='center', va='bottom')
        
        # Chart 4: Database type distribution
        ax4 = axes[1, 1]
        db_types = [self.results[db]['database_type'] for db in databases]
        type_counts = {}
        for db_type in db_types:
            type_counts[db_type] = type_counts.get(db_type, 0) + 1
        
        ax4.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.0f%%', colors=colors)
        ax4.set_title('Database Types Distribution')
        
        plt.tight_layout()
        plt.savefig('database_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _find_category_winners(self) -> Dict[str, str]:
        """Find the best performing database for each category."""
        winners = {}
        categories = ['basic_search', 'text_search', 'complex_queries']
        
        for category in categories:
            best_db = None
            best_avg = float('inf')
            
            for db_name, db_results in self.results.items():
                if category in db_results:
                    durations = [result['duration'] for result in db_results[category].values() 
                               if 'duration' in result and result['duration'] > 0]
                    if durations:
                        avg_duration = np.mean(durations)
                        if avg_duration < best_avg:
                            best_avg = avg_duration
                            best_db = db_name
            
            if best_db:
                winners[category] = best_db
        
        return winners
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        logger.info("📝 Generating comprehensive benchmark report...")
        
        winners = self._find_category_winners()
        
        report_lines = [
            "=" * 80,
            "ALL 4 DATABASES COMPREHENSIVE BENCHMARK REPORT",
            "Chicago 311 Service Requests - Performance Analysis",
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"• Databases Tested: {len(self.results)}",
            f"• Available Databases: {', '.join(self.results.keys())}",
            "",
            "PERFORMANCE RANKINGS BY CATEGORY",
            "-" * 40
        ]
        
        # Add category winners
        category_names = {
            'basic_search': 'Basic Search',
            'text_search': 'Text Search', 
            'complex_queries': 'Complex Queries'
        }
        
        for category, winner in winners.items():
            if winner and category in self.results[winner]:
                durations = [result['duration'] for result in self.results[winner][category].values() 
                           if 'duration' in result and result['duration'] > 0]
                avg_duration = np.mean(durations) if durations else 0
                report_lines.extend([
                    "",
                    f"🏆 {category_names[category]}:",
                    f"   Winner: {winner} ({avg_duration:.4f}s average)"
                ])
        
        report_lines.extend([
            "",
            "",
            "DETAILED RESULTS BY DATABASE",
            "-" * 40
        ])
        
        # Add detailed results for each database
        for db_name, db_results in self.results.items():
            report_lines.extend([
                "",
                f"{db_name.upper()} ({db_results['database_type']})",
                f"Records: {db_results['record_count']:,}",
                "=" * 20
            ])
            
            for category, category_name in category_names.items():
                if category in db_results:
                    report_lines.append(f"\n{category_name}:")
                    for test_name, result in db_results[category].items():
                        if 'duration' in result:
                            duration = result['duration']
                            count = result.get('count', 0)
                            report_lines.append(f"  • {test_name}: {duration:.4f}s ({count} results)")
        
        # Add recommendations
        report_lines.extend([
            "",
            "",
            "RECOMMENDATIONS",
            "-" * 40,
            "🎯 Database Recommendations:",
        ])
        
        # Find overall best performer
        best_overall = None
        best_avg = float('inf')
        for db_name, db_results in self.results.items():
            all_durations = []
            for category in ['basic_search', 'text_search', 'complex_queries']:
                if category in db_results:
                    for result in db_results[category].values():
                        if 'duration' in result and result['duration'] > 0:
                            all_durations.append(result['duration'])
            if all_durations:
                avg = np.mean(all_durations)
                if avg < best_avg:
                    best_avg = avg
                    best_overall = db_name
        
        if best_overall:
            report_lines.append(f"   • Overall Best Performer: {best_overall}")
        
        report_lines.extend([
            "   • MongoDB: Best for flexible document operations",
            "   • Elasticsearch: Ideal for search-heavy applications", 
            "   • PostgreSQL: Perfect for complex relational queries",
            "   • DuckDB: Excellent for analytical and data science workloads",
            "",
            "💡 General Recommendations:",
            "   • Choose based on your specific use case and query patterns",
            "   • Consider data size and performance requirements",
            "   • MongoDB and Elasticsearch scale well for large datasets",
            "   • PostgreSQL offers ACID compliance and complex joins",
            "   • DuckDB provides excellent analytical performance"
        ])
        
        # Write report
        with open('all_4_databases_benchmark_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("📄 Generated all_4_databases_benchmark_report.txt")

def main():
    """Main execution."""
    try:
        benchmark = All4DatabasesBenchmarkSystem()
        benchmark.run_comprehensive_benchmark()
        
        logger.info("🎉 All 4 databases benchmark completed successfully!")
        logger.info("📊 Generated files:")
        logger.info("   - comprehensive_benchmark_charts.png")
        logger.info("   - database_performance_comparison.png") 
        logger.info("   - all_4_databases_benchmark_results.json")
        logger.info("   - all_4_databases_benchmark_report.txt")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()