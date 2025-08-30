#!/usr/bin/env python3
"""
Comprehensive Database Connection and Benchmark System
Tests MongoDB, Elasticsearch, PostgreSQL, and DuckDB
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler
from src.databases.postgresql_handler import PostgreSQLHandler
from src.databases.duckdb_handler import DuckDBHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseBenchmarkSuite:
    """Comprehensive database benchmark and testing suite."""
    
    def __init__(self):
        self.results = {
            'MongoDB': {'connected': False, 'count': 0, 'benchmarks': {}},
            'Elasticsearch': {'connected': False, 'count': 0, 'benchmarks': {}},
            'PostgreSQL': {'connected': False, 'count': 0, 'benchmarks': {}},
            'DuckDB': {'connected': False, 'count': 0, 'benchmarks': {}}
        }
        
        # Standard benchmark queries (will be adapted per database)
        self.benchmark_queries = {
            'count_all': "Count all records",
            'status_filter': "Filter by status = 'Open'", 
            'date_range': "Records in last 30 days",
            'complex_aggregation': "Group by status and count",
            'text_search': "Search in service request types"
        }
    
    def test_database_connections(self):
        """Test connections to all four databases."""
        print("🔍 Comprehensive Database Connection Check")
        print("=" * 80)
        
        # Test MongoDB
        try:
            mongo_handler = MongoDBHandler()
            count = mongo_handler.collection.count_documents({})
            self.results['MongoDB'] = {
                'connected': True, 
                'count': count, 
                'handler': mongo_handler,
                'benchmarks': {}
            }
            print(f"✅ MongoDB: {count:,} records")
        except Exception as e:
            print(f"❌ MongoDB: {e}")
            self.results['MongoDB']['error'] = str(e)
        
        # Test Elasticsearch
        try:
            es_handler = ElasticsearchHandler()
            result = es_handler.es.count(index=es_handler.index_name)
            count = result['count']
            self.results['Elasticsearch'] = {
                'connected': True, 
                'count': count, 
                'handler': es_handler,
                'benchmarks': {}
            }
            print(f"✅ Elasticsearch: {count:,} documents")
        except Exception as e:
            print(f"❌ Elasticsearch: {e}")
            self.results['Elasticsearch']['error'] = str(e)
        
        # Test PostgreSQL
        try:
            pg_handler = PostgreSQLHandler()
            tables = pg_handler.list_tables()
            total_count = 0
            main_table = None
            
            for table in tables:
                count = pg_handler.get_table_count(table)
                total_count += count
                if count > 1000:  # Assume this is the main table
                    main_table = table
            
            self.results['PostgreSQL'] = {
                'connected': True, 
                'count': total_count, 
                'handler': pg_handler,
                'main_table': main_table,
                'tables': tables,
                'benchmarks': {}
            }
            print(f"✅ PostgreSQL: {total_count:,} records across {len(tables)} tables")
            if main_table:
                print(f"   📊 Main table: {main_table}")
        except Exception as e:
            print(f"❌ PostgreSQL: {e}")
            self.results['PostgreSQL']['error'] = str(e)
        
        # Test DuckDB
        try:
            duckdb_handler = DuckDBHandler()
            tables = duckdb_handler.list_tables()
            total_count = 0
            main_table = None
            
            for table in tables:
                count = duckdb_handler.get_table_count(table)
                total_count += count
                if count > 1000:  # Assume this is the main table
                    main_table = table
            
            self.results['DuckDB'] = {
                'connected': True, 
                'count': total_count, 
                'handler': duckdb_handler,
                'main_table': main_table,
                'tables': tables,
                'benchmarks': {}
            }
            print(f"✅ DuckDB: {total_count:,} records across {len(tables)} tables")
            if main_table:
                print(f"   📊 Main table: {main_table}")
        except Exception as e:
            print(f"❌ DuckDB: {e}")
            self.results['DuckDB']['error'] = str(e)
    
    def run_benchmarks(self):
        """Run benchmark queries on all connected databases."""
        print("\n🚀 Running Performance Benchmarks")
        print("=" * 80)
        
        for db_name, db_info in self.results.items():
            if not db_info['connected']:
                continue
                
            print(f"\n📊 Benchmarking {db_name}...")
            
            if db_name == 'MongoDB':
                self._benchmark_mongodb(db_info)
            elif db_name == 'Elasticsearch':
                self._benchmark_elasticsearch(db_info)
            elif db_name == 'PostgreSQL':
                self._benchmark_postgresql(db_info)
            elif db_name == 'DuckDB':
                self._benchmark_duckdb(db_info)
    
    def _benchmark_mongodb(self, db_info):
        """Run MongoDB-specific benchmarks."""
        handler = db_info['handler']
        benchmarks = {}
        
        # Query 1: Count all records
        start_time = time.time()
        count = handler.collection.count_documents({})
        benchmarks['count_all'] = time.time() - start_time
        
        # Query 2: Filter by status
        start_time = time.time()
        open_count = handler.collection.count_documents({'status': 'Open'})
        benchmarks['status_filter'] = time.time() - start_time
        
        # Query 3: Date range query
        start_time = time.time()
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_count = handler.collection.count_documents({
            'created_date': {'$gte': thirty_days_ago}
        })
        benchmarks['date_range'] = time.time() - start_time
        
        # Query 4: Aggregation
        start_time = time.time()
        pipeline = [
            {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        list(handler.collection.aggregate(pipeline))
        benchmarks['complex_aggregation'] = time.time() - start_time
        
        # Query 5: Text search
        start_time = time.time()
        text_search = handler.collection.count_documents({
            'sr_type': {'$regex': 'Pothole', '$options': 'i'}
        })
        benchmarks['text_search'] = time.time() - start_time
        
        db_info['benchmarks'] = benchmarks
        for query, duration in benchmarks.items():
            print(f"   {query}: {duration:.4f}s")
    
    def _benchmark_elasticsearch(self, db_info):
        """Run Elasticsearch-specific benchmarks."""
        handler = db_info['handler']
        benchmarks = {}
        
        # Query 1: Count all documents
        start_time = time.time()
        handler.es.count(index=handler.index_name)
        benchmarks['count_all'] = time.time() - start_time
        
        # Query 2: Filter by status
        start_time = time.time()
        query = {"query": {"term": {"status": "Open"}}}
        handler.es.count(index=handler.index_name, body=query)
        benchmarks['status_filter'] = time.time() - start_time
        
        # Query 3: Date range query
        start_time = time.time()
        query = {
            "query": {
                "range": {
                    "creation_date": {
                        "gte": "now-30d"
                    }
                }
            }
        }
        handler.es.count(index=handler.index_name, body=query)
        benchmarks['date_range'] = time.time() - start_time
        
        # Query 4: Aggregation
        start_time = time.time()
        query = {
            "size": 0,
            "aggs": {
                "status_counts": {
                    "terms": {"field": "status"}
                }
            }
        }
        handler.es.search(index=handler.index_name, body=query)
        benchmarks['complex_aggregation'] = time.time() - start_time
        
        # Query 5: Text search
        start_time = time.time()
        query = {
            "query": {
                "match": {
                    "sr_type": "Pothole"
                }
            }
        }
        handler.es.count(index=handler.index_name, body=query)
        benchmarks['text_search'] = time.time() - start_time
        
        db_info['benchmarks'] = benchmarks
        for query, duration in benchmarks.items():
            print(f"   {query}: {duration:.4f}s")
    
    def _benchmark_postgresql(self, db_info):
        """Run PostgreSQL-specific benchmarks."""
        if 'error' in db_info:
            print(f"   ❌ Cannot benchmark - {db_info['error']}")
            return
            
        handler = db_info['handler']
        benchmarks = {}
        main_table = db_info.get('main_table', 'chicago_311_requests')  # Default table name
        
        # Skip if no main table identified
        if not main_table or main_table not in db_info.get('tables', []):
            print("   ⚠️ No suitable table found for benchmarking")
            return
        
        try:
            # Query 1: Count all records
            query = f"SELECT COUNT(*) FROM {main_table}"
            _, duration = handler.execute_query(query)
            benchmarks['count_all'] = duration
            
            # Query 2: Filter by status (assuming status column exists)
            query = f"SELECT COUNT(*) FROM {main_table} WHERE status = 'Open'"
            _, duration = handler.execute_query(query)
            benchmarks['status_filter'] = duration
            
            db_info['benchmarks'] = benchmarks
            for query, duration in benchmarks.items():
                print(f"   {query}: {duration:.4f}s")
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
    
    def _benchmark_duckdb(self, db_info):
        """Run DuckDB-specific benchmarks."""
        if 'error' in db_info:
            print(f"   ❌ Cannot benchmark - {db_info['error']}")
            return
            
        handler = db_info['handler']
        benchmarks = {}
        main_table = db_info.get('main_table')
        
        if not main_table:
            # Try common table names
            for table_name in ['fact_requests_v3', 'chicago_311', 'requests']:
                if table_name in db_info.get('tables', []):
                    main_table = table_name
                    break
        
        if not main_table:
            print("   ⚠️ No suitable table found for benchmarking")
            return
        
        try:
            # Query 1: Count all records
            query = f"SELECT COUNT(*) FROM {main_table}"
            _, duration = handler.execute_query(query)
            benchmarks['count_all'] = duration
            
            # Query 2: Filter by status (if column exists)
            query = f"SELECT COUNT(*) FROM {main_table} WHERE status = 'Open'"
            _, duration = handler.execute_query(query)
            benchmarks['status_filter'] = duration
            
            db_info['benchmarks'] = benchmarks
            for query, duration in benchmarks.items():
                print(f"   {query}: {duration:.4f}s")
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
    
    def generate_charts(self):
        """Generate performance comparison charts."""
        print("\n📈 Generating Performance Charts...")
        
        # Collect benchmark data
        chart_data = []
        for db_name, db_info in self.results.items():
            if db_info['connected'] and db_info['benchmarks']:
                for query, duration in db_info['benchmarks'].items():
                    chart_data.append({
                        'Database': db_name,
                        'Query': query,
                        'Duration (seconds)': duration
                    })
        
        if not chart_data:
            print("❌ No benchmark data available for charting")
            return
        
        df = pd.DataFrame(chart_data)
        
        # Create performance comparison chart
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df, x='Query', y='Duration (seconds)', hue='Database')
        plt.title('Database Performance Comparison - Query Execution Times')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('database_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create database record count comparison
        plt.figure(figsize=(10, 6))
        db_names = []
        record_counts = []
        for db_name, db_info in self.results.items():
            if db_info['connected']:
                db_names.append(db_name)
                record_counts.append(db_info['count'])
        
        plt.bar(db_names, record_counts)
        plt.title('Database Record Counts')
        plt.ylabel('Number of Records')
        plt.yscale('log')  # Log scale due to potentially large differences
        for i, count in enumerate(record_counts):
            plt.text(i, count, f'{count:,}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('database_record_counts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Charts saved as 'database_performance_comparison.png' and 'database_record_counts.png'")
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        # Clean results for JSON serialization
        clean_results = {}
        for db_name, db_info in self.results.items():
            clean_results[db_name] = {
                'connected': db_info['connected'],
                'count': db_info['count'],
                'benchmarks': db_info['benchmarks']
            }
            if 'error' in db_info:
                clean_results[db_name]['error'] = db_info['error']
            if 'main_table' in db_info:
                clean_results[db_name]['main_table'] = db_info['main_table']
        
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'databases': clean_results
        }
        
        with open('comprehensive_benchmark_results.json', 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print("✅ Results saved to 'comprehensive_benchmark_results.json'")
    
    def print_summary(self):
        """Print a comprehensive summary."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE DATABASE SUMMARY")
        print("=" * 80)
        
        connected_dbs = [name for name, info in self.results.items() if info['connected']]
        print(f"🔗 Connected Databases: {len(connected_dbs)}/4 ({', '.join(connected_dbs)})")
        
        total_records = sum(info['count'] for info in self.results.values() if info['connected'])
        print(f"📊 Total Records Across All DBs: {total_records:,}")
        
        expected_count = 12_300_000
        if total_records < expected_count * 0.1:
            print(f"⚠️ WARNING: Total records ({total_records:,}) significantly below expected ({expected_count:,})")
            print("   This suggests the databases may not have the full Chicago 311 dataset loaded.")
        
        # Performance summary
        if any(info['benchmarks'] for info in self.results.values() if info['connected']):
            print("\n🏆 Performance Ranking (Count All Query):")
            count_all_times = []
            for db_name, db_info in self.results.items():
                if db_info['connected'] and 'count_all' in db_info['benchmarks']:
                    count_all_times.append((db_name, db_info['benchmarks']['count_all']))
            
            count_all_times.sort(key=lambda x: x[1])
            for i, (db_name, time) in enumerate(count_all_times, 1):
                print(f"   {i}. {db_name}: {time:.4f}s")
    
    def close_connections(self):
        """Close all database connections."""
        for db_info in self.results.values():
            if 'handler' in db_info:
                try:
                    db_info['handler'].close()
                except:
                    pass

def main():
    """Main function to run comprehensive database tests."""
    suite = DatabaseBenchmarkSuite()
    
    try:
        suite.test_database_connections()
        suite.run_benchmarks()
        suite.generate_charts()
        suite.save_results()
        suite.print_summary()
    finally:
        suite.close_connections()

if __name__ == "__main__":
    main()