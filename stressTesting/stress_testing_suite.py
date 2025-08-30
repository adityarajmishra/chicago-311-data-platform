#!/usr/bin/env python3
"""
Database Stress Testing Suite
Tests breaking points and performance limits of all databases
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
import time
import threading
import concurrent.futures
import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler
from src.databases.postgresql_handler import PostgreSQLHandler
from src.databases.duckdb_handler import DuckDBHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseStressTester:
    """Comprehensive database stress testing suite."""
    
    def __init__(self):
        self.results = {
            'MongoDB': {'connected': False, 'stress_results': {}},
            'Elasticsearch': {'connected': False, 'stress_results': {}},
            'PostgreSQL': {'connected': False, 'stress_results': {}},
            'DuckDB': {'connected': False, 'stress_results': {}}
        }
        self.test_configs = {
            'concurrent_reads': [1, 5, 10, 25, 50, 100],
            'concurrent_writes': [1, 5, 10, 25, 50],
            'bulk_insert_sizes': [100, 1000, 5000, 10000, 50000],
            'query_complexities': ['simple', 'medium', 'complex']
        }
    
    def setup_connections(self):
        """Setup connections to all databases."""
        print("🔄 Setting up database connections for stress testing...")
        
        # MongoDB
        try:
            mongo_handler = MongoDBHandler()
            self.results['MongoDB']['connected'] = True
            self.results['MongoDB']['handler'] = mongo_handler
            print("✅ MongoDB connected")
        except Exception as e:
            print(f"❌ MongoDB failed: {e}")
        
        # Elasticsearch
        try:
            es_handler = ElasticsearchHandler()
            self.results['Elasticsearch']['connected'] = True
            self.results['Elasticsearch']['handler'] = es_handler
            print("✅ Elasticsearch connected")
        except Exception as e:
            print(f"❌ Elasticsearch failed: {e}")
        
        # PostgreSQL
        try:
            pg_handler = PostgreSQLHandler()
            self.results['PostgreSQL']['connected'] = True
            self.results['PostgreSQL']['handler'] = pg_handler
            print("✅ PostgreSQL connected")
        except Exception as e:
            print(f"❌ PostgreSQL failed: {e}")
        
        # DuckDB
        try:
            duckdb_handler = DuckDBHandler()
            self.results['DuckDB']['connected'] = True
            self.results['DuckDB']['handler'] = duckdb_handler
            print("✅ DuckDB connected")
        except Exception as e:
            print(f"❌ DuckDB failed: {e}")
    
    def generate_test_data(self, count: int) -> List[Dict]:
        """Generate synthetic test data for stress testing."""
        test_data = []
        status_options = ['Open', 'Closed', 'In Progress', 'Duplicate']
        sr_types = ['Pothole', 'Street Light', 'Graffiti', 'Tree Trim', 'Snow Removal']
        
        for i in range(count):
            record = {
                'sr_number': f'TEST{i:08d}',
                'sr_type': random.choice(sr_types),
                'sr_short_code': f'T{random.randint(100, 999)}',
                'status': random.choice(status_options),
                'owner_department': 'TEST_DEPT',
                'created_date': datetime.now() - timedelta(days=random.randint(0, 365)),
                'street_address': f'{random.randint(1, 9999)} Test St',
                'city': 'Chicago',
                'ward': random.randint(1, 50),
                'latitude': 41.8781 + random.uniform(-0.1, 0.1),
                'longitude': -87.6298 + random.uniform(-0.1, 0.1)
            }
            test_data.append(record)
        
        return test_data
    
    def test_concurrent_reads(self, db_name: str, concurrent_users: List[int]) -> Dict:
        """Test concurrent read performance."""
        if not self.results[db_name]['connected']:
            return {}
        
        print(f"📖 Testing concurrent reads for {db_name}...")
        results = {}
        
        for user_count in concurrent_users:
            print(f"   Testing {user_count} concurrent users...")
            
            def read_worker():
                start_time = time.time()
                try:
                    if db_name == 'MongoDB':
                        handler = self.results[db_name]['handler']
                        handler.collection.count_documents({})
                    elif db_name == 'Elasticsearch':
                        handler = self.results[db_name]['handler']
                        handler.es.count(index=handler.index_name)
                    elif db_name == 'PostgreSQL':
                        handler = self.results[db_name]['handler']
                        handler.execute_query("SELECT COUNT(*) FROM chicago_311_requests")
                    elif db_name == 'DuckDB':
                        handler = self.results[db_name]['handler']
                        handler.execute_query("SELECT COUNT(*) FROM fact_requests_v3")
                    
                    return time.time() - start_time
                except Exception as e:
                    logger.error(f"Read worker failed: {e}")
                    return float('inf')
            
            # Run concurrent reads
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = [executor.submit(read_worker) for _ in range(user_count)]
                response_times = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            
            results[user_count] = {
                'total_time': total_time,
                'avg_response_time': np.mean(response_times),
                'max_response_time': np.max(response_times),
                'min_response_time': np.min(response_times),
                'queries_per_second': user_count / total_time,
                'success_rate': sum(1 for t in response_times if t != float('inf')) / len(response_times)
            }
            
            print(f"      QPS: {results[user_count]['queries_per_second']:.2f}, "
                  f"Avg: {results[user_count]['avg_response_time']:.4f}s")
        
        return results
    
    def test_bulk_inserts(self, db_name: str, batch_sizes: List[int]) -> Dict:
        """Test bulk insert performance."""
        if not self.results[db_name]['connected']:
            return {}
        
        print(f"📝 Testing bulk inserts for {db_name}...")
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size}...")
            test_data = self.generate_test_data(batch_size)
            
            start_time = time.time()
            try:
                if db_name == 'MongoDB':
                    handler = self.results[db_name]['handler']
                    handler.collection.insert_many(test_data)
                elif db_name == 'Elasticsearch':
                    handler = self.results[db_name]['handler']
                    actions = []
                    for doc in test_data:
                        actions.append({
                            "_index": handler.index_name,
                            "_source": doc
                        })
                    from elasticsearch.helpers import bulk
                    bulk(handler.es, actions)
                
                duration = time.time() - start_time
                records_per_second = batch_size / duration if duration > 0 else 0
                
                results[batch_size] = {
                    'duration': duration,
                    'records_per_second': records_per_second,
                    'success': True
                }
                
                print(f"      {records_per_second:.2f} records/sec")
                
                # Cleanup test data
                if db_name == 'MongoDB':
                    handler.collection.delete_many({'sr_number': {'$regex': '^TEST'}})
                elif db_name == 'Elasticsearch':
                    handler.es.delete_by_query(
                        index=handler.index_name,
                        body={"query": {"prefix": {"sr_number": "TEST"}}}
                    )
                
            except Exception as e:
                results[batch_size] = {
                    'duration': float('inf'),
                    'records_per_second': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"      ❌ Failed: {e}")
        
        return results
    
    def test_memory_limits(self, db_name: str) -> Dict:
        """Test memory usage limits."""
        if not self.results[db_name]['connected']:
            return {}
        
        print(f"🧠 Testing memory limits for {db_name}...")
        
        # Test increasingly large result sets
        result_sizes = [1000, 5000, 10000, 50000, 100000]
        results = {}
        
        for size in result_sizes:
            print(f"   Testing result set size: {size}")
            start_time = time.time()
            
            try:
                if db_name == 'MongoDB':
                    handler = self.results[db_name]['handler']
                    list(handler.collection.find().limit(size))
                elif db_name == 'Elasticsearch':
                    handler = self.results[db_name]['handler']
                    handler.es.search(
                        index=handler.index_name,
                        body={"query": {"match_all": {}}},
                        size=min(size, 10000)  # ES has a default limit
                    )
                
                duration = time.time() - start_time
                results[size] = {
                    'duration': duration,
                    'success': True,
                    'records_per_second': size / duration if duration > 0 else 0
                }
                print(f"      ✅ Success in {duration:.2f}s")
                
            except Exception as e:
                results[size] = {
                    'duration': float('inf'),
                    'success': False,
                    'error': str(e)
                }
                print(f"      ❌ Failed: {e}")
                break  # Stop testing larger sizes if current fails
        
        return results
    
    def run_comprehensive_stress_test(self):
        """Run all stress tests."""
        print("🚀 Starting Comprehensive Stress Testing")
        print("=" * 80)
        
        for db_name in self.results.keys():
            if not self.results[db_name]['connected']:
                continue
            
            print(f"\n🎯 Stress Testing {db_name}")
            print("-" * 40)
            
            # Test concurrent reads
            concurrent_read_results = self.test_concurrent_reads(
                db_name, self.test_configs['concurrent_reads']
            )
            
            # Test bulk inserts
            bulk_insert_results = self.test_bulk_inserts(
                db_name, self.test_configs['bulk_insert_sizes']
            )
            
            # Test memory limits
            memory_test_results = self.test_memory_limits(db_name)
            
            self.results[db_name]['stress_results'] = {
                'concurrent_reads': concurrent_read_results,
                'bulk_inserts': bulk_insert_results,
                'memory_limits': memory_test_results
            }
    
    def generate_stress_test_charts(self):
        """Generate comprehensive stress test visualization."""
        print("\n📊 Generating stress test charts...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Database Stress Testing Results', fontsize=16, fontweight='bold')
        
        # 1. Concurrent Read Performance
        ax1 = axes[0, 0]
        for db_name, db_data in self.results.items():
            if db_data['connected'] and 'concurrent_reads' in db_data['stress_results']:
                data = db_data['stress_results']['concurrent_reads']
                if data:
                    users = list(data.keys())
                    qps = [data[u]['queries_per_second'] for u in users]
                    ax1.plot(users, qps, marker='o', label=db_name, linewidth=2)
        
        ax1.set_title('Concurrent Read Performance')
        ax1.set_xlabel('Concurrent Users')
        ax1.set_ylabel('Queries Per Second')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bulk Insert Performance
        ax2 = axes[0, 1]
        for db_name, db_data in self.results.items():
            if db_data['connected'] and 'bulk_inserts' in db_data['stress_results']:
                data = db_data['stress_results']['bulk_inserts']
                if data:
                    batch_sizes = list(data.keys())
                    rps = [data[b]['records_per_second'] for b in batch_sizes if data[b]['success']]
                    valid_sizes = [b for b in batch_sizes if data[b]['success']]
                    if valid_sizes:
                        ax2.plot(valid_sizes, rps, marker='s', label=db_name, linewidth=2)
        
        ax2.set_title('Bulk Insert Performance')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Records Per Second')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 3. Response Time Distribution
        ax3 = axes[1, 0]
        response_data = []
        for db_name, db_data in self.results.items():
            if db_data['connected'] and 'concurrent_reads' in db_data['stress_results']:
                data = db_data['stress_results']['concurrent_reads']
                for users, metrics in data.items():
                    response_data.append({
                        'Database': db_name,
                        'Users': users,
                        'Avg Response Time': metrics['avg_response_time']
                    })
        
        if response_data:
            df = pd.DataFrame(response_data)
            for db in df['Database'].unique():
                db_data = df[df['Database'] == db]
                ax3.plot(db_data['Users'], db_data['Avg Response Time'], 
                        marker='o', label=db, linewidth=2)
        
        ax3.set_title('Average Response Time vs Load')
        ax3.set_xlabel('Concurrent Users')
        ax3.set_ylabel('Average Response Time (seconds)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Success Rate Analysis
        ax4 = axes[1, 1]
        success_data = []
        for db_name, db_data in self.results.items():
            if db_data['connected'] and 'concurrent_reads' in db_data['stress_results']:
                data = db_data['stress_results']['concurrent_reads']
                for users, metrics in data.items():
                    success_data.append({
                        'Database': db_name,
                        'Users': users,
                        'Success Rate': metrics['success_rate'] * 100
                    })
        
        if success_data:
            df = pd.DataFrame(success_data)
            for db in df['Database'].unique():
                db_data = df[df['Database'] == db]
                ax4.plot(db_data['Users'], db_data['Success Rate'], 
                        marker='^', label=db, linewidth=2)
        
        ax4.set_title('Success Rate Under Load')
        ax4.set_xlabel('Concurrent Users')
        ax4.set_ylabel('Success Rate (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig('database_stress_test_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Stress test charts saved as 'database_stress_test_results.png'")
    
    def generate_breaking_point_analysis(self):
        """Analyze and identify breaking points for each database."""
        print("\n💥 Analyzing Breaking Points...")
        
        breaking_points = {}
        
        for db_name, db_data in self.results.items():
            if not db_data['connected']:
                continue
            
            bp = {'database': db_name}
            
            # Concurrent read breaking point
            if 'concurrent_reads' in db_data['stress_results']:
                read_data = db_data['stress_results']['concurrent_reads']
                # Find where QPS starts declining or success rate drops below 95%
                for users in sorted(read_data.keys()):
                    metrics = read_data[users]
                    if metrics['success_rate'] < 0.95 or metrics['avg_response_time'] > 5.0:
                        bp['max_concurrent_reads'] = users
                        break
                else:
                    bp['max_concurrent_reads'] = max(read_data.keys()) if read_data else 0
            
            # Bulk insert breaking point
            if 'bulk_inserts' in db_data['stress_results']:
                insert_data = db_data['stress_results']['bulk_inserts']
                max_successful_batch = 0
                for batch_size in sorted(insert_data.keys()):
                    if insert_data[batch_size]['success']:
                        max_successful_batch = batch_size
                    else:
                        break
                bp['max_bulk_insert_size'] = max_successful_batch
            
            # Memory limit breaking point
            if 'memory_limits' in db_data['stress_results']:
                memory_data = db_data['stress_results']['memory_limits']
                max_result_set = 0
                for size in sorted(memory_data.keys()):
                    if memory_data[size]['success']:
                        max_result_set = size
                    else:
                        break
                bp['max_result_set_size'] = max_result_set
            
            breaking_points[db_name] = bp
        
        # Save breaking points analysis
        with open('database_breaking_points.json', 'w') as f:
            json.dump(breaking_points, f, indent=2)
        
        # Generate summary report
        report = []
        report.append("=" * 80)
        report.append("DATABASE BREAKING POINT ANALYSIS")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for db_name, bp in breaking_points.items():
            report.append(f"{db_name.upper()} BREAKING POINTS:")
            report.append("-" * 40)
            report.append(f"Max Concurrent Reads: {bp.get('max_concurrent_reads', 'N/A')} users")
            report.append(f"Max Bulk Insert Size: {bp.get('max_bulk_insert_size', 'N/A')} records")
            report.append(f"Max Result Set Size: {bp.get('max_result_set_size', 'N/A')} records")
            report.append("")
        
        with open('breaking_point_analysis.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("✅ Breaking point analysis saved")
        return breaking_points
    
    def save_stress_test_results(self):
        """Save complete stress test results."""
        # Clean results for JSON serialization
        clean_results = {}
        for db_name, db_data in self.results.items():
            if 'handler' in db_data:
                del db_data['handler']  # Remove handler objects
            clean_results[db_name] = db_data
        
        timestamp = datetime.now().isoformat()
        results_with_metadata = {
            'timestamp': timestamp,
            'test_configs': self.test_configs,
            'results': clean_results
        }
        
        with open('stress_test_results.json', 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print("✅ Complete stress test results saved")
    
    def close_connections(self):
        """Close all database connections."""
        for db_data in self.results.values():
            if 'handler' in db_data:
                try:
                    db_data['handler'].close()
                except:
                    pass

def main():
    """Run comprehensive stress testing."""
    tester = DatabaseStressTester()
    
    try:
        tester.setup_connections()
        tester.run_comprehensive_stress_test()
        tester.generate_stress_test_charts()
        tester.generate_breaking_point_analysis()
        tester.save_stress_test_results()
        
        print("\n🎉 Stress testing completed successfully!")
        print("📁 Generated files:")
        print("   • database_stress_test_results.png")
        print("   • database_breaking_points.json")
        print("   • breaking_point_analysis.txt")
        print("   • stress_test_results.json")
        
    except Exception as e:
        print(f"❌ Stress testing failed: {e}")
    finally:
        tester.close_connections()

if __name__ == "__main__":
    main()