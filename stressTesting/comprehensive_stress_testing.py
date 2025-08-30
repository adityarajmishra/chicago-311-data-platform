#!/usr/bin/env python3
"""
Comprehensive Stress Testing System
Tests all 4 databases under heavy load with dummy data injection
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
import random
import string
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
import duckdb
import uuid

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveStressTester:
    """Complete stress testing system for all databases."""
    
    def __init__(self):
        self.databases = {}
        self.stress_results = {}
        self.baseline_counts = {}
        
        # Stress test configurations
        self.test_configs = {
            'concurrent_reads': [1, 5, 10, 25, 50, 100, 200],
            'concurrent_writes': [1, 5, 10, 25, 50, 100],
            'bulk_insert_sizes': [1000, 5000, 10000, 25000, 50000, 100000],
            'load_test_durations': [30, 60, 120],  # seconds
            'dummy_data_scales': [10000, 50000, 100000, 500000]  # records to inject
        }
    
    def setup_database_connections(self):
        """Setup connections for stress testing."""
        print("🔄 Setting up databases for stress testing...")
        
        # MongoDB
        try:
            mongo_handler = MongoDBHandler()
            count = mongo_handler.collection.count_documents({})
            self.databases['MongoDB'] = {
                'handler': mongo_handler,
                'type': 'document',
                'baseline_count': count
            }
            self.baseline_counts['MongoDB'] = count
            print(f"✅ MongoDB: {count:,} records")
        except Exception as e:
            print(f"❌ MongoDB: {e}")
        
        # Elasticsearch
        try:
            es_handler = ElasticsearchHandler()
            result = es_handler.es.count(index=es_handler.index_name)
            count = result['count']
            self.databases['Elasticsearch'] = {
                'handler': es_handler,
                'type': 'search_engine',
                'baseline_count': count
            }
            self.baseline_counts['Elasticsearch'] = count
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
            
            self.databases['PostgreSQL'] = {
                'handler': conn,
                'type': 'relational',
                'baseline_count': count
            }
            self.baseline_counts['PostgreSQL'] = count
            print(f"✅ PostgreSQL: {count:,} records")
        except Exception as e:
            print(f"❌ PostgreSQL: {e}")
        
        # DuckDB
        try:
            conn = duckdb.connect("chicago_311_complete.duckdb")
            result = conn.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()
            count = result[0] if result else 0
            
            self.databases['DuckDB'] = {
                'handler': conn,
                'type': 'analytical',
                'baseline_count': count
            }
            self.baseline_counts['DuckDB'] = count
            print(f"✅ DuckDB: {count:,} records")
        except Exception as e:
            print(f"❌ DuckDB: {e}")
        
        if not self.databases:
            raise Exception("No databases available for stress testing!")
        
        print(f"📊 Stress testing ready: {list(self.databases.keys())}")
    
    def generate_dummy_data(self, count: int) -> List[Dict]:
        """Generate realistic dummy data for stress testing."""
        dummy_data = []
        
        # Realistic data options
        sr_types = [
            'Pothole in Street', 'Street Light Out', 'Graffiti Removal', 
            'Tree Trim', 'Alley Light Out', 'Abandoned Vehicle', 
            'Garbage Cart Black Maintenance', 'Water Leak'
        ]
        
        departments = [
            'Streets & Sanitation', 'Transportation', 'Water Management',
            'Buildings', 'Police', 'Fire', 'Aviation'
        ]
        
        statuses = ['Open', 'Completed', 'In Progress', 'Duplicate']
        origins = ['Phone', 'Internet', 'Mobile']
        
        # Chicago coordinates range
        lat_min, lat_max = 41.6, 42.0
        lng_min, lng_max = -87.9, -87.5
        
        for i in range(count):
            # Generate unique ID for stress test
            stress_id = f"STRESS_{int(time.time() * 1000000)}_{i}"
            
            dummy_record = {
                'sr_number': stress_id,
                'sr_type': random.choice(sr_types),
                'sr_short_code': 'STR',
                'owner_department': random.choice(departments),
                'status': random.choice(statuses),
                'origin': random.choice(origins),
                'created_date': datetime.now() - timedelta(days=random.randint(0, 365)),
                'last_modified_date': datetime.now() - timedelta(days=random.randint(0, 30)),
                'closed_date': datetime.now() - timedelta(days=random.randint(0, 10)) if random.random() > 0.3 else None,
                'street_address': f'{random.randint(1, 9999)} {random.choice(["N", "S", "E", "W"])} Test St',
                'city': 'Chicago',
                'state': 'Illinois',
                'zip_code': str(random.randint(60601, 60827)),
                'street_number': str(random.randint(1, 9999)),
                'street_direction': random.choice(['N', 'S', 'E', 'W']),
                'street_name': f'Test {random.choice(["Oak", "Main", "Park", "Lake", "State"])}',
                'street_type': random.choice(['St', 'Ave', 'Blvd', 'Dr']),
                'duplicate': random.choice([True, False]),
                'legacy_record': False,
                'latitude': round(random.uniform(lat_min, lat_max), 6),
                'longitude': round(random.uniform(lng_min, lng_max), 6),
                'ward': random.randint(1, 50),
                'community_area': random.randint(1, 77),
                'police_district': str(random.randint(1, 25)).zfill(3)
            }
            
            dummy_data.append(dummy_record)
        
        return dummy_data
    
    def inject_dummy_data(self, db_name: str, data_count: int) -> Dict:
        """Inject dummy data into a specific database."""
        print(f"💉 Injecting {data_count:,} dummy records into {db_name}...")
        
        if db_name not in self.databases:
            return {'success': False, 'error': 'Database not available'}
        
        try:
            dummy_data = self.generate_dummy_data(data_count)
            handler = self.databases[db_name]['handler']
            
            start_time = time.time()
            
            if db_name == 'MongoDB':
                # Insert in batches
                batch_size = 5000
                total_inserted = 0
                
                for i in range(0, len(dummy_data), batch_size):
                    batch = dummy_data[i:i + batch_size]
                    result = handler.collection.insert_many(batch, ordered=False)
                    total_inserted += len(result.inserted_ids)
                
                insertion_time = time.time() - start_time
                return {
                    'success': True,
                    'inserted_count': total_inserted,
                    'insertion_time': insertion_time,
                    'records_per_second': total_inserted / insertion_time if insertion_time > 0 else 0
                }
            
            elif db_name == 'Elasticsearch':
                from elasticsearch.helpers import bulk
                
                actions = []
                for doc in dummy_data:
                    # Convert datetime to string
                    es_doc = {}
                    for key, value in doc.items():
                        if isinstance(value, datetime):
                            es_doc[key] = value.isoformat()
                        else:
                            es_doc[key] = value
                    
                    actions.append({
                        "_index": handler.index_name,
                        "_id": doc['sr_number'],
                        "_source": es_doc
                    })
                
                success_count, failed = bulk(handler.es, actions, request_timeout=300)
                insertion_time = time.time() - start_time
                
                return {
                    'success': True,
                    'inserted_count': success_count,
                    'insertion_time': insertion_time,
                    'records_per_second': success_count / insertion_time if insertion_time > 0 else 0,
                    'failed_count': len(failed) if failed else 0
                }
            
            elif db_name == 'PostgreSQL':
                cursor = handler.cursor()
                
                insert_sql = """
                INSERT INTO chicago_311_requests (
                    sr_number, sr_type, sr_short_code, owner_department, status, origin,
                    created_date, last_modified_date, closed_date, street_address, city, state,
                    zip_code, street_number, street_direction, street_name, street_type,
                    duplicate, legacy_record, latitude, longitude, ward, community_area, police_district
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                # Prepare values
                values = []
                for record in dummy_data:
                    values.append((
                        record['sr_number'], record['sr_type'], record['sr_short_code'],
                        record['owner_department'], record['status'], record['origin'],
                        record['created_date'], record['last_modified_date'], record['closed_date'],
                        record['street_address'], record['city'], record['state'],
                        record['zip_code'], record['street_number'], record['street_direction'],
                        record['street_name'], record['street_type'], record['duplicate'],
                        record['legacy_record'], record['latitude'], record['longitude'],
                        record['ward'], record['community_area'], record['police_district']
                    ))
                
                cursor.executemany(insert_sql, values)
                handler.commit()
                inserted_count = cursor.rowcount
                cursor.close()
                
                insertion_time = time.time() - start_time
                return {
                    'success': True,
                    'inserted_count': inserted_count,
                    'insertion_time': insertion_time,
                    'records_per_second': inserted_count / insertion_time if insertion_time > 0 else 0
                }
            
            elif db_name == 'DuckDB':
                # Prepare data with unique IDs
                processed_data = []
                for i, record in enumerate(dummy_data):
                    row = record.copy()
                    row['id'] = int(time.time() * 1000000) + i
                    processed_data.append(row)
                
                df = pd.DataFrame(processed_data)
                handler.register('stress_df', df)
                handler.execute("INSERT INTO chicago_311_requests SELECT * FROM stress_df")
                handler.unregister('stress_df')
                
                insertion_time = time.time() - start_time
                return {
                    'success': True,
                    'inserted_count': len(dummy_data),
                    'insertion_time': insertion_time,
                    'records_per_second': len(dummy_data) / insertion_time if insertion_time > 0 else 0
                }
        
        except Exception as e:
            logger.error(f"Dummy data injection failed for {db_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'insertion_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def test_concurrent_read_performance(self, db_name: str, concurrent_users: List[int]) -> Dict:
        """Test concurrent read performance under stress."""
        print(f"📖 Testing concurrent reads for {db_name}...")
        results = {}
        
        def read_worker():
            """Worker function for concurrent reads."""
            start_time = time.time()
            try:
                handler = self.databases[db_name]['handler']
                
                if db_name == 'MongoDB':
                    # Random query
                    queries = [
                        {},  # All records
                        {'status': 'Open'},
                        {'ward': random.randint(1, 50)},
                        {'owner_department': 'Streets & Sanitation'}
                    ]
                    query = random.choice(queries)
                    handler.collection.count_documents(query)
                
                elif db_name == 'Elasticsearch':
                    queries = [
                        {"query": {"match_all": {}}},
                        {"query": {"term": {"status": "Open"}}},
                        {"query": {"range": {"ward": {"gte": 1, "lte": 50}}}},
                        {"query": {"term": {"owner_department": "Streets & Sanitation"}}}
                    ]
                    query = random.choice(queries)
                    handler.es.count(index=handler.index_name, body=query)
                
                elif db_name == 'PostgreSQL':
                    cursor = handler.cursor()
                    queries = [
                        "SELECT COUNT(*) FROM chicago_311_requests",
                        "SELECT COUNT(*) FROM chicago_311_requests WHERE status = 'Open'",
                        f"SELECT COUNT(*) FROM chicago_311_requests WHERE ward = {random.randint(1, 50)}",
                        "SELECT COUNT(*) FROM chicago_311_requests WHERE owner_department = 'Streets & Sanitation'"
                    ]
                    query = random.choice(queries)
                    cursor.execute(query)
                    cursor.fetchone()
                    cursor.close()
                
                elif db_name == 'DuckDB':
                    queries = [
                        "SELECT COUNT(*) FROM chicago_311_requests",
                        "SELECT COUNT(*) FROM chicago_311_requests WHERE status = 'Open'",
                        f"SELECT COUNT(*) FROM chicago_311_requests WHERE ward = {random.randint(1, 50)}",
                        "SELECT COUNT(*) FROM chicago_311_requests WHERE owner_department = 'Streets & Sanitation'"
                    ]
                    query = random.choice(queries)
                    handler.execute(query)
                
                return time.time() - start_time
            
            except Exception as e:
                logger.error(f"Read worker failed: {e}")
                return float('inf')
        
        for users in concurrent_users:
            print(f"   Testing {users} concurrent users...")
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=users) as executor:
                futures = [executor.submit(read_worker) for _ in range(users)]
                response_times = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            successful_queries = [t for t in response_times if t != float('inf')]
            
            if successful_queries:
                results[users] = {
                    'total_time': total_time,
                    'avg_response_time': np.mean(successful_queries),
                    'max_response_time': np.max(successful_queries),
                    'min_response_time': np.min(successful_queries),
                    'queries_per_second': len(successful_queries) / total_time,
                    'success_rate': len(successful_queries) / len(response_times),
                    'throughput': users / total_time if total_time > 0 else 0
                }
                
                print(f"      QPS: {results[users]['queries_per_second']:.2f}, "
                      f"Success: {results[users]['success_rate']*100:.1f}%")
            else:
                results[users] = {
                    'total_time': total_time,
                    'success_rate': 0,
                    'queries_per_second': 0,
                    'error': 'All queries failed'
                }
        
        return results
    
    def test_bulk_insert_performance(self, db_name: str, batch_sizes: List[int]) -> Dict:
        """Test bulk insert performance."""
        print(f"💾 Testing bulk inserts for {db_name}...")
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size:,}...")
            
            # Generate test data
            test_data = self.generate_dummy_data(batch_size)
            
            start_time = time.time()
            try:
                handler = self.databases[db_name]['handler']
                
                if db_name == 'MongoDB':
                    result = handler.collection.insert_many(test_data, ordered=False)
                    inserted_count = len(result.inserted_ids)
                
                elif db_name == 'Elasticsearch':
                    from elasticsearch.helpers import bulk
                    actions = []
                    for doc in test_data:
                        es_doc = {}
                        for key, value in doc.items():
                            if isinstance(value, datetime):
                                es_doc[key] = value.isoformat()
                            else:
                                es_doc[key] = value
                        
                        actions.append({
                            "_index": handler.index_name,
                            "_id": doc['sr_number'],
                            "_source": es_doc
                        })
                    
                    inserted_count, failed = bulk(handler.es, actions, request_timeout=300)
                
                elif db_name == 'PostgreSQL':
                    cursor = handler.cursor()
                    insert_sql = """
                    INSERT INTO chicago_311_requests (
                        sr_number, sr_type, sr_short_code, owner_department, status, origin,
                        created_date, last_modified_date, closed_date, street_address, city, state,
                        zip_code, street_number, street_direction, street_name, street_type,
                        duplicate, legacy_record, latitude, longitude, ward, community_area, police_district
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    values = []
                    for record in test_data:
                        values.append(tuple(record[key] for key in [
                            'sr_number', 'sr_type', 'sr_short_code', 'owner_department', 'status', 'origin',
                            'created_date', 'last_modified_date', 'closed_date', 'street_address', 'city', 'state',
                            'zip_code', 'street_number', 'street_direction', 'street_name', 'street_type',
                            'duplicate', 'legacy_record', 'latitude', 'longitude', 'ward', 'community_area', 'police_district'
                        ]))
                    
                    cursor.executemany(insert_sql, values)
                    handler.commit()
                    inserted_count = cursor.rowcount
                    cursor.close()
                
                elif db_name == 'DuckDB':
                    processed_data = []
                    for i, record in enumerate(test_data):
                        row = record.copy()
                        row['id'] = int(time.time() * 1000000) + i
                        processed_data.append(row)
                    
                    df = pd.DataFrame(processed_data)
                    handler.register('bulk_test_df', df)
                    handler.execute("INSERT INTO chicago_311_requests SELECT * FROM bulk_test_df")
                    handler.unregister('bulk_test_df')
                    inserted_count = len(test_data)
                
                duration = time.time() - start_time
                records_per_second = inserted_count / duration if duration > 0 else 0
                
                results[batch_size] = {
                    'duration': duration,
                    'inserted_count': inserted_count,
                    'records_per_second': records_per_second,
                    'success': True
                }
                
                print(f"      {records_per_second:.0f} records/sec")
                
                # Cleanup test data
                self.cleanup_test_data(db_name, [d['sr_number'] for d in test_data])
                
            except Exception as e:
                results[batch_size] = {
                    'duration': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
                print(f"      ❌ Failed: {e}")
        
        return results
    
    def cleanup_test_data(self, db_name: str, sr_numbers: List[str]):
        """Clean up test data after stress tests."""
        try:
            handler = self.databases[db_name]['handler']
            
            if db_name == 'MongoDB':
                handler.collection.delete_many({'sr_number': {'$in': sr_numbers}})
            
            elif db_name == 'Elasticsearch':
                for sr_number in sr_numbers:
                    try:
                        handler.es.delete(index=handler.index_name, id=sr_number, ignore=[404])
                    except:
                        pass
            
            elif db_name == 'PostgreSQL':
                cursor = handler.cursor()
                cursor.execute("DELETE FROM chicago_311_requests WHERE sr_number = ANY(%s)", (sr_numbers,))
                handler.commit()
                cursor.close()
            
            elif db_name == 'DuckDB':
                placeholders = ', '.join(['?' for _ in sr_numbers])
                handler.execute(f"DELETE FROM chicago_311_requests WHERE sr_number IN ({placeholders})", sr_numbers)
        
        except Exception as e:
            logger.warning(f"Cleanup failed for {db_name}: {e}")
    
    def run_comprehensive_stress_tests(self):
        """Run all stress tests."""
        print("\n🚀 COMPREHENSIVE STRESS TESTING SYSTEM")
        print("=" * 80)
        
        for db_name, db_info in self.databases.items():
            print(f"\n💪 Stress Testing {db_name} (Baseline: {db_info['baseline_count']:,} records)")
            print("-" * 70)
            
            # Initialize results
            self.stress_results[db_name] = {
                'baseline_count': db_info['baseline_count'],
                'database_type': db_info['type']
            }
            
            # 1. Test concurrent reads
            concurrent_results = self.test_concurrent_read_performance(
                db_name, self.test_configs['concurrent_reads']
            )
            self.stress_results[db_name]['concurrent_reads'] = concurrent_results
            
            # 2. Test bulk inserts
            bulk_results = self.test_bulk_insert_performance(
                db_name, self.test_configs['bulk_insert_sizes']
            )
            self.stress_results[db_name]['bulk_inserts'] = bulk_results
            
            # 3. Inject dummy data for load testing
            for data_scale in [50000, 100000]:  # Inject different amounts
                print(f"   🔄 Load testing with {data_scale:,} additional records...")
                
                injection_result = self.inject_dummy_data(db_name, data_scale)
                if injection_result['success']:
                    # Test performance with increased data
                    load_test_key = f'load_test_{data_scale}'
                    load_concurrent_results = self.test_concurrent_read_performance(
                        db_name, [10, 25, 50]  # Smaller test for load testing
                    )
                    
                    self.stress_results[db_name][load_test_key] = {
                        'injection_result': injection_result,
                        'concurrent_performance': load_concurrent_results
                    }
                    
                    print(f"      ✅ Injected {injection_result['inserted_count']:,} records in {injection_result['insertion_time']:.2f}s")
                else:
                    print(f"      ❌ Data injection failed: {injection_result.get('error', 'Unknown error')}")
            
            print(f"   ✅ {db_name} stress testing completed")
    
    def generate_stress_test_charts(self):
        """Generate comprehensive stress test charts."""
        print("\n📊 Generating Stress Test Charts...")
        
        # Set up plotting
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Chart 1: Concurrent Read Performance
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_concurrent_performance_chart(ax1)
        
        # Chart 2: Bulk Insert Performance
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_bulk_insert_chart(ax2)
        
        # Chart 3: Success Rate Under Load
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_success_rate_chart(ax3)
        
        # Chart 4: Response Time Distribution
        ax4 = fig.add_subplot(gs[1, :])
        self._create_response_time_distribution(ax4)
        
        # Chart 5: Load Test Results
        ax5 = fig.add_subplot(gs[2, 0])
        self._create_load_test_chart(ax5)
        
        # Chart 6: Breaking Point Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self._create_breaking_point_chart(ax6)
        
        # Chart 7: Database Comparison Summary
        ax7 = fig.add_subplot(gs[2, 2])
        self._create_database_summary_chart(ax7)
        
        # Chart 8: Performance Under Different Data Volumes
        ax8 = fig.add_subplot(gs[3, :])
        self._create_data_volume_performance_chart(ax8)
        
        plt.suptitle('Comprehensive Database Stress Test Results - Chicago 311 Platform', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('comprehensive_stress_test_charts.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Stress test charts saved as 'comprehensive_stress_test_charts.png'")
    
    def _create_concurrent_performance_chart(self, ax):
        """Create concurrent performance chart."""
        for db_name, results in self.stress_results.items():
            if 'concurrent_reads' in results:
                users = list(results['concurrent_reads'].keys())
                qps = [results['concurrent_reads'][u]['queries_per_second'] 
                       for u in users if 'queries_per_second' in results['concurrent_reads'][u]]
                valid_users = [u for u in users if 'queries_per_second' in results['concurrent_reads'][u]]
                
                if qps:
                    ax.plot(valid_users, qps, marker='o', label=db_name, linewidth=2)
        
        ax.set_title('Concurrent Read Performance', fontweight='bold')
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Queries Per Second')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_bulk_insert_chart(self, ax):
        """Create bulk insert performance chart."""
        for db_name, results in self.stress_results.items():
            if 'bulk_inserts' in results:
                batch_sizes = list(results['bulk_inserts'].keys())
                rps = [results['bulk_inserts'][b]['records_per_second'] 
                       for b in batch_sizes if results['bulk_inserts'][b]['success']]
                valid_sizes = [b for b in batch_sizes if results['bulk_inserts'][b]['success']]
                
                if rps:
                    ax.plot(valid_sizes, rps, marker='s', label=db_name, linewidth=2)
        
        ax.set_title('Bulk Insert Performance', fontweight='bold')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Records Per Second')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_success_rate_chart(self, ax):
        """Create success rate chart."""
        for db_name, results in self.stress_results.items():
            if 'concurrent_reads' in results:
                users = list(results['concurrent_reads'].keys())
                success_rates = [results['concurrent_reads'][u]['success_rate'] * 100 
                               for u in users if 'success_rate' in results['concurrent_reads'][u]]
                valid_users = [u for u in users if 'success_rate' in results['concurrent_reads'][u]]
                
                if success_rates:
                    ax.plot(valid_users, success_rates, marker='^', label=db_name, linewidth=2)
        
        ax.set_title('Success Rate Under Load', fontweight='bold')
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_response_time_distribution(self, ax):
        """Create response time distribution chart."""
        # Collect response time data
        all_response_times = {}
        
        for db_name, results in self.stress_results.items():
            if 'concurrent_reads' in results:
                response_times = []
                for user_count, metrics in results['concurrent_reads'].items():
                    if 'avg_response_time' in metrics:
                        response_times.append(metrics['avg_response_time'])
                all_response_times[db_name] = response_times
        
        if all_response_times:
            # Create violin plot
            data_for_plot = []
            labels = []
            for db_name, times in all_response_times.items():
                if times:
                    data_for_plot.append(times)
                    labels.append(db_name)
            
            if data_for_plot:
                ax.violinplot(data_for_plot, positions=range(len(labels)), showmeans=True)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                ax.set_title('Response Time Distribution', fontweight='bold')
                ax.set_ylabel('Response Time (seconds)')
    
    def _create_load_test_chart(self, ax):
        """Create load test performance chart."""
        databases = list(self.stress_results.keys())
        load_scales = [50000, 100000]
        
        # Performance degradation data
        baseline_qps = {}
        load_qps = {}
        
        for db_name in databases:
            if 'concurrent_reads' in self.stress_results[db_name]:
                # Get baseline QPS (25 users)
                if 25 in self.stress_results[db_name]['concurrent_reads']:
                    baseline_qps[db_name] = self.stress_results[db_name]['concurrent_reads'][25]['queries_per_second']
                
                # Get load test QPS
                for scale in load_scales:
                    load_key = f'load_test_{scale}'
                    if load_key in self.stress_results[db_name]:
                        if 25 in self.stress_results[db_name][load_key]['concurrent_performance']:
                            load_qps[f'{db_name}_{scale}'] = self.stress_results[db_name][load_key]['concurrent_performance'][25]['queries_per_second']
        
        # Plot performance comparison
        if baseline_qps:
            x_pos = np.arange(len(databases))
            baseline_values = [baseline_qps.get(db, 0) for db in databases]
            
            bars1 = ax.bar(x_pos - 0.2, baseline_values, 0.4, label='Baseline', alpha=0.8)
            
            for i, scale in enumerate(load_scales):
                load_values = [load_qps.get(f'{db}_{scale}', 0) for db in databases]
                bars = ax.bar(x_pos + 0.2 + i*0.2, load_values, 0.2, 
                             label=f'With +{scale//1000}K records', alpha=0.8)
            
            ax.set_xlabel('Database')
            ax.set_ylabel('Queries Per Second')
            ax.set_title('Performance Under Data Load', fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(databases)
            ax.legend()
    
    def _create_breaking_point_chart(self, ax):
        """Create breaking point analysis chart."""
        breaking_points = {}
        
        for db_name, results in self.stress_results.items():
            if 'concurrent_reads' in results:
                # Find where success rate drops below 95%
                breaking_point = None
                for users in sorted(results['concurrent_reads'].keys()):
                    if 'success_rate' in results['concurrent_reads'][users]:
                        if results['concurrent_reads'][users]['success_rate'] < 0.95:
                            breaking_point = users
                            break
                
                if breaking_point:
                    breaking_points[db_name] = breaking_point
                else:
                    # If no breaking point found, use max tested users
                    breaking_points[db_name] = max(results['concurrent_reads'].keys())
        
        if breaking_points:
            databases = list(breaking_points.keys())
            points = list(breaking_points.values())
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(databases)]
            bars = ax.bar(databases, points, color=colors, alpha=0.7)
            
            ax.set_title('Breaking Point Analysis', fontweight='bold')
            ax.set_ylabel('Concurrent Users (95% Success Rate)')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, points):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value}', ha='center', va='bottom')
    
    def _create_database_summary_chart(self, ax):
        """Create database performance summary."""
        summary_data = {}
        
        for db_name, results in self.stress_results.items():
            scores = []
            
            # Concurrent read score (higher QPS = better)
            if 'concurrent_reads' in results:
                max_qps = 0
                for metrics in results['concurrent_reads'].values():
                    if 'queries_per_second' in metrics:
                        max_qps = max(max_qps, metrics['queries_per_second'])
                scores.append(max_qps)
            
            # Bulk insert score (higher RPS = better)
            if 'bulk_inserts' in results:
                max_rps = 0
                for metrics in results['bulk_inserts'].values():
                    if metrics['success'] and 'records_per_second' in metrics:
                        max_rps = max(max_rps, metrics['records_per_second'])
                scores.append(max_rps / 1000)  # Scale down for visualization
            
            if scores:
                summary_data[db_name] = np.mean(scores)
        
        if summary_data:
            databases = list(summary_data.keys())
            scores = list(summary_data.values())
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(databases)]
            bars = ax.bar(databases, scores, color=colors, alpha=0.7)
            
            ax.set_title('Overall Performance Summary', fontweight='bold')
            ax.set_ylabel('Performance Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.0f}', ha='center', va='bottom')
    
    def _create_data_volume_performance_chart(self, ax):
        """Create data volume vs performance chart."""
        # Show how performance degrades with increased data volume
        for db_name, results in self.stress_results.items():
            volumes = [results['baseline_count']]
            performances = []
            
            # Get baseline performance (25 concurrent users)
            if ('concurrent_reads' in results and 
                25 in results['concurrent_reads'] and
                'queries_per_second' in results['concurrent_reads'][25]):
                performances.append(results['concurrent_reads'][25]['queries_per_second'])
                
                # Get performance with additional data
                for scale in [50000, 100000]:
                    load_key = f'load_test_{scale}'
                    if load_key in results:
                        volumes.append(results['baseline_count'] + scale)
                        if (25 in results[load_key]['concurrent_performance'] and
                            'queries_per_second' in results[load_key]['concurrent_performance'][25]):
                            performances.append(results[load_key]['concurrent_performance'][25]['queries_per_second'])
                
                if len(volumes) == len(performances) and len(performances) > 1:
                    ax.plot(volumes, performances, marker='o', label=db_name, linewidth=2)
        
        ax.set_title('Performance vs Data Volume', fontweight='bold')
        ax.set_xlabel('Total Records')
        ax.set_ylabel('Queries Per Second')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='x')
    
    def save_stress_test_results(self):
        """Save stress test results."""
        # Clean results for JSON serialization
        clean_results = {}
        for db_name, results in self.stress_results.items():
            clean_results[db_name] = results.copy()
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'comprehensive_stress_testing',
            'test_configurations': self.test_configs,
            'baseline_counts': self.baseline_counts,
            'results': clean_results
        }
        
        with open('comprehensive_stress_test_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print("✅ Results saved to 'comprehensive_stress_test_results.json'")
    
    def generate_stress_test_report(self):
        """Generate comprehensive stress test report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DATABASE STRESS TEST REPORT")
        report.append("Chicago 311 Service Requests - Load Testing Analysis")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"• Databases Tested: {len(self.stress_results)}")
        report.append(f"• Stress Test Categories: Concurrent Reads, Bulk Inserts, Load Testing")
        report.append(f"• Maximum Concurrent Users Tested: {max(self.test_configs['concurrent_reads'])}")
        report.append(f"• Maximum Bulk Insert Size: {max(self.test_configs['bulk_insert_sizes']):,} records")
        report.append("")
        
        # Breaking Point Analysis
        report.append("BREAKING POINT ANALYSIS")
        report.append("-" * 40)
        
        for db_name, results in self.stress_results.items():
            report.append(f"\n{db_name.upper()}:")
            
            # Find breaking points
            if 'concurrent_reads' in results:
                max_successful_users = 0
                for users, metrics in results['concurrent_reads'].items():
                    if 'success_rate' in metrics and metrics['success_rate'] >= 0.95:
                        max_successful_users = max(max_successful_users, users)
                
                report.append(f"  • Max Concurrent Users (95% success): {max_successful_users}")
            
            if 'bulk_inserts' in results:
                max_bulk_size = 0
                for size, metrics in results['bulk_inserts'].items():
                    if metrics['success']:
                        max_bulk_size = max(max_bulk_size, size)
                
                report.append(f"  • Max Bulk Insert Size: {max_bulk_size:,} records")
        
        # Performance Under Load
        report.append(f"\n\nPERFORMANCE UNDER LOAD")
        report.append("-" * 40)
        
        for db_name, results in self.stress_results.items():
            report.append(f"\n{db_name}:")
            baseline_count = results['baseline_count']
            report.append(f"  Baseline Records: {baseline_count:,}")
            
            # Show performance degradation
            if 'concurrent_reads' in results and 25 in results['concurrent_reads']:
                baseline_qps = results['concurrent_reads'][25]['queries_per_second']
                report.append(f"  Baseline QPS (25 users): {baseline_qps:.2f}")
                
                for scale in [50000, 100000]:
                    load_key = f'load_test_{scale}'
                    if load_key in results:
                        if 25 in results[load_key]['concurrent_performance']:
                            load_qps = results[load_key]['concurrent_performance'][25]['queries_per_second']
                            degradation = ((baseline_qps - load_qps) / baseline_qps) * 100
                            report.append(f"  QPS with +{scale:,} records: {load_qps:.2f} ({degradation:+.1f}%)")
        
        # Detailed Results
        report.append(f"\n\nDETAILED STRESS TEST RESULTS")
        report.append("-" * 40)
        
        for db_name, results in self.stress_results.items():
            report.append(f"\n{db_name.upper()} - {results.get('database_type', 'Unknown Type')}")
            report.append("=" * 30)
            
            # Concurrent Read Performance
            if 'concurrent_reads' in results:
                report.append("\nConcurrent Read Performance:")
                for users in sorted(results['concurrent_reads'].keys()):
                    metrics = results['concurrent_reads'][users]
                    if 'queries_per_second' in metrics:
                        qps = metrics['queries_per_second']
                        success_rate = metrics['success_rate'] * 100
                        avg_response = metrics['avg_response_time']
                        report.append(f"  {users:3d} users: {qps:6.1f} QPS, {success_rate:5.1f}% success, {avg_response:.4f}s avg")
            
            # Bulk Insert Performance
            if 'bulk_inserts' in results:
                report.append("\nBulk Insert Performance:")
                for size in sorted(results['bulk_inserts'].keys()):
                    metrics = results['bulk_inserts'][size]
                    if metrics['success']:
                        rps = metrics['records_per_second']
                        duration = metrics['duration']
                        report.append(f"  {size:6,} records: {rps:8.0f} RPS, {duration:6.2f}s total")
                    else:
                        report.append(f"  {size:6,} records: FAILED - {metrics.get('error', 'Unknown error')}")
        
        # Recommendations
        report.append(f"\n\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find best performers
        best_concurrent = None
        best_bulk = None
        best_concurrent_qps = 0
        best_bulk_rps = 0
        
        for db_name, results in self.stress_results.items():
            # Check concurrent performance
            if 'concurrent_reads' in results:
                for metrics in results['concurrent_reads'].values():
                    if 'queries_per_second' in metrics:
                        qps = metrics['queries_per_second']
                        if qps > best_concurrent_qps:
                            best_concurrent_qps = qps
                            best_concurrent = db_name
            
            # Check bulk insert performance
            if 'bulk_inserts' in results:
                for metrics in results['bulk_inserts'].values():
                    if metrics['success'] and 'records_per_second' in metrics:
                        rps = metrics['records_per_second']
                        if rps > best_bulk_rps:
                            best_bulk_rps = rps
                            best_bulk = db_name
        
        if best_concurrent:
            report.append(f"🏆 Best Concurrent Read Performance: {best_concurrent} ({best_concurrent_qps:.1f} QPS)")
        if best_bulk:
            report.append(f"🏆 Best Bulk Insert Performance: {best_bulk} ({best_bulk_rps:.0f} RPS)")
        
        report.append("\n💡 Usage Recommendations:")
        report.append("   • MongoDB: Excellent for high-concurrency read/write operations")
        report.append("   • Elasticsearch: Best for search-intensive applications")
        report.append("   • PostgreSQL: Ideal for transactional integrity under load")
        report.append("   • DuckDB: Optimized for analytical batch processing")
        report.append("   • Monitor performance degradation as data volume increases")
        report.append("   • Consider horizontal scaling for high-load scenarios")
        
        # Save report
        with open('comprehensive_stress_test_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("✅ Stress test report saved as 'comprehensive_stress_test_report.txt'")
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
    """Run comprehensive stress testing."""
    stress_tester = ComprehensiveStressTester()
    
    try:
        stress_tester.setup_database_connections()
        stress_tester.run_comprehensive_stress_tests()
        stress_tester.generate_stress_test_charts()
        stress_tester.save_stress_test_results()
        report = stress_tester.generate_stress_test_report()
        
        print(f"\n{'='*80}")
        print("🎉 COMPREHENSIVE STRESS TESTING COMPLETED!")
        print(f"{'='*80}")
        print("📁 Generated Files:")
        print("   • comprehensive_stress_test_charts.png")
        print("   • comprehensive_stress_test_results.json") 
        print("   • comprehensive_stress_test_report.txt")
        
        # Show quick summary
        print(f"\n📋 Quick Summary:")
        print(report.split("DETAILED STRESS TEST RESULTS")[0])
        
    except Exception as e:
        print(f"❌ Stress testing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stress_tester.close_connections()

if __name__ == "__main__":
    main()