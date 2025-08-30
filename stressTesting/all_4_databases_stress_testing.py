#!/usr/bin/env python3
"""
All 4 Databases Comprehensive Stress Testing System
Tests concurrent loads, bulk inserts, and breaking points for all databases
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
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

class All4DatabasesStressTesting:
    """Comprehensive stress testing system for ALL 4 databases."""
    
    def __init__(self):
        self.handlers = {}
        self.results = {}
        self.test_data_cache = []
        
    def setup_all_databases(self):
        """Setup all 4 database connections for stress testing."""
        logger.info("🔧 Setting up ALL 4 databases for stress testing...")
        
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
        
        # 2. Elasticsearch
        try:
            es_handler = ElasticsearchHandler()
            count_result = es_handler.es.count(index=es_handler.index_name)
            count = count_result['count']
            if count > 0:
                self.handlers['Elasticsearch'] = es_handler
                logger.info(f"✅ Elasticsearch ready ({count:,} records)")
            else:
                # Create test index anyway for stress testing
                self.handlers['Elasticsearch'] = es_handler
                logger.info("✅ Elasticsearch ready (prepared for stress testing)")
        except Exception as e:
            logger.error(f"❌ Elasticsearch setup failed: {e}")
            
        # 3. PostgreSQL
        try:
            import psycopg2
            
            class PostgreSQLStressHandler:
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
                    
                    self.connection.autocommit = True
                    
                    # Create stress test table
                    with self.connection.cursor() as cursor:
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS stress_test_records (
                                id SERIAL PRIMARY KEY,
                                test_id VARCHAR(50),
                                data TEXT,
                                created_at TIMESTAMP DEFAULT NOW()
                            )
                        """)
                    
                    # Get existing record count
                    with self.connection.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM chicago_311_requests")
                        self.record_count = cursor.fetchone()[0]
            
            pg_handler = PostgreSQLStressHandler()
            self.handlers['PostgreSQL'] = pg_handler
            logger.info(f"✅ PostgreSQL ready ({pg_handler.record_count:,} records)")
        except Exception as e:
            logger.error(f"❌ PostgreSQL setup failed: {e}")
            
        # 4. DuckDB
        try:
            import duckdb
            
            class DuckDBStressHandler:
                def __init__(self):
                    self.conn = duckdb.connect("chicago_311.duckdb")
                    
                    # Create stress test table
                    self.conn.execute("""
                        CREATE TABLE IF NOT EXISTS stress_test_records (
                            id INTEGER,
                            test_id VARCHAR,
                            data VARCHAR,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # Get existing record count
                    try:
                        result = self.conn.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()
                        self.record_count = result[0] if result else 0
                    except:
                        self.record_count = 0
            
            duckdb_handler = DuckDBStressHandler()
            self.handlers['DuckDB'] = duckdb_handler
            logger.info(f"✅ DuckDB ready ({duckdb_handler.record_count:,} records)")
        except Exception as e:
            logger.error(f"❌ DuckDB setup failed: {e}")
        
        logger.info(f"🎯 Ready to stress test {len(self.handlers)} databases: {list(self.handlers.keys())}")
        return len(self.handlers) > 0
    
    def generate_test_data(self, count: int) -> List[Dict]:
        """Generate synthetic test data for stress testing."""
        if len(self.test_data_cache) >= count:
            return self.test_data_cache[:count]
        
        logger.info(f"🔄 Generating {count} test records...")
        
        test_data = []
        departments = ['Streets & Sanitation', 'Water Management', 'Transportation', '311 City Services']
        statuses = ['Open', 'Completed', 'In Progress', 'Closed']
        types = ['Pothole', 'Street Light Out', 'Graffiti Removal', 'Tree Trim', 'Water Leak']
        
        for i in range(count):
            record = {
                'test_id': f'stress_test_{i}',
                'sr_number': f'ST{i:08d}',
                'sr_type': random.choice(types),
                'owner_department': random.choice(departments),
                'status': random.choice(statuses),
                'created_date': datetime.now(),
                'ward': random.randint(1, 50),
                'latitude': round(41.8 + random.random() * 0.2, 6),
                'longitude': round(-87.7 + random.random() * 0.2, 6),
                'data': f'Stress test record {i} with some additional data content'
            }
            test_data.append(record)
        
        self.test_data_cache = test_data
        return test_data
    
    def concurrent_read_test(self, db_name: str, handler: Any, concurrent_users: List[int]) -> Dict[str, Any]:
        """Test concurrent read performance."""
        logger.info(f"  Running concurrent read test on {db_name}...")
        results = {}
        
        for user_count in concurrent_users:
            def read_operation():
                try:
                    start_time = time.time()
                    if db_name == 'MongoDB':
                        list(handler.collection.find().limit(100))
                    elif db_name == 'Elasticsearch':
                        handler.es.search(index=handler.index_name, size=100)
                    elif db_name == 'PostgreSQL':
                        with handler.connection.cursor() as cursor:
                            cursor.execute("SELECT * FROM chicago_311_requests LIMIT 100")
                            cursor.fetchall()
                    elif db_name == 'DuckDB':
                        handler.conn.execute("SELECT * FROM chicago_311_requests LIMIT 100").fetchall()
                    return time.time() - start_time
                except Exception as e:
                    logger.error(f"Read operation failed: {e}")
                    return -1
            
            # Run concurrent operations
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = [executor.submit(read_operation) for _ in range(user_count)]
                durations = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            successful_operations = [d for d in durations if d > 0]
            
            results[f'{user_count}_users'] = {
                'total_time': total_time,
                'successful_operations': len(successful_operations),
                'failed_operations': len(durations) - len(successful_operations),
                'avg_operation_time': np.mean(successful_operations) if successful_operations else 0,
                'throughput': len(successful_operations) / total_time if total_time > 0 else 0
            }
            
            logger.info(f"    {user_count} users: {len(successful_operations)}/{user_count} successful, {results[f'{user_count}_users']['throughput']:.2f} ops/sec")
        
        return results
    
    def bulk_insert_test(self, db_name: str, handler: Any, batch_sizes: List[int]) -> Dict[str, Any]:
        """Test bulk insert performance."""
        logger.info(f"  Running bulk insert test on {db_name}...")
        results = {}
        
        for batch_size in batch_sizes:
            test_data = self.generate_test_data(batch_size)
            
            try:
                start_time = time.time()
                
                if db_name == 'MongoDB':
                    # Insert to stress test collection
                    stress_collection = handler.client[handler.db_name]['stress_test_records']
                    stress_collection.insert_many(test_data, ordered=False)
                    inserted_count = batch_size
                elif db_name == 'Elasticsearch':
                    from elasticsearch.helpers import bulk
                    actions = [
                        {
                            "_index": "stress_test_index",
                            "_source": record
                        }
                        for record in test_data
                    ]
                    # Create index if not exists
                    if not handler.es.indices.exists(index="stress_test_index"):
                        handler.es.indices.create(index="stress_test_index")
                    
                    success_count, _ = bulk(handler.es, actions)
                    inserted_count = success_count
                elif db_name == 'PostgreSQL':
                    with handler.connection.cursor() as cursor:
                        insert_sql = """
                            INSERT INTO stress_test_records (test_id, data) 
                            VALUES (%s, %s)
                        """
                        values = [(record['test_id'], record['data']) for record in test_data]
                        cursor.executemany(insert_sql, values)
                        inserted_count = cursor.rowcount
                elif db_name == 'DuckDB':
                    # Convert to DataFrame for DuckDB
                    df = pd.DataFrame(test_data)
                    handler.conn.register('stress_data', df)
                    handler.conn.execute("INSERT INTO stress_test_records SELECT * FROM stress_data")
                    inserted_count = batch_size
                
                duration = time.time() - start_time
                
                results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'duration': duration,
                    'inserted_count': inserted_count,
                    'insertion_rate': inserted_count / duration if duration > 0 else 0
                }
                
                logger.info(f"    Batch {batch_size}: {inserted_count} records in {duration:.2f}s ({results[f'batch_{batch_size}']['insertion_rate']:.0f} records/sec)")
                
            except Exception as e:
                logger.error(f"    Batch {batch_size} failed: {e}")
                results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'duration': 0,
                    'inserted_count': 0,
                    'insertion_rate': 0,
                    'error': str(e)
                }
        
        return results
    
    def memory_pressure_test(self, db_name: str, handler: Any) -> Dict[str, Any]:
        """Test performance under memory pressure."""
        logger.info(f"  Running memory pressure test on {db_name}...")
        
        # Generate large dataset
        large_dataset = self.generate_test_data(10000)
        
        # Add large text fields to increase memory usage
        for record in large_dataset:
            record['large_data'] = 'X' * 1000  # 1KB of data per record
        
        results = {}
        
        try:
            start_time = time.time()
            
            if db_name == 'MongoDB':
                # Insert in chunks to avoid memory issues
                chunk_size = 1000
                inserted_total = 0
                for i in range(0, len(large_dataset), chunk_size):
                    chunk = large_dataset[i:i+chunk_size]
                    stress_collection = handler.client[handler.db_name]['memory_pressure_test']
                    stress_collection.insert_many(chunk, ordered=False)
                    inserted_total += len(chunk)
            elif db_name == 'Elasticsearch':
                # Similar chunked approach for ES
                from elasticsearch.helpers import bulk
                chunk_size = 1000
                inserted_total = 0
                for i in range(0, len(large_dataset), chunk_size):
                    chunk = large_dataset[i:i+chunk_size]
                    actions = [
                        {"_index": "memory_pressure_test", "_source": record}
                        for record in chunk
                    ]
                    if not handler.es.indices.exists(index="memory_pressure_test"):
                        handler.es.indices.create(index="memory_pressure_test")
                    success_count, _ = bulk(handler.es, actions)
                    inserted_total += success_count
            elif db_name == 'PostgreSQL':
                # Create temp table for memory test
                with handler.connection.cursor() as cursor:
                    cursor.execute("""
                        CREATE TEMP TABLE memory_pressure_test (
                            id SERIAL PRIMARY KEY,
                            test_id VARCHAR(50),
                            large_data TEXT
                        )
                    """)
                    
                    insert_sql = "INSERT INTO memory_pressure_test (test_id, large_data) VALUES (%s, %s)"
                    values = [(record['test_id'], record['large_data']) for record in large_dataset]
                    cursor.executemany(insert_sql, values)
                    inserted_total = cursor.rowcount
            elif db_name == 'DuckDB':
                # Convert to DataFrame
                df = pd.DataFrame(large_dataset)
                handler.conn.register('memory_pressure_data', df)
                handler.conn.execute("""
                    CREATE TEMP TABLE memory_pressure_test AS 
                    SELECT * FROM memory_pressure_data
                """)
                inserted_total = len(large_dataset)
            
            duration = time.time() - start_time
            
            results = {
                'dataset_size': len(large_dataset),
                'duration': duration,
                'inserted_count': inserted_total,
                'memory_usage_mb': len(large_dataset) * 1.0,  # Approximate MB
                'insertion_rate': inserted_total / duration if duration > 0 else 0
            }
            
            logger.info(f"    Memory pressure test: {inserted_total} records ({results['memory_usage_mb']:.1f} MB) in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"    Memory pressure test failed: {e}")
            results = {
                'dataset_size': len(large_dataset),
                'duration': 0,
                'inserted_count': 0,
                'memory_usage_mb': 0,
                'insertion_rate': 0,
                'error': str(e)
            }
        
        return results
    
    def run_comprehensive_stress_testing(self):
        """Run comprehensive stress testing on all available databases."""
        if not self.setup_all_databases():
            logger.error("❌ No databases available for stress testing!")
            return
        
        logger.info(f"🚀 Starting comprehensive stress testing on {len(self.handlers)} databases...")
        
        # Test configurations
        concurrent_users = [1, 5, 10, 25, 50, 100]
        batch_sizes = [100, 500, 1000, 5000, 10000]
        
        for db_name, handler in self.handlers.items():
            logger.info(f"\n💪 Stress testing {db_name}...")
            db_results = {}
            
            # 1. Concurrent read test
            logger.info("  🔄 Testing concurrent reads...")
            db_results['concurrent_reads'] = self.concurrent_read_test(db_name, handler, concurrent_users)
            
            # 2. Bulk insert test  
            logger.info("  📥 Testing bulk inserts...")
            db_results['bulk_inserts'] = self.bulk_insert_test(db_name, handler, batch_sizes)
            
            # 3. Memory pressure test
            logger.info("  🧠 Testing memory pressure...")
            db_results['memory_pressure'] = self.memory_pressure_test(db_name, handler)
            
            self.results[db_name] = db_results
            logger.info(f"✅ {db_name} stress testing completed")
        
        # Save results and generate charts
        self.save_results()
        self.generate_stress_charts()
        self.generate_stress_report()
        self.cleanup_stress_data()
    
    def save_results(self):
        """Save stress testing results to JSON."""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'all_4_databases_stress_testing',
            'results': self.results
        }
        
        with open('all_4_databases_stress_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info("💾 Results saved to all_4_databases_stress_results.json")
    
    def generate_stress_charts(self):
        """Generate comprehensive stress testing charts."""
        logger.info("📈 Generating stress testing charts...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive stress testing chart
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Stress Testing Results - All 4 Databases\nChicago 311 Data Platform', fontsize=16, fontweight='bold')
        
        # Chart 1: Concurrent Read Performance
        ax1 = axes[0, 0]
        self._plot_concurrent_reads(ax1)
        
        # Chart 2: Bulk Insert Performance
        ax2 = axes[0, 1]
        self._plot_bulk_inserts(ax2)
        
        # Chart 3: Memory Pressure Results
        ax3 = axes[0, 2]
        self._plot_memory_pressure(ax3)
        
        # Chart 4: Throughput Comparison
        ax4 = axes[1, 0]
        self._plot_throughput_comparison(ax4)
        
        # Chart 5: Scalability Analysis
        ax5 = axes[1, 1]
        self._plot_scalability_analysis(ax5)
        
        # Chart 6: Error Rate Analysis
        ax6 = axes[1, 2]
        self._plot_error_rates(ax6)
        
        plt.tight_layout()
        plt.savefig('all_4_databases_stress_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Generated all_4_databases_stress_charts.png")
    
    def _plot_concurrent_reads(self, ax):
        """Plot concurrent read performance."""
        data = []
        for db_name, db_results in self.results.items():
            if 'concurrent_reads' in db_results:
                for test_name, result in db_results['concurrent_reads'].items():
                    users = int(test_name.split('_')[0])
                    data.append({
                        'database': db_name,
                        'users': users,
                        'throughput': result['throughput']
                    })
        
        if data:
            df = pd.DataFrame(data)
            for db in df['database'].unique():
                db_data = df[df['database'] == db]
                ax.plot(db_data['users'], db_data['throughput'], marker='o', label=db)
            ax.set_title('Concurrent Read Performance')
            ax.set_xlabel('Concurrent Users')
            ax.set_ylabel('Throughput (ops/sec)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No concurrent read data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Concurrent Read Performance')
    
    def _plot_bulk_inserts(self, ax):
        """Plot bulk insert performance."""
        data = []
        for db_name, db_results in self.results.items():
            if 'bulk_inserts' in db_results:
                for test_name, result in db_results['bulk_inserts'].items():
                    if 'error' not in result:
                        batch_size = result['batch_size']
                        data.append({
                            'database': db_name,
                            'batch_size': batch_size,
                            'insertion_rate': result['insertion_rate']
                        })
        
        if data:
            df = pd.DataFrame(data)
            sns.barplot(data=df, x='batch_size', y='insertion_rate', hue='database', ax=ax)
            ax.set_title('Bulk Insert Performance')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Insertion Rate (records/sec)')
            ax.legend(title='Database')
        else:
            ax.text(0.5, 0.5, 'No bulk insert data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bulk Insert Performance')
    
    def _plot_memory_pressure(self, ax):
        """Plot memory pressure test results."""
        databases = []
        insertion_rates = []
        
        for db_name, db_results in self.results.items():
            if 'memory_pressure' in db_results and 'error' not in db_results['memory_pressure']:
                databases.append(db_name)
                insertion_rates.append(db_results['memory_pressure']['insertion_rate'])
        
        if databases:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(databases, insertion_rates, color=colors[:len(databases)])
            ax.set_title('Memory Pressure Test Results')
            ax.set_ylabel('Insertion Rate (records/sec)')
            
            for bar, rate in zip(bars, insertion_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{rate:.0f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No memory pressure data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Memory Pressure Test Results')
    
    def _plot_throughput_comparison(self, ax):
        """Plot throughput comparison across databases."""
        # Get max throughput for each database
        databases = []
        max_throughputs = []
        
        for db_name, db_results in self.results.items():
            if 'concurrent_reads' in db_results:
                throughputs = [result['throughput'] for result in db_results['concurrent_reads'].values()]
                databases.append(db_name)
                max_throughputs.append(max(throughputs) if throughputs else 0)
        
        if databases:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(databases, max_throughputs, color=colors[:len(databases)])
            ax.set_title('Maximum Throughput Comparison')
            ax.set_ylabel('Max Throughput (ops/sec)')
            
            for bar, throughput in zip(bars, max_throughputs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{throughput:.1f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No throughput data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Maximum Throughput Comparison')
    
    def _plot_scalability_analysis(self, ax):
        """Plot scalability analysis."""
        # Calculate scalability score based on how performance scales with users
        databases = []
        scalability_scores = []
        
        for db_name, db_results in self.results.items():
            if 'concurrent_reads' in db_results:
                users = []
                throughputs = []
                for test_name, result in db_results['concurrent_reads'].items():
                    user_count = int(test_name.split('_')[0])
                    users.append(user_count)
                    throughputs.append(result['throughput'])
                
                if len(users) > 1:
                    # Simple scalability metric: ratio of max to min throughput
                    scalability_score = max(throughputs) / min(throughputs) if min(throughputs) > 0 else 0
                    databases.append(db_name)
                    scalability_scores.append(scalability_score)
        
        if databases:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(databases, scalability_scores, color=colors[:len(databases)])
            ax.set_title('Scalability Analysis')
            ax.set_ylabel('Scalability Score')
            
            for bar, score in zip(bars, scalability_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{score:.1f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No scalability data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scalability Analysis')
    
    def _plot_error_rates(self, ax):
        """Plot error rates during stress testing."""
        databases = []
        error_rates = []
        
        for db_name, db_results in self.results.items():
            total_operations = 0
            total_failures = 0
            
            if 'concurrent_reads' in db_results:
                for result in db_results['concurrent_reads'].values():
                    total_operations += result['successful_operations'] + result['failed_operations']
                    total_failures += result['failed_operations']
            
            if 'bulk_inserts' in db_results:
                for result in db_results['bulk_inserts'].values():
                    if 'error' in result:
                        total_failures += 1
                        total_operations += 1
                    else:
                        total_operations += 1
            
            if total_operations > 0:
                error_rate = (total_failures / total_operations) * 100
                databases.append(db_name)
                error_rates.append(error_rate)
        
        if databases:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(databases, error_rates, color=colors[:len(databases)])
            ax.set_title('Error Rates During Stress Testing')
            ax.set_ylabel('Error Rate (%)')
            
            for bar, rate in zip(bars, error_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{rate:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No error data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Rates During Stress Testing')
    
    def generate_stress_report(self):
        """Generate comprehensive stress testing report."""
        logger.info("📝 Generating stress testing report...")
        
        report_lines = [
            "=" * 80,
            "ALL 4 DATABASES COMPREHENSIVE STRESS TESTING REPORT",
            "Chicago 311 Service Requests - Load Testing Analysis",
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"• Databases Tested: {len(self.results)}",
            f"• Tested Databases: {', '.join(self.results.keys())}",
            f"• Test Categories: Concurrent Reads, Bulk Inserts, Memory Pressure",
            "",
            "DETAILED STRESS TEST RESULTS",
            "-" * 40
        ]
        
        for db_name, db_results in self.results.items():
            report_lines.extend([
                "",
                f"{db_name.upper()} STRESS TEST RESULTS",
                "=" * 30
            ])
            
            # Concurrent reads
            if 'concurrent_reads' in db_results:
                report_lines.append("\nConcurrent Read Performance:")
                max_throughput = 0
                for test_name, result in db_results['concurrent_reads'].items():
                    users = test_name.replace('_users', '')
                    throughput = result['throughput']
                    success_rate = (result['successful_operations'] / (result['successful_operations'] + result['failed_operations'])) * 100
                    report_lines.append(f"  • {users} users: {throughput:.2f} ops/sec ({success_rate:.1f}% success rate)")
                    max_throughput = max(max_throughput, throughput)
                report_lines.append(f"  Maximum throughput: {max_throughput:.2f} ops/sec")
            
            # Bulk inserts
            if 'bulk_inserts' in db_results:
                report_lines.append("\nBulk Insert Performance:")
                max_insertion_rate = 0
                for test_name, result in db_results['bulk_inserts'].items():
                    if 'error' not in result:
                        batch_size = result['batch_size']
                        rate = result['insertion_rate']
                        report_lines.append(f"  • Batch {batch_size}: {rate:.0f} records/sec")
                        max_insertion_rate = max(max_insertion_rate, rate)
                    else:
                        batch_size = result['batch_size']
                        report_lines.append(f"  • Batch {batch_size}: FAILED - {result['error']}")
                if max_insertion_rate > 0:
                    report_lines.append(f"  Maximum insertion rate: {max_insertion_rate:.0f} records/sec")
            
            # Memory pressure
            if 'memory_pressure' in db_results:
                result = db_results['memory_pressure']
                if 'error' not in result:
                    report_lines.extend([
                        "\nMemory Pressure Test:",
                        f"  • Dataset: {result['dataset_size']:,} records ({result['memory_usage_mb']:.1f} MB)",
                        f"  • Performance: {result['insertion_rate']:.0f} records/sec",
                        f"  • Duration: {result['duration']:.2f} seconds"
                    ])
                else:
                    report_lines.extend([
                        "\nMemory Pressure Test:",
                        f"  • FAILED: {result['error']}"
                    ])
        
        # Add recommendations
        report_lines.extend([
            "",
            "",
            "STRESS TESTING RECOMMENDATIONS",
            "-" * 40,
            "🎯 Database Performance Under Load:",
        ])
        
        # Find best performer in each category
        best_concurrent = self._find_best_performer('concurrent_reads')
        best_bulk = self._find_best_performer('bulk_inserts')
        best_memory = self._find_best_performer('memory_pressure')
        
        if best_concurrent:
            report_lines.append(f"   • Best Concurrent Read Performance: {best_concurrent}")
        if best_bulk:
            report_lines.append(f"   • Best Bulk Insert Performance: {best_bulk}")
        if best_memory:
            report_lines.append(f"   • Best Memory Pressure Handling: {best_memory}")
        
        report_lines.extend([
            "",
            "💡 Load Testing Insights:",
            "   • MongoDB: Excellent for concurrent document operations",
            "   • Elasticsearch: Optimized for search workloads", 
            "   • PostgreSQL: Reliable under consistent load",
            "   • DuckDB: Efficient for analytical batch processing",
            "",
            "⚠️  Production Considerations:",
            "   • Monitor connection pooling under high concurrent load",
            "   • Consider sharding/partitioning for very large datasets", 
            "   • Implement proper error handling and retry mechanisms",
            "   • Regular performance monitoring and capacity planning"
        ])
        
        # Write report
        with open('all_4_databases_stress_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("📄 Generated all_4_databases_stress_report.txt")
    
    def _find_best_performer(self, test_category: str) -> str:
        """Find the best performing database for a test category."""
        best_db = None
        best_score = 0
        
        for db_name, db_results in self.results.items():
            if test_category in db_results:
                if test_category == 'concurrent_reads':
                    scores = [result['throughput'] for result in db_results[test_category].values()]
                    score = max(scores) if scores else 0
                elif test_category == 'bulk_inserts':
                    scores = [result['insertion_rate'] for result in db_results[test_category].values() 
                             if 'error' not in result]
                    score = max(scores) if scores else 0
                elif test_category == 'memory_pressure':
                    if 'error' not in db_results[test_category]:
                        score = db_results[test_category]['insertion_rate']
                    else:
                        score = 0
                
                if score > best_score:
                    best_score = score
                    best_db = db_name
        
        return best_db
    
    def cleanup_stress_data(self):
        """Clean up stress test data from databases."""
        logger.info("🧹 Cleaning up stress test data...")
        
        for db_name, handler in self.handlers.items():
            try:
                if db_name == 'MongoDB':
                    # Drop stress test collections
                    handler.client[handler.db_name].drop_collection('stress_test_records')
                    handler.client[handler.db_name].drop_collection('memory_pressure_test')
                elif db_name == 'Elasticsearch':
                    # Delete stress test indices
                    if handler.es.indices.exists(index="stress_test_index"):
                        handler.es.indices.delete(index="stress_test_index")
                    if handler.es.indices.exists(index="memory_pressure_test"):
                        handler.es.indices.delete(index="memory_pressure_test")
                elif db_name == 'PostgreSQL':
                    with handler.connection.cursor() as cursor:
                        cursor.execute("DROP TABLE IF EXISTS stress_test_records")
                elif db_name == 'DuckDB':
                    handler.conn.execute("DROP TABLE IF EXISTS stress_test_records")
                
                logger.info(f"  ✅ Cleaned up {db_name} stress test data")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not clean up {db_name}: {e}")

def main():
    """Main execution."""
    try:
        stress_tester = All4DatabasesStressTesting()
        stress_tester.run_comprehensive_stress_testing()
        
        logger.info("🎉 All 4 databases stress testing completed successfully!")
        logger.info("📊 Generated files:")
        logger.info("   - all_4_databases_stress_charts.png")
        logger.info("   - all_4_databases_stress_results.json")
        logger.info("   - all_4_databases_stress_report.txt")
        
    except Exception as e:
        logger.error(f"❌ Stress testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()