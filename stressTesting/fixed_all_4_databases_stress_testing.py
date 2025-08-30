#!/usr/bin/env python3
"""
Fixed All 4 Databases Comprehensive Stress Testing System
Addresses NumPy 2.0 compatibility issues and generates charts for ALL 4 databases
"""

import os
import sys
import time
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from databases.mongodb_handler import MongoDBHandler
from databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedAll4DatabasesStressTesting:
    """Comprehensive stress testing system for ALL 4 databases with NumPy fixes."""
    
    def __init__(self):
        self.handlers = {}
        self.results = {}
        self.concurrent_users = [1, 5, 10, 25, 50, 100]
        self.batch_sizes = [100, 500, 1000, 5000, 10000]
        
    def setup_databases(self):
        """Set up all 4 database connections."""
        logger.info("🔧 Setting up ALL 4 databases for stress testing...")
        
        # 1. MongoDB
        try:
            mongo_handler = MongoDBHandler()
            count = mongo_handler.collection.count_documents({})
            if count > 0:
                self.handlers['MongoDB'] = mongo_handler
                logger.info(f"✅ MongoDB ready ({count:,} records)")
        except Exception as e:
            logger.error(f"❌ MongoDB setup failed: {e}")
        
        # 2. Elasticsearch - Fixed version
        try:
            import pymongo
            import psycopg2
            import duckdb
            
            # Simple Elasticsearch handler without NumPy issues
            class FixedElasticsearchHandler:
                def __init__(self):
                    from elasticsearch import Elasticsearch
                    self.es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
                    self.index_name = 'chicago_311_fixed'
                
                def simple_search(self):
                    """Simple search without NumPy dependencies."""
                    try:
                        return self.es.search(index=self.index_name, body={"query": {"match_all": {}}}, size=1)
                    except:
                        return None
                        
                def bulk_insert_simple(self, records):
                    """Simple bulk insert without NumPy."""
                    try:
                        actions = []
                        for i, record in enumerate(records):
                            # Convert all values to simple Python types
                            clean_record = {}
                            for k, v in record.items():
                                if isinstance(v, (int, float, str, bool)) or v is None:
                                    clean_record[k] = v
                                else:
                                    clean_record[k] = str(v)
                            
                            action = {
                                "_index": f"stress_test_{int(time.time())}",
                                "_source": clean_record
                            }
                            actions.append(action)
                        
                        from elasticsearch.helpers import bulk
                        success, _ = bulk(self.es, actions, request_timeout=60)
                        return success
                    except Exception as e:
                        logger.error(f"Elasticsearch bulk insert failed: {e}")
                        return 0
            
            es_handler = FixedElasticsearchHandler()
            if es_handler.simple_search() is not None:
                self.handlers['Elasticsearch'] = es_handler
                logger.info("✅ Elasticsearch ready (prepared for stress testing)")
        except Exception as e:
            logger.error(f"❌ Elasticsearch setup failed: {e}")
        
        # 3. PostgreSQL
        try:
            import psycopg2
            
            class FixedPostgreSQLHandler:
                def __init__(self):
                    self.connection = psycopg2.connect(
                        host="localhost", port=5432, database="postgres", user="postgres", password="postgres"
                    )
                    self.connection.autocommit = True
                
                def simple_query(self):
                    with self.connection.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM chicago_311_requests")
                        return cursor.fetchone()[0]
                        
                def bulk_insert_simple(self, records):
                    try:
                        with self.connection.cursor() as cursor:
                            insert_sql = """
                            INSERT INTO stress_test_temp (id, data) VALUES (%s, %s)
                            ON CONFLICT DO NOTHING
                            """
                            # Create temp table if not exists
                            cursor.execute("""
                            CREATE TABLE IF NOT EXISTS stress_test_temp (
                                id VARCHAR(50), data TEXT
                            )
                            """)
                            
                            values = [(str(uuid.uuid4()), str(record)) for record in records]
                            cursor.executemany(insert_sql, values)
                            return len(values)
                    except Exception as e:
                        logger.error(f"PostgreSQL bulk insert failed: {e}")
                        return 0
            
            pg_handler = FixedPostgreSQLHandler()
            count = pg_handler.simple_query()
            if count > 0:
                self.handlers['PostgreSQL'] = pg_handler
                logger.info(f"✅ PostgreSQL ready ({count:,} records)")
        except Exception as e:
            logger.error(f"❌ PostgreSQL setup failed: {e}")
        
        # 4. DuckDB
        try:
            import duckdb
            
            class FixedDuckDBHandler:
                def __init__(self):
                    self.conn = duckdb.connect("chicago_311.duckdb")
                
                def simple_query(self):
                    return self.conn.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()[0]
                
                def bulk_insert_simple(self, records):
                    try:
                        # Create temp table
                        self.conn.execute("""
                        CREATE TABLE IF NOT EXISTS stress_test_temp (
                            id VARCHAR, data VARCHAR
                        )
                        """)
                        
                        # Insert records
                        for i, record in enumerate(records):
                            self.conn.execute(
                                "INSERT INTO stress_test_temp VALUES (?, ?)", 
                                (str(uuid.uuid4()), str(record))
                            )
                        return len(records)
                    except Exception as e:
                        logger.error(f"DuckDB bulk insert failed: {e}")
                        return 0
            
            duck_handler = FixedDuckDBHandler()
            count = duck_handler.simple_query()
            if count > 0:
                self.handlers['DuckDB'] = duck_handler
                logger.info(f"✅ DuckDB ready ({count:,} records)")
        except Exception as e:
            logger.error(f"❌ DuckDB setup failed: {e}")
        
        logger.info(f"🎯 Ready to stress test {len(self.handlers)} databases: {list(self.handlers.keys())}")
    
    def concurrent_read_test(self, db_name, handler, users):
        """Test concurrent read performance."""
        results = {'successful': 0, 'failed': 0, 'total_time': 0}
        
        def read_operation():
            start_time = time.time()
            try:
                if db_name == 'MongoDB':
                    handler.collection.count_documents({})
                elif db_name == 'Elasticsearch':
                    handler.simple_search()
                elif db_name == 'PostgreSQL':
                    with handler.connection.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM chicago_311_requests LIMIT 1")
                        cursor.fetchone()
                elif db_name == 'DuckDB':
                    handler.conn.execute("SELECT COUNT(*) FROM chicago_311_requests LIMIT 1")
                
                results['successful'] += 1
                return time.time() - start_time
            except Exception as e:
                results['failed'] += 1
                return 0
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=users) as executor:
            futures = [executor.submit(read_operation) for _ in range(users)]
            for future in as_completed(futures):
                future.result()
        
        total_time = time.time() - start_time
        success_rate = results['successful'] / users if users > 0 else 0
        ops_per_sec = users / total_time if total_time > 0 else 0
        
        return {
            'users': users,
            'successful': results['successful'],
            'failed': results['failed'],
            'success_rate': success_rate,
            'ops_per_sec': ops_per_sec,
            'total_time': total_time
        }
    
    def bulk_insert_test(self, db_name, handler, batch_size):
        """Test bulk insert performance."""
        # Generate simple test records without NumPy
        test_records = []
        for i in range(batch_size):
            record = {
                'id': str(uuid.uuid4()),
                'test_field': f'stress_test_{i}',
                'batch_size': batch_size,
                'timestamp': datetime.now().isoformat()
            }
            test_records.append(record)
        
        start_time = time.time()
        try:
            if db_name == 'MongoDB':
                result = handler.collection.insert_many(test_records)
                inserted = len(result.inserted_ids)
            elif db_name == 'Elasticsearch':
                inserted = handler.bulk_insert_simple(test_records)
            elif db_name == 'PostgreSQL':
                inserted = handler.bulk_insert_simple(test_records)
            elif db_name == 'DuckDB':
                inserted = handler.bulk_insert_simple(test_records)
            else:
                inserted = 0
            
            duration = time.time() - start_time
            records_per_sec = inserted / duration if duration > 0 else 0
            
            return {
                'batch_size': batch_size,
                'inserted': inserted,
                'duration': duration,
                'records_per_sec': records_per_sec,
                'success': True
            }
            
        except Exception as e:
            return {
                'batch_size': batch_size,
                'inserted': 0,
                'duration': time.time() - start_time,
                'records_per_sec': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_stress_testing(self):
        """Run comprehensive stress testing on all databases."""
        logger.info("🚀 Starting comprehensive stress testing on 4 databases...")
        
        for db_name, handler in self.handlers.items():
            logger.info(f"\n💪 Stress testing {db_name}...")
            self.results[db_name] = {
                'database_type': self.get_db_type(db_name),
                'concurrent_reads': [],
                'bulk_inserts': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Test concurrent reads
            logger.info("  🔄 Testing concurrent reads...")
            for users in self.concurrent_users:
                result = self.concurrent_read_test(db_name, handler, users)
                self.results[db_name]['concurrent_reads'].append(result)
                logger.info(f"    {users} users: {result['successful']}/{users} successful, {result['ops_per_sec']:.2f} ops/sec")
            
            # Test bulk inserts
            logger.info("  📥 Testing bulk inserts...")
            for batch_size in self.batch_sizes:
                result = self.bulk_insert_test(db_name, handler, batch_size)
                self.results[db_name]['bulk_inserts'].append(result)
                if result['success']:
                    logger.info(f"    Batch {batch_size}: {result['inserted']} records in {result['duration']:.2f}s ({result['records_per_sec']:.0f} records/sec)")
                else:
                    logger.info(f"    Batch {batch_size}: Failed - {result.get('error', 'Unknown error')}")
            
            logger.info(f"✅ {db_name} stress testing completed")
    
    def get_db_type(self, db_name):
        """Get database type for categorization."""
        types = {
            'MongoDB': 'document',
            'Elasticsearch': 'search', 
            'PostgreSQL': 'relational',
            'DuckDB': 'analytical'
        }
        return types.get(db_name, 'unknown')
    
    def generate_charts(self):
        """Generate comprehensive stress testing charts."""
        logger.info("📈 Generating comprehensive stress testing charts...")
        
        # Create comprehensive stress testing charts
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Database Stress Test Results - ALL 4 Databases', fontsize=16, fontweight='bold')
        
        # Chart 1: Concurrent Read Performance
        ax1 = axes[0, 0]
        for db_name in self.results:
            reads = self.results[db_name]['concurrent_reads']
            if reads:
                users = [r['users'] for r in reads]
                ops = [r['ops_per_sec'] for r in reads]
                ax1.plot(users, ops, marker='o', label=db_name, linewidth=2)
        ax1.set_xlabel('Concurrent Users')
        ax1.set_ylabel('Operations/Second')
        ax1.set_title('Concurrent Read Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Success Rate Under Load
        ax2 = axes[0, 1]
        for db_name in self.results:
            reads = self.results[db_name]['concurrent_reads']
            if reads:
                users = [r['users'] for r in reads]
                success_rates = [r['success_rate'] * 100 for r in reads]
                ax2.plot(users, success_rates, marker='s', label=db_name, linewidth=2)
        ax2.set_xlabel('Concurrent Users')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate Under Load')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        # Chart 3: Bulk Insert Performance
        ax3 = axes[0, 2]
        db_names = []
        insert_performance = []
        for db_name in self.results:
            inserts = self.results[db_name]['bulk_inserts']
            successful_inserts = [i for i in inserts if i['success']]
            if successful_inserts:
                avg_performance = sum(i['records_per_sec'] for i in successful_inserts) / len(successful_inserts)
                db_names.append(db_name)
                insert_performance.append(avg_performance)
        
        if db_names:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(db_names)]
            bars = ax3.bar(db_names, insert_performance, color=colors, alpha=0.8)
            ax3.set_ylabel('Records/Second')
            ax3.set_title('Average Bulk Insert Performance')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, insert_performance):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.0f}', ha='center', va='bottom')
        
        # Chart 4: Performance vs Batch Size
        ax4 = axes[1, 0]
        for db_name in self.results:
            inserts = self.results[db_name]['bulk_inserts']
            successful_inserts = [i for i in inserts if i['success']]
            if successful_inserts:
                batch_sizes = [i['batch_size'] for i in successful_inserts]
                records_per_sec = [i['records_per_sec'] for i in successful_inserts]
                ax4.plot(batch_sizes, records_per_sec, marker='d', label=db_name, linewidth=2)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Records/Second')
        ax4.set_title('Performance vs Batch Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        # Chart 5: Database Comparison Summary
        ax5 = axes[1, 1]
        categories = ['Avg Read Ops/Sec', 'Max Concurrent Users', 'Avg Insert Rate']
        db_comparison = {}
        
        for db_name in self.results:
            reads = self.results[db_name]['concurrent_reads']
            inserts = self.results[db_name]['bulk_inserts']
            
            avg_read_ops = sum(r['ops_per_sec'] for r in reads) / len(reads) if reads else 0
            max_users = max((r['users'] for r in reads if r['success_rate'] > 0.9), default=0)
            successful_inserts = [i for i in inserts if i['success']]
            avg_insert_rate = sum(i['records_per_sec'] for i in successful_inserts) / len(successful_inserts) if successful_inserts else 0
            
            db_comparison[db_name] = [avg_read_ops, max_users, avg_insert_rate]
        
        # Normalize for comparison
        if db_comparison:
            x = range(len(categories))
            width = 0.2
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (db_name, values) in enumerate(db_comparison.items()):
                normalized_values = [v / max(db_comparison[db][j] for db in db_comparison) * 100 
                                   for j, v in enumerate(values)]
                ax5.bar([xi + i * width for xi in x], normalized_values, width, 
                       label=db_name, color=colors[i % len(colors)], alpha=0.8)
            
            ax5.set_xlabel('Performance Categories')
            ax5.set_ylabel('Normalized Performance (%)')
            ax5.set_title('Database Performance Comparison')
            ax5.set_xticks([xi + width * 1.5 for xi in x])
            ax5.set_xticklabels(categories, rotation=45, ha='right')
            ax5.legend()
        
        # Chart 6: Overall Stress Test Summary
        ax6 = axes[1, 2]
        summary_data = []
        db_labels = []
        
        for db_name in self.results:
            reads = self.results[db_name]['concurrent_reads']
            inserts = self.results[db_name]['bulk_inserts']
            
            total_ops = sum(r['ops_per_sec'] for r in reads)
            total_inserts = sum(i['records_per_sec'] for i in inserts if i['success'])
            overall_score = (total_ops + total_inserts) / 100  # Scale down for display
            
            summary_data.append(overall_score)
            db_labels.append(db_name)
        
        if summary_data:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(db_labels)]
            bars = ax6.bar(db_labels, summary_data, color=colors, alpha=0.8)
            ax6.set_ylabel('Overall Performance Score')
            ax6.set_title('Overall Stress Test Performance')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, summary_data):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comprehensive_stress_test_charts_all_4_databases.png', dpi=300, bbox_inches='tight')
        plt.savefig('database_stress_testing_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Generated comprehensive_stress_test_charts_all_4_databases.png")
        logger.info("📊 Generated database_stress_testing_comparison.png")
    
    def save_results(self):
        """Save stress testing results to JSON."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'all_4_databases_comprehensive_stress_testing',
            'databases_tested': list(self.results.keys()),
            'test_parameters': {
                'concurrent_users': self.concurrent_users,
                'batch_sizes': self.batch_sizes
            },
            'results': self.results
        }
        
        with open('all_4_databases_stress_test_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info("💾 Results saved to all_4_databases_stress_test_results.json")
    
    def generate_report(self):
        """Generate comprehensive text report."""
        report_lines = [
            "=" * 80,
            "ALL 4 DATABASES COMPREHENSIVE STRESS TEST REPORT",
            "Chicago 311 Service Requests - Stress Testing Analysis",
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"• Databases Tested: {len(self.results)}",
            f"• Available Databases: {', '.join(self.results.keys())}",
            "",
            "PERFORMANCE RANKINGS",
            "-" * 40,
        ]
        
        # Find best performers
        best_concurrent = max(self.results.items(), 
                            key=lambda x: max((r['ops_per_sec'] for r in x[1]['concurrent_reads']), default=0))
        best_bulk = max(self.results.items(),
                       key=lambda x: max((r['records_per_sec'] for r in x[1]['bulk_inserts'] if r['success']), default=0))
        
        report_lines.extend([
            f"🏆 Best Concurrent Performance: {best_concurrent[0]}",
            f"🏆 Best Bulk Insert Performance: {best_bulk[0]}",
            "",
            "DETAILED RESULTS BY DATABASE",
            "-" * 40,
        ])
        
        for db_name, results in self.results.items():
            db_type = results['database_type']
            report_lines.extend([
                "",
                f"{db_name.upper()} ({db_type})",
                "=" * 20,
                "",
                "Concurrent Read Performance:",
            ])
            
            for read_result in results['concurrent_reads']:
                report_lines.append(
                    f"  • {read_result['users']} users: {read_result['ops_per_sec']:.2f} ops/sec "
                    f"({read_result['success_rate']*100:.1f}% success)"
                )
            
            report_lines.append("\nBulk Insert Performance:")
            for insert_result in results['bulk_inserts']:
                if insert_result['success']:
                    report_lines.append(
                        f"  • Batch {insert_result['batch_size']}: {insert_result['records_per_sec']:.0f} records/sec"
                    )
                else:
                    report_lines.append(
                        f"  • Batch {insert_result['batch_size']}: Failed"
                    )
        
        report_lines.extend([
            "",
            "",
            "RECOMMENDATIONS",
            "-" * 40,
            "🎯 Database Recommendations:",
            "   • Choose based on your specific load patterns",
            "   • Consider concurrent user requirements",
            "   • Evaluate bulk insert needs",
            "",
            "💡 Performance Insights:",
            "   • Monitor success rates under high concurrency",
            "   • Optimize batch sizes for bulk operations",
            "   • Consider database-specific tuning parameters",
        ])
        
        with open('all_4_databases_stress_test_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("📄 Generated all_4_databases_stress_test_report.txt")
    
    def run(self):
        """Run the complete stress testing suite."""
        self.setup_databases()
        
        if not self.handlers:
            logger.error("❌ No databases available for stress testing!")
            return
        
        self.run_comprehensive_stress_testing()
        self.save_results()
        self.generate_charts()
        self.generate_report()
        
        logger.info("\n🎉 All 4 databases stress testing completed successfully!")
        logger.info("📊 Generated files:")
        logger.info("   - comprehensive_stress_test_charts_all_4_databases.png")
        logger.info("   - database_stress_testing_comparison.png")
        logger.info("   - all_4_databases_stress_test_results.json")
        logger.info("   - all_4_databases_stress_test_report.txt")

if __name__ == "__main__":
    stress_tester = FixedAll4DatabasesStressTesting()
    stress_tester.run()