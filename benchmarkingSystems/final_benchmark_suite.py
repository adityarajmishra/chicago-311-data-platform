#!/usr/bin/env python3
"""
Final Comprehensive Benchmark Suite
Runs complete performance analysis on all available databases
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
from datetime import datetime
from typing import Dict, List, Any, Tuple
import concurrent.futures
import random

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalBenchmarkSuite:
    """Complete benchmark suite for Chicago 311 databases."""
    
    def __init__(self):
        self.databases = {}
        self.benchmark_results = {}
        self.queries = {
            'simple_count': {
                'name': 'Count All Records',
                'description': 'Basic count query to test simple performance'
            },
            'status_filter': {
                'name': 'Filter by Status',
                'description': 'Filter records by status = "Open"'
            },
            'date_range': {
                'name': 'Date Range Query',
                'description': 'Records created in the last 30 days'
            },
            'text_search': {
                'name': 'Text Search',
                'description': 'Search for specific service request types'
            },
            'geospatial': {
                'name': 'Geospatial Query',
                'description': 'Find records within geographic bounds'
            },
            'aggregation': {
                'name': 'Group Aggregation',
                'description': 'Group by department and count'
            }
        }
        
    def setup_connections(self):
        """Setup connections to available databases."""
        print("🔄 Setting up database connections...")
        
        # MongoDB
        try:
            mongo_handler = MongoDBHandler()
            count = mongo_handler.collection.count_documents({})
            if count > 0:
                self.databases['MongoDB'] = {
                    'handler': mongo_handler,
                    'count': count
                }
                print(f"✅ MongoDB: {count:,} records")
            else:
                print("⚠️ MongoDB: No data found")
        except Exception as e:
            print(f"❌ MongoDB failed: {e}")
        
        # Elasticsearch
        try:
            es_handler = ElasticsearchHandler()
            result = es_handler.es.count(index=es_handler.index_name)
            count = result['count']
            if count > 0:
                self.databases['Elasticsearch'] = {
                    'handler': es_handler,
                    'count': count
                }
                print(f"✅ Elasticsearch: {count:,} records")
            else:
                print("⚠️ Elasticsearch: No data found")
        except Exception as e:
            print(f"❌ Elasticsearch failed: {e}")
        
        if not self.databases:
            raise Exception("No databases with data available!")
        
        print(f"📊 Ready for benchmarking: {list(self.databases.keys())}")
    
    def run_mongodb_benchmarks(self, handler) -> Dict:
        """Run MongoDB-specific benchmark queries."""
        results = {}
        collection = handler.collection
        
        # 1. Simple count
        start_time = time.time()
        count = collection.count_documents({})
        results['simple_count'] = {
            'duration': time.time() - start_time,
            'result_count': count
        }
        
        # 2. Status filter
        start_time = time.time()
        open_count = collection.count_documents({'status': 'Open'})
        results['status_filter'] = {
            'duration': time.time() - start_time,
            'result_count': open_count
        }
        
        # 3. Date range (last 30 days)
        start_time = time.time()
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_count = collection.count_documents({
            'created_date': {'$gte': thirty_days_ago}
        })
        results['date_range'] = {
            'duration': time.time() - start_time,
            'result_count': recent_count
        }
        
        # 4. Text search
        start_time = time.time()
        pothole_count = collection.count_documents({
            'sr_type': {'$regex': 'Pothole', '$options': 'i'}
        })
        results['text_search'] = {
            'duration': time.time() - start_time,
            'result_count': pothole_count
        }
        
        # 5. Geospatial query (Chicago downtown area)
        start_time = time.time()
        geo_count = collection.count_documents({
            'latitude': {'$gte': 41.8, '$lte': 41.9},
            'longitude': {'$gte': -87.7, '$lte': -87.6}
        })
        results['geospatial'] = {
            'duration': time.time() - start_time,
            'result_count': geo_count
        }
        
        # 6. Aggregation
        start_time = time.time()
        pipeline = [
            {'$group': {'_id': '$owner_department', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 10}
        ]
        agg_results = list(collection.aggregate(pipeline))
        results['aggregation'] = {
            'duration': time.time() - start_time,
            'result_count': len(agg_results)
        }
        
        return results
    
    def run_elasticsearch_benchmarks(self, handler) -> Dict:
        """Run Elasticsearch-specific benchmark queries."""
        results = {}
        es = handler.es
        index = handler.index_name
        
        # 1. Simple count
        start_time = time.time()
        result = es.count(index=index)
        count = result['count']
        results['simple_count'] = {
            'duration': time.time() - start_time,
            'result_count': count
        }
        
        # 2. Status filter
        start_time = time.time()
        query = {"query": {"term": {"status": "Open"}}}
        result = es.count(index=index, body=query)
        results['status_filter'] = {
            'duration': time.time() - start_time,
            'result_count': result['count']
        }
        
        # 3. Date range
        start_time = time.time()
        query = {
            "query": {
                "range": {
                    "created_date": {
                        "gte": "now-30d"
                    }
                }
            }
        }
        result = es.count(index=index, body=query)
        results['date_range'] = {
            'duration': time.time() - start_time,
            'result_count': result['count']
        }
        
        # 4. Text search
        start_time = time.time()
        query = {
            "query": {
                "match": {
                    "sr_type": {
                        "query": "Pothole",
                        "operator": "and"
                    }
                }
            }
        }
        result = es.count(index=index, body=query)
        results['text_search'] = {
            'duration': time.time() - start_time,
            'result_count': result['count']
        }
        
        # 5. Geospatial query
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
        result = es.count(index=index, body=query)
        results['geospatial'] = {
            'duration': time.time() - start_time,
            'result_count': result['count']
        }
        
        # 6. Aggregation
        start_time = time.time()
        query = {
            "size": 0,
            "aggs": {
                "departments": {
                    "terms": {
                        "field": "owner_department",
                        "size": 10
                    }
                }
            }
        }
        result = es.search(index=index, body=query)
        agg_count = len(result['aggregations']['departments']['buckets'])
        results['aggregation'] = {
            'duration': time.time() - start_time,
            'result_count': agg_count
        }
        
        return results
    
    def run_concurrent_load_test(self, db_name: str, concurrent_users: List[int]) -> Dict:
        """Run concurrent load testing."""
        if db_name not in self.databases:
            return {}
        
        print(f"🚀 Running concurrent load test for {db_name}...")
        results = {}
        
        def worker_query():
            """Simple worker query for load testing."""
            start_time = time.time()
            try:
                if db_name == 'MongoDB':
                    handler = self.databases[db_name]['handler']
                    handler.collection.count_documents({'status': 'Open'})
                elif db_name == 'Elasticsearch':
                    handler = self.databases[db_name]['handler']
                    query = {"query": {"term": {"status": "Open"}}}
                    handler.es.count(index=handler.index_name, body=query)
                
                return time.time() - start_time
            except Exception as e:
                logger.error(f"Worker query failed: {e}")
                return float('inf')
        
        for users in concurrent_users:
            print(f"   Testing {users} concurrent users...")
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=users) as executor:
                futures = [executor.submit(worker_query) for _ in range(users)]
                response_times = []
                
                for future in concurrent.futures.as_completed(futures):
                    response_times.append(future.result())
            
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
                    'throughput': users / total_time
                }
                
                print(f"      QPS: {results[users]['queries_per_second']:.2f}, "
                      f"Success: {results[users]['success_rate']*100:.1f}%")
            else:
                results[users] = {
                    'total_time': total_time,
                    'success_rate': 0,
                    'queries_per_second': 0
                }
        
        return results
    
    def run_comprehensive_benchmarks(self):
        """Run all benchmarks for all databases."""
        print("\n🎯 Starting Comprehensive Benchmarks")
        print("=" * 80)
        
        for db_name, db_info in self.databases.items():
            print(f"\n📊 Benchmarking {db_name} ({db_info['count']:,} records)...")
            
            # Run query benchmarks
            if db_name == 'MongoDB':
                query_results = self.run_mongodb_benchmarks(db_info['handler'])
            elif db_name == 'Elasticsearch':
                query_results = self.run_elasticsearch_benchmarks(db_info['handler'])
            else:
                query_results = {}
            
            # Run concurrent load tests
            concurrent_users = [1, 5, 10, 25, 50]
            load_test_results = self.run_concurrent_load_test(db_name, concurrent_users)
            
            self.benchmark_results[db_name] = {
                'record_count': db_info['count'],
                'query_benchmarks': query_results,
                'load_test_results': load_test_results
            }
            
            # Display query results
            print(f"   Query Performance Results:")
            for query_name, result in query_results.items():
                duration = result['duration']
                count = result['result_count']
                print(f"   • {self.queries[query_name]['name']}: {duration:.4f}s ({count:,} results)")
    
    def generate_comprehensive_charts(self):
        """Generate comprehensive benchmark visualization."""
        print("\n📊 Generating comprehensive charts...")
        
        # Set style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Query Performance Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        query_data = []
        for db_name, results in self.benchmark_results.items():
            for query_name, metrics in results['query_benchmarks'].items():
                query_data.append({
                    'Database': db_name,
                    'Query': self.queries[query_name]['name'].replace(' ', '\n'),
                    'Duration': metrics['duration']
                })
        
        if query_data:
            df = pd.DataFrame(query_data)
            df_pivot = df.pivot(index='Query', columns='Database', values='Duration')
            df_pivot.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Query Performance Comparison', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Duration (seconds)')
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.legend(fontsize=8)
        
        # 2. Throughput Comparison (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        throughput_data = []
        for db_name, results in self.benchmark_results.items():
            if 'load_test_results' in results:
                for users, metrics in results['load_test_results'].items():
                    throughput_data.append({
                        'Database': db_name,
                        'Concurrent_Users': users,
                        'QPS': metrics['queries_per_second']
                    })
        
        if throughput_data:
            df_throughput = pd.DataFrame(throughput_data)
            for db in df_throughput['Database'].unique():
                db_data = df_throughput[df_throughput['Database'] == db]
                ax2.plot(db_data['Concurrent_Users'], db_data['QPS'], 
                        marker='o', label=db, linewidth=2)
            
            ax2.set_title('Concurrent Load Performance', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Concurrent Users')
            ax2.set_ylabel('Queries Per Second')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Response Time Heat Map (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        heatmap_data = []
        query_names = []
        db_names = list(self.benchmark_results.keys())
        
        for query_name in self.queries.keys():
            query_names.append(self.queries[query_name]['name'])
            row = []
            for db_name in db_names:
                if (db_name in self.benchmark_results and 
                    query_name in self.benchmark_results[db_name]['query_benchmarks']):
                    duration = self.benchmark_results[db_name]['query_benchmarks'][query_name]['duration']
                    row.append(duration)
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, index=query_names, columns=db_names)
            sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax3,
                       cbar_kws={'label': 'Duration (seconds)'})
            ax3.set_title('Response Time Heatmap', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            ax3.tick_params(axis='y', rotation=0, labelsize=8)
        
        # 4. Database Record Counts (Second Row Left)
        ax4 = fig.add_subplot(gs[1, 0])
        db_names = list(self.benchmark_results.keys())
        record_counts = [self.benchmark_results[db]['record_count'] for db in db_names]
        
        bars = ax4.bar(db_names, record_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(db_names)])
        ax4.set_title('Database Record Counts', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Records')
        
        # Add value labels on bars
        for bar, count in zip(bars, record_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # 5. Success Rate Analysis (Second Row Center)
        ax5 = fig.add_subplot(gs[1, 1])
        for db_name, results in self.benchmark_results.items():
            if 'load_test_results' in results:
                users = list(results['load_test_results'].keys())
                success_rates = [results['load_test_results'][u]['success_rate'] * 100 for u in users]
                ax5.plot(users, success_rates, marker='^', label=db_name, linewidth=2)
        
        ax5.set_title('Success Rate Under Load', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Concurrent Users')
        ax5.set_ylabel('Success Rate (%)')
        ax5.set_ylim(0, 105)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Average Response Time vs Load (Second Row Right)
        ax6 = fig.add_subplot(gs[1, 2])
        for db_name, results in self.benchmark_results.items():
            if 'load_test_results' in results:
                users = list(results['load_test_results'].keys())
                avg_times = [results['load_test_results'][u]['avg_response_time'] for u in users]
                ax6.plot(users, avg_times, marker='s', label=db_name, linewidth=2)
        
        ax6.set_title('Response Time vs Load', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Concurrent Users')
        ax6.set_ylabel('Avg Response Time (s)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7-9. Individual Database Deep Dive (Bottom Row)
        for i, (db_name, results) in enumerate(self.benchmark_results.items()):
            if i >= 3:  # Only show first 3 databases
                break
                
            ax = fig.add_subplot(gs[2 + i//3, i%3])
            
            # Show query performance breakdown for this database
            if 'query_benchmarks' in results:
                query_names = [self.queries[q]['name'] for q in results['query_benchmarks'].keys()]
                durations = [results['query_benchmarks'][q]['duration'] for q in results['query_benchmarks'].keys()]
                
                bars = ax.bar(range(len(query_names)), durations, color=f'C{i}', alpha=0.7)
                ax.set_title(f'{db_name} Query Breakdown', fontsize=11, fontweight='bold')
                ax.set_ylabel('Duration (seconds)')
                ax.set_xticks(range(len(query_names)))
                ax.set_xticklabels(query_names, rotation=45, ha='right', fontsize=8)
                
                # Add value labels on bars
                for bar, duration in zip(bars, durations):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{duration:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Comprehensive Database Benchmark Results', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('comprehensive_benchmark_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Comprehensive benchmark chart saved as 'comprehensive_benchmark_results.png'")
    
    def generate_performance_report(self):
        """Generate a detailed performance analysis report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DATABASE PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        total_records = sum(db['record_count'] for db in self.benchmark_results.values())
        report.append(f"• Total Records Analyzed: {total_records:,}")
        report.append(f"• Databases Tested: {len(self.benchmark_results)}")
        report.append(f"• Query Types Tested: {len(self.queries)}")
        report.append("")
        
        # Performance Rankings
        report.append("PERFORMANCE RANKINGS")
        report.append("-" * 40)
        
        # Find fastest database for each query type
        for query_name, query_info in self.queries.items():
            fastest_db = None
            fastest_time = float('inf')
            
            for db_name, results in self.benchmark_results.items():
                if query_name in results['query_benchmarks']:
                    duration = results['query_benchmarks'][query_name]['duration']
                    if duration < fastest_time:
                        fastest_time = duration
                        fastest_db = db_name
            
            if fastest_db:
                report.append(f"🏆 {query_info['name']}: {fastest_db} ({fastest_time:.4f}s)")
        
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS BY DATABASE")
        report.append("-" * 40)
        
        for db_name, results in self.benchmark_results.items():
            report.append(f"\n{db_name.upper()}")
            report.append("-" * 20)
            report.append(f"Records: {results['record_count']:,}")
            
            if 'query_benchmarks' in results:
                report.append("Query Performance:")
                for query_name, metrics in results['query_benchmarks'].items():
                    query_title = self.queries[query_name]['name']
                    duration = metrics['duration']
                    result_count = metrics['result_count']
                    report.append(f"  • {query_title}: {duration:.4f}s ({result_count:,} results)")
            
            if 'load_test_results' in results:
                report.append("Load Test Performance:")
                for users, metrics in results['load_test_results'].items():
                    qps = metrics['queries_per_second']
                    success_rate = metrics['success_rate'] * 100
                    report.append(f"  • {users} users: {qps:.2f} QPS, {success_rate:.1f}% success")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find overall best performing database
        overall_scores = {}
        for db_name, results in self.benchmark_results.items():
            score = 0
            query_count = 0
            
            if 'query_benchmarks' in results:
                for query_name, metrics in results['query_benchmarks'].items():
                    # Lower duration = better score
                    score += 1.0 / (metrics['duration'] + 0.001)
                    query_count += 1
            
            if query_count > 0:
                overall_scores[db_name] = score / query_count
        
        if overall_scores:
            best_db = max(overall_scores.keys(), key=lambda x: overall_scores[x])
            report.append(f"🎯 Overall Best Performance: {best_db}")
            
            # Specific recommendations
            if best_db == 'MongoDB':
                report.append("   - Excellent for complex queries and aggregations")
                report.append("   - Best choice for document-based operations")
            elif best_db == 'Elasticsearch':
                report.append("   - Superior text search and analytics capabilities")
                report.append("   - Best choice for search-heavy applications")
        
        report.append("\n💡 General Recommendations:")
        report.append("   - MongoDB: Use for complex document queries and transactions")
        report.append("   - Elasticsearch: Use for full-text search and analytics")
        report.append("   - Consider data size when choosing between databases")
        report.append("   - Implement proper indexing for optimal performance")
        
        # Save report
        with open('comprehensive_performance_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("✅ Performance report saved as 'comprehensive_performance_report.txt'")
        return '\n'.join(report)
    
    def save_results(self):
        """Save benchmark results to JSON."""
        # Clean results for JSON serialization
        clean_results = {}
        for db_name, results in self.benchmark_results.items():
            clean_results[db_name] = {
                'record_count': results['record_count'],
                'query_benchmarks': results.get('query_benchmarks', {}),
                'load_test_results': results.get('load_test_results', {})
            }
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_results': clean_results,
            'query_definitions': self.queries
        }
        
        with open('final_benchmark_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("✅ Results saved to 'final_benchmark_results.json'")
    
    def close_connections(self):
        """Close all database connections."""
        for db_info in self.databases.values():
            try:
                db_info['handler'].close()
            except:
                pass

def main():
    """Run the complete benchmark suite."""
    suite = FinalBenchmarkSuite()
    
    try:
        suite.setup_connections()
        suite.run_comprehensive_benchmarks()
        suite.generate_comprehensive_charts()
        report = suite.generate_performance_report()
        suite.save_results()
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE BENCHMARKING COMPLETED!")
        print("="*80)
        print("📁 Generated files:")
        print("   • comprehensive_benchmark_results.png")
        print("   • comprehensive_performance_report.txt")
        print("   • final_benchmark_results.json")
        print("\n📋 Quick Summary:")
        print(report.split("DETAILED RESULTS")[0])
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        suite.close_connections()

if __name__ == "__main__":
    main()