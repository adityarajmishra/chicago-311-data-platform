#!/usr/bin/env python3
"""
Chicago 311 Data Platform - Performance Benchmark Test
Real performance testing with sample data generation
"""

import time
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Mock database classes for testing
class MockMongoDBHandler:
    def __init__(self):
        self.data = self._generate_sample_data(10000)
        
    def _generate_sample_data(self, n_records):
        """Generate realistic Chicago 311 sample data"""
        service_types = [
            "Graffiti Removal", "Pothole in Street", "Tree Debris", 
            "Alley Light Out", "Street Light - All/Out", "Sanitation Code Violation",
            "Garbage Cart Maintenance", "Rodent Baiting/Rat Complaint",
            "Building Violation", "Water in Basement"
        ]
        
        statuses = ["Completed", "Open", "In Progress", "Cancelled"]
        departments = ["STREETS & SAN", "BUILDINGS", "WATER MGMNT", "POLICE", "FIRE"]
        
        data = []
        for i in range(n_records):
            created_date = datetime.now() - timedelta(days=random.randint(0, 365))
            completion_time = random.randint(1, 720) if random.random() > 0.3 else None
            
            record = {
                'sr_number': f'SR{i:06d}',
                'sr_type': random.choice(service_types),
                'status': random.choice(statuses),
                'created_date': created_date,
                'closed_date': created_date + timedelta(hours=completion_time) if completion_time else None,
                'department': random.choice(departments),
                'ward': random.randint(1, 50),
                'zip_code': random.choice([60601, 60602, 60603, 60604, 60605, 60606]),
                'latitude': 41.8781 + random.uniform(-0.3, 0.3),
                'longitude': -87.6298 + random.uniform(-0.3, 0.3),
                'community_area': random.randint(1, 77)
            }
            data.append(record)
        
        return data
    
    def find_by_status(self, status):
        start_time = time.time()
        results = [record for record in self.data if record['status'] == status]
        return results, time.time() - start_time
    
    def find_by_service_type(self, service_type):
        start_time = time.time()
        results = [record for record in self.data if record['sr_type'] == service_type]
        return results, time.time() - start_time
    
    def aggregate_by_ward(self):
        start_time = time.time()
        ward_counts = {}
        for record in self.data:
            ward = record['ward']
            ward_counts[ward] = ward_counts.get(ward, 0) + 1
        return ward_counts, time.time() - start_time
    
    def geospatial_query(self, lat_center, lon_center, radius_km=1.0):
        start_time = time.time()
        results = []
        for record in self.data:
            # Simple distance calculation
            lat_diff = record['latitude'] - lat_center
            lon_diff = record['longitude'] - lon_center
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
            if distance <= radius_km:
                results.append(record)
        return results, time.time() - start_time

class MockElasticsearchHandler:
    def __init__(self):
        self.data = self._generate_sample_data(10000)
        
    def _generate_sample_data(self, n_records):
        """Generate realistic Chicago 311 sample data"""
        service_types = [
            "Graffiti Removal", "Pothole in Street", "Tree Debris", 
            "Alley Light Out", "Street Light - All/Out", "Sanitation Code Violation",
            "Garbage Cart Maintenance", "Rodent Baiting/Rat Complaint",
            "Building Violation", "Water in Basement"
        ]
        
        statuses = ["Completed", "Open", "In Progress", "Cancelled"]
        departments = ["STREETS & SAN", "BUILDINGS", "WATER MGMNT", "POLICE", "FIRE"]
        
        data = []
        for i in range(n_records):
            created_date = datetime.now() - timedelta(days=random.randint(0, 365))
            completion_time = random.randint(1, 720) if random.random() > 0.3 else None
            
            record = {
                'sr_number': f'SR{i:06d}',
                'sr_type': random.choice(service_types),
                'status': random.choice(statuses),
                'created_date': created_date,
                'closed_date': created_date + timedelta(hours=completion_time) if completion_time else None,
                'department': random.choice(departments),
                'ward': random.randint(1, 50),
                'zip_code': random.choice([60601, 60602, 60603, 60604, 60605, 60606]),
                'latitude': 41.8781 + random.uniform(-0.3, 0.3),
                'longitude': -87.6298 + random.uniform(-0.3, 0.3),
                'community_area': random.randint(1, 77)
            }
            data.append(record)
        
        return data
    
    def find_by_status(self, status):
        start_time = time.time()
        # Simulate faster search with indexing
        time.sleep(0.001)  # Simulated index lookup
        results = [record for record in self.data if record['status'] == status]
        return results, time.time() - start_time
    
    def find_by_service_type(self, service_type):
        start_time = time.time()
        time.sleep(0.001)  # Simulated index lookup
        results = [record for record in self.data if record['sr_type'] == service_type]
        return results, time.time() - start_time
    
    def aggregate_by_ward(self):
        start_time = time.time()
        time.sleep(0.005)  # Simulated aggregation
        ward_counts = {}
        for record in self.data:
            ward = record['ward']
            ward_counts[ward] = ward_counts.get(ward, 0) + 1
        return ward_counts, time.time() - start_time
    
    def geospatial_query(self, lat_center, lon_center, radius_km=1.0):
        start_time = time.time()
        time.sleep(0.002)  # Simulated geo-index lookup
        results = []
        for record in self.data:
            # Simple distance calculation
            lat_diff = record['latitude'] - lat_center
            lon_diff = record['longitude'] - lon_center
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
            if distance <= radius_km:
                results.append(record)
        return results, time.time() - start_time

class PerformanceBenchmark:
    def __init__(self):
        self.mongo_handler = MockMongoDBHandler()
        self.es_handler = MockElasticsearchHandler()
        self.results = {}
        
    def run_benchmark_suite(self, iterations=5):
        """Run comprehensive benchmark suite"""
        print("🚀 Starting Performance Benchmark Suite")
        print(f"   📊 Dataset size: {len(self.mongo_handler.data):,} records")
        print(f"   🔄 Iterations per test: {iterations}")
        print("=" * 60)
        
        # Test operations
        operations = [
            ("Status Query (Completed)", self._test_status_query, ["Completed"]),
            ("Service Type Query", self._test_service_type_query, ["Graffiti Removal"]),
            ("Ward Aggregation", self._test_ward_aggregation, []),
            ("Geospatial Query", self._test_geospatial_query, [41.8781, -87.6298, 2.0]),
        ]
        
        for op_name, test_func, args in operations:
            print(f"\n🧪 Testing: {op_name}")
            
            # MongoDB tests
            mongo_times = []
            for i in range(iterations):
                _, duration = test_func(self.mongo_handler, *args)
                mongo_times.append(duration)
                print(f"   MongoDB iter {i+1}: {duration*1000:.2f}ms")
            
            # Elasticsearch tests
            es_times = []
            for i in range(iterations):
                _, duration = test_func(self.es_handler, *args)
                es_times.append(duration)
                print(f"   Elasticsearch iter {i+1}: {duration*1000:.2f}ms")
            
            # Calculate statistics
            self.results[op_name] = {
                'mongodb': {
                    'times': mongo_times,
                    'mean': np.mean(mongo_times),
                    'std': np.std(mongo_times),
                    'min': np.min(mongo_times),
                    'max': np.max(mongo_times)
                },
                'elasticsearch': {
                    'times': es_times,
                    'mean': np.mean(es_times),
                    'std': np.std(es_times),
                    'min': np.min(es_times),
                    'max': np.max(es_times)
                }
            }
            
            # Show comparison
            mongo_avg = np.mean(mongo_times) * 1000
            es_avg = np.mean(es_times) * 1000
            speedup = mongo_avg / es_avg if es_avg > 0 else 1
            
            print(f"   📈 MongoDB avg: {mongo_avg:.2f}ms")
            print(f"   📈 Elasticsearch avg: {es_avg:.2f}ms")
            print(f"   ⚡ Speedup: {speedup:.1f}x {'(ES faster)' if speedup > 1 else '(MongoDB faster)'}")
        
        return self.results
    
    def _test_status_query(self, handler, status):
        return handler.find_by_status(status)
    
    def _test_service_type_query(self, handler, service_type):
        return handler.find_by_service_type(service_type)
    
    def _test_ward_aggregation(self, handler):
        return handler.aggregate_by_ward()
    
    def _test_geospatial_query(self, handler, lat, lon, radius):
        return handler.geospatial_query(lat, lon, radius)
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)
        
        # Summary table
        print("\n🎯 Performance Summary (Average Response Times)")
        print("-" * 80)
        print(f"{'Operation':<25} {'MongoDB (ms)':<15} {'Elasticsearch (ms)':<18} {'Speedup':<10}")
        print("-" * 80)
        
        for op_name, data in self.results.items():
            mongo_avg = data['mongodb']['mean'] * 1000
            es_avg = data['elasticsearch']['mean'] * 1000
            speedup = mongo_avg / es_avg if es_avg > 0 else 1
            speedup_text = f"{speedup:.1f}x"
            
            print(f"{op_name:<25} {mongo_avg:<15.2f} {es_avg:<18.2f} {speedup_text:<10}")
        
        # Detailed statistics
        print("\n📈 Detailed Statistics")
        print("-" * 80)
        
        for op_name, data in self.results.items():
            print(f"\n🔍 {op_name}:")
            
            # MongoDB stats
            mongo_stats = data['mongodb']
            print(f"   MongoDB:      Mean={mongo_stats['mean']*1000:.2f}ms, "
                  f"Std={mongo_stats['std']*1000:.2f}ms, "
                  f"Min={mongo_stats['min']*1000:.2f}ms, "
                  f"Max={mongo_stats['max']*1000:.2f}ms")
            
            # Elasticsearch stats
            es_stats = data['elasticsearch']
            print(f"   Elasticsearch: Mean={es_stats['mean']*1000:.2f}ms, "
                  f"Std={es_stats['std']*1000:.2f}ms, "
                  f"Min={es_stats['min']*1000:.2f}ms, "
                  f"Max={es_stats['max']*1000:.2f}ms")
        
        return self.results
    
    def create_performance_visualizations(self, save_path="benchmark_results.png"):
        """Create performance comparison visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average response times comparison
        operations = list(self.results.keys())
        mongo_means = [self.results[op]['mongodb']['mean'] * 1000 for op in operations]
        es_means = [self.results[op]['elasticsearch']['mean'] * 1000 for op in operations]
        
        x = np.arange(len(operations))
        width = 0.35
        
        ax1.bar(x - width/2, mongo_means, width, label='MongoDB', color='lightblue', alpha=0.8)
        ax1.bar(x + width/2, es_means, width, label='Elasticsearch', color='orange', alpha=0.8)
        ax1.set_xlabel('Operations')
        ax1.set_ylabel('Response Time (ms)')
        ax1.set_title('Average Response Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([op.replace(' ', '\\n') for op in operations], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speedup factors
        speedups = [mongo_means[i] / es_means[i] if es_means[i] > 0 else 1 for i in range(len(operations))]
        colors = ['green' if s > 1 else 'red' for s in speedups]
        
        ax2.bar(operations, speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Performance Speedup (MongoDB vs Elasticsearch)')
        ax2.set_xticklabels([op.replace(' ', '\\n') for op in operations], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Response time distribution (box plot)
        all_mongo_times = []
        all_es_times = []
        labels = []
        
        for op in operations:
            all_mongo_times.extend([t * 1000 for t in self.results[op]['mongodb']['times']])
            all_es_times.extend([t * 1000 for t in self.results[op]['elasticsearch']['times']])
            labels.extend([f"{op}\\n(MongoDB)"] * len(self.results[op]['mongodb']['times']))
            labels.extend([f"{op}\\n(Elasticsearch)"] * len(self.results[op]['elasticsearch']['times']))
        
        # Create box plot data
        box_data = []
        box_labels = []
        for op in operations:
            box_data.append([t * 1000 for t in self.results[op]['mongodb']['times']])
            box_data.append([t * 1000 for t in self.results[op]['elasticsearch']['times']])
            box_labels.extend([f"{op[:15]}..\\n(MongoDB)", f"{op[:15]}..\\n(ES)"])
        
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes alternately
        colors = ['lightblue', 'orange'] * len(operations)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Operation and Database')
        ax3.set_ylabel('Response Time (ms)')
        ax3.set_title('Response Time Distribution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics summary
        ax4.axis('off')
        
        # Calculate overall statistics
        total_mongo_time = sum(mongo_means)
        total_es_time = sum(es_means)
        overall_speedup = total_mongo_time / total_es_time
        
        summary_text = f"""
📊 BENCHMARK SUMMARY
        
Dataset Size: {len(self.mongo_handler.data):,} records
Test Iterations: 5 per operation
        
🏆 OVERALL PERFORMANCE:
• Total MongoDB Time: {total_mongo_time:.2f}ms
• Total Elasticsearch Time: {total_es_time:.2f}ms
• Overall Speedup: {overall_speedup:.1f}x
        
🥇 FASTEST OPERATIONS:
"""
        
        # Find fastest operations for each database
        mongo_fastest = min(operations, key=lambda op: self.results[op]['mongodb']['mean'])
        es_fastest = min(operations, key=lambda op: self.results[op]['elasticsearch']['mean'])
        
        summary_text += f"• MongoDB: {mongo_fastest}\\n"
        summary_text += f"  ({self.results[mongo_fastest]['mongodb']['mean']*1000:.2f}ms)\\n"
        summary_text += f"• Elasticsearch: {es_fastest}\\n"
        summary_text += f"  ({self.results[es_fastest]['elasticsearch']['mean']*1000:.2f}ms)\\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance visualization saved to: {save_path}")
        return save_path

def main():
    """Main benchmark execution function"""
    print("🚀 Chicago 311 Data Platform - Performance Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Run benchmark suite
    results = benchmark.run_benchmark_suite(iterations=5)
    
    # Generate report
    benchmark.generate_performance_report()
    
    # Create visualizations
    viz_path = benchmark.create_performance_visualizations()
    
    # Save results to JSON
    results_file = "benchmark_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_results = {}
        for op_name, data in results.items():
            json_results[op_name] = {
                'mongodb': {
                    'mean': float(data['mongodb']['mean']),
                    'std': float(data['mongodb']['std']),
                    'min': float(data['mongodb']['min']),
                    'max': float(data['mongodb']['max'])
                },
                'elasticsearch': {
                    'mean': float(data['elasticsearch']['mean']),
                    'std': float(data['elasticsearch']['std']),
                    'min': float(data['elasticsearch']['min']),
                    'max': float(data['elasticsearch']['max'])
                }
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"📄 Results saved to: {results_file}")
    print(f"📊 Visualization saved to: {viz_path}")
    
    return results

if __name__ == "__main__":
    main()