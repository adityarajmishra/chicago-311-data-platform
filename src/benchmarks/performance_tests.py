"""Performance testing and benchmarking module."""
import logging
import time
import statistics
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import random

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmark suite for MongoDB and Elasticsearch."""
    
    def __init__(self, mongo_handler=None, es_handler=None):
        """Initialize benchmark suite."""
        self.mongo_handler = mongo_handler
        self.es_handler = es_handler
        self.results = {}
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        
        if operation_name not in self.results:
            self.results[operation_name] = []
        self.results[operation_name].append(execution_time)
    
    def run_search_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run search operation benchmarks."""
        logger.info("🔍 Running search benchmarks...")
        
        # Test queries
        test_queries = [
            {"status": "Completed"},
            {"status": "Open"},
            {"sr_type": "Pothole in Street"},
            {"ward": 1},
            {"created_date": {"$gte": "2023-01-01"}},
        ]
        
        search_results = {}
        
        for i in range(iterations):
            logger.info(f"Search iteration {i+1}/{iterations}")
            
            for j, query in enumerate(test_queries):
                query_name = f"search_query_{j+1}"
                
                # MongoDB search
                if self.mongo_handler:
                    with self.timer(f"mongodb_{query_name}"):
                        results = self.mongo_handler.find_by_criteria(query, limit=1000)
                    search_results[f"mongodb_{query_name}_count"] = len(results)
                
                # Elasticsearch search
                if self.es_handler:
                    # Convert MongoDB query to ES format
                    es_query = self._convert_query_to_es(query)
                    with self.timer(f"elasticsearch_{query_name}"):
                        results = self.es_handler.search(es_query)
                    search_results[f"elasticsearch_{query_name}_count"] = results.get('total', 0)
        
        return search_results
    
    def run_text_search_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run text search benchmarks."""
        logger.info("📝 Running text search benchmarks...")
        
        text_queries = [
            "pothole",
            "street light",
            "garbage collection",
            "water leak",
            "tree removal"
        ]
        
        text_results = {}
        
        for i in range(iterations):
            logger.info(f"Text search iteration {i+1}/{iterations}")
            
            for j, query_text in enumerate(text_queries):
                query_name = f"text_search_{j+1}"
                
                # MongoDB text search
                if self.mongo_handler:
                    with self.timer(f"mongodb_{query_name}"):
                        results = self.mongo_handler.search_text(query_text, limit=1000)
                    text_results[f"mongodb_{query_name}_count"] = len(results)
                
                # Elasticsearch text search
                if self.es_handler:
                    es_query = {"text": query_text, "size": 1000}
                    with self.timer(f"elasticsearch_{query_name}"):
                        results = self.es_handler.search(es_query)
                    text_results[f"elasticsearch_{query_name}_count"] = results.get('total', 0)
        
        return text_results
    
    def run_geospatial_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run geospatial search benchmarks."""
        logger.info("🗺️ Running geospatial benchmarks...")
        
        # Chicago locations for testing
        test_locations = [
            (-87.6298, 41.8781),  # Downtown Chicago
            (-87.6244, 41.8756),  # Millennium Park
            (-87.6198, 41.8819),  # Navy Pier
            (-87.6073, 41.7943),  # University of Chicago
            (-87.6431, 41.9022),  # Lincoln Park
        ]
        
        geo_results = {}
        
        for i in range(iterations):
            logger.info(f"Geospatial iteration {i+1}/{iterations}")
            
            for j, (lon, lat) in enumerate(test_locations):
                query_name = f"geo_search_{j+1}"
                
                # MongoDB geospatial search
                if self.mongo_handler:
                    with self.timer(f"mongodb_{query_name}"):
                        results = self.mongo_handler.find_nearby(lon, lat, max_distance=1000, limit=1000)
                    geo_results[f"mongodb_{query_name}_count"] = len(results)
                
                # Elasticsearch geospatial search
                if self.es_handler:
                    es_query = {
                        "lat": lat,
                        "lon": lon,
                        "radius": "1km",
                        "size": 1000
                    }
                    with self.timer(f"elasticsearch_{query_name}"):
                        results = self.es_handler.search(es_query)
                    geo_results[f"elasticsearch_{query_name}_count"] = results.get('total', 0)
        
        return geo_results
    
    def run_aggregation_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run aggregation benchmarks."""
        logger.info("📊 Running aggregation benchmarks...")
        
        agg_results = {}
        
        for i in range(iterations):
            logger.info(f"Aggregation iteration {i+1}/{iterations}")
            
            # Status aggregation
            if self.mongo_handler:
                with self.timer("mongodb_status_aggregation"):
                    results = self.mongo_handler.aggregate_by_status()
                agg_results[f"mongodb_status_agg_count"] = len(results)
            
            if self.es_handler:
                with self.timer("elasticsearch_status_aggregation"):
                    results = self.es_handler.aggregate_by_field("status")
                agg_results[f"elasticsearch_status_agg_count"] = len(results)
            
            # Ward aggregation
            if self.mongo_handler:
                with self.timer("mongodb_ward_aggregation"):
                    results = self.mongo_handler.aggregate_by_ward()
                agg_results[f"mongodb_ward_agg_count"] = len(results)
            
            if self.es_handler:
                with self.timer("elasticsearch_ward_aggregation"):
                    results = self.es_handler.aggregate_by_field("ward")
                agg_results[f"elasticsearch_ward_agg_count"] = len(results)
            
            # SR Type aggregation
            if self.mongo_handler:
                with self.timer("mongodb_srtype_aggregation"):
                    results = self.mongo_handler.aggregate_by_sr_type()
                agg_results[f"mongodb_srtype_agg_count"] = len(results)
            
            if self.es_handler:
                with self.timer("elasticsearch_srtype_aggregation"):
                    results = self.es_handler.aggregate_by_field("sr_type.keyword")
                agg_results[f"elasticsearch_srtype_agg_count"] = len(results)
        
        return agg_results
    
    def run_complex_query_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run complex multi-criteria query benchmarks."""
        logger.info("🔍 Running complex query benchmarks...")
        
        complex_results = {}
        
        for i in range(iterations):
            logger.info(f"Complex query iteration {i+1}/{iterations}")
            
            # Complex MongoDB query
            if self.mongo_handler:
                with self.timer("mongodb_complex_query"):
                    complex_query = {
                        "status": "Open",
                        "ward": {"$in": [1, 2, 3, 4, 5]},
                        "created_date": {"$gte": "2023-01-01"}
                    }
                    results = self.mongo_handler.find_by_criteria(complex_query, limit=1000)
                complex_results[f"mongodb_complex_count"] = len(results)
            
            # Complex Elasticsearch query
            if self.es_handler:
                with self.timer("elasticsearch_complex_query"):
                    es_query = {
                        "status": "Open",
                        "ward": [1, 2, 3, 4, 5],  # Will be converted to terms query
                        "date_from": "2023-01-01",
                        "size": 1000
                    }
                    results = self.es_handler.search(es_query)
                complex_results[f"elasticsearch_complex_count"] = results.get('total', 0)
        
        return complex_results
    
    def run_all_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run all benchmark suites."""
        logger.info(f"🚀 Running complete benchmark suite with {iterations} iterations...")
        
        all_results = {
            'search_results': self.run_search_benchmarks(iterations),
            'text_search_results': self.run_text_search_benchmarks(iterations),
            'geospatial_results': self.run_geospatial_benchmarks(iterations),
            'aggregation_results': self.run_aggregation_benchmarks(iterations),
            'complex_query_results': self.run_complex_query_benchmarks(iterations),
            'timing_results': self._calculate_statistics(),
        }
        
        return all_results
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from timing results."""
        stats = {}
        
        for operation, times in self.results.items():
            if times:
                stats[operation] = {
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'min': min(times),
                    'max': max(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                    'iterations': len(times)
                }
        
        return stats
    
    def print_comparison_report(self) -> None:
        """Print detailed comparison report."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        stats = self._calculate_statistics()
        
        # Group results by operation type
        mongodb_results = {k: v for k, v in stats.items() if 'mongodb' in k}
        es_results = {k: v for k, v in stats.items() if 'elasticsearch' in k}
        
        # Print summary statistics
        if mongodb_results:
            mongo_avg = statistics.mean([v['mean'] for v in mongodb_results.values()])
            print(f"MongoDB Average Response Time: {mongo_avg*1000:.2f}ms")
        
        if es_results:
            es_avg = statistics.mean([v['mean'] for v in es_results.values()])
            print(f"Elasticsearch Average Response Time: {es_avg*1000:.2f}ms")
        
        if mongodb_results and es_results:
            speedup = mongo_avg / es_avg
            print(f"Elasticsearch Speedup: {speedup:.2f}x faster")
        
        print("\n" + "-"*80)
        print("DETAILED RESULTS (milliseconds)")
        print("-"*80)
        
        # Print detailed results
        operation_types = [
            ('Search Operations', 'search'),
            ('Text Search Operations', 'text_search'),
            ('Geospatial Operations', 'geo_search'),
            ('Aggregation Operations', 'aggregation'),
            ('Complex Query Operations', 'complex')
        ]
        
        for operation_type, operation_prefix in operation_types:
            print(f"\n{operation_type}:")
            print("-" * len(operation_type))
            
            relevant_ops = {k: v for k, v in stats.items() if operation_prefix in k}
            
            for operation, data in relevant_ops.items():
                db_type = "MongoDB" if "mongodb" in operation else "Elasticsearch"
                op_name = operation.replace("mongodb_", "").replace("elasticsearch_", "")
                
                print(f"  {db_type:15} {op_name:25} "
                      f"Mean: {data['mean']*1000:6.1f}ms "
                      f"Median: {data['median']*1000:6.1f}ms "
                      f"Min: {data['min']*1000:6.1f}ms "
                      f"Max: {data['max']*1000:6.1f}ms")
        
        print("="*80)
    
    def _convert_query_to_es(self, mongo_query: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MongoDB query to Elasticsearch format."""
        es_query = {}
        
        for key, value in mongo_query.items():
            if key == "created_date" and isinstance(value, dict):
                if "$gte" in value:
                    es_query["date_from"] = value["$gte"]
                if "$lte" in value:
                    es_query["date_to"] = value["$lte"]
            else:
                es_query[key] = value
        
        return es_query

def run_quick_benchmark():
    """Run a quick benchmark for testing."""
    from src.databases.mongodb_handler import MongoDBHandler
    from src.databases.elasticsearch_handler import ElasticsearchHandler
    
    try:
        mongo_handler = MongoDBHandler()
        es_handler = ElasticsearchHandler()
        
        benchmark = PerformanceBenchmark(mongo_handler, es_handler)
        results = benchmark.run_all_benchmarks(iterations=3)
        
        benchmark.print_comparison_report()
        
        mongo_handler.close()
        es_handler.close()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Quick benchmark failed: {e}")
        return None

if __name__ == "__main__":
    run_quick_benchmark()