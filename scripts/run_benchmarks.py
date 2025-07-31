"""Script to run performance benchmarks."""
import argparse
import logging
import sys
from pathlib import Path
import time
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler
from src.benchmarks.performance_tests import PerformanceBenchmark
from src.utils.helpers import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    
    parser.add_argument('--mongodb-only', action='store_true',
                       help='Run benchmarks only on MongoDB')
    parser.add_argument('--elasticsearch-only', action='store_true',
                       help='Run benchmarks only on Elasticsearch')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations per test')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting performance benchmarks...")
    
    # Initialize handlers
    mongo_handler = None
    es_handler = None
    
    if not args.elasticsearch_only:
        try:
            mongo_handler = MongoDBHandler()
            logger.info("✅ MongoDB handler initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB: {e}")
            if args.mongodb_only:
                sys.exit(1)
    
    if not args.mongodb_only:
        try:
            es_handler = ElasticsearchHandler()
            logger.info("✅ Elasticsearch handler initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Elasticsearch: {e}")
            if args.elasticsearch_only:
                sys.exit(1)
    
    # Run benchmarks
    try:
        benchmark = PerformanceBenchmark(mongo_handler, es_handler)
        results = benchmark.run_all_benchmarks(iterations=args.iterations)
        
        # Save results
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'iterations': args.iterations,
            'mongodb_enabled': mongo_handler is not None,
            'elasticsearch_enabled': es_handler is not None
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Benchmark results saved to {args.output}")
        
        # Print summary
        benchmark.print_comparison_report()
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1)
    finally:
        # Close connections
        if mongo_handler:
            mongo_handler.close()
        if es_handler:
            es_handler.close()

if __name__ == "__main__":
    main()