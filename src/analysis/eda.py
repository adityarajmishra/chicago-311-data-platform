"""Exploratory Data Analysis module for Chicago 311 data."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class Chicago311EDA:
    """Comprehensive EDA for Chicago 311 data."""
    
    def __init__(self, mongo_handler=None, es_handler=None):
        """Initialize EDA with database handlers."""
        self.mongo_handler = mongo_handler
        self.es_handler = es_handler
        self.analysis_results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def run_basic_statistics(self) -> Dict[str, Any]:
        """Generate basic statistical overview."""
        logger.info("📊 Running basic statistics analysis...")
        
        if not self.mongo_handler:
            logger.error("MongoDB handler required for basic statistics")
            return {}
        
        try:
            # Get overall statistics
            stats = self.mongo_handler.get_stats()
            
            # Get status distribution
            status_dist = self.mongo_handler.aggregate_by_status()
            
            # Get top service request types
            sr_types = self.mongo_handler.aggregate_by_sr_type(limit=20)
            
            # Get ward distribution
            ward_dist = self.mongo_handler.aggregate_by_ward()
            
            # Get temporal overview
            temporal_data = self.mongo_handler.get_temporal_trends()
            
            basic_stats = {
                'overview': stats,
                'status_distribution': status_dist,
                'top_service_types': sr_types,
                'ward_distribution': ward_dist[:20],  # Top 20 wards
                'temporal_overview': temporal_data[-12:] if len(temporal_data) > 12 else temporal_data  # Last 12 months
            }
            
            self.analysis_results['basic_statistics'] = basic_stats
            logger.info("✅ Basic statistics completed")
            
            return basic_stats
            
        except Exception as e:
            logger.error(f"❌ Error in basic statistics: {e}")
            return {}
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in service requests."""
        logger.info("📅 Analyzing temporal patterns...")
        
        if not self.mongo_handler:
            logger.error("MongoDB handler required for temporal analysis")
            return {}
        
        try:
            # Monthly trends
            monthly_trends = self.mongo_handler.get_temporal_trends("created_date")
            
            # Response time analysis
            response_times = self.mongo_handler.analyze_response_times()
            
            # Day of week analysis
            dow_pipeline = [
                {"$match": {"created_date": {"$ne": None}}},
                {
                    "$group": {
                        "_id": {"$dayOfWeek": "$created_date"},
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            day_of_week = list(self.mongo_handler.collection.aggregate(dow_pipeline))
            
            # Hour of day analysis
            hour_pipeline = [
                {"$match": {"created_date": {"$ne": None}}},
                {
                    "$group": {
                        "_id": {"$hour": "$created_date"},
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            hour_of_day = list(self.mongo_handler.collection.aggregate(hour_pipeline))
            
            # Seasonal analysis
            seasonal_pipeline = [
                {"$match": {"created_date": {"$ne": None}}},
                {
                    "$group": {
                        "_id": {"$month": "$created_date"},
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            seasonal_data = list(self.mongo_handler.collection.aggregate(seasonal_pipeline))
            
            temporal_analysis = {
                'monthly_trends': monthly_trends,
                'response_times': response_times,
                'day_of_week_pattern': day_of_week,
                'hour_of_day_pattern': hour_of_day,
                'seasonal_pattern': seasonal_data
            }
            
            self.analysis_results['temporal_patterns'] = temporal_analysis
            logger.info("✅ Temporal analysis completed")
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in temporal analysis: {e}")
            return {}
    
    def analyze_geospatial_patterns(self) -> Dict[str, Any]:
        """Analyze geospatial patterns in service requests."""
        logger.info("🗺️ Analyzing geospatial patterns...")
        
        if not self.mongo_handler:
            logger.error("MongoDB handler required for geospatial analysis")
            return {}
        
        try:
            # Ward analysis
            ward_analysis = self.mongo_handler.aggregate_by_ward()
            
            # Community area analysis
            community_pipeline = [
                {"$match": {"community_area": {"$ne": None}}},
                {"$group": {"_id": "$community_area", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            community_analysis = list(self.mongo_handler.collection.aggregate(community_pipeline))
            
            # Zip code analysis
            zip_pipeline = [
                {"$match": {"zip_code": {"$ne": None}}},
                {"$group": {"_id": "$zip_code", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 20}
            ]
            zip_analysis = list(self.mongo_handler.collection.aggregate(zip_pipeline))
            
            # Service type by location analysis
            location_service_pipeline = [
                {"$match": {"ward": {"$ne": None}, "sr_type": {"$ne": None}}},
                {
                    "$group": {
                        "_id": {"ward": "$ward", "sr_type": "$sr_type"},
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": 100}
            ]
            location_services = list(self.mongo_handler.collection.aggregate(location_service_pipeline))
            
            geospatial_analysis = {
                'ward_distribution': ward_analysis,
                'community_area_distribution': community_analysis[:20],
                'zip_code_distribution': zip_analysis,
                'service_by_location': location_services
            }
            
            self.analysis_results['geospatial_patterns'] = geospatial_analysis
            logger.info("✅ Geospatial analysis completed")
            
            return geospatial_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in geospatial analysis: {e}")
            return {}
    
    def analyze_service_performance(self) -> Dict[str, Any]:
        """Analyze service performance metrics."""
        logger.info("⚡ Analyzing service performance...")
        
        if not self.mongo_handler:
            logger.error("MongoDB handler required for performance analysis")
            return {}
        
        try:
            # Response time analysis by service type
            response_times = self.mongo_handler.analyze_response_times()
            
            # Completion rate analysis
            completion_pipeline = [
                {
                    "$group": {
                        "_id": "$sr_type",
                        "total": {"$sum": 1},
                        "completed": {
                            "$sum": {
                                "$cond": [{"$eq": ["$status", "Completed"]}, 1, 0]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "sr_type": "$_id",
                        "total": 1,
                        "completed": 1,
                        "completion_rate": {
                            "$multiply": [
                                {"$divide": ["$completed", "$total"]}, 100
                            ]
                        }
                    }
                },
                {"$match": {"total": {"$gte": 100}}},  # Only types with at least 100 requests
                {"$sort": {"completion_rate": -1}}
            ]
            completion_rates = list(self.mongo_handler.collection.aggregate(completion_pipeline))
            
            # Department performance
            dept_performance_pipeline = [
                {"$match": {"owner_department": {"$ne": None}}},
                {
                    "$group": {
                        "_id": "$owner_department",
                        "total_requests": {"$sum": 1},
                        "completed_requests": {
                            "$sum": {
                                "$cond": [{"$eq": ["$status", "Completed"]}, 1, 0]
                            }
                        },
                        "avg_response_time": {
                            "$avg": {
                                "$cond": [
                                    {
                                        "$and": [
                                            {"$ne": ["$created_date", None]},
                                            {"$ne": ["$closed_date", None]}
                                        ]
                                    },
                                    {
                                        "$divide": [
                                            {"$subtract": ["$closed_date", "$created_date"]},
                                            1000 * 60 * 60 * 24  # Convert to days
                                        ]
                                    },
                                    None
                                ]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "department": "$_id",
                        "total_requests": 1,
                        "completed_requests": 1,
                        "completion_rate": {
                            "$multiply": [
                                {"$divide": ["$completed_requests", "$total_requests"]}, 100
                            ]
                        },
                        "avg_response_time_days": "$avg_response_time"
                    }
                },
                {"$match": {"total_requests": {"$gte": 50}}},
                {"$sort": {"completion_rate": -1}}
            ]
            dept_performance = list(self.mongo_handler.collection.aggregate(dept_performance_pipeline))
            
            performance_analysis = {
                'response_times_by_type': response_times,
                'completion_rates_by_type': completion_rates,
                'department_performance': dept_performance
            }
            
            self.analysis_results['service_performance'] = performance_analysis
            logger.info("✅ Service performance analysis completed")
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in performance analysis: {e}")
            return {}
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analysis modules."""
        logger.info("🚀 Running comprehensive EDA...")
        
        # Run all analyses
        basic_stats = self.run_basic_statistics()
        temporal_patterns = self.analyze_temporal_patterns()
        geospatial_patterns = self.analyze_geospatial_patterns()
        service_performance = self.analyze_service_performance()
        
        # Compile comprehensive results
        comprehensive_results = {
            'basic_statistics': basic_stats,
            'temporal_patterns': temporal_patterns,
            'geospatial_patterns': geospatial_patterns,
            'service_performance': service_performance,
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_analyses': 4,
                'data_sources': ['MongoDB'] + (['Elasticsearch'] if self.es_handler else [])
            }
        }
        
        logger.info("✅ Comprehensive EDA completed")
        return comprehensive_results
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        if not self.analysis_results:
            return "No analysis results available. Please run analysis first."
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("CHICAGO 311 SERVICE REQUESTS - ANALYSIS SUMMARY")
        report_lines.append("="*80)
        
        # Basic statistics summary
        if 'basic_statistics' in self.analysis_results:
            basic = self.analysis_results['basic_statistics']
            if 'overview' in basic:
                total_records = basic['overview'].get('total_records', 0)
                report_lines.append(f"\nTotal Service Requests: {total_records:,}")
            
            if 'status_distribution' in basic:
                report_lines.append("\nTop Request Statuses:")
                for status in basic['status_distribution'][:5]:
                    report_lines.append(f"  - {status['_id']}: {status['count']:,}")
            
            if 'top_service_types' in basic:
                report_lines.append("\nTop Service Request Types:")
                for sr_type in basic['top_service_types'][:5]:
                    report_lines.append(f"  - {sr_type['_id']}: {sr_type['count']:,}")
        
        # Performance insights
        if 'service_performance' in self.analysis_results:
            perf = self.analysis_results['service_performance']
            if 'response_times_by_type' in perf and perf['response_times_by_type']:
                report_lines.append("\nResponse Time Analysis:")
                fastest = min(perf['response_times_by_type'], 
                            key=lambda x: x['avg_response_time_hours'])
                slowest = max(perf['response_times_by_type'], 
                            key=lambda x: x['avg_response_time_hours'])
                
                report_lines.append(f"  - Fastest: {fastest['_id']} "
                                  f"({fastest['avg_response_time_hours']:.1f} hours)")
                report_lines.append(f"  - Slowest: {slowest['_id']} "
                                  f"({slowest['avg_response_time_hours']:.1f} hours)")
        
        report_lines.append("="*80)
        
        return "\n".join(report_lines)

def run_sample_eda():
    """Run sample EDA for testing."""
    from src.databases.mongodb_handler import MongoDBHandler
    
    try:
        mongo_handler = MongoDBHandler()
        eda = Chicago311EDA(mongo_handler=mongo_handler)
        
        # Run basic analysis
        results = eda.run_basic_statistics()
        
        # Generate summary
        summary = eda.generate_summary_report()
        print(summary)
        
        mongo_handler.close()
        return results
        
    except Exception as e:
        logger.error(f"❌ Sample EDA failed: {e}")
        return None

if __name__ == "__main__":
    run_sample_eda()