"""MongoDB handler for Chicago 311 data."""
import logging
from typing import Dict, Any, List, Optional, Generator
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import BulkWriteError, ConnectionFailure, OperationFailure
from pymongo.collection import Collection
from pymongo.database import Database
import gridfs
from datetime import datetime, timedelta
import pandas as pd

from config.database import DatabaseConfig

logger = logging.getLogger(__name__)

class MongoDBHandler:
    """Handle MongoDB operations for Chicago 311 data."""
    
    def __init__(self, connection_string: str = None):
        """Initialize MongoDB handler."""
        if connection_string is None:
            connection_string = DatabaseConfig.get_connection_string()
        
        try:
            self.client = MongoClient(connection_string)
            self.db_name = DatabaseConfig.get_mongodb_config()['database']
            self.db: Database = self.client[self.db_name]
            self.collection: Collection = self.db.service_requests
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {self.db_name}")
            
            # Setup indexes
            self._setup_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _setup_indexes(self) -> None:
        """Create optimized indexes for fast queries."""
        try:
            # Define indexes for optimal performance
            indexes = [
                # Single field indexes
                [("sr_number", ASCENDING)],
                [("creation_date", DESCENDING)],
                [("status", ASCENDING)],
                [("sr_type", ASCENDING)],
                [("ward", ASCENDING)],
                [("community_area", ASCENDING)],
                [("zip_code", ASCENDING)],
                [("owner_department", ASCENDING)],
                
                # Geospatial index
                [("location", "2dsphere")],
                
                # Compound indexes for common queries
                [("status", ASCENDING), ("creation_date", DESCENDING)],
                [("sr_type", ASCENDING), ("creation_date", DESCENDING)],
                [("ward", ASCENDING), ("creation_date", DESCENDING)],
                [("creation_date", DESCENDING), ("status", ASCENDING)],
                [("owner_department", ASCENDING), ("status", ASCENDING)],
                
                # Text index for search
                [("sr_type", TEXT), ("street_address", TEXT)]
            ]
            
            # Create indexes
            for index_spec in indexes:
                try:
                    index_name = self.collection.create_index(index_spec)
                    logger.debug(f"Created index: {index_name}")
                except OperationFailure as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Failed to create index {index_spec}: {e}")
            
            logger.info("✅ MongoDB indexes setup completed")
            
        except Exception as e:
            logger.error(f"❌ Error setting up indexes: {e}")
    
    def insert_batch(self, records: List[Dict[str, Any]], 
                    validate: bool = True) -> Dict[str, Any]:
        """Insert a batch of records."""
        if not records:
            return {'inserted': 0, 'errors': 0, 'error_details': []}
        
        processed_records = []
        error_details = []
        
        for i, record in enumerate(records):
            try:
                processed_record = self._preprocess_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                error_details.append({
                    'index': i,
                    'sr_number': record.get('sr_number', 'Unknown'),
                    'error': str(e)
                })
        
        if not processed_records:
            return {'inserted': 0, 'errors': len(records), 'error_details': error_details}
        
        try:
            # Bulk insert with ordered=False for better performance
            result = self.collection.insert_many(processed_records, ordered=False)
            inserted_count = len(result.inserted_ids)
            
            logger.info(f"✅ Inserted {inserted_count} records into MongoDB")
            
            return {
                'inserted': inserted_count,
                'errors': len(error_details),
                'error_details': error_details
            }
            
        except BulkWriteError as e:
            # Handle partial success in bulk operations
            inserted_count = e.details.get('nInserted', 0)
            write_errors = e.details.get('writeErrors', [])
            
            logger.warning(f"⚠️ Bulk insert partial success: {inserted_count} inserted, {len(write_errors)} errors")
            
            return {
                'inserted': inserted_count,
                'errors': len(write_errors) + len(error_details),
                'error_details': error_details + [
                    {'error': err.get('errmsg', 'Unknown error')} for err in write_errors
                ]
            }
        
        except Exception as e:
            logger.error(f"❌ Error inserting batch: {e}")
            return {'inserted': 0, 'errors': len(records), 'error_details': [{'error': str(e)}]}
    
    def _preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess record for MongoDB storage."""
        processed = record.copy()
        
        # Handle geospatial data
        if record.get('latitude') and record.get('longitude'):
            try:
                lat = float(record['latitude'])
                lon = float(record['longitude'])
                
                # Validate coordinates are in Chicago area
                if 41.6 <= lat <= 42.1 and -87.9 <= lon <= -87.5:
                    processed['location'] = {
                        'type': 'Point',
                        'coordinates': [lon, lat]  # GeoJSON format: [longitude, latitude]
                    }
            except (ValueError, TypeError):
                logger.debug(f"Invalid coordinates: lat={record.get('latitude')}, lon={record.get('longitude')}")
        
        # Convert date strings to datetime objects
        date_fields = ['creation_date', 'completion_date', 'due_date']
        for field in date_fields:
            if record.get(field):
                try:
                    processed[field] = pd.to_datetime(record[field]).to_pydatetime()
                except:
                    logger.debug(f"Invalid date format in {field}: {record[field]}")
                    processed[field] = None
        
        # Convert numeric fields
        numeric_fields = {'ward': int, 'community_area': int}
        for field, dtype in numeric_fields.items():
            if record.get(field):
                try:
                    processed[field] = dtype(record[field])
                except (ValueError, TypeError):
                    processed[field] = None
        
        # Convert boolean fields
        boolean_fields = ['duplicate', 'legacy_record']
        for field in boolean_fields:
            if field in record and record[field] is not None:
                if isinstance(record[field], str):
                    processed[field] = record[field].lower() in ['true', '1', 'yes']
                else:
                    processed[field] = bool(record[field])
        
        # Add processing metadata
        processed['_processed_at'] = datetime.utcnow()
        
        return processed
    
    def find_by_criteria(self, criteria: Dict[str, Any], 
                        limit: int = 1000, 
                        skip: int = 0) -> List[Dict[str, Any]]:
        """Find documents by criteria."""
        try:
            cursor = self.collection.find(criteria).limit(limit).skip(skip)
            return list(cursor)
        except Exception as e:
            logger.error(f"❌ Error finding documents: {e}")
            return []
    
    def search_text(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Perform text search."""
        try:
            cursor = self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"❌ Error in text search: {e}")
            return []
    
    def find_nearby(self, longitude: float, latitude: float, 
                   max_distance: int = 1000, limit: int = 100) -> List[Dict[str, Any]]:
        """Find records near a geographic point."""
        try:
            cursor = self.collection.find({
                "location": {
                    "$near": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [longitude, latitude]
                        },
                        "$maxDistance": max_distance
                    }
                }
            }).limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"❌ Error in geospatial search: {e}")
            return []
    
    def aggregate_by_status(self) -> List[Dict[str, Any]]:
        """Aggregate records by status."""
        try:
            pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"❌ Error in status aggregation: {e}")
            return []
    
    def aggregate_by_ward(self) -> List[Dict[str, Any]]:
        """Aggregate records by ward."""
        try:
            pipeline = [
                {"$match": {"ward": {"$ne": None}}},
                {"$group": {"_id": "$ward", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"❌ Error in ward aggregation: {e}")
            return []
    
    def aggregate_by_sr_type(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Aggregate records by service request type."""
        try:
            pipeline = [
                {"$group": {"_id": "$sr_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": limit}
            ]
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"❌ Error in SR type aggregation: {e}")
            return []
    
    def analyze_response_times(self) -> List[Dict[str, Any]]:
        """Analyze response times by service request type."""
        try:
            pipeline = [
                {
                    "$match": {
                        "creation_date": {"$ne": None},
                        "completion_date": {"$ne": None}
                    }
                },
                {
                    "$project": {
                        "sr_type": 1,
                        "response_time_hours": {
                            "$divide": [
                                {"$subtract": ["$completion_date", "$creation_date"]},
                                1000 * 60 * 60  # Convert to hours
                            ]
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$sr_type",
                        "avg_response_time_hours": {"$avg": "$response_time_hours"},
                        "min_response_time_hours": {"$min": "$response_time_hours"},
                        "max_response_time_hours": {"$max": "$response_time_hours"},
                        "count": {"$sum": 1}
                    }
                },
                {"$match": {"count": {"$gte": 10}}},  # Only types with at least 10 completed requests
                {"$sort": {"avg_response_time_hours": -1}}
            ]
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"❌ Error in response time analysis: {e}")
            return []
    
    def get_temporal_trends(self, date_field: str = "creation_date") -> List[Dict[str, Any]]:
        """Get temporal trends by month."""
        try:
            pipeline = [
                {"$match": {date_field: {"$ne": None}}},
                {
                    "$group": {
                        "_id": {
                            "year": {"$year": f"${date_field}"},
                            "month": {"$month": f"${date_field}"}
                        },
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id.year": 1, "_id.month": 1}}
            ]
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"❌ Error in temporal trends analysis: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            total_count = self.collection.count_documents({})
            
            # Get status distribution
            status_dist = self.aggregate_by_status()
            
            # Get date range
            date_range = list(self.collection.aggregate([
                {"$match": {"creation_date": {"$ne": None}}},
                {
                    "$group": {
                        "_id": None,
                        "min_date": {"$min": "$creation_date"},
                        "max_date": {"$max": "$creation_date"}
                    }
                }
            ]))
            
            stats = {
                'total_records': total_count,
                'status_distribution': status_dist,
                'date_range': date_range[0] if date_range else None,
                'collection_name': self.collection.name,
                'database_name': self.db.name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create a backup of the collection."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            backup_collection = self.db[backup_name]
            
            # Copy all documents
            pipeline = [{"$out": backup_name}]
            list(self.collection.aggregate(pipeline))
            
            logger.info(f"✅ Backup created: {backup_name}")
            return backup_name
            
        except Exception as e:
            logger.error(f"❌ Error creating backup: {e}")
            raise
    
    def drop_collection(self) -> bool:
        """Drop the collection (use with caution!)."""
        try:
            self.collection.drop()
            logger.info("✅ Collection dropped")
            return True
        except Exception as e:
            logger.error(f"❌ Error dropping collection: {e}")
            return False
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("✅ MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Utility functions
def test_mongodb_connection() -> bool:
    """Test MongoDB connection."""
    try:
        with MongoDBHandler() as handler:
            stats = handler.get_stats()
            logger.info(f"✅ MongoDB connection test successful. Records: {stats.get('total_records', 0)}")
            return True
    except Exception as e:
        logger.error(f"❌ MongoDB connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the MongoDB handler
    if test_mongodb_connection():
        print("MongoDB handler test passed!")
    else:
        print("MongoDB handler test failed!")