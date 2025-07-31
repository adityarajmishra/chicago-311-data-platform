"""Elasticsearch handler for Chicago 311 data."""
import logging
from typing import Dict, Any, List, Optional
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, RequestError, NotFoundError
import json
from datetime import datetime
import pandas as pd

from config.elasticsearch import ElasticsearchConfig

logger = logging.getLogger(__name__)

class ElasticsearchHandler:
    """Handle Elasticsearch operations for Chicago 311 data."""
    
    def __init__(self, hosts: List[str] = None, index_name: str = None):
        """Initialize Elasticsearch handler."""
        config = ElasticsearchConfig.get_elasticsearch_config()
        
        self.hosts = hosts or config['hosts']
        self.index_name = index_name or config['index_name']
        
        try:
            # Initialize Elasticsearch client
            es_config = {
                'hosts': self.hosts,
                'request_timeout': config['timeout'],
                'max_retries': config['max_retries'],
                'retry_on_timeout': config['retry_on_timeout']
            }
            
            # Add authentication if provided
            if config['username'] and config['password']:
                es_config['http_auth'] = (config['username'], config['password'])
            
            self.es = Elasticsearch(**es_config)
            
            # Test connection
            if not self.es.ping():
                raise ConnectionError("Cannot connect to Elasticsearch")
            
            logger.info(f"✅ Connected to Elasticsearch: {self.hosts}")
            
            # Create index if it doesn't exist
            self._create_index()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            raise
    
    def _create_index(self) -> None:
        """Create index with optimized mapping."""
        try:
            if self.es.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return
            
            index_settings = ElasticsearchConfig.get_index_settings()
            
            self.es.indices.create(
                index=self.index_name,
                body=index_settings
            )
            
            logger.info(f"✅ Created Elasticsearch index: {self.index_name}")
            
        except RequestError as e:
            if "already_exists_exception" not in str(e):
                logger.error(f"❌ Error creating index: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ Unexpected error creating index: {e}")
            raise
    
    def bulk_index(self, documents: List[Dict[str, Any]], 
                  chunk_size: int = 1000) -> Dict[str, Any]:
        """Bulk index documents."""
        if not documents:
            return {'indexed': 0, 'errors': 0, 'error_details': []}
        
        actions = []
        error_details = []
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self._preprocess_document(doc)
                if processed_doc:
                    action = {
                        "_index": self.index_name,
                        "_source": processed_doc
                    }
                    
                    # Use sr_number as document ID if available
                    if 'sr_number' in processed_doc:
                        action["_id"] = processed_doc['sr_number']
                    
                    actions.append(action)
                    
            except Exception as e:
                error_details.append({
                    'index': i,
                    'sr_number': doc.get('sr_number', 'Unknown'),
                    'error': str(e)
                })
        
        if not actions:
            return {'indexed': 0, 'errors': len(documents), 'error_details': error_details}
        
        try:
            # Perform bulk indexing
            success, failed = helpers.bulk(
                self.es,
                actions,
                chunk_size=chunk_size,
                request_timeout=60,
                max_retries=3,
                initial_backoff=2,
                max_backoff=600
            )
            
            logger.info(f"✅ Indexed {success} documents to Elasticsearch")
            
            return {
                'indexed': success,
                'errors': len(failed) + len(error_details),
                'error_details': error_details + failed
            }
            
        except Exception as e:
            logger.error(f"❌ Error during bulk indexing: {e}")
            return {
                'indexed': 0,
                'errors': len(documents),
                'error_details': [{'error': str(e)}]
            }
    
    def _preprocess_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess document for Elasticsearch indexing."""
        processed = doc.copy()
        
        # Handle location data for geo_point mapping
        if doc.get('latitude') and doc.get('longitude'):
            try:
                lat = float(doc['latitude'])
                lon = float(doc['longitude'])
                
                # Validate coordinates are in Chicago area
                if 41.6 <= lat <= 42.1 and -87.9 <= lon <= -87.5:
                    processed['location'] = {
                        'lat': lat,
                        'lon': lon
                    }
            except (ValueError, TypeError):
                logger.debug(f"Invalid coordinates: lat={doc.get('latitude')}, lon={doc.get('longitude')}")
        
        # Convert date strings to ISO format
        date_fields = ['creation_date', 'completion_date', 'due_date']
        for field in date_fields:
            if doc.get(field):
                try:
                    dt = pd.to_datetime(doc[field])
                    processed[field] = dt.isoformat()
                except:
                    logger.debug(f"Invalid date format in {field}: {doc[field]}")
                    processed[field] = None
        
        # Convert numeric fields
        numeric_fields = {'ward': int, 'community_area': int}
        for field, dtype in numeric_fields.items():
            if doc.get(field):
                try:
                    processed[field] = dtype(doc[field])
                except (ValueError, TypeError):
                    processed[field] = None
        
        # Convert boolean fields
        boolean_fields = ['duplicate', 'legacy_record']
        for field in boolean_fields:
            if field in doc and doc[field] is not None:
                if isinstance(doc[field], str):
                    processed[field] = doc[field].lower() in ['true', '1', 'yes']
                else:
                    processed[field] = bool(doc[field])
        
        # Clean string fields
        string_fields = ['sr_type', 'owner_department', 'street_address', 'city', 'state']
        for field in string_fields:
            if field in processed and processed[field]:
                processed[field] = str(processed[field]).strip()
        
        # Add indexing metadata
        processed['_indexed_at'] = datetime.utcnow().isoformat()
        
        return processed
    
    def search(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform search with various parameters."""
        try:
            body = self._build_search_query(query_params)
            
            response = self.es.search(
                index=self.index_name,
                body=body,
                timeout='30s'
            )
            
            return {
                'total': response['hits']['total']['value'],
                'max_score': response['hits']['max_score'],
                'hits': response['hits']['hits'],
                'took': response['took']
            }
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return {'total': 0, 'hits': [], 'error': str(e)}
    
    def _build_search_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build Elasticsearch query from parameters."""
        query = {
            "bool": {
                "must": [],
                "filter": [],
                "should": []
            }
        }
        
        # Text search
        if params.get('text'):
            query["bool"]["must"].append({
                "multi_match": {
                    "query": params['text'],
                    "fields": ["sr_type^2", "street_address", "owner_department"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        
        # Exact match filters
        exact_fields = ['status', 'sr_type', 'ward', 'owner_department', 'community_area']
        for field in exact_fields:
            if params.get(field):
                query["bool"]["filter"].append({
                    "term": {f"{field}": params[field]}
                })
        
        # Date range filter
        if params.get('date_from') or params.get('date_to'):
            date_range = {}
            if params.get('date_from'):
                date_range['gte'] = params['date_from']
            if params.get('date_to'):
                date_range['lte'] = params['date_to']
            
            query["bool"]["filter"].append({
                "range": {"creation_date": date_range}
            })
        
        # Geospatial search
        if params.get('lat') and params.get('lon'):
            query["bool"]["filter"].append({
                "geo_distance": {
                    "distance": params.get('radius', '1km'),
                    "location": {
                        "lat": params['lat'],
                        "lon": params['lon']
                    }
                }
            })
        
        # Build final query body
        body = {
            "query": query,
            "sort": [{"creation_date": {"order": "desc"}}],
            "size": params.get('size', 100),
            "from": params.get('from', 0)
        }
        
        # Add aggregations if requested
        if params.get('include_aggs'):
            body["aggs"] = self._get_default_aggregations()
        
        return body
    
    def _get_default_aggregations(self) -> Dict[str, Any]:
        """Get default aggregations for search results."""
        return {
            "status_breakdown": {
                "terms": {"field": "status", "size": 10}
            },
            "sr_type_breakdown": {
                "terms": {"field": "sr_type.keyword", "size": 10}
            },
            "ward_breakdown": {
                "terms": {"field": "ward", "size": 50}
            },
            "monthly_trend": {
                "date_histogram": {
                    "field": "creation_date",
                    "calendar_interval": "month"
                }
            }
        }
    
    def aggregate_by_field(self, field: str, size: int = 20) -> List[Dict[str, Any]]:
        """Aggregate documents by a specific field."""
        try:
            body = {
                "size": 0,
                "aggs": {
                    "field_aggregation": {
                        "terms": {"field": field, "size": size}
                    }
                }
            }
            
            response = self.es.search(index=self.index_name, body=body)
            buckets = response['aggregations']['field_aggregation']['buckets']
            
            return [{'key': bucket['key'], 'count': bucket['doc_count']} 
                   for bucket in buckets]
            
        except Exception as e:
            logger.error(f"❌ Aggregation error for field {field}: {e}")
            return []
    
    def get_date_histogram(self, date_field: str = "creation_date", 
                          interval: str = "month") -> List[Dict[str, Any]]:
        """Get date histogram aggregation."""
        try:
            body = {
                "size": 0,
                "aggs": {
                    "date_histogram": {
                        "date_histogram": {
                            "field": date_field,
                            "calendar_interval": interval
                        }
                    }
                }
            }
            
            response = self.es.search(index=self.index_name, body=body)
            buckets = response['aggregations']['date_histogram']['buckets']
            
            return [{'date': bucket['key_as_string'], 'count': bucket['doc_count']} 
                   for bucket in buckets]
            
        except Exception as e:
            logger.error(f"❌ Date histogram error: {e}")
            return []
    
    def geo_aggregation(self, precision: int = 5) -> Dict[str, Any]:
        """Get geographic aggregation using geohash grid."""
        try:
            body = {
                "size": 0,
                "aggs": {
                    "geo_grid": {
                        "geohash_grid": {
                            "field": "location",
                            "precision": precision
                        }
                    }
                }
            }
            
            response = self.es.search(index=self.index_name, body=body)
            return response['aggregations']['geo_grid']
            
        except Exception as e:
            logger.error(f"❌ Geo aggregation error: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            # Get index stats
            index_stats = self.es.indices.stats(index=self.index_name)
            
            # Get document count
            count_response = self.es.count(index=self.index_name)
            doc_count = count_response['count']
            
            # Get mapping info
            mapping = self.es.indices.get_mapping(index=self.index_name)
            
            stats = {
                'total_documents': doc_count,
                'index_size_bytes': index_stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                'index_size_mb': round(index_stats['indices'][self.index_name]['total']['store']['size_in_bytes'] / (1024 * 1024), 2),
                'number_of_shards': len(index_stats['indices'][self.index_name]['shards']),
                'field_count': len(mapping[self.index_name]['mappings']['properties']),
                'index_name': self.index_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting Elasticsearch stats: {e}")
            return {}
    
    def delete_index(self) -> bool:
        """Delete the index (use with caution!)."""
        try:
            self.es.indices.delete(index=self.index_name)
            logger.info(f"✅ Index {self.index_name} deleted")
            return True
        except NotFoundError:
            logger.warning(f"Index {self.index_name} not found")
            return False
        except Exception as e:
            logger.error(f"❌ Error deleting index: {e}")
            return False
    
    def refresh_index(self) -> bool:
        """Refresh the index to make recent changes searchable."""
        try:
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"✅ Index {self.index_name} refreshed")
            return True
        except Exception as e:
            logger.error(f"❌ Error refreshing index: {e}")
            return False
    
    def close(self) -> None:
        """Close Elasticsearch connection."""
        # Elasticsearch client doesn't need explicit closing
        logger.info("✅ Elasticsearch handler closed")

# Utility functions
def test_elasticsearch_connection() -> bool:
    """Test Elasticsearch connection."""
    try:
        handler = ElasticsearchHandler()
        stats = handler.get_stats()
        logger.info(f"✅ Elasticsearch connection test successful. Documents: {stats.get('total_documents', 0)}")
        return True
    except Exception as e:
        logger.error(f"❌ Elasticsearch connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the Elasticsearch handler
    if test_elasticsearch_connection():
        print("Elasticsearch handler test passed!")
    else:
        print("Elasticsearch handler test failed!")