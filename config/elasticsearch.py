"""Elasticsearch configuration settings."""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class ElasticsearchConfig:
    """Elasticsearch configuration class."""
    
    @staticmethod
    def get_elasticsearch_config() -> Dict[str, Any]:
        """Get Elasticsearch configuration."""
        return {
            'hosts': [f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', 9200)}"],
            'index_name': os.getenv('ELASTICSEARCH_INDEX', 'chicago_311'),
            'username': os.getenv('ELASTICSEARCH_USERNAME'),
            'password': os.getenv('ELASTICSEARCH_PASSWORD'),
            'timeout': 30,
            'max_retries': 3,
            'retry_on_timeout': True,
        }
    
    @staticmethod
    def get_index_settings() -> Dict[str, Any]:
        """Get index settings for optimal performance."""
        return {
            "settings": {
                "number_of_shards": int(os.getenv('ES_NUMBER_OF_SHARDS', 3)),
                "number_of_replicas": int(os.getenv('ES_NUMBER_OF_REPLICAS', 1)),
                "refresh_interval": os.getenv('ES_REFRESH_INTERVAL', '30s'),
                "index": {
                    "max_result_window": 50000,
                    "mapping": {
                        "total_fields": {"limit": 2000}
                    }
                },
                "analysis": {
                    "analyzer": {
                        "chicago_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "porter_stem"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "sr_number": {"type": "keyword"},
                    "sr_type": {
                        "type": "text",
                        "analyzer": "chicago_analyzer",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "sr_short_code": {"type": "keyword"},
                    "owner_department": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "creation_date": {"type": "date"},
                    "completion_date": {"type": "date"},
                    "due_date": {"type": "date"},
                    "street_address": {
                        "type": "text",
                        "analyzer": "chicago_analyzer"
                    },
                    "city": {"type": "keyword"},
                    "state": {"type": "keyword"},
                    "zip_code": {"type": "keyword"},
                    "ward": {"type": "integer"},
                    "police_district": {"type": "keyword"},
                    "community_area": {"type": "integer"},
                    "ssa": {"type": "keyword"},
                    "location": {"type": "geo_point"},
                    "latitude": {"type": "float"},
                    "longitude": {"type": "float"},
                    "source": {"type": "keyword"},
                    "duplicate": {"type": "boolean"},
                    "legacy_record": {"type": "boolean"},
                    "legacy_sr_number": {"type": "keyword"},
                    "parent_sr_number": {"type": "keyword"}
                }
            }
        }