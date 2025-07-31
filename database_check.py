#!/usr/bin/env python3
"""
Database Connection and Record Count Check
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_databases():
    """Check both databases for record counts and connection status."""
    print("🔍 Chicago 311 Database Connection Check")
    print("=" * 60)
    
    mongo_count = 0
    es_count = 0
    
    # Check MongoDB
    print("\n📊 MongoDB Connection Check:")
    try:
        mongo_handler = MongoDBHandler()
        mongo_count = mongo_handler.collection.count_documents({})
        print(f"   ✅ MongoDB Connected")
        print(f"   📈 Total Records: {mongo_count:,}")
        
        # Get sample record
        sample = mongo_handler.collection.find_one()
        if sample:
            print(f"   🔍 Sample record keys: {list(sample.keys())[:10]}...")
        
        # Check indexes
        indexes = mongo_handler.collection.list_indexes()
        index_names = [idx['name'] for idx in indexes]
        print(f"   🎯 Indexes: {len(index_names)} ({', '.join(index_names[:5])}...)")
        
        mongo_handler.close()
        
    except Exception as e:
        print(f"   ❌ MongoDB Error: {e}")
    
    # Check Elasticsearch
    print("\n🔍 Elasticsearch Connection Check:")
    try:
        es_handler = ElasticsearchHandler()
        
        # Get document count
        result = es_handler.es.count(index=es_handler.index_name)
        es_count = result['count']
        print(f"   ✅ Elasticsearch Connected")
        print(f"   📈 Total Documents: {es_count:,}")
        
        # Get sample document
        search_result = es_handler.es.search(index=es_handler.index_name, size=1)
        if search_result['hits']['hits']:
            sample_doc = search_result['hits']['hits'][0]['_source']
            print(f"   🔍 Sample document keys: {list(sample_doc.keys())[:10]}...")
        
        # Check index settings
        index_settings = es_handler.es.indices.get_settings(index=es_handler.index_name)
        shards = index_settings[es_handler.index_name]['settings']['index']['number_of_shards']
        replicas = index_settings[es_handler.index_name]['settings']['index']['number_of_replicas']
        print(f"   ⚙️ Index config - Shards: {shards}, Replicas: {replicas}")
        
        es_handler.close()
        
    except Exception as e:
        print(f"   ❌ Elasticsearch Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DATABASE SUMMARY")
    print("=" * 60)
    print(f"MongoDB Records:      {mongo_count:,}")
    print(f"Elasticsearch Docs:   {es_count:,}")
    
    if mongo_count > 0 and es_count > 0:
        sync_percentage = min(mongo_count, es_count) / max(mongo_count, es_count) * 100
        print(f"Data Sync Rate:       {sync_percentage:.1f}%")
        
        if abs(mongo_count - es_count) > 1000:
            print("⚠️  Warning: Significant difference in record counts detected!")
        else:
            print("✅ Databases appear to be in sync")
    
    expected_count = 12_300_000  # 12.3 million
    if mongo_count < expected_count * 0.9:
        print(f"⚠️  Warning: MongoDB count ({mongo_count:,}) is significantly below expected ({expected_count:,})")
    
    if es_count < expected_count * 0.9:
        print(f"⚠️  Warning: Elasticsearch count ({es_count:,}) is significantly below expected ({expected_count:,})")
    
    return mongo_count, es_count

if __name__ == "__main__":
    check_databases()