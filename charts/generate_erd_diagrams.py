#!/usr/bin/env python3
"""
ERD Diagram Generator for Chicago 311 Data Platform
Generates Entity Relationship Diagrams for MongoDB and Elasticsearch
Based on 12M Chicago 311 Service Request records
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os
from datetime import datetime

class ChicagoERDGenerator:
    """Generate ERD diagrams for Chicago 311 data platform."""
    
    def __init__(self):
        """Initialize ERD generator."""
        # Ensure directory exists
        os.makedirs('erdDiagram', exist_ok=True)
        
        # Chicago 311 data schema based on the actual API structure
        self.chicago_311_fields = {
            # Primary identifiers
            'sr_number': {'type': 'String', 'primary': True, 'description': 'Unique service request number'},
            
            # Request details
            'sr_type': {'type': 'String', 'indexed': True, 'description': 'Type of service request'},
            'sr_short_code': {'type': 'String', 'description': 'Short code for request type'},
            'owner_department': {'type': 'String', 'indexed': True, 'description': 'Department responsible'},
            'status': {'type': 'String', 'indexed': True, 'description': 'Current status'},
            
            # Temporal data
            'creation_date': {'type': 'DateTime', 'indexed': True, 'description': 'When request was created'},
            'completion_date': {'type': 'DateTime', 'description': 'When request was completed'},
            'due_date': {'type': 'DateTime', 'description': 'When request is due'},
            
            # Location data
            'street_address': {'type': 'String', 'description': 'Street address'},
            'city': {'type': 'String', 'description': 'City'},
            'state': {'type': 'String', 'description': 'State'},
            'zip_code': {'type': 'String', 'indexed': True, 'description': 'ZIP code'},
            'street_number': {'type': 'String', 'description': 'Street number'},
            'street_direction': {'type': 'String', 'description': 'Street direction'},
            'street_name': {'type': 'String', 'description': 'Street name'},
            'street_type': {'type': 'String', 'description': 'Street type'},
            
            # Geographic coordinates
            'latitude': {'type': 'Float', 'geo': True, 'description': 'Latitude coordinate'},
            'longitude': {'type': 'Float', 'geo': True, 'description': 'Longitude coordinate'},
            'location': {'type': 'GeoPoint', 'indexed': True, 'description': 'Geographic location'},
            
            # Administrative areas
            'ward': {'type': 'Integer', 'indexed': True, 'description': 'Ward number'},
            'community_area': {'type': 'Integer', 'indexed': True, 'description': 'Community area number'},
            'ssa': {'type': 'String', 'description': 'Special Service Area'},
            'census_tract': {'type': 'String', 'description': 'Census tract'},
            'historical_wards_03_15': {'type': 'String', 'description': 'Historical ward data'},
            
            # Police districts
            'police_district': {'type': 'String', 'description': 'Police district'},
            
            # Flags and metadata
            'duplicate': {'type': 'Boolean', 'description': 'Is duplicate request'},
            'legacy_record': {'type': 'Boolean', 'description': 'Legacy system record'},
            'legacy_sr_number': {'type': 'String', 'description': 'Legacy system SR number'},
            
            # Processing metadata
            '_processed_at': {'type': 'DateTime', 'description': 'When record was processed'},
            '_indexed_at': {'type': 'DateTime', 'description': 'When record was indexed'}
        }
        
    def generate_mongodb_erd(self):
        """Generate MongoDB ERD diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title
        ax.text(10, 15.5, 'MongoDB Schema - Chicago 311 Service Requests', 
                fontsize=20, fontweight='bold', ha='center')
        ax.text(10, 15, 'Document-Based Schema for 12M+ Records', 
                fontsize=14, ha='center', style='italic')
        
        # Main collection box
        collection_box = FancyBboxPatch(
            (1, 2), 18, 12,
            boxstyle="round,pad=0.1",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax.add_patch(collection_box)
        
        # Collection header
        ax.text(10, 13.5, 'Collection: service_requests', 
                fontsize=16, fontweight='bold', ha='center')
        ax.text(10, 13, 'MongoDB Document Structure', 
                fontsize=12, ha='center')
        
        # Field categories
        categories = {
            'Primary Key': {'fields': ['sr_number'], 'color': '#FFEBEE', 'border': '#C62828'},
            'Request Info': {'fields': ['sr_type', 'sr_short_code', 'owner_department', 'status'], 'color': '#F3E5F5', 'border': '#7B1FA2'},
            'Temporal': {'fields': ['creation_date', 'completion_date', 'due_date'], 'color': '#E8F5E8', 'border': '#388E3C'},
            'Location': {'fields': ['street_address', 'city', 'state', 'zip_code', 'street_number', 'street_direction', 'street_name', 'street_type'], 'color': '#FFF3E0', 'border': '#F57C00'},
            'Geographic': {'fields': ['latitude', 'longitude', 'location'], 'color': '#E0F2F1', 'border': '#00695C'},
            'Administrative': {'fields': ['ward', 'community_area', 'ssa', 'census_tract', 'police_district'], 'color': '#FCE4EC', 'border': '#C2185B'},
            'Metadata': {'fields': ['duplicate', 'legacy_record', '_processed_at', '_indexed_at'], 'color': '#F1F8E9', 'border': '#689F38'}
        }
        
        y_start = 12.5
        y_offset = 0
        
        for category, info in categories.items():
            # Category header
            cat_box = FancyBboxPatch(
                (2, y_start - y_offset), 16, 0.4,
                boxstyle="round,pad=0.05",
                facecolor=info['border'],
                edgecolor=info['border'],
                alpha=0.8
            )
            ax.add_patch(cat_box)
            ax.text(10, y_start - y_offset + 0.2, category, 
                    fontsize=12, fontweight='bold', ha='center', color='white')
            
            y_offset += 0.6
            
            # Fields in this category
            for field in info['fields']:
                field_info = self.chicago_311_fields[field]
                
                # Field box
                field_box = FancyBboxPatch(
                    (2.5, y_start - y_offset), 15, 0.3,
                    boxstyle="round,pad=0.02",
                    facecolor=info['color'],
                    edgecolor=info['border'],
                    alpha=0.7
                )
                ax.add_patch(field_box)
                
                # Field details
                field_text = f"{field}: {field_info['type']}"
                if field_info.get('primary'):
                    field_text += " [PRIMARY]"
                if field_info.get('indexed'):
                    field_text += " [INDEXED]"
                if field_info.get('geo'):
                    field_text += " [GEO]"
                
                ax.text(3, y_start - y_offset + 0.15, field_text, 
                        fontsize=9, fontweight='bold' if field_info.get('primary') else 'normal')
                
                # Description
                ax.text(14, y_start - y_offset + 0.15, field_info['description'], 
                        fontsize=8, style='italic', ha='right')
                
                y_offset += 0.35
            
            y_offset += 0.2
        
        # Indexes section
        index_box = FancyBboxPatch(
            (1, 0.2), 8.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#FFF8E1',
            edgecolor='#F57F17',
            linewidth=2
        )
        ax.add_patch(index_box)
        
        ax.text(5.25, 1.5, 'MongoDB Indexes', fontsize=12, fontweight='bold', ha='center')
        indexes = [
            '• sr_number (unique)',
            '• creation_date (desc)',
            '• status + creation_date',
            '• location (2dsphere)',
            '• sr_type + creation_date',
            '• ward + creation_date'
        ]
        for i, idx in enumerate(indexes):
            ax.text(1.2, 1.3 - i*0.15, idx, fontsize=9)
        
        # GeoJSON example
        geo_box = FancyBboxPatch(
            (10.5, 0.2), 8.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(geo_box)
        
        ax.text(14.75, 1.5, 'GeoJSON Location Format', fontsize=12, fontweight='bold', ha='center')
        geojson_example = '''{
  "location": {
    "type": "Point",
    "coordinates": [-87.623177, 41.881832]
  }
}'''
        ax.text(10.7, 1.3, geojson_example, fontsize=8, fontfamily='monospace', va='top')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/01_mongodb_schema_chicago_311_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 01_mongodb_schema_chicago_311_12m_records.png")
    
    def generate_elasticsearch_erd(self):
        """Generate Elasticsearch ERD diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title
        ax.text(10, 15.5, 'Elasticsearch Schema - Chicago 311 Service Requests', 
                fontsize=20, fontweight='bold', ha='center')
        ax.text(10, 15, 'Search-Optimized Schema for 12M+ Records', 
                fontsize=14, ha='center', style='italic')
        
        # Main index box
        index_box = FancyBboxPatch(
            (1, 2), 18, 12,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax.add_patch(index_box)
        
        # Index header
        ax.text(10, 13.5, 'Index: chicago_311_requests', 
                fontsize=16, fontweight='bold', ha='center')
        ax.text(10, 13, 'Elasticsearch Mapping Structure', 
                fontsize=12, ha='center')
        
        # Field categories with Elasticsearch-specific types
        es_categories = {
            'Identifiers': {
                'fields': [
                    ('sr_number', 'keyword', 'Unique service request identifier'),
                    ('legacy_sr_number', 'keyword', 'Legacy system identifier')
                ],
                'color': '#FFEBEE', 'border': '#C62828'
            },
            'Searchable Text': {
                'fields': [
                    ('sr_type', 'text + keyword', 'Service request type (analyzed + exact)'),
                    ('owner_department', 'text + keyword', 'Department name (analyzed + exact)'),
                    ('street_address', 'text', 'Full street address (analyzed)')
                ],
                'color': '#E8F5E8', 'border': '#388E3C'
            },
            'Categorical': {
                'fields': [
                    ('status', 'keyword', 'Request status (exact match)'),
                    ('city', 'keyword', 'City name'),
                    ('state', 'keyword', 'State code'),
                    ('street_direction', 'keyword', 'Street direction')
                ],
                'color': '#FFF3E0', 'border': '#F57C00'
            },
            'Temporal': {
                'fields': [
                    ('creation_date', 'date', 'Request creation timestamp'),
                    ('completion_date', 'date', 'Request completion timestamp'),
                    ('due_date', 'date', 'Request due date')
                ],
                'color': '#E3F2FD', 'border': '#1976D2'
            },
            'Geographic': {
                'fields': [
                    ('location', 'geo_point', 'Geographic coordinates'),
                    ('latitude', 'float', 'Latitude coordinate'),
                    ('longitude', 'float', 'Longitude coordinate')
                ],
                'color': '#E0F2F1', 'border': '#00695C'
            },
            'Numeric': {
                'fields': [
                    ('ward', 'integer', 'Ward number'),
                    ('community_area', 'integer', 'Community area number'),
                    ('zip_code', 'keyword', 'ZIP code (as string)')
                ],
                'color': '#FCE4EC', 'border': '#C2185B'
            },
            'Boolean Flags': {
                'fields': [
                    ('duplicate', 'boolean', 'Duplicate request flag'),
                    ('legacy_record', 'boolean', 'Legacy system record flag')
                ],
                'color': '#F1F8E9', 'border': '#689F38'
            },
            'Metadata': {
                'fields': [
                    ('_indexed_at', 'date', 'Indexing timestamp'),
                    ('census_tract', 'keyword', 'Census tract identifier'),
                    ('police_district', 'keyword', 'Police district')
                ],
                'color': '#FFF8E1', 'border': '#F57F17'
            }
        }
        
        y_start = 12.5
        y_offset = 0
        
        for category, info in es_categories.items():
            # Category header
            cat_box = FancyBboxPatch(
                (2, y_start - y_offset), 16, 0.4,
                boxstyle="round,pad=0.05",
                facecolor=info['border'],
                edgecolor=info['border'],
                alpha=0.8
            )
            ax.add_patch(cat_box)
            ax.text(10, y_start - y_offset + 0.2, category, 
                    fontsize=12, fontweight='bold', ha='center', color='white')
            
            y_offset += 0.6
            
            # Fields in this category
            for field_name, field_type, description in info['fields']:
                # Field box
                field_box = FancyBboxPatch(
                    (2.5, y_start - y_offset), 15, 0.3,
                    boxstyle="round,pad=0.02",
                    facecolor=info['color'],
                    edgecolor=info['border'],
                    alpha=0.7
                )
                ax.add_patch(field_box)
                
                # Field details
                field_text = f"{field_name}: {field_type}"
                ax.text(3, y_start - y_offset + 0.15, field_text, fontsize=9, fontweight='bold')
                
                # Description
                ax.text(14, y_start - y_offset + 0.15, description, 
                        fontsize=8, style='italic', ha='right')
                
                y_offset += 0.35
            
            y_offset += 0.2
        
        # Search capabilities section
        search_box = FancyBboxPatch(
            (1, 0.2), 6, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(search_box)
        
        ax.text(4, 1.5, 'Search Capabilities', fontsize=12, fontweight='bold', ha='center')
        search_features = [
            '• Full-text search on sr_type, address',
            '• Geo-distance queries on location',
            '• Date range filters',
            '• Term aggregations',
            '• Fuzzy matching',
            '• Auto-complete suggestions'
        ]
        for i, feature in enumerate(search_features):
            ax.text(1.2, 1.3 - i*0.15, feature, fontsize=9)
        
        # Index settings
        settings_box = FancyBboxPatch(
            (7.5, 0.2), 6, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#FFF3E0',
            edgecolor='#F57C00',
            linewidth=2
        )
        ax.add_patch(settings_box)
        
        ax.text(10.5, 1.5, 'Index Settings', fontsize=12, fontweight='bold', ha='center')
        settings = [
            '• Shards: 5',
            '• Replicas: 1',
            '• Refresh interval: 1s',
            '• Max result window: 10000',
            '• Analysis: standard analyzer',
            '• Doc values: enabled'
        ]
        for i, setting in enumerate(settings):
            ax.text(7.7, 1.3 - i*0.15, setting, fontsize=9)
        
        # Geo-point example
        geo_example_box = FancyBboxPatch(
            (14, 0.2), 5.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#E0F2F1',
            edgecolor='#00695C',
            linewidth=2
        )
        ax.add_patch(geo_example_box)
        
        ax.text(16.75, 1.5, 'Geo-Point Format', fontsize=12, fontweight='bold', ha='center')
        geo_example = '''{
  "location": {
    "lat": 41.881832,
    "lon": -87.623177
  }
}'''
        ax.text(14.2, 1.3, geo_example, fontsize=8, fontfamily='monospace', va='top')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/02_elasticsearch_schema_chicago_311_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 02_elasticsearch_schema_chicago_311_12m_records.png")
    
    def generate_comparison_erd(self):
        """Generate comparison ERD showing differences between MongoDB and Elasticsearch."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
        
        # MongoDB side (left)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 14)
        ax1.axis('off')
        
        ax1.text(5, 13.5, 'MongoDB Document Model', fontsize=16, fontweight='bold', ha='center')
        ax1.text(5, 13, 'Document-based storage optimized for operations', fontsize=12, ha='center', style='italic')
        
        # MongoDB characteristics
        mongo_box = FancyBboxPatch(
            (0.5, 1), 9, 11.5,
            boxstyle="round,pad=0.1",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax1.add_patch(mongo_box)
        
        mongo_features = [
            ('Schema Flexibility', '• Dynamic schema\n• Nested documents\n• Arrays supported', '#E8F5E8'),
            ('Data Types', '• ObjectId (primary key)\n• GeoJSON locations\n• DateTime objects\n• Embedded documents', '#FFF3E0'),
            ('Indexing', '• B-tree indexes\n• 2dsphere geo indexes\n• Compound indexes\n• Text indexes', '#FCE4EC'),
            ('Query Capabilities', '• Rich query language\n• Aggregation pipelines\n• Map-reduce\n• Geospatial queries', '#F1F8E9'),
            ('Optimized For', '• CRUD operations\n• Complex aggregations\n• Geospatial queries\n• Document relationships', '#FFEBEE')
        ]
        
        y_pos = 11.5
        for title, features, color in mongo_features:
            # Feature box
            feature_box = FancyBboxPatch(
                (1, y_pos), 8, 1.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='#757575',
                alpha=0.7
            )
            ax1.add_patch(feature_box)
            
            ax1.text(1.2, y_pos + 1.5, title, fontsize=11, fontweight='bold')
            ax1.text(1.2, y_pos + 0.2, features, fontsize=9, va='bottom')
            
            y_pos -= 2.2
        
        # Elasticsearch side (right)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 14)
        ax2.axis('off')
        
        ax2.text(5, 13.5, 'Elasticsearch Search Model', fontsize=16, fontweight='bold', ha='center')
        ax2.text(5, 13, 'Search-optimized with inverted indexes', fontsize=12, ha='center', style='italic')
        
        # Elasticsearch characteristics
        es_box = FancyBboxPatch(
            (0.5, 1), 9, 11.5,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax2.add_patch(es_box)
        
        es_features = [
            ('Schema Flexibility', '• Explicit mapping\n• Dynamic mapping\n• Field data types', '#E8F5E8'),
            ('Data Types', '• Keyword fields\n• Text fields (analyzed)\n• Geo-point coordinates\n• Date fields', '#FFF3E0'),
            ('Indexing', '• Inverted indexes\n• Doc values\n• Geo-spatial indexes\n• Completion suggesters', '#FCE4EC'),
            ('Query Capabilities', '• Full-text search\n• Fuzzy matching\n• Range queries\n• Geo-distance search', '#F1F8E9'),
            ('Optimized For', '• Text search\n• Real-time search\n• Analytics\n• Log analysis', '#FFEBEE')
        ]
        
        y_pos = 11.5
        for title, features, color in es_features:
            # Feature box
            feature_box = FancyBboxPatch(
                (1, y_pos), 8, 1.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='#757575',
                alpha=0.7
            )
            ax2.add_patch(feature_box)
            
            ax2.text(1.2, y_pos + 1.5, title, fontsize=11, fontweight='bold')
            ax2.text(1.2, y_pos + 0.2, features, fontsize=9, va='bottom')
            
            y_pos -= 2.2
        
        # Main title
        fig.suptitle('MongoDB vs Elasticsearch Schema Comparison\nChicago 311 Service Requests - 12M Records', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig('erdDiagram/03_mongodb_vs_elasticsearch_schema_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 03_mongodb_vs_elasticsearch_schema_comparison.png")
    
    def generate_data_flow_erd(self):
        """Generate data flow ERD showing how data moves through the system."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Title
        ax.text(10, 13.5, 'Chicago 311 Data Platform - Data Flow Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 13, 'ETL Pipeline for 12M+ Service Request Records', 
                fontsize=14, ha='center', style='italic')
        
        # Data source
        source_box = FancyBboxPatch(
            (1, 10.5), 4, 2,
            boxstyle="round,pad=0.1",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax.add_patch(source_box)
        ax.text(3, 11.8, 'Chicago Data Portal', fontsize=12, fontweight='bold', ha='center')
        ax.text(3, 11.4, 'Socrata API', fontsize=10, ha='center')
        ax.text(3, 11, '• 12.4M+ records\n• Real-time updates\n• REST API access', fontsize=9, ha='center')
        
        # ETL Process
        etl_box = FancyBboxPatch(
            (8, 10.5), 4, 2,
            boxstyle="round,pad=0.1",
            facecolor='#FFF3E0',
            edgecolor='#F57C00',
            linewidth=2
        )
        ax.add_patch(etl_box)
        ax.text(10, 11.8, 'ETL Pipeline', fontsize=12, fontweight='bold', ha='center')
        ax.text(10, 11.4, 'Data Processing', fontsize=10, ha='center')
        ax.text(10, 11, '• Data validation\n• Coordinate conversion\n• Date parsing\n• Batch processing', fontsize=9, ha='center')
        
        # Data validation
        validation_box = FancyBboxPatch(
            (15, 10.5), 4, 2,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(validation_box)
        ax.text(17, 11.8, 'Data Validation', fontsize=12, fontweight='bold', ha='center')
        ax.text(17, 11.4, 'Quality Assurance', fontsize=10, ha='center')
        ax.text(17, 11, '• Schema validation\n• Coordinate bounds\n• Date format check\n• Duplicate detection', fontsize=9, ha='center')
        
        # MongoDB storage
        mongo_box = FancyBboxPatch(
            (2, 6), 6, 3,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(mongo_box)
        ax.text(5, 8.5, 'MongoDB Storage', fontsize=14, fontweight='bold', ha='center')
        ax.text(5, 8, 'service_requests collection', fontsize=11, ha='center')
        
        mongo_details = '''Document Structure:
• sr_number (ObjectId)
• location (GeoJSON Point)
• creation_date (ISODate)
• sr_type (String)
• status (String)
• ward (NumberInt)
• Indexes: 2dsphere, compound'''
        ax.text(2.2, 7.7, mongo_details, fontsize=9, va='top')
        
        # Elasticsearch storage
        es_box = FancyBboxPatch(
            (12, 6), 6, 3,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax.add_patch(es_box)
        ax.text(15, 8.5, 'Elasticsearch Storage', fontsize=14, fontweight='bold', ha='center')
        ax.text(15, 8, 'chicago_311_requests index', fontsize=11, ha='center')
        
        es_details = '''Mapping Structure:
• sr_number (keyword)
• location (geo_point)
• creation_date (date)
• sr_type (text + keyword)
• status (keyword)
• ward (integer)
• Full-text search enabled'''
        ax.text(12.2, 7.7, es_details, fontsize=9, va='top')
        
        # Applications
        app_boxes = [
            {'name': 'Analytics\nDashboard', 'x': 1, 'color': '#FCE4EC', 'border': '#C2185B'},
            {'name': 'Search\nApplication', 'x': 5, 'color': '#F1F8E9', 'border': '#689F38'},
            {'name': 'Geospatial\nAnalysis', 'x': 9, 'color': '#FFF8E1', 'border': '#F57F17'},
            {'name': 'Performance\nMonitoring', 'x': 13, 'color': '#E0F2F1', 'border': '#00695C'},
            {'name': 'API\nEndpoints', 'x': 17, 'color': '#FFEBEE', 'border': '#C62828'}
        ]
        
        for app in app_boxes:
            app_box = FancyBboxPatch(
                (app['x'], 2), 3, 1.5,
                boxstyle="round,pad=0.05",
                facecolor=app['color'],
                edgecolor=app['border'],
                linewidth=1.5
            )
            ax.add_patch(app_box)
            ax.text(app['x'] + 1.5, 2.75, app['name'], fontsize=10, fontweight='bold', ha='center')
        
        # Arrows showing data flow
        # Source to ETL
        arrow1 = ConnectionPatch((5, 11.5), (8, 11.5), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
        ax.add_patch(arrow1)
        
        # ETL to Validation
        arrow2 = ConnectionPatch((12, 11.5), (15, 11.5), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
        ax.add_patch(arrow2)
        
        # Validation to databases
        arrow3 = ConnectionPatch((16, 10.5), (5, 9), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="green")
        ax.add_patch(arrow3)
        
        arrow4 = ConnectionPatch((18, 10.5), (15, 9), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, fc="purple")
        ax.add_patch(arrow4)
        
        # Databases to applications
        for i, app in enumerate(app_boxes):
            if i < 3:  # MongoDB connections
                arrow = ConnectionPatch((5, 6), (app['x'] + 1.5, 3.5), "data", "data",
                                       arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, fc="green", alpha=0.6)
                ax.add_patch(arrow)
            else:  # Elasticsearch connections
                arrow = ConnectionPatch((15, 6), (app['x'] + 1.5, 3.5), "data", "data",
                                       arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, fc="purple", alpha=0.6)
                ax.add_patch(arrow)
        
        # Legend
        legend_box = FancyBboxPatch(
            (8, 0.2), 4, 1.2,
            boxstyle="round,pad=0.05",
            facecolor='#F5F5F5',
            edgecolor='#757575',
            linewidth=1
        )
        ax.add_patch(legend_box)
        ax.text(10, 1.2, 'Data Flow Legend', fontsize=11, fontweight='bold', ha='center')
        ax.text(8.2, 0.9, '→ ETL Process Flow', fontsize=9)
        ax.text(8.2, 0.7, '→ MongoDB Connection', fontsize=9, color='green')
        ax.text(8.2, 0.5, '→ Elasticsearch Connection', fontsize=9, color='purple')
        ax.text(8.2, 0.3, '12M+ records processed', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/04_data_flow_architecture_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 04_data_flow_architecture_12m_records.png")
    
    def generate_all_erds(self):
        """Generate all ERD diagrams."""
        print("🚀 Generating ERD Diagrams for Chicago 311 Data Platform...")
        print("📊 Based on 12 million Chicago 311 service request records")
        print("=" * 70)
        
        self.generate_mongodb_erd()
        self.generate_elasticsearch_erd()
        self.generate_comparison_erd()
        self.generate_data_flow_erd()
        
        print("=" * 70)
        print("🎉 ALL ERD DIAGRAMS GENERATED!")
        print("\n📁 ERD diagrams saved in erdDiagram/ folder:")
        print("   1. 01_mongodb_schema_chicago_311_12m_records.png")
        print("   2. 02_elasticsearch_schema_chicago_311_12m_records.png")
        print("   3. 03_mongodb_vs_elasticsearch_schema_comparison.png")
        print("   4. 04_data_flow_architecture_12m_records.png")
        print("\n✅ All diagrams reflect the actual data structure used for 12M+ records!")
        print("✅ Proper naming conventions applied")

if __name__ == "__main__":
    generator = ChicagoERDGenerator()
    generator.generate_all_erds()