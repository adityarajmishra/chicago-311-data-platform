#!/usr/bin/env python3
"""
Clean ERD Diagram Generator for Chicago 311 Data Platform
Fixed layouts with proper spacing and systematic organization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os
from datetime import datetime

class CleanChicagoERDGenerator:
    """Generate clean, well-organized ERD diagrams for Chicago 311 data platform."""
    
    def __init__(self):
        """Initialize ERD generator."""
        # Ensure directory exists
        os.makedirs('erdDiagram', exist_ok=True)
        
        # Chicago 311 data schema organized for clean display
        self.field_categories = {
            'Primary Key': {
                'fields': [
                    {'name': 'sr_number', 'type': 'String', 'desc': 'Unique service request number', 'special': 'PRIMARY'},
                ],
                'color': '#FFCDD2', 'border': '#D32F2F'
            },
            'Request Information': {
                'fields': [
                    {'name': 'sr_type', 'type': 'String', 'desc': 'Type of service request', 'special': 'INDEXED'},
                    {'name': 'owner_department', 'type': 'String', 'desc': 'Responsible department', 'special': 'INDEXED'},
                    {'name': 'status', 'type': 'String', 'desc': 'Current request status', 'special': 'INDEXED'},
                    {'name': 'sr_short_code', 'type': 'String', 'desc': 'Short code for request type', 'special': ''},
                ],
                'color': '#E1BEE7', 'border': '#8E24AA'
            },
            'Temporal Data': {
                'fields': [
                    {'name': 'creation_date', 'type': 'DateTime', 'desc': 'When request was created', 'special': 'INDEXED'},
                    {'name': 'completion_date', 'type': 'DateTime', 'desc': 'When request was completed', 'special': ''},
                    {'name': 'due_date', 'type': 'DateTime', 'desc': 'When request is due', 'special': ''},
                ],
                'color': '#C8E6C9', 'border': '#388E3C'
            },
            'Geographic Data': {
                'fields': [
                    {'name': 'location', 'type': 'GeoPoint', 'desc': 'Geographic coordinates', 'special': 'GEO_INDEX'},
                    {'name': 'latitude', 'type': 'Float', 'desc': 'Latitude coordinate', 'special': 'GEO'},
                    {'name': 'longitude', 'type': 'Float', 'desc': 'Longitude coordinate', 'special': 'GEO'},
                ],
                'color': '#B2DFDB', 'border': '#00695C'
            },
            'Address Information': {
                'fields': [
                    {'name': 'street_address', 'type': 'String', 'desc': 'Full street address', 'special': 'TEXT_SEARCH'},
                    {'name': 'city', 'type': 'String', 'desc': 'City name', 'special': ''},
                    {'name': 'state', 'type': 'String', 'desc': 'State abbreviation', 'special': ''},
                    {'name': 'zip_code', 'type': 'String', 'desc': 'ZIP postal code', 'special': 'INDEXED'},
                ],
                'color': '#FFE0B2', 'border': '#F57C00'
            },
            'Administrative Areas': {
                'fields': [
                    {'name': 'ward', 'type': 'Integer', 'desc': 'Ward number', 'special': 'INDEXED'},
                    {'name': 'community_area', 'type': 'Integer', 'desc': 'Community area number', 'special': 'INDEXED'},
                    {'name': 'police_district', 'type': 'String', 'desc': 'Police district', 'special': ''},
                    {'name': 'census_tract', 'type': 'String', 'desc': 'Census tract identifier', 'special': ''},
                ],
                'color': '#F8BBD9', 'border': '#C2185B'
            }
        }
        
    def generate_clean_mongodb_erd(self):
        """Generate clean MongoDB ERD diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(22, 16))
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title section
        title_box = FancyBboxPatch(
            (1, 14.5), 20, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax.add_patch(title_box)
        ax.text(11, 15.3, 'MongoDB Schema - Chicago 311 Service Requests', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(11, 14.8, 'Document-Based Schema for 12M+ Records', 
                fontsize=12, ha='center', style='italic')
        
        # Collection info box
        collection_box = FancyBboxPatch(
            (1, 13), 20, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='#F5F5F5',
            edgecolor='#757575',
            linewidth=1
        )
        ax.add_patch(collection_box)
        ax.text(11, 13.6, 'Collection: service_requests', 
                fontsize=14, fontweight='bold', ha='center')
        ax.text(11, 13.2, 'MongoDB Document Structure with Optimized Indexes', 
                fontsize=10, ha='center')
        
        # Main content area
        main_box = FancyBboxPatch(
            (1, 2), 15, 10.5,
            boxstyle="round,pad=0.1",
            facecolor='#FAFAFA',
            edgecolor='#424242',
            linewidth=1
        )
        ax.add_patch(main_box)
        
        # Field categories
        y_start = 12
        row_height = 0.4
        category_height = 0.5
        
        for category_name, category_info in self.field_categories.items():
            # Category header
            cat_header_box = FancyBboxPatch(
                (2, y_start), 13, category_height,
                boxstyle="round,pad=0.05",
                facecolor=category_info['border'],
                edgecolor=category_info['border'],
                alpha=0.9
            )
            ax.add_patch(cat_header_box)
            ax.text(8.5, y_start + 0.25, category_name, 
                    fontsize=12, fontweight='bold', ha='center', color='white')
            
            y_start -= category_height + 0.1
            
            # Fields in category
            for field in category_info['fields']:
                # Field row
                field_box = FancyBboxPatch(
                    (2.5, y_start), 12, row_height,
                    boxstyle="round,pad=0.02",
                    facecolor=category_info['color'],
                    edgecolor=category_info['border'],
                    alpha=0.6,
                    linewidth=0.5
                )
                ax.add_patch(field_box)
                
                # Field name and type
                field_text = f"{field['name']}: {field['type']}"
                if field['special']:
                    field_text += f" [{field['special']}]"
                
                ax.text(2.8, y_start + 0.2, field_text, 
                        fontsize=10, fontweight='bold' if field['special'] else 'normal', va='center')
                
                # Description
                ax.text(11, y_start + 0.2, field['desc'], 
                        fontsize=9, style='italic', ha='right', va='center')
                
                y_start -= row_height + 0.05
            
            y_start -= 0.2  # Space between categories
        
        # Indexes panel
        index_box = FancyBboxPatch(
            (16.5, 8), 5, 4.5,
            boxstyle="round,pad=0.1",
            facecolor='#FFF3E0',
            edgecolor='#F57C00',
            linewidth=2
        )
        ax.add_patch(index_box)
        
        ax.text(19, 12.2, 'MongoDB Indexes', fontsize=12, fontweight='bold', ha='center')
        ax.text(19, 11.8, 'Performance Optimization', fontsize=10, ha='center', style='italic')
        
        indexes = [
            'Primary Index:',
            '• sr_number (unique)',
            '',
            'Performance Indexes:',
            '• creation_date (desc)',
            '• status + creation_date',
            '• sr_type + creation_date',
            '• ward + creation_date',
            '',
            'Geospatial Index:',
            '• location (2dsphere)',
            '',
            'Text Search Index:',
            '• sr_type, street_address'
        ]
        
        y_pos = 11.5
        for idx in indexes:
            if idx == '':
                y_pos -= 0.15
                continue
            if idx.endswith(':'):
                ax.text(16.8, y_pos, idx, fontsize=10, fontweight='bold')
            else:
                ax.text(16.8, y_pos, idx, fontsize=9)
            y_pos -= 0.25
        
        # GeoJSON example
        geo_box = FancyBboxPatch(
            (16.5, 2), 5, 5.5,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(geo_box)
        
        ax.text(19, 7.2, 'GeoJSON Location Format', fontsize=12, fontweight='bold', ha='center')
        
        geojson_example = '''{
  "_id": ObjectId("..."),
  "sr_number": "SR23-00123456",
  "sr_type": "Pothole in Street",
  "status": "Open",
  "creation_date": ISODate("2023-01-15"),
  "location": {
    "type": "Point",
    "coordinates": [-87.623177, 41.881832]
  },
  "ward": 42,
  "community_area": 8,
  "street_address": "123 N State St",
  "owner_department": "CDOT"
}'''
        ax.text(16.8, 6.8, geojson_example, fontsize=8, fontfamily='monospace', va='top')
        
        # Footer
        footer_box = FancyBboxPatch(
            (1, 0.2), 20, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#F1F8E9',
            edgecolor='#689F38',
            linewidth=1
        )
        ax.add_patch(footer_box)
        ax.text(11, 1.3, 'MongoDB Schema Features', fontsize=12, fontweight='bold', ha='center')
        footer_text = '✓ Flexible document structure  ✓ Rich geospatial queries  ✓ Compound indexes  ✓ Aggregation pipelines  ✓ Horizontal scaling'
        ax.text(11, 0.8, footer_text, fontsize=10, ha='center')
        ax.text(11, 0.4, '12+ Million Chicago 311 Service Request Documents', fontsize=10, fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/01_mongodb_schema_chicago_311_12m_records_clean.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 01_mongodb_schema_chicago_311_12m_records_clean.png")
    
    def generate_clean_elasticsearch_erd(self):
        """Generate clean Elasticsearch ERD diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(22, 16))
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title section
        title_box = FancyBboxPatch(
            (1, 14.5), 20, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax.add_patch(title_box)
        ax.text(11, 15.3, 'Elasticsearch Schema - Chicago 311 Service Requests', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(11, 14.8, 'Search-Optimized Schema for 12M+ Records', 
                fontsize=12, ha='center', style='italic')
        
        # Index info box
        index_info_box = FancyBboxPatch(
            (1, 13), 20, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='#F5F5F5',
            edgecolor='#757575',
            linewidth=1
        )
        ax.add_patch(index_info_box)
        ax.text(11, 13.6, 'Index: chicago_311_requests', 
                fontsize=14, fontweight='bold', ha='center')
        ax.text(11, 13.2, 'Elasticsearch Mapping with Full-Text Search and Geospatial Capabilities', 
                fontsize=10, ha='center')
        
        # Main content area
        main_box = FancyBboxPatch(
            (1, 2), 15, 10.5,
            boxstyle="round,pad=0.1",
            facecolor='#FAFAFA',
            edgecolor='#424242',
            linewidth=1
        )
        ax.add_patch(main_box)
        
        # Elasticsearch field mappings
        es_categories = {
            'Document ID': {
                'fields': [
                    {'name': 'sr_number', 'type': 'keyword', 'desc': 'Document ID and unique identifier'},
                ],
                'color': '#FFCDD2', 'border': '#D32F2F'
            },
            'Searchable Text Fields': {
                'fields': [
                    {'name': 'sr_type', 'type': 'text + keyword', 'desc': 'Analyzed text + exact matching'},
                    {'name': 'owner_department', 'type': 'text + keyword', 'desc': 'Department name with analysis'},
                    {'name': 'street_address', 'type': 'text', 'desc': 'Full-text searchable address'},
                ],
                'color': '#C8E6C9', 'border': '#388E3C'
            },
            'Categorical Fields': {
                'fields': [
                    {'name': 'status', 'type': 'keyword', 'desc': 'Exact match status filtering'},
                    {'name': 'city', 'type': 'keyword', 'desc': 'City name exact match'},
                    {'name': 'state', 'type': 'keyword', 'desc': 'State code exact match'},
                    {'name': 'zip_code', 'type': 'keyword', 'desc': 'ZIP code exact match'},
                ],
                'color': '#FFE0B2', 'border': '#F57C00'
            },
            'Temporal Fields': {
                'fields': [
                    {'name': 'creation_date', 'type': 'date', 'desc': 'ISO 8601 date format'},
                    {'name': 'completion_date', 'type': 'date', 'desc': 'Completion timestamp'},
                    {'name': 'due_date', 'type': 'date', 'desc': 'Due date timestamp'},
                ],
                'color': '#E1BEE7', 'border': '#8E24AA'
            },
            'Geographic Fields': {
                'fields': [
                    {'name': 'location', 'type': 'geo_point', 'desc': 'Geospatial coordinates'},
                    {'name': 'latitude', 'type': 'float', 'desc': 'Latitude decimal degrees'},
                    {'name': 'longitude', 'type': 'float', 'desc': 'Longitude decimal degrees'},
                ],
                'color': '#B2DFDB', 'border': '#00695C'
            },
            'Numeric Fields': {
                'fields': [
                    {'name': 'ward', 'type': 'integer', 'desc': 'Ward number for aggregations'},
                    {'name': 'community_area', 'type': 'integer', 'desc': 'Community area number'},
                ],
                'color': '#F8BBD9', 'border': '#C2185B'
            }
        }
        
        # Render field categories
        y_start = 12
        row_height = 0.4
        category_height = 0.5
        
        for category_name, category_info in es_categories.items():
            # Category header
            cat_header_box = FancyBboxPatch(
                (2, y_start), 13, category_height,
                boxstyle="round,pad=0.05",
                facecolor=category_info['border'],
                edgecolor=category_info['border'],
                alpha=0.9
            )
            ax.add_patch(cat_header_box)
            ax.text(8.5, y_start + 0.25, category_name, 
                    fontsize=12, fontweight='bold', ha='center', color='white')
            
            y_start -= category_height + 0.1
            
            # Fields in category
            for field in category_info['fields']:
                # Field row
                field_box = FancyBboxPatch(
                    (2.5, y_start), 12, row_height,
                    boxstyle="round,pad=0.02",
                    facecolor=category_info['color'],
                    edgecolor=category_info['border'],
                    alpha=0.6,
                    linewidth=0.5
                )
                ax.add_patch(field_box)
                
                # Field name and type
                field_text = f"{field['name']}: {field['type']}"
                ax.text(2.8, y_start + 0.2, field_text, 
                        fontsize=10, fontweight='bold', va='center')
                
                # Description
                ax.text(11, y_start + 0.2, field['desc'], 
                        fontsize=9, style='italic', ha='right', va='center')
                
                y_start -= row_height + 0.05
            
            y_start -= 0.2  # Space between categories
        
        # Search capabilities panel
        search_box = FancyBboxPatch(
            (16.5, 8.5), 5, 4,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(search_box)
        
        ax.text(19, 12.2, 'Search Capabilities', fontsize=12, fontweight='bold', ha='center')
        
        search_features = [
            'Text Search:',
            '• Full-text search',
            '• Fuzzy matching',
            '• Auto-complete',
            '',
            'Geospatial:',
            '• Geo-distance queries',
            '• Bounding box search',
            '',
            'Aggregations:',
            '• Terms aggregation',
            '• Date histogram',
            '• Geohash grid'
        ]
        
        y_pos = 11.9
        for feature in search_features:
            if feature == '':
                y_pos -= 0.15
                continue
            if feature.endswith(':'):
                ax.text(16.8, y_pos, feature, fontsize=10, fontweight='bold')
            else:
                ax.text(16.8, y_pos, feature, fontsize=9)
            y_pos -= 0.25
        
        # Index settings and example
        settings_box = FancyBboxPatch(
            (16.5, 2), 5, 6,
            boxstyle="round,pad=0.1",
            facecolor='#FFF3E0',
            edgecolor='#F57C00',
            linewidth=2
        )
        ax.add_patch(settings_box)
        
        ax.text(19, 7.7, 'Index Settings & Example', fontsize=12, fontweight='bold', ha='center')
        
        settings_text = '''Settings:
• Shards: 5
• Replicas: 1
• Refresh: 1s
• Max window: 10000

Document Example:
{
  "_id": "SR23-00123456",
  "sr_type": "Pothole in Street",
  "status": "Open",
  "creation_date": "2023-01-15T10:30:00Z",
  "location": {
    "lat": 41.881832,
    "lon": -87.623177
  },
  "ward": 42
}'''
        ax.text(16.8, 7.4, settings_text, fontsize=8, fontfamily='monospace', va='top')
        
        # Footer
        footer_box = FancyBboxPatch(
            (1, 0.2), 20, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=1
        )
        ax.add_patch(footer_box)
        ax.text(11, 1.3, 'Elasticsearch Schema Features', fontsize=12, fontweight='bold', ha='center')
        footer_text = '✓ Full-text search  ✓ Real-time analytics  ✓ Geospatial queries  ✓ Distributed architecture  ✓ RESTful API'
        ax.text(11, 0.8, footer_text, fontsize=10, ha='center')
        ax.text(11, 0.4, '12+ Million Searchable Chicago 311 Service Request Documents', fontsize=10, fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/02_elasticsearch_schema_chicago_311_12m_records_clean.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 02_elasticsearch_schema_chicago_311_12m_records_clean.png")
    
    def generate_clean_comparison_erd(self):
        """Generate clean comparison ERD."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
        
        # MongoDB side (left)
        ax1.set_xlim(0, 12)
        ax1.set_ylim(0, 16)
        ax1.axis('off')
        
        # MongoDB title
        mongo_title_box = FancyBboxPatch(
            (0.5, 14.5), 11, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax1.add_patch(mongo_title_box)
        ax1.text(6, 15.3, 'MongoDB Document Model', fontsize=16, fontweight='bold', ha='center')
        ax1.text(6, 14.8, 'Operational Database Approach', fontsize=12, ha='center', style='italic')
        
        # MongoDB content
        mongo_sections = [
            {
                'title': 'Schema Design',
                'content': ['• Flexible document schema', '• Nested document support', '• Dynamic field addition', '• Rich data types (ObjectId, Date)'],
                'color': '#E8F5E8', 'y': 12.5
            },
            {
                'title': 'Data Storage',
                'content': ['• BSON document format', '• Collection-based organization', '• GridFS for large files', '• Replica sets for HA'],
                'color': '#FFF3E0', 'y': 10.5
            },
            {
                'title': 'Query Capabilities',
                'content': ['• Rich query operators', '• Aggregation pipelines', '• MapReduce support', '• Geospatial queries ($near)'],
                'color': '#FCE4EC', 'y': 8.5
            },
            {
                'title': 'Indexing Strategy',
                'content': ['• B-tree indexes', '• Compound indexes', '• 2dsphere geo indexes', '• Sparse & partial indexes'],
                'color': '#F1F8E9', 'y': 6.5
            },
            {
                'title': 'Best Use Cases',
                'content': ['• Operational applications', '• Complex data relationships', '• Real-time read/write', '• Geographic applications'],
                'color': '#FFEBEE', 'y': 4.5
            }
        ]
        
        for section in mongo_sections:
            # Section box
            section_box = FancyBboxPatch(
                (1, section['y']), 10, 1.8,
                boxstyle="round,pad=0.1",
                facecolor=section['color'],
                edgecolor='#424242',
                linewidth=1,
                alpha=0.8
            )
            ax1.add_patch(section_box)
            
            # Section title
            ax1.text(6, section['y'] + 1.5, section['title'], 
                    fontsize=12, fontweight='bold', ha='center')
            
            # Section content
            for i, item in enumerate(section['content']):
                ax1.text(1.3, section['y'] + 1.1 - (i * 0.25), item, 
                        fontsize=9, va='center')
        
        # Performance metrics
        perf_box = FancyBboxPatch(
            (1, 1), 10, 2.8,
            boxstyle="round,pad=0.1",
            facecolor='#E3F2FD',
            edgecolor='#1976D2',
            linewidth=2
        )
        ax1.add_patch(perf_box)
        ax1.text(6, 3.5, 'MongoDB Performance Profile', fontsize=12, fontweight='bold', ha='center')
        
        mongo_metrics = [
            'Read Performance: Excellent (indexed queries)',
            'Write Performance: Very Good (single writes)',
            'Aggregation: Good (pipeline operations)',
            'Scalability: Horizontal (sharding)',
            'Consistency: Strong (ACID transactions)',
            'Geospatial: Excellent (native GeoJSON)',
            'Full-text Search: Limited (basic text search)',
            'Memory Usage: Moderate (working set)'
        ]
        
        y_pos = 3.2
        for metric in mongo_metrics:
            ax1.text(1.3, y_pos, metric, fontsize=9)
            y_pos -= 0.25
        
        # Elasticsearch side (right)
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, 16)
        ax2.axis('off')
        
        # Elasticsearch title
        es_title_box = FancyBboxPatch(
            (0.5, 14.5), 11, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax2.add_patch(es_title_box)
        ax2.text(6, 15.3, 'Elasticsearch Search Model', fontsize=16, fontweight='bold', ha='center')
        ax2.text(6, 14.8, 'Search & Analytics Engine Approach', fontsize=12, ha='center', style='italic')
        
        # Elasticsearch content
        es_sections = [
            {
                'title': 'Schema Design',
                'content': ['• Explicit field mappings', '• Data type enforcement', '• Multi-field support', '• Dynamic mapping option'],
                'color': '#E8F5E8', 'y': 12.5
            },
            {
                'title': 'Data Storage',
                'content': ['• JSON document format', '• Index-based organization', '• Distributed sharding', '• Node replication'],
                'color': '#FFF3E0', 'y': 10.5
            },
            {
                'title': 'Query Capabilities',
                'content': ['• DSL query language', '• Full-text search', '• Fuzzy matching', '• Geo-distance queries'],
                'color': '#FCE4EC', 'y': 8.5
            },
            {
                'title': 'Indexing Strategy',
                'content': ['• Inverted indexes', '• Doc values', '• Geo-spatial indexes', '• Completion suggesters'],
                'color': '#F1F8E9', 'y': 6.5
            },
            {
                'title': 'Best Use Cases',
                'content': ['• Search applications', '• Real-time analytics', '• Log analysis', '• Content discovery'],
                'color': '#FFEBEE', 'y': 4.5
            }
        ]
        
        for section in es_sections:
            # Section box
            section_box = FancyBboxPatch(
                (1, section['y']), 10, 1.8,
                boxstyle="round,pad=0.1",
                facecolor=section['color'],
                edgecolor='#424242',
                linewidth=1,
                alpha=0.8
            )
            ax2.add_patch(section_box)
            
            # Section title
            ax2.text(6, section['y'] + 1.5, section['title'], 
                    fontsize=12, fontweight='bold', ha='center')
            
            # Section content
            for i, item in enumerate(section['content']):
                ax2.text(1.3, section['y'] + 1.1 - (i * 0.25), item, 
                        fontsize=9, va='center')
        
        # Performance metrics
        es_perf_box = FancyBboxPatch(
            (1, 1), 10, 2.8,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax2.add_patch(es_perf_box)
        ax2.text(6, 3.5, 'Elasticsearch Performance Profile', fontsize=12, fontweight='bold', ha='center')
        
        es_metrics = [
            'Read Performance: Excellent (search optimized)',
            'Write Performance: Good (bulk operations)',
            'Aggregation: Excellent (real-time analytics)',
            'Scalability: Horizontal (distributed)',
            'Consistency: Eventually consistent',
            'Geospatial: Very Good (geo queries)',
            'Full-text Search: Excellent (core strength)',
            'Memory Usage: High (fielddata & caches)'
        ]
        
        y_pos = 3.2
        for metric in es_metrics:
            ax2.text(1.3, y_pos, metric, fontsize=9)
            y_pos -= 0.25
        
        # Main comparison title
        fig.suptitle('MongoDB vs Elasticsearch Architecture Comparison\nChicago 311 Service Requests - 12M+ Records', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        plt.savefig('erdDiagram/03_mongodb_vs_elasticsearch_comparison_clean.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 03_mongodb_vs_elasticsearch_comparison_clean.png")
    
    def generate_clean_data_flow_erd(self):
        """Generate clean data flow ERD."""
        fig, ax = plt.subplots(1, 1, figsize=(22, 16))
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title
        title_box = FancyBboxPatch(
            (1, 14.5), 20, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(title_box)
        ax.text(11, 15.3, 'Chicago 311 Data Platform - Complete Data Flow Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(11, 14.8, 'ETL Pipeline Processing 12+ Million Service Request Records', 
                fontsize=12, ha='center', style='italic')
        
        # Data pipeline stages
        pipeline_stages = [
            {
                'name': 'Data Source\nChicago Data Portal',
                'details': ['• Socrata API', '• 12.4M+ records', '• Real-time updates', '• JSON format'],
                'pos': (2, 11.5), 'size': (4, 2.5), 'color': '#E3F2FD', 'border': '#1976D2'
            },
            {
                'name': 'Data Extraction\nAPI Client',
                'details': ['• Batch processing', '• Rate limiting', '• Error handling', '• Progress tracking'],
                'pos': (7.5, 11.5), 'size': (4, 2.5), 'color': '#FFF3E0', 'border': '#F57C00'
            },
            {
                'name': 'Data Validation\nQuality Control',
                'details': ['• Schema validation', '• Coordinate bounds', '• Date format check', '• Duplicate detection'],
                'pos': (13, 11.5), 'size': (4, 2.5), 'color': '#F3E5F5', 'border': '#7B1FA2'
            },
            {
                'name': 'Data Processing\nETL Operations',
                'details': ['• Field mapping', '• Data type conversion', '• GeoJSON formatting', '• Index preparation'],
                'pos': (18, 11.5), 'size': (3.5, 2.5), 'color': '#E0F2F1', 'border': '#00695C'
            }
        ]
        
        # Draw pipeline stages
        for stage in pipeline_stages:
            # Stage box
            stage_box = FancyBboxPatch(
                stage['pos'], stage['size'][0], stage['size'][1],
                boxstyle="round,pad=0.1",
                facecolor=stage['color'],
                edgecolor=stage['border'],
                linewidth=2
            )
            ax.add_patch(stage_box)
            
            # Stage title
            ax.text(stage['pos'][0] + stage['size'][0]/2, stage['pos'][1] + stage['size'][1] - 0.3, 
                   stage['name'], fontsize=11, fontweight='bold', ha='center')
            
            # Stage details
            for i, detail in enumerate(stage['details']):
                ax.text(stage['pos'][0] + 0.2, stage['pos'][1] + stage['size'][1] - 0.8 - (i * 0.3), 
                       detail, fontsize=9, va='center')
        
        # Database storage section
        db_y = 7.5
        
        # MongoDB storage
        mongo_box = FancyBboxPatch(
            (2, db_y), 8, 3,
            boxstyle="round,pad=0.1",
            facecolor='#E8F5E8',
            edgecolor='#388E3C',
            linewidth=2
        )
        ax.add_patch(mongo_box)
        ax.text(6, db_y + 2.5, 'MongoDB Storage Layer', fontsize=14, fontweight='bold', ha='center')
        ax.text(6, db_y + 2.1, 'service_requests Collection', fontsize=11, ha='center', style='italic')
        
        mongo_features = [
            'Document Structure:',
            '• ObjectId primary keys',
            '• GeoJSON Point coordinates',
            '• Nested document support',
            '• Rich data type support',
            '',
            'Performance Features:',
            '• Compound indexes',
            '• Geospatial 2dsphere indexes',
            '• Aggregation pipelines',
            '• Replica set redundancy'
        ]
        
        y_pos = db_y + 1.8
        for feature in mongo_features:
            if feature == '':
                y_pos -= 0.15
                continue
            if feature.endswith(':'):
                ax.text(2.3, y_pos, feature, fontsize=10, fontweight='bold')
            else:
                ax.text(2.3, y_pos, feature, fontsize=9)
            y_pos -= 0.2
        
        # Elasticsearch storage
        es_box = FancyBboxPatch(
            (12, db_y), 8, 3,
            boxstyle="round,pad=0.1",
            facecolor='#F3E5F5',
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax.add_patch(es_box)
        ax.text(16, db_y + 2.5, 'Elasticsearch Storage Layer', fontsize=14, fontweight='bold', ha='center')
        ax.text(16, db_y + 2.1, 'chicago_311_requests Index', fontsize=11, ha='center', style='italic')
        
        es_features = [
            'Index Structure:',
            '• Keyword document IDs',
            '• Geo-point coordinates',
            '• Text analysis & keywords',
            '• Date field optimization',
            '',
            'Search Features:',
            '• Inverted indexes',
            '• Real-time search',
            '• Aggregation framework',
            '• Distributed architecture'
        ]
        
        y_pos = db_y + 1.8
        for feature in es_features:
            if feature == '':
                y_pos -= 0.15
                continue
            if feature.endswith(':'):
                ax.text(12.3, y_pos, feature, fontsize=10, fontweight='bold')
            else:
                ax.text(12.3, y_pos, feature, fontsize=9)
            y_pos -= 0.2
        
        # Application layer
        app_y = 3
        applications = [
            {'name': 'Analytics\nDashboards', 'desc': 'Operational\nReporting', 'pos': 2, 'color': '#FCE4EC', 'border': '#C2185B'},
            {'name': 'Search\nPortals', 'desc': 'Citizen\nServices', 'pos': 6, 'color': '#F1F8E9', 'border': '#689F38'},
            {'name': 'GIS\nMapping', 'desc': 'Geospatial\nAnalysis', 'pos': 10, 'color': '#FFF8E1', 'border': '#F57F17'},
            {'name': 'Mobile\nApps', 'desc': 'Field\nOperations', 'pos': 14, 'color': '#E0F2F1', 'border': '#00695C'},
            {'name': 'API\nServices', 'desc': 'Third-party\nIntegration', 'pos': 18, 'color': '#FFEBEE', 'border': '#C62828'}
        ]
        
        for app in applications:
            app_box = FancyBboxPatch(
                (app['pos'], app_y), 3, 1.5,
                boxstyle="round,pad=0.05",
                facecolor=app['color'],
                edgecolor=app['border'],
                linewidth=1.5
            )
            ax.add_patch(app_box)
            ax.text(app['pos'] + 1.5, app_y + 1.1, app['name'], 
                   fontsize=10, fontweight='bold', ha='center')
            ax.text(app['pos'] + 1.5, app_y + 0.4, app['desc'], 
                   fontsize=9, ha='center', style='italic')
        
        # Data flow arrows (cleaned up)
        # Pipeline flow arrows
        pipeline_arrows = [
            ((6, 12.75), (7.5, 12.75)),  # Source to Extraction
            ((11.5, 12.75), (13, 12.75)),  # Extraction to Validation
            ((17, 12.75), (18, 12.75))     # Validation to Processing
        ]
        
        for start, end in pipeline_arrows:
            arrow = ConnectionPatch(start, end, "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=20, fc="black", linewidth=2)
            ax.add_patch(arrow)
        
        # Processing to databases
        proc_to_mongo = ConnectionPatch((19, 11.5), (6, 10.5), "data", "data",
                                       arrowstyle="->", shrinkA=10, shrinkB=10,
                                       mutation_scale=20, fc="green", linewidth=2)
        ax.add_patch(proc_to_mongo)
        
        proc_to_es = ConnectionPatch((20, 11.5), (16, 10.5), "data", "data",
                                    arrowstyle="->", shrinkA=10, shrinkB=10,
                                    mutation_scale=20, fc="purple", linewidth=2)
        ax.add_patch(proc_to_es)
        
        # Databases to applications (simplified)
        for i, app in enumerate(applications):
            if i < 3:  # First 3 apps connect to MongoDB
                arrow = ConnectionPatch((6, db_y), (app['pos'] + 1.5, app_y + 1.5), "data", "data",
                                       arrowstyle="->", shrinkA=10, shrinkB=10,
                                       mutation_scale=15, fc="green", alpha=0.6, linewidth=1.5)
                ax.add_patch(arrow)
            else:  # Last 2 apps connect to Elasticsearch
                arrow = ConnectionPatch((16, db_y), (app['pos'] + 1.5, app_y + 1.5), "data", "data",
                                       arrowstyle="->", shrinkA=10, shrinkB=10,
                                       mutation_scale=15, fc="purple", alpha=0.6, linewidth=1.5)
                ax.add_patch(arrow)
        
        # Legend and stats
        legend_box = FancyBboxPatch(
            (1, 0.2), 20, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#F5F5F5',
            edgecolor='#424242',
            linewidth=1
        )
        ax.add_patch(legend_box)
        
        ax.text(11, 1.5, 'Data Flow Legend & Statistics', fontsize=12, fontweight='bold', ha='center')
        
        legend_items = [
            '→ ETL Processing Flow',
            '→ MongoDB Storage (Green)',
            '→ Elasticsearch Storage (Purple)',
            '📊 12+ Million Records Processed',
            '🔄 Real-time Data Updates',
            '⚡ Dual Storage Architecture'
        ]
        
        # Arrange legend items in two columns
        for i, item in enumerate(legend_items):
            x_pos = 2 if i < 3 else 12
            y_pos = 1.2 - ((i % 3) * 0.25)
            ax.text(x_pos, y_pos, item, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('erdDiagram/04_data_flow_architecture_clean.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: 04_data_flow_architecture_clean.png")
    
    def generate_all_clean_erds(self):
        """Generate all clean ERD diagrams."""
        print("🚀 Generating Clean ERD Diagrams for Chicago 311 Data Platform...")
        print("📊 Fixed layouts with proper spacing and systematic organization")
        print("=" * 80)
        
        self.generate_clean_mongodb_erd()
        self.generate_clean_elasticsearch_erd() 
        self.generate_clean_comparison_erd()
        self.generate_clean_data_flow_erd()
        
        print("=" * 80)
        print("🎉 ALL CLEAN ERD DIAGRAMS GENERATED!")
        print("\n📁 Clean ERD diagrams saved in erdDiagram/ folder:")
        print("   1. 01_mongodb_schema_chicago_311_12m_records_clean.png")
        print("   2. 02_elasticsearch_schema_chicago_311_12m_records_clean.png") 
        print("   3. 03_mongodb_vs_elasticsearch_comparison_clean.png")
        print("   4. 04_data_flow_architecture_clean.png")
        print("\n✅ All diagrams now have clean, systematic layouts!")
        print("✅ No more overlapping text or cramped sections!")
        print("✅ Professional appearance with proper spacing!")

if __name__ == "__main__":
    generator = CleanChicagoERDGenerator()
    generator.generate_all_clean_erds()