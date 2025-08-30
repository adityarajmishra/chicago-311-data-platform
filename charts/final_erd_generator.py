#!/usr/bin/env python3
"""
Final ERD Generator - Single Clean Version
Properly spaced layouts with no overlapping text
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import os

class FinalERDGenerator:
    """Generate final clean ERD diagrams with proper spacing."""
    
    def __init__(self):
        """Initialize ERD generator."""
        os.makedirs('erdDiagram', exist_ok=True)
        
    def generate_mongodb_erd(self):
        """Generate MongoDB ERD with proper spacing."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 24))  # Increased height
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 24)
        ax.axis('off')
        
        # Title
        ax.text(10, 23, 'MongoDB Schema - Chicago 311 Service Requests', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 22.5, 'Document-Based Schema for 12M+ Records', 
                fontsize=14, ha='center', style='italic')
        
        # Collection info
        ax.text(10, 21.8, 'Collection: service_requests', 
                fontsize=16, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Field definitions with proper spacing
        fields = [
            ('Primary Key', [
                'sr_number: String [PRIMARY KEY] - Unique service request identifier'
            ], 'red'),
            
            ('Request Information', [
                'sr_type: String [INDEXED] - Type of service request (e.g., Pothole, Graffiti)',
                'owner_department: String [INDEXED] - Department responsible for request',
                'status: String [INDEXED] - Current status (Open, Closed, In Progress)',
                'sr_short_code: String - Abbreviated code for request type'
            ], 'purple'),
            
            ('Temporal Data', [
                'creation_date: DateTime [INDEXED] - When request was submitted',
                'completion_date: DateTime - When request was completed',
                'due_date: DateTime - Expected completion date'
            ], 'green'),
            
            ('Geographic Information', [
                'location: GeoJSON Point [2DSPHERE INDEX] - Geographic coordinates',
                'latitude: Float [GEO] - Latitude coordinate (-90 to 90)',
                'longitude: Float [GEO] - Longitude coordinate (-180 to 180)'
            ], 'teal'),
            
            ('Address Details', [
                'street_address: String [TEXT SEARCH] - Full street address',
                'city: String - City name (typically Chicago)',
                'state: String - State abbreviation (IL)',
                'zip_code: String [INDEXED] - ZIP postal code'
            ], 'orange'),
            
            ('Administrative Areas', [
                'ward: Integer [INDEXED] - Ward number (1-50)',
                'community_area: Integer [INDEXED] - Community area (1-77)',
                'police_district: String - Police district identifier',
                'census_tract: String - Census tract number'
            ], 'pink'),
            
            ('Metadata', [
                'duplicate: Boolean - Flag indicating duplicate request',
                'legacy_record: Boolean - Flag for legacy system records',
                '_processed_at: DateTime - Processing timestamp'
            ], 'lightgreen')
        ]
        
        # Draw fields with proper spacing
        y_pos = 20.5
        for category, field_list, color in fields:
            # Category header
            ax.text(1, y_pos, category, fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            y_pos -= 0.8
            
            # Fields in category with proper spacing
            for field in field_list:
                ax.text(2, y_pos, field, fontsize=11, wrap=True)
                y_pos -= 0.6  # Proper spacing between fields
            
            y_pos -= 0.4  # Extra space between categories
        
        # Indexes section
        ax.text(1, y_pos, 'MongoDB Indexes', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
        y_pos -= 0.8
        
        indexes = [
            '• Primary: sr_number (unique identifier)',
            '• Performance: creation_date (descending order)',
            '• Compound: status + creation_date (filtered queries)',
            '• Compound: sr_type + creation_date (type-based queries)',
            '• Compound: ward + creation_date (ward-based queries)',
            '• Geospatial: location (2dsphere for geo queries)',
            '• Text Search: sr_type, street_address (full-text search)'
        ]
        
        for idx in indexes:
            ax.text(2, y_pos, idx, fontsize=11)
            y_pos -= 0.6
        
        # Example document
        y_pos -= 0.8
        ax.text(1, y_pos, 'Example Document Structure', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.5))
        y_pos -= 0.8
        
        example = '''
{
  "_id": ObjectId("64a1b2c3d4e5f6789abcdef0"),
  "sr_number": "SR23-00123456",
  "sr_type": "Pothole in Street", 
  "status": "Open",
  "creation_date": ISODate("2023-01-15T10:30:00Z"),
  "location": {
    "type": "Point",
    "coordinates": [-87.623177, 41.881832]
  },
  "ward": 42,
  "community_area": 8,
  "street_address": "123 N State St, Chicago, IL",
  "owner_department": "CDOT"
}
        '''
        
        ax.text(2, y_pos, example, fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('erdDiagram/mongodb_schema_chicago_311_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: mongodb_schema_chicago_311_12m_records.png")
    
    def generate_elasticsearch_erd(self):
        """Generate Elasticsearch ERD with proper spacing."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 24))  # Increased height
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 24)
        ax.axis('off')
        
        # Title
        ax.text(10, 23, 'Elasticsearch Schema - Chicago 311 Service Requests', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 22.5, 'Search-Optimized Schema for 12M+ Records', 
                fontsize=14, ha='center', style='italic')
        
        # Index info
        ax.text(10, 21.8, 'Index: chicago_311_requests', 
                fontsize=16, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink', alpha=0.7))
        
        # Field mappings with proper spacing
        mappings = [
            ('Document Identifiers', [
                'sr_number: keyword - Unique service request number (exact match only)'
            ], 'red'),
            
            ('Searchable Text Fields', [
                'sr_type: text + keyword - Service type (analyzed for search + exact match)',
                'owner_department: text + keyword - Department name (searchable + exact)',
                'street_address: text - Full address (analyzed for full-text search)'
            ], 'green'),
            
            ('Categorical Fields', [
                'status: keyword - Request status (exact match filtering)',
                'city: keyword - City name (exact match)',
                'state: keyword - State code (exact match)', 
                'zip_code: keyword - ZIP code (exact match)'
            ], 'orange'),
            
            ('Temporal Fields', [
                'creation_date: date - Request creation timestamp (ISO 8601 format)',
                'completion_date: date - Request completion timestamp',
                'due_date: date - Expected completion date'
            ], 'purple'),
            
            ('Geographic Fields', [
                'location: geo_point - Geographic coordinates (lat, lon format)',
                'latitude: float - Latitude coordinate for calculations',
                'longitude: float - Longitude coordinate for calculations'
            ], 'teal'),
            
            ('Numeric Fields', [
                'ward: integer - Ward number for aggregations',
                'community_area: integer - Community area for grouping'
            ], 'pink'),
            
            ('Boolean Fields', [
                'duplicate: boolean - Duplicate request flag',
                'legacy_record: boolean - Legacy system record flag'
            ], 'lightgreen'),
            
            ('Metadata Fields', [
                '_indexed_at: date - Document indexing timestamp',
                'census_tract: keyword - Census tract identifier',
                'police_district: keyword - Police district code'
            ], 'lightblue')
        ]
        
        # Draw mappings with proper spacing
        y_pos = 20.5
        for category, field_list, color in mappings:
            # Category header
            ax.text(1, y_pos, category, fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            y_pos -= 0.8
            
            # Fields in category with proper spacing
            for field in field_list:
                ax.text(2, y_pos, field, fontsize=11, wrap=True)
                y_pos -= 0.6  # Proper spacing between fields
            
            y_pos -= 0.4  # Extra space between categories
        
        # Search capabilities section
        ax.text(1, y_pos, 'Search Capabilities', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
        y_pos -= 0.8
        
        capabilities = [
            '• Full-text search on sr_type and street_address fields',
            '• Geo-distance queries using location geo_point field',
            '• Date range filtering on temporal fields',
            '• Fuzzy matching for typo tolerance',
            '• Auto-completion and suggestions',
            '• Real-time aggregations (terms, date histogram, geo grid)',
            '• Boolean filtering on categorical fields',
            '• Multi-field search across different field types'
        ]
        
        for cap in capabilities:
            ax.text(2, y_pos, cap, fontsize=11)
            y_pos -= 0.6
        
        # Index settings
        y_pos -= 0.8
        ax.text(1, y_pos, 'Index Settings', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.5))
        y_pos -= 0.8
        
        settings = [
            '• Number of shards: 5 (distributed across cluster)',
            '• Number of replicas: 1 (redundancy)',
            '• Refresh interval: 1s (near real-time search)',
            '• Max result window: 10,000 documents',
            '• Default analyzer: standard (tokenization + lowercasing)',
            '• Doc values: enabled (for aggregations and sorting)'
        ]
        
        for setting in settings:
            ax.text(2, y_pos, setting, fontsize=11)
            y_pos -= 0.6
        
        # Example document
        y_pos -= 0.8
        ax.text(1, y_pos, 'Example Document', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.5))
        y_pos -= 0.8
        
        example = '''
{
  "_id": "SR23-00123456",
  "sr_type": "Pothole in Street",
  "status": "Open", 
  "creation_date": "2023-01-15T10:30:00Z",
  "location": {
    "lat": 41.881832,
    "lon": -87.623177
  },
  "ward": 42,
  "community_area": 8,
  "street_address": "123 N State St Chicago IL",
  "owner_department": "CDOT"
}
        '''
        
        ax.text(2, y_pos, example, fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('erdDiagram/elasticsearch_schema_chicago_311_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: elasticsearch_schema_chicago_311_12m_records.png")
    
    def generate_comparison_erd(self):
        """Generate clean comparison between MongoDB and Elasticsearch."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
        
        # MongoDB side
        ax1.set_xlim(0, 12)
        ax1.set_ylim(0, 16)
        ax1.axis('off')
        
        ax1.text(6, 15.5, 'MongoDB Approach', fontsize=16, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax1.text(6, 15, 'Document-Based Operational Database', fontsize=12, ha='center', style='italic')
        
        mongo_features = [
            ('Schema Flexibility', [
                '• Dynamic schema - add fields anytime',
                '• Nested documents and arrays supported',
                '• Rich data types (ObjectId, Date, GeoJSON)',
                '• No predefined structure required'
            ]),
            ('Query Capabilities', [
                '• MongoDB Query Language (MQL)',
                '• Complex aggregation pipelines',
                '• Geospatial queries with GeoJSON',
                '• Map-reduce operations'
            ]),
            ('Performance Features', [
                '• B-tree indexes for fast lookups',
                '• Compound indexes for complex queries',
                '• 2dsphere indexes for geospatial data',
                '• Covered queries using indexes only'
            ]),
            ('Scaling & Reliability', [
                '• Horizontal scaling via sharding',
                '• Replica sets for high availability',
                '• ACID transactions (single document)',
                '• Automatic failover capabilities'
            ])
        ]
        
        y_pos = 14
        for title, features in mongo_features:
            ax1.text(0.5, y_pos, title, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.5))
            y_pos -= 0.7
            
            for feature in features:
                ax1.text(1, y_pos, feature, fontsize=10)
                y_pos -= 0.5
            y_pos -= 0.3
        
        # Elasticsearch side
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, 16)
        ax2.axis('off')
        
        ax2.text(6, 15.5, 'Elasticsearch Approach', fontsize=16, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink', alpha=0.7))
        ax2.text(6, 15, 'Search-Optimized Analytics Engine', fontsize=12, ha='center', style='italic')
        
        es_features = [
            ('Schema Design', [
                '• Explicit field mappings required',
                '• Data type enforcement',
                '• Multi-field support (text + keyword)',
                '• Dynamic mapping available'
            ]),
            ('Search Capabilities', [
                '• Domain Specific Language (DSL)',
                '• Full-text search with analyzers',
                '• Fuzzy matching and suggestions',
                '• Geo-distance and bounding box queries'
            ]),
            ('Performance Features', [
                '• Inverted indexes for text search',
                '• Doc values for aggregations',
                '• Field data caching',
                '• Query result caching'
            ]),
            ('Analytics & Scaling', [
                '• Real-time aggregations',
                '• Distributed architecture',
                '• Horizontal scaling via sharding',
                '• Eventually consistent'
            ])
        ]
        
        y_pos = 14
        for title, features in es_features:
            ax2.text(0.5, y_pos, title, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.5))
            y_pos -= 0.7
            
            for feature in features:
                ax2.text(1, y_pos, feature, fontsize=10)
                y_pos -= 0.5
            y_pos -= 0.3
        
        plt.suptitle('MongoDB vs Elasticsearch: Architecture Comparison\nChicago 311 Service Requests - 12M+ Records', 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/mongodb_vs_elasticsearch_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: mongodb_vs_elasticsearch_comparison.png")
    
    def generate_data_flow_erd(self):
        """Generate data flow architecture diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title
        ax.text(10, 15.5, 'Chicago 311 Data Platform - Data Flow Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 15, '12+ Million Service Request Records Processing Pipeline', 
                fontsize=14, ha='center', style='italic')
        
        # Data source
        ax.add_patch(Rectangle((1, 12), 4, 2.5, facecolor='lightblue', edgecolor='blue', alpha=0.7))
        ax.text(3, 13.7, 'Chicago Data Portal', fontsize=12, fontweight='bold', ha='center')
        ax.text(3, 13.3, 'Socrata API', fontsize=10, ha='center')
        ax.text(3, 12.9, '12.4M+ Records', fontsize=10, ha='center')
        ax.text(3, 12.5, 'JSON Format', fontsize=10, ha='center')
        ax.text(3, 12.1, 'Real-time Updates', fontsize=10, ha='center')
        
        # ETL Process
        ax.add_patch(Rectangle((7, 12), 4, 2.5, facecolor='lightyellow', edgecolor='orange', alpha=0.7))
        ax.text(9, 13.7, 'ETL Pipeline', fontsize=12, fontweight='bold', ha='center')
        ax.text(9, 13.3, 'Data Processing', fontsize=10, ha='center')
        ax.text(9, 12.9, 'Validation & Cleaning', fontsize=10, ha='center')
        ax.text(9, 12.5, 'Format Conversion', fontsize=10, ha='center')
        ax.text(9, 12.1, 'Batch Processing', fontsize=10, ha='center')
        
        # Data validation
        ax.add_patch(Rectangle((13, 12), 4, 2.5, facecolor='lightgreen', edgecolor='green', alpha=0.7))
        ax.text(15, 13.7, 'Data Validation', fontsize=12, fontweight='bold', ha='center')
        ax.text(15, 13.3, 'Quality Control', fontsize=10, ha='center')
        ax.text(15, 12.9, 'Schema Validation', fontsize=10, ha='center')
        ax.text(15, 12.5, 'Coordinate Bounds', fontsize=10, ha='center')
        ax.text(15, 12.1, 'Duplicate Detection', fontsize=10, ha='center')
        
        # MongoDB storage
        ax.add_patch(Rectangle((2, 8), 6, 3, facecolor='lightcyan', edgecolor='teal', alpha=0.7))
        ax.text(5, 10.5, 'MongoDB Storage', fontsize=14, fontweight='bold', ha='center')
        ax.text(5, 10, 'service_requests Collection', fontsize=11, ha='center')
        
        mongo_details = [
            '• Document-based storage',
            '• GeoJSON Point coordinates', 
            '• Rich indexes (2dsphere, compound)',
            '• Aggregation pipelines',
            '• ACID transactions'
        ]
        
        y_pos = 9.5
        for detail in mongo_details:
            ax.text(2.2, y_pos, detail, fontsize=9)
            y_pos -= 0.25
        
        # Elasticsearch storage
        ax.add_patch(Rectangle((10, 8), 6, 3, facecolor='lavender', edgecolor='purple', alpha=0.7))
        ax.text(13, 10.5, 'Elasticsearch Storage', fontsize=14, fontweight='bold', ha='center')
        ax.text(13, 10, 'chicago_311_requests Index', fontsize=11, ha='center')
        
        es_details = [
            '• Search-optimized storage',
            '• Geo-point coordinates',
            '• Inverted indexes',
            '• Real-time aggregations', 
            '• Full-text search'
        ]
        
        y_pos = 9.5
        for detail in es_details:
            ax.text(10.2, y_pos, detail, fontsize=9)
            y_pos -= 0.25
        
        # Applications
        applications = [
            ('Analytics\nDashboard', 2, 'lightpink'),
            ('Search\nPortal', 5, 'lightgreen'), 
            ('GIS\nMapping', 8, 'lightyellow'),
            ('Mobile\nApp', 11, 'lightblue'),
            ('API\nEndpoints', 14, 'lightcoral')
        ]
        
        for app_name, x_pos, color in applications:
            ax.add_patch(Rectangle((x_pos, 4), 3, 1.5, facecolor=color, edgecolor='black', alpha=0.7))
            ax.text(x_pos + 1.5, 4.75, app_name, fontsize=11, fontweight='bold', ha='center')
        
        # Arrows - simplified and clear
        # Source to ETL
        ax.arrow(5, 13.25, 1.8, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        # ETL to Validation  
        ax.arrow(11, 13.25, 1.8, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        # Validation to MongoDB
        ax.arrow(14, 12, -3, -1, head_width=0.2, head_length=0.2, fc='green', ec='green')
        
        # Validation to Elasticsearch
        ax.arrow(16, 12, -1, -1, head_width=0.2, head_length=0.2, fc='purple', ec='purple')
        
        # Legend
        ax.text(10, 2.5, 'Data Flow Legend', fontsize=12, fontweight='bold', ha='center')
        ax.text(2, 2, 'Black arrows: ETL processing flow', fontsize=10)
        ax.text(2, 1.6, 'Green arrows: MongoDB data flow', fontsize=10, color='green')
        ax.text(2, 1.2, 'Purple arrows: Elasticsearch data flow', fontsize=10, color='purple')
        ax.text(2, 0.8, 'Total Records Processed: 12+ Million', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/data_flow_architecture_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: data_flow_architecture_12m_records.png")
    
    def generate_all_erds(self):
        """Generate all ERD diagrams."""
        print("🚀 Generating Final Clean ERD Diagrams...")
        print("📊 Proper spacing, no overlapping text")
        print("=" * 60)
        
        self.generate_mongodb_erd()
        self.generate_elasticsearch_erd()
        self.generate_comparison_erd()
        self.generate_data_flow_erd()
        
        print("=" * 60)
        print("🎉 ALL ERD DIAGRAMS GENERATED!")
        print("\n📁 ERD diagrams in erdDiagram/ folder:")
        print("   1. mongodb_schema_chicago_311_12m_records.png")
        print("   2. elasticsearch_schema_chicago_311_12m_records.png")
        print("   3. mongodb_vs_elasticsearch_comparison.png")
        print("   4. data_flow_architecture_12m_records.png")
        print("\n✅ Clean layouts with proper spacing!")
        print("✅ No overlapping text!")

if __name__ == "__main__":
    generator = FinalERDGenerator()
    generator.generate_all_erds()