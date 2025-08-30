#!/usr/bin/env python3
"""
Fixed ERD Generator - Resolves overlapping text and missing arrows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

class FixedERDGenerator:
    """Generate properly formatted ERD diagrams with no overlapping issues."""
    
    def __init__(self):
        """Initialize ERD generator."""
        os.makedirs('erdDiagram', exist_ok=True)
        
    def generate_mongodb_erd(self):
        """Generate MongoDB ERD with fixed spacing."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 28))  # Increased height further
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 28)
        ax.axis('off')
        
        # Title
        ax.text(10, 27, 'MongoDB Schema - Chicago 311 Service Requests', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 26.3, 'Document-Based Schema for 12M+ Records', 
                fontsize=14, ha='center', style='italic')
        
        # Collection info
        ax.add_patch(Rectangle((3, 25), 14, 1, facecolor='lightblue', edgecolor='blue', alpha=0.7))
        ax.text(10, 25.5, 'Collection: service_requests', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Field definitions with more spacing
        fields = [
            ('Primary Key', [
                'sr_number: String [PRIMARY KEY] - Unique service request identifier'
            ], '#FFE6E6'),
            
            ('Request Information', [
                'sr_type: String [INDEXED] - Type of service request (e.g., Pothole, Graffiti)',
                'owner_department: String [INDEXED] - Department responsible for request',
                'status: String [INDEXED] - Current status (Open, Closed, In Progress)',
                'sr_short_code: String - Abbreviated code for request type'
            ], '#E6E6FF'),
            
            ('Temporal Data', [
                'creation_date: DateTime [INDEXED] - When request was submitted',
                'completion_date: DateTime - When request was completed',
                'due_date: DateTime - Expected completion date'
            ], '#E6FFE6'),
            
            ('Geographic Information', [
                'location: GeoJSON Point [2DSPHERE INDEX] - Geographic coordinates',
                'latitude: Float [GEO] - Latitude coordinate (-90 to 90)',
                'longitude: Float [GEO] - Longitude coordinate (-180 to 180)'
            ], '#E6FFFF'),
            
            ('Address Details', [
                'street_address: String [TEXT SEARCH] - Full street address',
                'city: String - City name (typically Chicago)',
                'state: String - State abbreviation (IL)',
                'zip_code: String [INDEXED] - ZIP postal code'
            ], '#FFFEE6'),
            
            ('Administrative Areas', [
                'ward: Integer [INDEXED] - Ward number (1-50)',
                'community_area: Integer [INDEXED] - Community area (1-77)',
                'police_district: String - Police district identifier',
                'census_tract: String - Census tract number'
            ], '#FFE6FF'),
            
            ('Metadata', [
                'duplicate: Boolean - Flag indicating duplicate request',
                'legacy_record: Boolean - Flag for legacy system records',
                '_processed_at: DateTime - Processing timestamp'
            ], '#F0FFF0')
        ]
        
        # Draw fields with proper spacing
        y_pos = 23.5
        for category, field_list, color in fields:
            # Category header with background
            ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor=color, edgecolor='gray', alpha=0.8))
            ax.text(2, y_pos, category, fontsize=14, fontweight='bold')
            y_pos -= 1.0  # Space after header
            
            # Fields in category with proper spacing
            for field in field_list:
                ax.text(2.5, y_pos, field, fontsize=11)
                y_pos -= 0.8  # More spacing between fields
            
            y_pos -= 0.5  # Extra space between categories
        
        # Indexes section with background
        y_pos -= 0.5
        ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor='#FFFFCC', edgecolor='orange', alpha=0.8))
        ax.text(2, y_pos, 'MongoDB Indexes', fontsize=14, fontweight='bold')
        y_pos -= 1.0
        
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
            ax.text(2.5, y_pos, idx, fontsize=11)
            y_pos -= 0.8
        
        # Example document with clear separation
        y_pos -= 1.0  # Extra space before example
        ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor='#F0F8FF', edgecolor='blue', alpha=0.8))
        ax.text(2, y_pos, 'Example Document Structure', fontsize=14, fontweight='bold')
        y_pos -= 1.0
        
        # Example in a clearly separated box
        example_box_height = 8
        ax.add_patch(Rectangle((2, y_pos - example_box_height), 16, example_box_height, 
                              facecolor='white', edgecolor='black', alpha=0.9))
        
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
  "owner_department": "CDOT",
  "duplicate": false,
  "_processed_at": ISODate("2023-01-15T10:35:00Z")
}'''
        
        ax.text(2.5, y_pos - 0.5, example, fontsize=10, fontfamily='monospace', va='top')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/mongodb_schema_chicago_311_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: mongodb_schema_chicago_311_12m_records.png")
    
    def generate_elasticsearch_erd(self):
        """Generate Elasticsearch ERD with fixed spacing."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 28))  # Increased height
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 28)
        ax.axis('off')
        
        # Title
        ax.text(10, 27, 'Elasticsearch Schema - Chicago 311 Service Requests', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 26.3, 'Search-Optimized Schema for 12M+ Records', 
                fontsize=14, ha='center', style='italic')
        
        # Index info
        ax.add_patch(Rectangle((3, 25), 14, 1, facecolor='lightpink', edgecolor='purple', alpha=0.7))
        ax.text(10, 25.5, 'Index: chicago_311_requests', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Field mappings with proper spacing
        mappings = [
            ('Document Identifiers', [
                'sr_number: keyword - Unique service request number (exact match only)'
            ], '#FFE6E6'),
            
            ('Searchable Text Fields', [
                'sr_type: text + keyword - Service type (analyzed for search + exact match)',
                'owner_department: text + keyword - Department name (searchable + exact)',
                'street_address: text - Full address (analyzed for full-text search)'
            ], '#E6FFE6'),
            
            ('Categorical Fields', [
                'status: keyword - Request status (exact match filtering)',
                'city: keyword - City name (exact match)',
                'state: keyword - State code (exact match)', 
                'zip_code: keyword - ZIP code (exact match)'
            ], '#FFFEE6'),
            
            ('Temporal Fields', [
                'creation_date: date - Request creation timestamp (ISO 8601 format)',
                'completion_date: date - Request completion timestamp',
                'due_date: date - Expected completion date'
            ], '#E6E6FF'),
            
            ('Geographic Fields', [
                'location: geo_point - Geographic coordinates (lat, lon format)',
                'latitude: float - Latitude coordinate for calculations',
                'longitude: float - Longitude coordinate for calculations'
            ], '#E6FFFF'),
            
            ('Numeric Fields', [
                'ward: integer - Ward number for aggregations',
                'community_area: integer - Community area for grouping'
            ], '#FFE6FF'),
            
            ('Boolean Fields', [
                'duplicate: boolean - Duplicate request flag',
                'legacy_record: boolean - Legacy system record flag'
            ], '#F0FFF0'),
            
            ('Metadata Fields', [
                '_indexed_at: date - Document indexing timestamp',
                'census_tract: keyword - Census tract identifier',
                'police_district: keyword - Police district code'
            ], '#F0F8FF')
        ]
        
        # Draw mappings with proper spacing
        y_pos = 23.5
        for category, field_list, color in mappings:
            # Category header with background
            ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor=color, edgecolor='gray', alpha=0.8))
            ax.text(2, y_pos, category, fontsize=14, fontweight='bold')
            y_pos -= 1.0  # Space after header
            
            # Fields in category with proper spacing
            for field in field_list:
                ax.text(2.5, y_pos, field, fontsize=11)
                y_pos -= 0.8  # More spacing between fields
            
            y_pos -= 0.5  # Extra space between categories
        
        # Search capabilities section
        y_pos -= 0.5
        ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor='#FFFFCC', edgecolor='orange', alpha=0.8))
        ax.text(2, y_pos, 'Search Capabilities', fontsize=14, fontweight='bold')
        y_pos -= 1.0
        
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
            ax.text(2.5, y_pos, cap, fontsize=11)
            y_pos -= 0.8
        
        # Index settings
        y_pos -= 1.0
        ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor='#F0FFFF', edgecolor='teal', alpha=0.8))
        ax.text(2, y_pos, 'Index Settings', fontsize=14, fontweight='bold')
        y_pos -= 1.0
        
        settings = [
            '• Number of shards: 5 (distributed across cluster)',
            '• Number of replicas: 1 (redundancy)',
            '• Refresh interval: 1s (near real-time search)',
            '• Max result window: 10,000 documents',
            '• Default analyzer: standard (tokenization + lowercasing)',
            '• Doc values: enabled (for aggregations and sorting)'
        ]
        
        for setting in settings:
            ax.text(2.5, y_pos, setting, fontsize=11)
            y_pos -= 0.8
        
        # Example document with clear separation
        y_pos -= 1.0  # Extra space before example
        ax.add_patch(Rectangle((1, y_pos - 0.3), 18, 0.6, facecolor='#FFFACD', edgecolor='gold', alpha=0.8))
        ax.text(2, y_pos, 'Example Document', fontsize=14, fontweight='bold')
        y_pos -= 1.0
        
        # Example in a clearly separated box
        example_box_height = 7
        ax.add_patch(Rectangle((2, y_pos - example_box_height), 16, example_box_height, 
                              facecolor='white', edgecolor='black', alpha=0.9))
        
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
  "owner_department": "CDOT",
  "duplicate": false,
  "_indexed_at": "2023-01-15T10:35:00Z"
}'''
        
        ax.text(2.5, y_pos - 0.5, example, fontsize=10, fontfamily='monospace', va='top')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/elasticsearch_schema_chicago_311_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: elasticsearch_schema_chicago_311_12m_records.png")
    
    def generate_comparison_erd(self):
        """Generate clean comparison between MongoDB and Elasticsearch."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 18))
        
        # MongoDB side
        ax1.set_xlim(0, 12)
        ax1.set_ylim(0, 18)
        ax1.axis('off')
        
        ax1.add_patch(Rectangle((1, 16.5), 10, 1.2, facecolor='lightblue', edgecolor='blue', alpha=0.8))
        ax1.text(6, 17.1, 'MongoDB Approach', fontsize=16, fontweight='bold', ha='center')
        ax1.text(6, 16.7, 'Document-Based Operational Database', fontsize=12, ha='center', style='italic')
        
        mongo_features = [
            ('Schema Flexibility', [
                '• Dynamic schema - add fields anytime',
                '• Nested documents and arrays supported',
                '• Rich data types (ObjectId, Date, GeoJSON)',
                '• No predefined structure required'
            ], '#E6F3FF'),
            ('Query Capabilities', [
                '• MongoDB Query Language (MQL)',
                '• Complex aggregation pipelines',
                '• Geospatial queries with GeoJSON',
                '• Map-reduce operations'
            ], '#E6FFE6'),
            ('Performance Features', [
                '• B-tree indexes for fast lookups',
                '• Compound indexes for complex queries',
                '• 2dsphere indexes for geospatial data',
                '• Covered queries using indexes only'
            ], '#FFF9E6'),
            ('Scaling & Reliability', [
                '• Horizontal scaling via sharding',
                '• Replica sets for high availability',
                '• ACID transactions (single document)',
                '• Automatic failover capabilities'
            ], '#FFE6F3')
        ]
        
        y_pos = 15.5
        for title, features, color in mongo_features:
            # Section header
            ax1.add_patch(Rectangle((1.5, y_pos - 0.2), 9, 0.4, facecolor=color, edgecolor='gray', alpha=0.8))
            ax1.text(2, y_pos, title, fontsize=12, fontweight='bold')
            y_pos -= 0.8
            
            for feature in features:
                ax1.text(2.2, y_pos, feature, fontsize=10)
                y_pos -= 0.5
            y_pos -= 0.3
        
        # Elasticsearch side
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, 18)
        ax2.axis('off')
        
        ax2.add_patch(Rectangle((1, 16.5), 10, 1.2, facecolor='lightpink', edgecolor='purple', alpha=0.8))
        ax2.text(6, 17.1, 'Elasticsearch Approach', fontsize=16, fontweight='bold', ha='center')
        ax2.text(6, 16.7, 'Search-Optimized Analytics Engine', fontsize=12, ha='center', style='italic')
        
        es_features = [
            ('Schema Design', [
                '• Explicit field mappings required',
                '• Data type enforcement',
                '• Multi-field support (text + keyword)',
                '• Dynamic mapping available'
            ], '#FFE6E6'),
            ('Search Capabilities', [
                '• Domain Specific Language (DSL)',
                '• Full-text search with analyzers',
                '• Fuzzy matching and suggestions',
                '• Geo-distance and bounding box queries'
            ], '#E6FFE6'),
            ('Performance Features', [
                '• Inverted indexes for text search',
                '• Doc values for aggregations',
                '• Field data caching',
                '• Query result caching'
            ], '#FFFEE6'),
            ('Analytics & Scaling', [
                '• Real-time aggregations',
                '• Distributed architecture',
                '• Horizontal scaling via sharding',
                '• Eventually consistent'
            ], '#F0E6FF')
        ]
        
        y_pos = 15.5
        for title, features, color in es_features:
            # Section header
            ax2.add_patch(Rectangle((1.5, y_pos - 0.2), 9, 0.4, facecolor=color, edgecolor='gray', alpha=0.8))
            ax2.text(2, y_pos, title, fontsize=12, fontweight='bold')
            y_pos -= 0.8
            
            for feature in features:
                ax2.text(2.2, y_pos, feature, fontsize=10)
                y_pos -= 0.5
            y_pos -= 0.3
        
        plt.suptitle('MongoDB vs Elasticsearch: Architecture Comparison\nChicago 311 Service Requests - 12M+ Records', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        plt.savefig('erdDiagram/mongodb_vs_elasticsearch_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: mongodb_vs_elasticsearch_comparison.png")
    
    def generate_data_flow_erd(self):
        """Generate data flow architecture diagram with proper arrows."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 16)
        ax.axis('off')
        
        # Title
        ax.add_patch(Rectangle((2, 14.5), 16, 1.2, facecolor='lightgreen', edgecolor='darkgreen', alpha=0.8))
        ax.text(10, 15.3, 'Chicago 311 Data Platform - Data Flow Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        ax.text(10, 14.8, '12+ Million Service Request Records Processing Pipeline', 
                fontsize=14, ha='center', style='italic')
        
        # Data source
        ax.add_patch(Rectangle((1, 11.5), 4, 2.5, facecolor='lightblue', edgecolor='blue', linewidth=2))
        ax.text(3, 13.5, 'Chicago Data Portal', fontsize=12, fontweight='bold', ha='center')
        ax.text(3, 13.1, 'Socrata API', fontsize=10, ha='center')
        ax.text(3, 12.7, '12.4M+ Records', fontsize=10, ha='center')
        ax.text(3, 12.3, 'JSON Format', fontsize=10, ha='center')
        ax.text(3, 11.9, 'Real-time Updates', fontsize=10, ha='center')
        
        # ETL Process
        ax.add_patch(Rectangle((7, 11.5), 4, 2.5, facecolor='lightyellow', edgecolor='orange', linewidth=2))
        ax.text(9, 13.5, 'ETL Pipeline', fontsize=12, fontweight='bold', ha='center')
        ax.text(9, 13.1, 'Data Processing', fontsize=10, ha='center')
        ax.text(9, 12.7, 'Validation & Cleaning', fontsize=10, ha='center')
        ax.text(9, 12.3, 'Format Conversion', fontsize=10, ha='center')
        ax.text(9, 11.9, 'Batch Processing', fontsize=10, ha='center')
        
        # Data validation
        ax.add_patch(Rectangle((13, 11.5), 4, 2.5, facecolor='lightgreen', edgecolor='green', linewidth=2))
        ax.text(15, 13.5, 'Data Validation', fontsize=12, fontweight='bold', ha='center')
        ax.text(15, 13.1, 'Quality Control', fontsize=10, ha='center')
        ax.text(15, 12.7, 'Schema Validation', fontsize=10, ha='center')
        ax.text(15, 12.3, 'Coordinate Bounds', fontsize=10, ha='center')
        ax.text(15, 11.9, 'Duplicate Detection', fontsize=10, ha='center')
        
        # MongoDB storage
        ax.add_patch(Rectangle((2, 7.5), 6, 3, facecolor='lightcyan', edgecolor='teal', linewidth=2))
        ax.text(5, 9.8, 'MongoDB Storage', fontsize=14, fontweight='bold', ha='center')
        ax.text(5, 9.3, 'service_requests Collection', fontsize=11, ha='center')
        
        mongo_details = [
            '• Document-based storage',
            '• GeoJSON Point coordinates', 
            '• Rich indexes (2dsphere, compound)',
            '• Aggregation pipelines',
            '• ACID transactions'
        ]
        
        y_pos = 8.8
        for detail in mongo_details:
            ax.text(2.2, y_pos, detail, fontsize=9)
            y_pos -= 0.25
        
        # Elasticsearch storage
        ax.add_patch(Rectangle((10, 7.5), 6, 3, facecolor='lavender', edgecolor='purple', linewidth=2))
        ax.text(13, 9.8, 'Elasticsearch Storage', fontsize=14, fontweight='bold', ha='center')
        ax.text(13, 9.3, 'chicago_311_requests Index', fontsize=11, ha='center')
        
        es_details = [
            '• Search-optimized storage',
            '• Geo-point coordinates',
            '• Inverted indexes',
            '• Real-time aggregations', 
            '• Full-text search'
        ]
        
        y_pos = 8.8
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
            ax.add_patch(Rectangle((x_pos, 4), 3, 1.5, facecolor=color, edgecolor='black', linewidth=1))
            ax.text(x_pos + 1.5, 4.75, app_name, fontsize=11, fontweight='bold', ha='center')
        
        # Add proper arrows using FancyArrowPatch
        # Source to ETL
        arrow1 = FancyArrowPatch((5, 12.75), (7, 12.75), 
                                arrowstyle='->', mutation_scale=20, 
                                color='black', linewidth=3)
        ax.add_patch(arrow1)
        
        # ETL to Validation  
        arrow2 = FancyArrowPatch((11, 12.75), (13, 12.75), 
                                arrowstyle='->', mutation_scale=20, 
                                color='black', linewidth=3)
        ax.add_patch(arrow2)
        
        # Validation to MongoDB
        arrow3 = FancyArrowPatch((14, 11.5), (6, 10.5), 
                                arrowstyle='->', mutation_scale=20, 
                                color='teal', linewidth=3)
        ax.add_patch(arrow3)
        
        # Validation to Elasticsearch
        arrow4 = FancyArrowPatch((16, 11.5), (12, 10.5), 
                                arrowstyle='->', mutation_scale=20, 
                                color='purple', linewidth=3)
        ax.add_patch(arrow4)
        
        # MongoDB to applications (first 3)
        for i in range(3):
            app_x = applications[i][1] + 1.5
            arrow = FancyArrowPatch((5, 7.5), (app_x, 5.5), 
                                   arrowstyle='->', mutation_scale=15, 
                                   color='teal', linewidth=2, alpha=0.7)
            ax.add_patch(arrow)
        
        # Elasticsearch to applications (last 2)
        for i in range(3, 5):
            app_x = applications[i][1] + 1.5
            arrow = FancyArrowPatch((13, 7.5), (app_x, 5.5), 
                                   arrowstyle='->', mutation_scale=15, 
                                   color='purple', linewidth=2, alpha=0.7)
            ax.add_patch(arrow)
        
        # Legend with background
        ax.add_patch(Rectangle((1, 0.5), 18, 2.5, facecolor='#F5F5F5', edgecolor='gray', alpha=0.9))
        ax.text(10, 2.7, 'Data Flow Legend & Statistics', fontsize=12, fontweight='bold', ha='center')
        
        # Legend items arranged clearly
        legend_items = [
            ('Black arrows: ETL processing flow', 2, 2.2, 'black'),
            ('Teal arrows: MongoDB data connections', 2, 1.8, 'teal'),
            ('Purple arrows: Elasticsearch data connections', 2, 1.4, 'purple'),
            ('Total Records Processed: 12+ Million', 2, 1.0, 'black'),
            ('Architecture: Dual storage approach', 11, 2.2, 'black'),
            ('Performance: Real-time updates', 11, 1.8, 'black'),
            ('Scalability: Distributed processing', 11, 1.4, 'black'),
            ('Applications: 5 consumer applications', 11, 1.0, 'black')
        ]
        
        for text, x, y, color in legend_items:
            ax.text(x, y, text, fontsize=10, color=color, fontweight='bold' if 'Total' in text else 'normal')
        
        plt.tight_layout()
        plt.savefig('erdDiagram/data_flow_architecture_12m_records.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: data_flow_architecture_12m_records.png")
    
    def generate_all_erds(self):
        """Generate all fixed ERD diagrams."""
        print("🚀 Generating Fixed ERD Diagrams...")
        print("📊 Fixed overlapping text and missing arrows")
        print("=" * 60)
        
        self.generate_mongodb_erd()
        self.generate_elasticsearch_erd()
        self.generate_comparison_erd()
        self.generate_data_flow_erd()
        
        print("=" * 60)
        print("🎉 ALL FIXED ERD DIAGRAMS GENERATED!")
        print("\n📁 ERD diagrams in erdDiagram/ folder:")
        print("   1. mongodb_schema_chicago_311_12m_records.png")
        print("   2. elasticsearch_schema_chicago_311_12m_records.png")
        print("   3. mongodb_vs_elasticsearch_comparison.png")
        print("   4. data_flow_architecture_12m_records.png")
        print("\n✅ Fixed overlapping example text!")
        print("✅ Added proper arrows to data flow!")
        print("✅ Professional spacing throughout!")

if __name__ == "__main__":
    generator = FixedERDGenerator()
    generator.generate_all_erds()