# MongoDB Data Models for Chicago 311 Service Requests

## Overview

This document outlines the MongoDB data models designed for the Chicago 311 Service Requests platform, including normalization strategies, schema design decisions, and justification for choosing MongoDB as a primary database solution.

## Original Data Structure

The Chicago 311 dataset from `https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy` contains the following key fields:

- **sr_number**: Unique service request identifier
- **sr_type**: Type of service request (e.g., "Pothole", "Street Light Out")
- **sr_short_code**: Abbreviated code for the service type
- **owner_department**: Department responsible for handling the request
- **status**: Current status (Open, Completed, Duplicate, etc.)
- **created_date**: When the request was created
- **last_modified_date**: When the request was last updated
- **closed_date**: When the request was closed (if applicable)
- **street_address**: Address where service is needed
- **city**: City (typically Chicago)
- **state**: State (typically IL)
- **zip_code**: ZIP code
- **street_number**: Street number
- **street_direction**: Street direction (N, S, E, W)
- **street_name**: Street name
- **street_type**: Street type (St, Ave, Blvd)
- **duplicate_ssr_number**: If duplicate, reference to original request
- **latitude**: Geographic latitude
- **longitude**: Geographic longitude
- **location**: Combined lat/long field
- **ward**: Chicago ward number
- **police_district**: Police district number
- **community_area**: Community area identifier
- **x_coordinate**: State plane coordinate
- **y_coordinate**: State plane coordinate

## MongoDB Normalization Strategy

### 1. Main Service Request Collection

```javascript
// Collection: service_requests
{
  _id: ObjectId("..."),
  sr_number: "SR24000001",
  sr_type_id: ObjectId("ref_to_service_types"),
  department_id: ObjectId("ref_to_departments"),
  status_id: ObjectId("ref_to_status_types"),
  
  // Temporal data
  dates: {
    created: ISODate("2024-01-15T10:30:00Z"),
    last_modified: ISODate("2024-01-16T14:20:00Z"),
    closed: ISODate("2024-01-17T16:45:00Z")  // null if not closed
  },
  
  // Location data (embedded for performance)
  location: {
    address: {
      street_number: "123",
      street_direction: "N",
      street_name: "State",
      street_type: "St",
      full_address: "123 N State St",
      city: "Chicago",
      state: "IL",
      zip_code: "60601"
    },
    coordinates: {
      type: "Point",
      coordinates: [-87.6298, 41.8781]  // [longitude, latitude]
    },
    state_plane: {
      x: 1176536.25,
      y: 1901234.75
    }
  },
  
  // Administrative boundaries (embedded for query performance)
  boundaries: {
    ward: 42,
    police_district: 18,
    community_area: 32
  },
  
  // Reference fields
  duplicate_of: "SR24000002",  // null if not duplicate
  
  // Metadata
  created_at: ISODate("2024-01-15T10:30:00Z"),
  updated_at: ISODate("2024-01-16T14:20:00Z")
}
```

### 2. Reference Collections (Normalized)

#### Service Types Collection
```javascript
// Collection: service_types
{
  _id: ObjectId("..."),
  name: "Pothole in Street",
  short_code: "PHI",
  category: "Streets & Sanitation",
  description: "Report potholes that need repair",
  priority_level: "Medium",
  avg_completion_days: 7,
  active: true,
  created_at: ISODate("2024-01-01T00:00:00Z")
}
```

#### Departments Collection
```javascript
// Collection: departments
{
  _id: ObjectId("..."),
  name: "Streets & Sanitation",
  short_code: "DSS",
  contact_info: {
    phone: "311",
    email: "streets@cityofchicago.org"
  },
  active: true,
  created_at: ISODate("2024-01-01T00:00:00Z")
}
```

#### Status Types Collection
```javascript
// Collection: status_types
{
  _id: ObjectId("..."),
  name: "Completed",
  description: "Service request has been completed",
  is_final: true,
  display_order: 3,
  active: true,
  created_at: ISODate("2024-01-01T00:00:00Z")
}
```

### 3. Supporting Collections

#### Location Index Collection (for geospatial optimization)
```javascript
// Collection: location_index
{
  _id: ObjectId("..."),
  sr_number: "SR24000001",
  location: {
    type: "Point",
    coordinates: [-87.6298, 41.8781]
  },
  ward: 42,
  community_area: 32,
  created_at: ISODate("2024-01-15T10:30:00Z")
}
```

#### Analytics Summary Collection (pre-aggregated data)
```javascript
// Collection: analytics_summary
{
  _id: ObjectId("..."),
  date: ISODate("2024-01-15T00:00:00Z"),
  ward: 42,
  summary: {
    total_requests: 145,
    completed_requests: 98,
    open_requests: 47,
    avg_completion_time_hours: 72.5,
    top_service_types: [
      { type: "Pothole", count: 45 },
      { type: "Street Light", count: 32 }
    ]
  },
  created_at: ISODate("2024-01-16T02:00:00Z")
}
```

## Indexing Strategy

### Primary Indexes
```javascript
// Compound index for common queries
db.service_requests.createIndex({ 
  "dates.created": -1, 
  "status_id": 1, 
  "sr_type_id": 1 
})

// Geospatial index for location-based queries
db.service_requests.createIndex({ 
  "location.coordinates": "2dsphere" 
})

// Text index for search functionality
db.service_requests.createIndex({
  "sr_number": "text",
  "location.address.full_address": "text"
})

// Ward-based queries
db.service_requests.createIndex({ 
  "boundaries.ward": 1, 
  "dates.created": -1 
})

// Status and date queries
db.service_requests.createIndex({ 
  "status_id": 1, 
  "dates.created": -1 
})
```

## Design Decisions & Justification

### 1. Hybrid Normalization Approach

**Decision**: Use a hybrid approach with normalized reference data and embedded location/boundary data.

**Rationale**:
- **Reference data normalization**: Service types, departments, and status types change infrequently but are referenced often. Normalizing prevents data duplication and ensures consistency.
- **Location embedding**: Address and coordinate data is accessed with almost every query and rarely changes independently, making embedding optimal for read performance.
- **Boundary embedding**: Ward, police district, and community area data is frequently queried together and rarely changes.

### 2. Geospatial Data Structure

**Decision**: Use GeoJSON Point format for coordinates with 2dsphere index.

**Rationale**:
- MongoDB's native geospatial capabilities with 2dsphere indexes
- Enables efficient radius, polygon, and proximity queries
- Supports both simple distance queries and complex geospatial analytics

### 3. Date Handling Strategy

**Decision**: Embed all related dates in a `dates` subdocument using ISODate format.

**Rationale**:
- Groups related temporal data logically
- Enables efficient date range queries
- Supports MongoDB's native date operations and aggregation pipeline

### 4. Pre-aggregated Analytics

**Decision**: Maintain separate analytics summary collections for common aggregations.

**Rationale**:
- With 12.3M+ records, real-time aggregations can be slow
- Pre-computed daily/weekly summaries enable instant dashboard responses
- Reduces load on primary collection during peak usage

## Why MongoDB for Chicago 311 Data?

### 1. **Document-Oriented Nature**
- 311 service requests have variable fields depending on service type
- MongoDB's flexible schema accommodates different request types without complex joins
- Nested address and location data fits naturally in document structure

### 2. **Geospatial Capabilities**
- Native geospatial indexing and queries (2dsphere)
- Efficient radius searches for "find nearby requests"
- Integration with GIS tools and mapping services

### 3. **Scalability**
- Horizontal scaling capabilities for growing dataset (12.3M+ records)
- Replica sets provide high availability for 24/7 city services
- Sharding potential based on ward or date ranges

### 4. **Performance Benefits**
- Single document reads eliminate complex joins
- Embedded data reduces network round trips
- Compound indexes support complex query patterns

### 5. **Analytics Integration**
- MongoDB Aggregation Pipeline for complex analytics
- MapReduce capabilities for large-scale data processing
- Time-series collections for temporal analysis

### 6. **Operational Advantages**
- JSON-native format matches API responses
- Flexible schema evolution as city processes change
- Rich query language for complex filtering

## Performance Considerations

### Read Optimization
- **Embedded frequently-accessed data** (location, boundaries)
- **Compound indexes** for common query patterns
- **Read replicas** for analytics and reporting workloads

### Write Optimization
- **Normalized reference data** to minimize update operations
- **Bulk operations** for data imports and updates
- **Write concerns** appropriate for data criticality

### Storage Optimization
- **Index selectivity** to minimize storage overhead
- **Compression** for historical data
- **TTL indexes** for temporary or log data

## Migration and ETL Strategy

### Initial Data Load
1. **Extract** from Chicago Data Portal API
2. **Transform** flat records into normalized document structure
3. **Load** with proper indexing and validation

### Incremental Updates
1. **Stream** updates from API using change detection
2. **Upsert** operations to handle updates and new records
3. **Validation** to ensure data integrity

### Data Validation
1. **Schema validation** using MongoDB JSON Schema
2. **Referential integrity** checks for normalized references
3. **Geospatial validation** for coordinate accuracy

## Conclusion

This MongoDB data model provides an optimal balance of normalization and performance for the Chicago 311 dataset. The hybrid approach leverages MongoDB's strengths while maintaining data integrity and query performance at scale. The design supports both operational queries and analytical workloads, making it ideal for a civic data platform serving multiple stakeholders.