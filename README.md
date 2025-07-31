# Chicago 311 Service Requests Data Platform

A high-performance data platform for analyzing Chicago's 311 service requests using MongoDB and Elasticsearch.

## 🚀 Features

- **Dual Database Implementation**: MongoDB for flexible document storage, Elasticsearch for lightning-fast search
- **Real-time Data Ingestion**: Extract and load 12.2M+ records from Chicago Data Portal
- **Advanced Analytics**: Comprehensive EDA with geospatial and temporal analysis
- **Performance Benchmarking**: Compare query performance between databases
- **Scalable Architecture**: Designed to handle large-scale city data

## 📊 Dataset

- **Source**: [Chicago 311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy/about_data)
- **Records**: 12.2M+ service requests
- **Updates**: Real-time via Chicago Data Portal API
- **Attributes**: 22 fields including location, timestamps, service types, and status

## 🛠️ Tech Stack

- **Databases**: MongoDB, Elasticsearch
- **Python**: 3.8+
- **Key Libraries**: pymongo, elasticsearch, pandas, sodapy
- **Visualization**: matplotlib, seaborn, plotly
- **Containerization**: Docker & Docker Compose

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- 16GB+ RAM (recommended for full dataset)
- 50GB+ free disk space

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/chicago-311-data-platform.git
cd chicago-311-data-platform
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt

# Make setup script executable
chmod +x setup.sh
```

### 3. Start Services
```bash
# Start MongoDB and Elasticsearch with Docker
docker-compose up -d

# Or run setup script for local installation
./setup.sh
```

###  4. Load Data
```bash
# Load sample data (1000 records)
python scripts/load_data.py --sample

# Load full dataset (12.2M records - takes 2-4 hours)
python scripts/load_data.py --full
```

### 5. Run Analysis
```bash
# Run performance benchmarks
python scripts/run_benchmarks.py

# Start Jupyter notebooks
jupyter notebook notebooks/
```

📈 Performance Results
----------------------

| Operation | MongoDB | Elasticsearch | Speedup |
| --- | --- | --- | --- |
| Simple Search | 245ms | 23ms | **10.7x** |
| Text Search | 1.2s | 45ms | **26.7x** |
| Geospatial Query | 890ms | 67ms | **13.3x** |
| Aggregations | 2.1s | 156ms | **13.5x** |


🏗️ Architecture
---------------
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chicago 311   │────│  Data Extractor  │────│   Data Loader   │
│   Data Portal   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                              ┌──────────────────────────┼──────────────────────────┐
                              │                          │                          │
                              ▼                          ▼                          ▼
                    ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
                    │    MongoDB      │        │  Elasticsearch  │        │   Analytics     │
                    │                 │        │                 │        │    Engine       │
                    │ • Document Store│        │ • Search Engine │        │                 │
                    │ • Geospatial    │        │ • Full-text     │        │ • EDA           │
                    │ • Aggregations  │        │ • Real-time     │        │ • Benchmarks    │
                    └─────────────────┘        └─────────────────┘        └─────────────────┘

```

📁 Project Structure
--------------------
```
chicago-311-data-platform/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
├── setup.sh
├── config/
│   ├── __init__.py
│   ├── database.py
│   └── elasticsearch.py
├── src/
│   ├── __init__.py
│   ├── data_extraction/
│   │   ├── __init__.py
│   │   ├── chicago_data_extractor.py
│   │   └── data_validator.py
│   ├── databases/
│   │   ├── __init__.py
│   │   ├── mongodb_handler.py
│   │   └── elasticsearch_handler.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py
│   │   └── visualizations.py
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── performance_tests.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── scripts/
│   ├── setup_mongodb.sh
│   ├── setup_elasticsearch.sh
│   ├── load_data.py
│   └── run_benchmarks.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_performance_analysis.ipynb
│   └── 03_visualization_dashboard.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_mongodb.py
│   ├── test_elasticsearch.py
│   └── test_data_extraction.py
├── docs/
│   ├── setup_guide.md
│   ├── api_documentation.md
│   └── performance_benchmarks.md
└── data/
    ├── sample/
    └── exports/
```

🔧 Configuration
----------------
### Environment Variables (.env)
```bash
# MongoDB Settings
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=chicago_311

# Elasticsearch Settings
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=chicago_311

# Chicago Data Portal
CHICAGO_API_TOKEN=your_token_here
BATCH_SIZE=10000
```

📊 Usage Examples
-----------------
### Quick Search Examples

```commandline
from src.databases.elasticsearch_handler import ElasticsearchHandler

es = ElasticsearchHandler()

# Search for potholes in ward 1
results = es.search({
    'text': 'pothole',
    'ward': 1,
    'status': 'Open'
})

# Geospatial search near downtown
results = es.geo_search(
    lat=41.8781,
    lon=-87.6298,
    radius='2km'
)
```
### MongoDB Aggregation Examples
```commandline
from src.databases.mongodb_handler import MongoDBHandler

mongo = MongoDBHandler()

# Get service requests by ward
ward_stats = mongo.aggregate_by_ward()

# Analyze response times
response_times = mongo.analyze_response_times()
```
### 🧪 Testing
```commandline
# Run all tests
python -m pytest tests/

# Test specific database
python -m pytest tests/test_mongodb.py
python -m pytest tests/test_elasticsearch.py

# Run with coverage
python -m pytest --cov=src tests/
```
📚 Documentation
----------------

-   [Setup Guide](docs/setup_guide.md) - Detailed installation instructions
-   [API Documentation](docs/api_documentation.md) - Code API reference
-   [Performance Benchmarks](docs/performance_benchmarks.md) - Detailed performance analysis

🤝 Contributing
---------------

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

📄 License
----------

This project is licensed under the MIT License - see the <LICENSE> file for details.

🙏 Acknowledgments
------------------

-   Chicago City Government for providing open data
-   Socrata for the Open Data API
-   MongoDB and Elasticsearch communities

📞 Contact
----------

Rahul Mishra - <adityaraj.pilot@hotmail.conm> 
Project Link: <https://github.com/adityarajmishra/chicago-311-data-platform>



