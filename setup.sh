#!/bin/bash

echo "🚀 Setting up Chicago 311 Data Platform..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv chicago_311_env
source chicago_311_env/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Installing databases locally..."
    
    # Setup MongoDB locally
    echo "🍃 Setting up MongoDB..."
    ./scripts/setup_mongodb.sh
    
    # Setup Elasticsearch locally
    echo "🔍 Setting up Elasticsearch..."
    ./scripts/setup_elasticsearch.sh
else
    echo "🐳 Docker found. Starting services with Docker Compose..."
    docker-compose up -d
    
    # Wait for services to start
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Check service health
    echo "🏥 Checking service health..."
    ./scripts/check_services.sh
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/sample
mkdir -p data/exports
mkdir -p logs

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️  Creating environment file..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: python scripts/load_data.py --sample"
echo "3. Run: python scripts/run_benchmarks.py"
echo ""
echo "🎉 Happy analyzing!"