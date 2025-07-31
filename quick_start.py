#!/usr/bin/env python3
"""
Quick Start Script for Chicago 311 Data Pipeline
This script helps you get started quickly with the data pipeline
"""
import os
import sys
import subprocess
import time

def print_banner():
    """Print welcome banner."""
    print("="*70)
    print("🏙️  CHICAGO 311 DATA PLATFORM - QUICK START")
    print("="*70)
    print("Welcome! This script will help you set up and run the data pipeline.")
    print()

def check_requirements():
    """Check if required packages are installed."""
    print("📋 Checking requirements...")
    try:
        import pandas
        import pymongo
        import elasticsearch
        import sodapy
        import requests
        import tqdm
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Please run: pip install -r requirements.txt")
        return False

def check_services():
    """Check if MongoDB and Elasticsearch are running."""
    print("\n🔍 Checking services...")
    
    # Check MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        client.server_info()
        print("✅ MongoDB is running")
        mongo_ok = True
    except Exception:
        print("❌ MongoDB is not running")
        print("   Please start MongoDB with: docker-compose up -d mongodb")
        mongo_ok = False
    
    # Check Elasticsearch
    try:
        import requests
        response = requests.get('http://localhost:9200', timeout=2)
        if response.status_code == 200:
            print("✅ Elasticsearch is running")
            es_ok = True
        else:
            es_ok = False
    except Exception:
        print("❌ Elasticsearch is not running")
        print("   Please start Elasticsearch with: docker-compose up -d elasticsearch")
        es_ok = False
    
    return mongo_ok and es_ok

def start_services():
    """Start services using docker-compose."""
    print("\n🚀 Starting services with Docker Compose...")
    try:
        # Check if docker-compose.yml exists
        if not os.path.exists('docker-compose.yml'):
            print("❌ docker-compose.yml not found in current directory")
            return False
        
        # Start services
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        print("✅ Services started successfully")
        
        # Wait for services to be ready
        print("⏱️  Waiting for services to be ready...")
        time.sleep(10)
        
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start services")
        return False
    except FileNotFoundError:
        print("❌ Docker Compose not found. Please install Docker and Docker Compose")
        return False

def run_test():
    """Run the test pipeline."""
    print("\n🧪 Running pipeline test...")
    try:
        result = subprocess.run([sys.executable, 'scripts/test_pipeline.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pipeline test passed!")
            print(result.stdout)
            return True
        else:
            print("❌ Pipeline test failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def run_sample():
    """Run sample data pipeline."""
    print("\n📊 Running sample data pipeline (100 records)...")
    try:
        result = subprocess.run([
            sys.executable, 'scripts/data_pipeline.py', 
            '--mode', 'sample', '--sample-size', '100'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample pipeline completed!")
            print(result.stdout)
            return True
        else:
            print("❌ Sample pipeline failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running sample pipeline: {e}")
        return False

def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 SETUP COMPLETE!")
    print("="*50)
    print("Your Chicago 311 data pipeline is ready to use!")
    print()
    print("📚 Next steps:")
    print("1. Run a larger sample:")
    print("   python scripts/data_pipeline.py --mode sample --sample-size 5000")
    print()
    print("2. Run the full pipeline (processes all data):")
    print("   python scripts/data_pipeline.py --mode full")
    print()
    print("3. Run incremental updates:")
    print("   python scripts/data_pipeline.py --mode incremental --start-date 2024-01-01 --end-date 2024-01-31")
    print()
    print("4. Check database statistics:")
    print("   python scripts/data_pipeline.py --stats-only")
    print()
    print("5. Explore the data in Jupyter notebooks:")
    print("   jupyter notebook notebooks/")
    print()
    print("6. Access Kibana dashboard:")
    print("   http://localhost:5601")
    print()
    print("💡 Tip: Check the logs in 'pipeline.log' for detailed information")

def main():
    """Main function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check if services are running
    if not check_services():
        print("\n🔧 Services are not running. Would you like to start them? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            if not start_services():
                return 1
            # Re-check services
            if not check_services():
                print("❌ Services still not available after starting")
                return 1
        else:
            print("❌ Services are required to run the pipeline")
            return 1
    
    # Run test
    if not run_test():
        print("❌ Pipeline test failed. Please check the configuration")
        return 1
    
    # Run sample
    if not run_sample():
        print("❌ Sample pipeline failed")
        return 1
    
    # Show next steps
    show_next_steps()
    return 0

if __name__ == "__main__":
    exit(main())