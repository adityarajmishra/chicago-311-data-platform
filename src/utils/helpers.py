"""Utility functions and helpers."""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime
import pandas as pd

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_dir / log_file))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(log_dir / f"chicago_311_{timestamp}.log"))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_config(config_file: str = ".env") -> Dict[str, str]:
    """Load configuration from file."""
    config = {}
    config_path = Path(config_file)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    return config

def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """Save data as JSON file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Any]:
    """Load data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return None

def progress_bar(current: int, total: int, width: int = 50) -> str:
    """Generate a simple progress bar string."""
    if total == 0:
        percentage = 0
    else:
        percentage = current / total
    
    filled = int(width * percentage)
    bar = '=' * filled + '-' * (width - filled)
    
    return f"[{bar}] {current}/{total} ({percentage:.1%})"

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def validate_chicago_coordinates(lat: float, lon: float) -> bool:
    """Validate if coordinates are within Chicago area."""
    # Chicago boundaries (approximate)
    chicago_bounds = {
        'lat_min': 41.6,
        'lat_max': 42.1,
        'lon_min': -87.9,
        'lon_max': -87.5
    }
    
    return (chicago_bounds['lat_min'] <= lat <= chicago_bounds['lat_max'] and
            chicago_bounds['lon_min'] <= lon <= chicago_bounds['lon_max'])

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Basic text cleaning
    text = text.strip()
    text = ' '.join(text.split())  # Remove extra whitespace
    
    return text

def get_chicago_ward_info() -> Dict[int, str]:
    """Get Chicago ward information."""
    # This is a simplified mapping - in production you'd load from a file
    return {
        1: "Downtown/Loop",
        2: "Near North Side",
        3: "Near South Side",
        4: "South Side",
        5: "Hyde Park",
        # ... add all 50 wards
    }

def create_sample_data(n_records: int = 100) -> List[Dict[str, Any]]:
    """Create sample data for testing."""
    import random
    from datetime import datetime, timedelta
    
    sample_data = []
    
    service_types = [
        "Pothole in Street",
        "Street Light - 1/Out",
        "Garbage Cart - Damaged",
        "Tree Removal",
        "Water Leak in Street",
        "Graffiti Removal",
        "Street Cleaning",
        "Abandoned Vehicle"
    ]
    
    statuses = ["Open", "Completed", "In Progress", "Cancelled"]
    departments = ["CDOT", "Streets & Sanitation", "Water Management", "Forestry"]
    
    for i in range(n_records):
        # Generate random Chicago coordinates
        lat = random.uniform(41.6, 42.1)
        lon = random.uniform(-87.9, -87.5)
        
        # Random date within last year
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        creation_date = start_date + timedelta(days=random_days)
        
        record = {
            'sr_number': f"{random.randint(10000000, 99999999)}",
            'sr_type': random.choice(service_types),
            'status': random.choice(statuses),
            'creation_date': creation_date.isoformat(),
            'owner_department': random.choice(departments),
            'street_address': f"{random.randint(1, 9999)} N State St",
            'city': "Chicago",
            'state': "IL",
            'zip_code': f"606{random.randint(10, 99)}",
            'ward': random.randint(1, 50),
            'community_area': random.randint(1, 77),
            'latitude': lat,
            'longitude': lon,
            'duplicate': random.choice([True, False]),
            'legacy_record': random.choice([True, False])
        }
        
        # Add completion date for completed requests
        if record['status'] == 'Completed':
            completion_days = random.randint(1, 30)
            completion_date = creation_date + timedelta(days=completion_days)
            record['completion_date'] = completion_date.isoformat()
        
        sample_data.append(record)
    
    return sample_data

def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements for the application."""
    requirements = {}
    
    # Check Python version
    requirements['python_version'] = sys.version_info >= (3, 8)
    
    # Check available memory (basic check)
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        requirements['memory'] = available_memory_gb >= 8  # At least 8GB
    except ImportError:
        requirements['memory'] = True  # Can't check, assume OK
    
    # Check disk space
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        requirements['disk_space'] = free_space_gb >= 20  # At least 20GB
    except:
        requirements['disk_space'] = True  # Can't check, assume OK
    
    return requirements

def print_system_info() -> None:
    """Print system information."""
    print("System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total Memory: {memory.total / (1024**3):.1f} GB")
        print(f"Available Memory: {memory.available / (1024**3):.1f} GB")
        
        disk = psutil.disk_usage('.')
        print(f"Total Disk Space: {disk.total / (1024**3):.1f} GB")
        print(f"Free Disk Space: {disk.free / (1024**3):.1f} GB")
    except ImportError:
        print("psutil not available - can't show memory/disk info")

if __name__ == "__main__":
    # Test utilities
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Testing utilities...")
    
    # Test system requirements
    requirements = check_system_requirements()
    print("System Requirements Check:")
    for req, status in requirements.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"  {req}: {status_str}")
    
    # Test sample data generation
    sample = create_sample_data(5)
    print(f"\nGenerated {len(sample)} sample records")
    
    logger.info("Utilities test completed")