"""Database configuration settings."""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    """Database configuration class."""
    
    @staticmethod
    def get_mongodb_config() -> Dict[str, Any]:
        """Get MongoDB configuration."""
        return {
            'host': os.getenv('MONGODB_HOST', 'localhost'),
            'port': int(os.getenv('MONGODB_PORT', 27017)),
            'database': os.getenv('MONGODB_DB', 'chicago_311'),
            'username': os.getenv('MONGODB_USERNAME'),
            'password': os.getenv('MONGODB_PASSWORD'),
            'pool_size': int(os.getenv('MONGO_POOL_SIZE', 100)),
            'connect_timeout': 20000,
            'server_selection_timeout': 20000,
        }
    
    @staticmethod
    def get_connection_string() -> str:
        """Get MongoDB connection string."""
        config = DatabaseConfig.get_mongodb_config()
        
        if config['username'] and config['password']:
            return (f"mongodb://{config['username']}:{config['password']}@"
                   f"{config['host']}:{config['port']}/{config['database']}?authSource={config['database']}")
        else:
            return f"mongodb://{config['host']}:{config['port']}/{config['database']}"