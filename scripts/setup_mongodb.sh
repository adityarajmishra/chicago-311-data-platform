#!/bin/bash

echo "🍃 Setting up MongoDB locally..."

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: $MACHINE"

# Install MongoDB based on OS
case "${MACHINE}" in
    Mac)
        echo "Installing MongoDB on macOS..."
        if command -v brew &> /dev/null; then
            brew tap mongodb/brew
            brew install mongodb-community@6.0
            brew services start mongodb/brew/mongodb-community
        else
            echo "❌ Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        ;;
    Linux)
        echo "Installing MongoDB on Linux..."
        # Ubuntu/Debian
        if command -v apt &> /dev/null; then
            wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
            echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
            sudo apt-get update
            sudo apt-get install -y mongodb-org
            sudo systemctl start mongod
            sudo systemctl enable mongod
        # CentOS/RHEL
        elif command -v yum &> /dev/null; then
            cat <<EOF | sudo tee /etc/yum.repos.d/mongodb-org-6.0.repo
[mongodb-org-6.0]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/8/mongodb-org/6.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-6.0.asc
EOF
            sudo yum install -y mongodb-org
            sudo systemctl start mongod
            sudo systemctl enable mongod
        else
            echo "❌ Unsupported Linux distribution"
            exit 1
        fi
        ;;
    *)
        echo "❌ Unsupported operating system: $MACHINE"
        echo "Please install MongoDB manually from https://docs.mongodb.com/manual/installation/"
        exit 1
        ;;
esac

# Wait for MongoDB to start
echo "⏳ Waiting for MongoDB to start..."
sleep 10

# Test connection
if mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo "✅ MongoDB is running successfully"
    
    # Create database and user
    echo "🔧 Setting up database and user..."
    mongosh --eval "
        use chicago_311;
        db.createUser({
            user: 'chicago_user',
            pwd: 'chicago_pass',
            roles: [
                { role: 'readWrite', db: 'chicago_311' },
                { role: 'dbAdmin', db: 'chicago_311' }
            ]
        });
        print('Database and user created successfully');
    "
    
    echo "📊 MongoDB setup completed!"
    echo "Database: chicago_311"
    echo "User: chicago_user"
    echo "Password: chicago_pass"
    echo "Connection string: mongodb://chicago_user:chicago_pass@localhost:27017/chicago_311"
    
else
    echo "❌ MongoDB failed to start"
    exit 1
fi