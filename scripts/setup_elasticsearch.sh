#!/bin/bash

echo "🔍 Setting up Elasticsearch locally..."

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: $MACHINE"

# Check Java installation
if ! command -v java &> /dev/null; then
    echo "❌ Java is required for Elasticsearch. Installing Java..."
    case "${MACHINE}" in
        Mac)
            if command -v brew &> /dev/null; then
                brew install openjdk@11
                echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
                export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
            fi
            ;;
        Linux)
            if command -v apt &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y openjdk-11-jdk
            elif command -v yum &> /dev/null; then
                sudo yum install -y java-11-openjdk
            fi
            ;;
    esac
fi

# Install Elasticsearch based on OS
case "${MACHINE}" in
    Mac)
        echo "Installing Elasticsearch on macOS..."
        if command -v brew &> /dev/null; then
            brew tap elastic/tap
            brew install elastic/tap/elasticsearch-full
            
            # Start Elasticsearch
            brew services start elastic/tap/elasticsearch-full
        else
            echo "❌ Homebrew not found. Installing manually..."
            ES_VERSION="8.8.2"
            wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-darwin-x86_64.tar.gz
            tar -xzf elasticsearch-${ES_VERSION}-darwin-x86_64.tar.gz
            mv elasticsearch-${ES_VERSION} elasticsearch
            
            # Configure Elasticsearch
            cat >> elasticsearch/config/elasticsearch.yml <<EOF
cluster.name: chicago-311-cluster
node.name: chicago-311-node
network.host: localhost
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
EOF
            
            # Start Elasticsearch
            elasticsearch/bin/elasticsearch -d
        fi
        ;;
    Linux)
        echo "Installing Elasticsearch on Linux..."
        ES_VERSION="8.8.2"
        
        # Download and install
        wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz
        tar -xzf elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz
        sudo mv elasticsearch-${ES_VERSION} /opt/elasticsearch
        
        # Create elasticsearch user
        sudo useradd -r -s /bin/false elasticsearch
        sudo chown -R elasticsearch:elasticsearch /opt/elasticsearch
        
        # Configure Elasticsearch
        sudo tee /opt/elasticsearch/config/elasticsearch.yml > /dev/null <<EOF
cluster.name: chicago-311-cluster
node.name: chicago-311-node
path.data: /opt/elasticsearch/data
path.logs: /opt/elasticsearch/logs
network.host: localhost
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
EOF
        
        # Create systemd service
        sudo tee /etc/systemd/system/elasticsearch.service > /dev/null <<EOF
[Unit]
Description=Elasticsearch
Documentation=https://www.elastic.co
Wants=network-online.target
After=network-online.target
RequiresMountsFor=/tmp

[Service]
Type=notify
User=elasticsearch
Group=elasticsearch
RuntimeDirectory=elasticsearch
Environment=ES_HOME=/opt/elasticsearch
Environment=ES_PATH_CONF=/opt/elasticsearch/config
Environment=PID_DIR=/var/run/elasticsearch
Environment=ES_SD_NOTIFY=true
EnvironmentFile=-/etc/default/elasticsearch
WorkingDirectory=/opt/elasticsearch
ExecStart=/opt/elasticsearch/bin/elasticsearch
LimitNOFILE=65535
LimitNPROC=4096
LimitAS=infinity
LimitFSIZE=infinity
TimeoutStopSec=0
KillSignal=SIGTERM
KillMode=process
SendSIGKILL=no
SuccessExitStatus=143
TimeoutStartSec=75

[Install]
WantedBy=multi-user.target
EOF
        
        # Start Elasticsearch
        sudo systemctl daemon-reload
        sudo systemctl enable elasticsearch
        sudo systemctl start elasticsearch
        ;;
    *)
        echo "❌ Unsupported operating system: $MACHINE"
        echo "Please install Elasticsearch manually from https://www.elastic.co/downloads/elasticsearch"
        exit 1
        ;;
esac

# Wait for Elasticsearch to start
echo "⏳ Waiting for Elasticsearch to start..."
sleep 30

# Test connection
for i in {1..30}; do
    if curl -s http://localhost:9200 > /dev/null; then
        echo "✅ Elasticsearch is running successfully"
        
        # Display cluster info
        echo "📊 Elasticsearch cluster info:"
        curl -s http://localhost:9200 | python3 -m json.tool
        
        echo ""
        echo "🎉 Elasticsearch setup completed!"
        echo "URL: http://localhost:9200"
        echo "Kibana (if installed): http://localhost:5601"
        
        exit 0
    fi
    echo "Waiting for Elasticsearch to be ready... ($i/30)"
    sleep 5
done

echo "❌ Elasticsearch failed to start within 150 seconds"
echo "Check logs for more information"
exit 1