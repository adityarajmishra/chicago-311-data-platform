// MongoDB initialization script for Chicago 311 data platform
print('Starting MongoDB initialization...');

// Switch to the chicago_311 database
db = db.getSiblingDB('chicago_311');

// Create a user for the chicago_311 database
db.createUser({
  user: 'chicago311user',
  pwd: 'chicago311pass',
  roles: [
    {
      role: 'readWrite',
      db: 'chicago_311'
    }
  ]
});

// Create the service_requests collection with some initial indexes
db.createCollection('service_requests');

print('MongoDB initialization completed successfully!');