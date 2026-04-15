#!/bin/bash

# e-KYC Server Restart Script
# Solves port conflict issues automatically

PORT=8502
APP_DIR="/Users/apple/Downloads/ekyc sssss"

echo "Restarting e-KYC Server on port $PORT..."

# Step 1: Find and kill any process using the port
echo "Checking for processes on port $PORT..."
PID=$(lsof -ti:$PORT)

if [ ! -z "$PID" ]; then
    echo "Killing process $PID using port $PORT..."
    kill -9 $PID 2>/dev/null
    sleep 2
    echo "Old process killed"
else
    echo "Port $PORT is free"
fi

# Step 2: Kill any streamlit processes (backup)
pkill -9 -f streamlit 2>/dev/null
sleep 1

# Step 3: Verify port is free
if lsof -i:$PORT > /dev/null 2>&1; then
    echo "ERROR: Port $PORT still in use!"
    echo "Manual cleanup required:"
    lsof -i:$PORT
    exit 1
fi

# Step 4: Start the server
echo "Starting Streamlit server..."
cd "$APP_DIR"
source .venv/bin/activate

# Start in background with proper logging
nohup python -m streamlit run app.py --server.port $PORT > streamlit.log 2>&1 &
NEW_PID=$!

# Step 5: Wait and verify
sleep 4

if lsof -i:$PORT > /dev/null 2>&1; then
    echo "Server started successfully!"
    echo "URL: http://localhost:$PORT"
    echo "Database: MySQL (XAMPP)"
    echo "OCR: Speed-optimized (2-3 min)"
    echo "PID: $NEW_PID"
    echo ""
    echo "Logs: tail -f streamlit.log"
else
    echo "Server failed to start!"
    echo "Check logs: tail -30 streamlit.log"
    exit 1
fi
