#!/bin/bash

# Kill any existing Streamlit server processes
kill $(lsof -ti :8501) 2>/dev/null

# Start the Streamlit app without opening the browser
streamlit run app.py --browser.serverAddress 0.0.0.0 --server.port 8501 --server.headless true &

# Save the Streamlit server's PID
echo $! > streamlit.pid

# Wait for the Streamlit app to start
sleep 2

# Change to the reentry-electron directory
cd reentry-electron

# Start the Electron app
npm start
