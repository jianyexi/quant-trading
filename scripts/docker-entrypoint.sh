#!/bin/bash
# Start ML inference sidecar (HTTP + TCP MQ) in background
if [ -f /app/ml_models/ml_serve.py ]; then
    echo "🧠 Starting ML inference sidecar..."
    cd /app && python3 ml_models/ml_serve.py \
        --host 0.0.0.0 --port 18091 --tcp-port 18094 \
        --model ml_models/factor_model.model \
        > /app/logs/ml_serve.log 2>&1 &
    ML_PID=$!
    echo "🧠 ML sidecar PID: $ML_PID (HTTP:18091, TCP:18094)"
    # Wait briefly for sidecar to start
    sleep 2
fi

# Start main application
exec "$@"
