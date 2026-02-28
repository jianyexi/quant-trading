#!/bin/bash
# Start ML inference sidecar (HTTP + TCP MQ) in background
if [ -f /app/ml_models/ml_serve.py ]; then
    # Find best available model file
    MODEL_FILE=""
    for f in ml_models/factor_model.lgb.txt ml_models/factor_model.model; do
        if [ -f "/app/$f" ]; then
            MODEL_FILE="$f"
            break
        fi
    done

    if [ -n "$MODEL_FILE" ]; then
        echo "🧠 Starting ML inference sidecar (model: $MODEL_FILE)..."
        cd /app && python3 ml_models/ml_serve.py \
            --host 0.0.0.0 --port 18091 --tcp-port 18094 \
            --model "$MODEL_FILE" \
            > /app/logs/ml_serve.log 2>&1 &
        ML_PID=$!
        echo "🧠 ML sidecar PID: $ML_PID (HTTP:18091, TCP:18094)"
        sleep 2
    else
        echo "🧠 No model file found, skipping ML sidecar"
    fi
fi

# Start main application
exec "$@"
