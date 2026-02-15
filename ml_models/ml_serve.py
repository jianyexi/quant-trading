#!/usr/bin/env python3
"""
ML Inference Sidecar for Quant Trading System

A Flask HTTP server that loads a trained ML model and serves predictions
via REST API. Supports GPU acceleration via CUDA (auto-detected).

Supported model formats:
  - LightGBM (.lgb.txt)
  - ONNX (.onnx) via onnxruntime with CUDA/CPU
  - PyTorch (.pt) via torch with CUDA/CPU

Usage:
    python ml_serve.py --model factor_model.onnx [--port 18091] [--device auto]

Endpoints:
    GET  /health       - Health check
    GET  /model_info   - Model metadata (type, device, features)
    POST /predict      - Run prediction on feature vector
    POST /predict_batch - Run batch predictions
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global model state
MODEL = None
MODEL_TYPE = None  # "lightgbm", "onnx", "pytorch"
DEVICE = "cpu"
NUM_FEATURES = 24
FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "volatility_5d", "volatility_20d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio", "ma5_ma20_cross",
    "rsi_14",
    "macd_histogram", "macd_normalized",
    "volume_ratio_5_20", "volume_change",
    "price_position",
    "gap",
    "intraday_range",
    "upper_shadow_ratio", "lower_shadow_ratio",
    "bollinger_pctb",
    "body_ratio",
    "close_to_open",
]


def detect_device():
    """Auto-detect best available device."""
    # Try CUDA via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  🎮 GPU detected: {name} ({mem:.1f} GB)")
            return "cuda"
    except ImportError:
        pass

    # Try CUDA via ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print("  🎮 ONNX Runtime CUDA provider available")
            return "cuda"
    except ImportError:
        pass

    print("  💻 Using CPU (install torch with CUDA or onnxruntime-gpu for GPU acceleration)")
    return "cpu"


def load_model(model_path, device):
    """Load model from file, auto-detecting format."""
    global MODEL, MODEL_TYPE, DEVICE
    DEVICE = device

    if model_path.endswith(".onnx"):
        load_onnx_model(model_path, device)
    elif model_path.endswith(".lgb.txt") or model_path.endswith(".lgb"):
        load_lightgbm_model(model_path)
    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        load_pytorch_model(model_path, device)
    else:
        print(f"  ⚠️ Unknown model format: {model_path}")
        print("  Using dummy model (returns 0.5 for all predictions)")
        MODEL = None
        MODEL_TYPE = "dummy"


def load_onnx_model(model_path, device):
    global MODEL, MODEL_TYPE
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("  🚀 ONNX Runtime: Using CUDA GPU")
    else:
        print("  💻 ONNX Runtime: Using CPU")

    MODEL = ort.InferenceSession(model_path, providers=providers)
    MODEL_TYPE = "onnx"
    print(f"  ✅ ONNX model loaded: {model_path}")

    # Print input/output info
    for inp in MODEL.get_inputs():
        print(f"     Input: {inp.name} shape={inp.shape} type={inp.type}")
    for out in MODEL.get_outputs():
        print(f"     Output: {out.name} shape={out.shape} type={out.type}")


def load_lightgbm_model(model_path):
    global MODEL, MODEL_TYPE
    import lightgbm as lgb

    MODEL = lgb.Booster(model_file=model_path)
    MODEL_TYPE = "lightgbm"
    print(f"  ✅ LightGBM model loaded: {model_path}")
    print(f"     Features: {MODEL.num_feature()}")


def load_pytorch_model(model_path, device):
    global MODEL, MODEL_TYPE, DEVICE
    import torch

    MODEL = torch.jit.load(model_path, map_location=device)
    MODEL.eval()
    MODEL_TYPE = "pytorch"
    if device == "cuda":
        MODEL = MODEL.cuda()
    print(f"  ✅ PyTorch model loaded: {model_path} (device={device})")


def predict_single(features):
    """Run prediction on a single feature vector. Returns probability [0, 1]."""
    if MODEL is None or MODEL_TYPE == "dummy":
        return 0.5

    features = np.array(features, dtype=np.float32).reshape(1, -1)

    if MODEL_TYPE == "onnx":
        input_name = MODEL.get_inputs()[0].name
        outputs = MODEL.run(None, {input_name: features})
        # LightGBM ONNX: output[1] is probabilities [batch, 2]
        if len(outputs) >= 2 and outputs[1].shape[-1] >= 2:
            return float(outputs[1][0, 1])
        return float(outputs[0][0])

    elif MODEL_TYPE == "lightgbm":
        prob = MODEL.predict(features)[0]
        return float(prob)

    elif MODEL_TYPE == "pytorch":
        import torch
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if DEVICE == "cuda":
                x = x.cuda()
            out = MODEL(x)
            if out.dim() == 2 and out.shape[1] >= 2:
                prob = torch.softmax(out, dim=1)[0, 1].item()
            else:
                prob = torch.sigmoid(out)[0].item()
            return float(prob)

    return 0.5


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_type": MODEL_TYPE,
        "device": DEVICE,
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    info = {
        "model_type": MODEL_TYPE,
        "device": DEVICE,
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "gpu_available": DEVICE == "cuda",
    }

    if MODEL_TYPE == "lightgbm" and MODEL:
        info["num_trees"] = MODEL.num_trees()
        info["num_model_features"] = MODEL.num_feature()
    elif MODEL_TYPE == "onnx" and MODEL:
        info["inputs"] = [
            {"name": i.name, "shape": i.shape, "type": i.type}
            for i in MODEL.get_inputs()
        ]
        info["outputs"] = [
            {"name": o.name, "shape": o.shape, "type": o.type}
            for o in MODEL.get_outputs()
        ]

    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features", [])

    if len(features) != NUM_FEATURES:
        return jsonify({"error": f"Expected {NUM_FEATURES} features, got {len(features)}"}), 400

    start = time.time()
    prob = predict_single(features)
    latency_ms = (time.time() - start) * 1000

    return jsonify({
        "probability": prob,
        "signal": "buy" if prob > 0.6 else ("sell" if prob < 0.35 else "hold"),
        "latency_ms": round(latency_ms, 2),
        "device": DEVICE,
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.json
    batch = data.get("batch", [])

    start = time.time()
    results = []
    for item in batch:
        features = item.get("features", [])
        if len(features) == NUM_FEATURES:
            prob = predict_single(features)
            results.append({"probability": prob})
        else:
            results.append({"error": f"Expected {NUM_FEATURES} features"})

    latency_ms = (time.time() - start) * 1000

    return jsonify({
        "predictions": results,
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
        "device": DEVICE,
    })


@app.route("/reload", methods=["POST"])
def api_reload():
    """
    Hot-reload model from a new file path.
    Body: { "model_path": "path/to/model.lgb.txt" }
    """
    global MODEL, MODEL_TYPE, DEVICE
    data = request.get_json(silent=True)
    if not data or "model_path" not in data:
        return jsonify({"error": "model_path required"}), 400

    model_path = data["model_path"]
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model file not found: {model_path}"}), 404

    try:
        load_model(model_path, DEVICE)
        print(f"🔄 Model hot-reloaded: {model_path} ({MODEL_TYPE})")
        return jsonify({
            "status": "reloaded",
            "model_path": model_path,
            "model_type": MODEL_TYPE,
            "device": DEVICE,
        })
    except Exception as e:
        return jsonify({"error": f"Reload failed: {e}"}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Inference Sidecar")
    parser.add_argument("--model", "-m", default="ml_models/factor_model.onnx",
                       help="Path to model file (.onnx, .lgb.txt, .pt)")
    parser.add_argument("--port", "-p", type=int, default=18091,
                       help="HTTP port (default: 18091)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", "-d", default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device for inference (default: auto-detect)")
    args = parser.parse_args()

    print("🧠 ML Inference Sidecar for QuantTrader")
    print(f"   Model: {args.model}")
    print(f"   Port:  {args.port}")

    # Detect device
    device = args.device
    if device == "auto":
        device = detect_device()
    else:
        print(f"  Device: {device}")
    DEVICE = device

    # Load model
    try:
        load_model(args.model, device)
    except Exception as e:
        print(f"  ⚠️ Failed to load model: {e}")
        print(f"  Running with dummy model (returns 0.5)")
        MODEL = None
        MODEL_TYPE = "dummy"

    print(f"\n🚀 Starting ML server on {args.host}:{args.port}")
    print(f"   Health:  http://{args.host}:{args.port}/health")
    print(f"   Predict: POST http://{args.host}:{args.port}/predict")
    print()

    app.run(host=args.host, port=args.port, debug=False)
