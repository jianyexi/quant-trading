#!/usr/bin/env python3
"""
ML Inference Sidecar for Quant Trading System

A Flask HTTP server that loads a trained ML model and serves predictions
via REST API. Supports GPU acceleration via CUDA (auto-detected).

Supported model formats:
  - LightGBM (.lgb.txt)
  - XGBoost (.xgb.json)
  - CatBoost (.catboost.bin)
  - ONNX (.onnx) via onnxruntime with CUDA/CPU
  - PyTorch/TorchScript (.pt) — LSTM, Transformer via torch with CUDA/CPU

Usage:
    python ml_serve.py --model factor_model.onnx [--port 18091] [--device auto]

Endpoints:
    GET  /health       - Health check
    GET  /model_info   - Model metadata (type, device, features)
    POST /predict      - Run prediction on feature vector
    POST /predict_batch - Run batch predictions
    POST /reload       - Hot-reload model
    POST /ensemble/load - Load multiple models for ensemble
    GET  /ensemble/status - Ensemble status
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

# Ensemble support: list of (model, model_type, weight) tuples
ENSEMBLE_MODELS = []
ENSEMBLE_ENABLED = False
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
    elif model_path.endswith(".xgb.json") or model_path.endswith(".xgb"):
        load_xgboost_model(model_path)
    elif model_path.endswith(".catboost.bin") or model_path.endswith(".catboost"):
        load_catboost_model(model_path)
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


def load_xgboost_model(model_path):
    global MODEL, MODEL_TYPE
    import xgboost as xgb

    MODEL = xgb.Booster()
    MODEL.load_model(model_path)
    MODEL_TYPE = "xgboost"
    print(f"  ✅ XGBoost model loaded: {model_path}")


def load_catboost_model(model_path):
    global MODEL, MODEL_TYPE
    from catboost import CatBoostClassifier

    MODEL = CatBoostClassifier()
    MODEL.load_model(model_path)
    MODEL_TYPE = "catboost"
    print(f"  ✅ CatBoost model loaded: {model_path}")


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
        if len(outputs) >= 2 and outputs[1].shape[-1] >= 2:
            return float(outputs[1][0, 1])
        return float(outputs[0][0])

    elif MODEL_TYPE == "lightgbm":
        prob = MODEL.predict(features)[0]
        return float(prob)

    elif MODEL_TYPE == "xgboost":
        import xgboost as xgb
        dmat = xgb.DMatrix(features)
        prob = MODEL.predict(dmat)[0]
        return float(prob)

    elif MODEL_TYPE == "catboost":
        prob = MODEL.predict_proba(features)[0, 1]
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


def predict_ensemble(features):
    """Run ensemble prediction — weighted average of all loaded models."""
    if not ENSEMBLE_MODELS:
        return predict_single(features)

    features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
    total_weight = 0.0
    weighted_sum = 0.0

    for model, mtype, weight in ENSEMBLE_MODELS:
        try:
            if mtype == "lightgbm":
                prob = float(model.predict(features_arr)[0])
            elif mtype == "xgboost":
                import xgboost as xgb
                prob = float(model.predict(xgb.DMatrix(features_arr))[0])
            elif mtype == "catboost":
                prob = float(model.predict_proba(features_arr)[0, 1])
            elif mtype == "onnx":
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: features_arr})
                if len(outputs) >= 2 and outputs[1].shape[-1] >= 2:
                    prob = float(outputs[1][0, 1])
                else:
                    prob = float(outputs[0][0])
            elif mtype == "pytorch":
                import torch
                with torch.no_grad():
                    x = torch.tensor(features_arr, dtype=torch.float32)
                    out = model(x)
                    prob = float(torch.sigmoid(out)[0].item())
            else:
                prob = 0.5
            weighted_sum += prob * weight
            total_weight += weight
        except Exception:
            continue

    if total_weight > 0:
        return weighted_sum / total_weight
    return predict_single(features)


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
    prob = predict_ensemble(features) if ENSEMBLE_ENABLED else predict_single(features)
    latency_ms = (time.time() - start) * 1000

    return jsonify({
        "probability": prob,
        "signal": "buy" if prob > 0.6 else ("sell" if prob < 0.35 else "hold"),
        "latency_ms": round(latency_ms, 2),
        "device": DEVICE,
        "ensemble": ENSEMBLE_ENABLED,
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


@app.route("/ensemble/load", methods=["POST"])
def api_ensemble_load():
    """
    Load multiple models for ensemble prediction.
    Body: { "models": [{"path": "model1.lgb.txt", "weight": 1.0}, ...] }
    """
    global ENSEMBLE_MODELS, ENSEMBLE_ENABLED
    data = request.get_json(silent=True)
    if not data or "models" not in data:
        return jsonify({"error": "models list required"}), 400

    loaded = []
    ENSEMBLE_MODELS = []

    for entry in data["models"]:
        path = entry.get("path", "")
        weight = float(entry.get("weight", 1.0))
        if not os.path.exists(path):
            loaded.append({"path": path, "status": "not_found"})
            continue

        try:
            if path.endswith(".lgb.txt") or path.endswith(".lgb"):
                import lightgbm as lgb
                m = lgb.Booster(model_file=path)
                ENSEMBLE_MODELS.append((m, "lightgbm", weight))
                loaded.append({"path": path, "type": "lightgbm", "weight": weight, "status": "ok"})
            elif path.endswith(".xgb.json") or path.endswith(".xgb"):
                import xgboost as xgb
                m = xgb.Booster()
                m.load_model(path)
                ENSEMBLE_MODELS.append((m, "xgboost", weight))
                loaded.append({"path": path, "type": "xgboost", "weight": weight, "status": "ok"})
            elif path.endswith(".catboost.bin") or path.endswith(".catboost"):
                from catboost import CatBoostClassifier
                m = CatBoostClassifier()
                m.load_model(path)
                ENSEMBLE_MODELS.append((m, "catboost", weight))
                loaded.append({"path": path, "type": "catboost", "weight": weight, "status": "ok"})
            elif path.endswith(".onnx"):
                import onnxruntime as ort
                m = ort.InferenceSession(path)
                ENSEMBLE_MODELS.append((m, "onnx", weight))
                loaded.append({"path": path, "type": "onnx", "weight": weight, "status": "ok"})
            elif path.endswith(".pt") or path.endswith(".pth"):
                import torch
                m = torch.jit.load(path, map_location=DEVICE)
                m.eval()
                ENSEMBLE_MODELS.append((m, "pytorch", weight))
                loaded.append({"path": path, "type": "pytorch", "weight": weight, "status": "ok"})
            else:
                loaded.append({"path": path, "status": "unsupported_format"})
        except Exception as e:
            loaded.append({"path": path, "status": f"error: {e}"})

    ENSEMBLE_ENABLED = len(ENSEMBLE_MODELS) > 0
    print(f"🎯 Ensemble loaded: {len(ENSEMBLE_MODELS)} models, enabled={ENSEMBLE_ENABLED}")

    return jsonify({
        "ensemble_enabled": ENSEMBLE_ENABLED,
        "models_loaded": len(ENSEMBLE_MODELS),
        "details": loaded,
    })


@app.route("/ensemble/status", methods=["GET"])
def api_ensemble_status():
    """Get ensemble status."""
    return jsonify({
        "ensemble_enabled": ENSEMBLE_ENABLED,
        "models_count": len(ENSEMBLE_MODELS),
        "models": [{"type": t, "weight": w} for _, t, w in ENSEMBLE_MODELS],
        "primary_model_type": MODEL_TYPE,
    })


# ── TCP Message Queue Server ─────────────────────────────────────────
# Binary protocol: [4 bytes: msg_len (big-endian u32)] [msg_len bytes: JSON]
# Same protocol as market_data_server TCP MQ.

import socket
import struct
import threading


def tcp_send(conn, msg: dict):
    """Send a length-prefixed JSON message."""
    data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(data))
    try:
        conn.sendall(header + data)
    except Exception:
        pass


def tcp_recv(conn, timeout=30.0):
    """Receive a length-prefixed JSON message."""
    conn.settimeout(timeout)
    try:
        header = b""
        while len(header) < 4:
            chunk = conn.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        msg_len = struct.unpack(">I", header)[0]
        if msg_len > 10_000_000:
            return None
        body = b""
        while len(body) < msg_len:
            chunk = conn.recv(min(msg_len - len(body), 65536))
            if not chunk:
                return None
            body += chunk
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def handle_ml_tcp_client(conn, addr):
    """Handle a single TCP ML inference client."""
    print(f"🔗 ML TCP client connected: {addr}")
    tcp_send(conn, {"type": "connected", "model_type": MODEL_TYPE, "device": DEVICE})

    try:
        while True:
            msg = tcp_recv(conn, timeout=60.0)
            if msg is None:
                break

            cmd = msg.get("cmd", "")

            if cmd == "predict":
                features = msg.get("features", [])
                if len(features) != NUM_FEATURES:
                    tcp_send(conn, {"type": "error",
                                    "error": f"Expected {NUM_FEATURES} features, got {len(features)}"})
                    continue

                t0 = time.time()
                prob = predict_ensemble(features) if ENSEMBLE_ENABLED else predict_single(features)
                latency_ms = (time.time() - t0) * 1000

                tcp_send(conn, {
                    "type": "prediction",
                    "probability": prob,
                    "latency_ms": round(latency_ms, 3),
                    "device": DEVICE,
                })

            elif cmd == "ping":
                tcp_send(conn, {"type": "pong", "ts": time.time()})

            elif cmd == "model_info":
                tcp_send(conn, {
                    "type": "model_info",
                    "model_type": MODEL_TYPE,
                    "device": DEVICE,
                    "num_features": NUM_FEATURES,
                })

            elif cmd == "reload":
                model_path = msg.get("model_path", "")
                if os.path.exists(model_path):
                    try:
                        load_model(model_path, DEVICE)
                        tcp_send(conn, {"type": "reloaded", "model_path": model_path})
                    except Exception as e:
                        tcp_send(conn, {"type": "error", "error": str(e)})
                else:
                    tcp_send(conn, {"type": "error", "error": f"File not found: {model_path}"})

    except Exception as e:
        print(f"🔗 ML TCP client error: {e}")
    finally:
        conn.close()
        print(f"🔗 ML TCP client disconnected: {addr}")


def ml_tcp_server_thread(port: int):
    """TCP message queue server for ML inference."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    print(f"🔗 ML TCP MQ listening on port {port}")

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t = threading.Thread(target=handle_ml_tcp_client, args=(conn, addr), daemon=True)
        t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Inference Sidecar")
    parser.add_argument("--model", "-m", default="ml_models/factor_model.onnx",
                       help="Path to model file (.onnx, .lgb.txt, .pt)")
    parser.add_argument("--port", "-p", type=int, default=18091,
                       help="HTTP port (default: 18091)")
    parser.add_argument("--tcp-port", type=int, default=18094,
                       help="TCP MQ port (default: 18094)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", "-d", default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device for inference (default: auto-detect)")
    args = parser.parse_args()

    print("🧠 ML Inference Sidecar for QuantTrader")
    print(f"   Model: {args.model}")
    print(f"   HTTP:  {args.port} | TCP MQ: {args.tcp_port}")

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

    # Start TCP MQ server thread
    threading.Thread(target=ml_tcp_server_thread, args=(args.tcp_port,), daemon=True).start()

    print(f"\n🚀 Starting ML server")
    print(f"   HTTP:    http://{args.host}:{args.port}/predict")
    print(f"   TCP MQ:  {args.host}:{args.tcp_port} (binary protocol)")
    print()

    app.run(host=args.host, port=args.port, debug=False)
