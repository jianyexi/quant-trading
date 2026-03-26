#!/usr/bin/env python3
"""LLM Signal Server — serves fine-tuned LLM trading signals.

Loads a base model + LoRA adapter, accepts market context via HTTP,
returns structured {action, confidence, reasoning} for the trading engine.

Usage:
    python llm_signal_serve.py --base-model Qwen/Qwen2.5-7B-Instruct \
                               --adapter ml_models/llm_adapters/sft/adapter \
                               --port 18095
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Global model state ───────────────────────────────────────────
MODEL = None
TOKENIZER = None
BASE_MODEL_NAME = None
ADAPTER_PATH = None
DEVICE = "cpu"
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

SIGNAL_SYSTEM_PROMPT = """You are a quantitative trading signal generator. Given market data and technical indicators for a stock, output a JSON trading signal.

You MUST respond with ONLY a valid JSON object in this exact format:
{"action": "buy" or "sell" or "hold", "confidence": 0.0-1.0, "reasoning": "brief explanation"}

Rules:
- action: "buy" if bullish, "sell" if bearish, "hold" if neutral
- confidence: 0.0 (no confidence) to 1.0 (maximum confidence)
- reasoning: 1-2 sentences explaining the signal
- Do NOT include any text outside the JSON object"""


# ── Device detection ─────────────────────────────────────────────

def detect_device():
    """Auto-detect best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Model loading ────────────────────────────────────────────────

def load_model(base_model: str, adapter_path: str = None,
               device: str = "auto", quantize_4bit: bool = True):
    """Load base model with optional LoRA adapter."""
    global MODEL, TOKENIZER, BASE_MODEL_NAME, ADAPTER_PATH, DEVICE

    if device == "auto":
        device = detect_device()
    DEVICE = device

    print(f"[llm_signal_serve] Loading base model: {base_model}")
    print(f"[llm_signal_serve] Device: {device}, 4-bit: {quantize_4bit}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
    except ImportError:
        print("ERROR: transformers and torch are required. Install with:")
        print("  pip install transformers torch peft bitsandbytes accelerate")
        sys.exit(1)

    # Tokenizer — prefer adapter dir (may have merged tokenizer)
    tok_source = adapter_path if adapter_path and Path(adapter_path).exists() else base_model
    TOKENIZER = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    # Quantization config
    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if quantize_4bit and device == "cuda":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
            print("[llm_signal_serve] Using 4-bit quantization (QLoRA)")
        except Exception as e:
            print(f"[llm_signal_serve] 4-bit unavailable: {e}, falling back")
            load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["device_map"] = {"": device}

    MODEL = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    # Apply LoRA adapter
    if adapter_path and Path(adapter_path).exists():
        try:
            from peft import PeftModel
            print(f"[llm_signal_serve] Applying LoRA adapter: {adapter_path}")
            MODEL = PeftModel.from_pretrained(MODEL, adapter_path)
            ADAPTER_PATH = adapter_path
            print("[llm_signal_serve] LoRA adapter applied successfully")
        except Exception as e:
            print(f"[llm_signal_serve] WARNING: Failed to load adapter: {e}")
            ADAPTER_PATH = None
    else:
        ADAPTER_PATH = None
        if adapter_path:
            print(f"[llm_signal_serve] WARNING: Adapter not found: {adapter_path}")

    MODEL.eval()
    BASE_MODEL_NAME = base_model
    print("[llm_signal_serve] Model loaded successfully")


# ── Prompt construction ──────────────────────────────────────────

def build_signal_prompt(data: dict) -> str:
    """Build a structured prompt from market context."""
    symbol = data.get("symbol", "UNKNOWN")
    bars = data.get("bars", [])
    indicators = data.get("indicators", {})

    bar_lines = []
    for b in bars[-10:]:
        bar_lines.append(
            f"  {b.get('datetime', '?')}: "
            f"O={b.get('open', 0):.2f} H={b.get('high', 0):.2f} "
            f"L={b.get('low', 0):.2f} C={b.get('close', 0):.2f} "
            f"V={b.get('volume', 0):.0f}"
        )

    ind_lines = []
    for key, val in sorted(indicators.items()):
        if isinstance(val, float):
            ind_lines.append(f"  {key}: {val:.4f}")
        else:
            ind_lines.append(f"  {key}: {val}")

    prompt = (
        f"Analyze the following market data for {symbol} "
        f"and generate a trading signal.\n\n"
        f"Recent Price Bars (OHLCV):\n"
        f"{chr(10).join(bar_lines) if bar_lines else '  No bar data available'}\n\n"
        f"Technical Indicators:\n"
        f"{chr(10).join(ind_lines) if ind_lines else '  No indicators available'}\n\n"
        f"Based on this data, provide your trading signal as a JSON object."
    )
    return prompt


# ── Signal generation ────────────────────────────────────────────

def generate_signal(prompt: str) -> dict:
    """Generate trading signal from the loaded model."""
    import torch

    messages = [
        {"role": "system", "content": SIGNAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if hasattr(TOKENIZER, "apply_chat_template"):
        text = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = (
            f"<|system|>{SIGNAL_SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>{prompt}<|end|>\n<|assistant|>"
        )

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            do_sample=GENERATION_CONFIG["do_sample"],
            repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            pad_token_id=TOKENIZER.pad_token_id,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = TOKENIZER.decode(gen_tokens, skip_special_tokens=True).strip()
    return parse_signal_response(response_text)


def parse_signal_response(text: str) -> dict:
    """Parse LLM response into structured signal."""
    # Direct JSON parse
    try:
        return normalize_signal(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Extract JSON from surrounding text
    json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return normalize_signal(json.loads(json_match.group()))
        except json.JSONDecodeError:
            pass

    # Fallback: detect action keyword from text
    text_lower = text.lower()
    if "buy" in text_lower:
        action = "buy"
    elif "sell" in text_lower:
        action = "sell"
    else:
        action = "hold"

    return {
        "action": action,
        "confidence": 0.5,
        "reasoning": f"Parsed from unstructured response: {text[:200]}",
        "raw_response": text,
    }


def normalize_signal(result: dict) -> dict:
    """Normalize and validate signal JSON."""
    action = str(result.get("action", "hold")).lower().strip()
    if action not in ("buy", "sell", "hold"):
        action = "hold"
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    reasoning = str(result.get("reasoning", ""))
    return {"action": action, "confidence": confidence, "reasoning": reasoning}


# ── Flask Endpoints ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "llm_signal_serve",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
        "base_model": BASE_MODEL_NAME,
        "adapter": ADAPTER_PATH,
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    info = {
        "base_model": BASE_MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "device": DEVICE,
        "generation_config": GENERATION_CONFIG,
    }
    if MODEL is not None:
        try:
            total = sum(p.numel() for p in MODEL.parameters())
            trainable = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
            info["total_params"] = total
            info["trainable_params"] = trainable
        except Exception:
            pass
    return jsonify(info)


@app.route("/signal", methods=["POST"])
def signal():
    """Generate trading signal from market context.

    Request: {"symbol": "...", "bars": [...], "indicators": {...}}
    Response: {"action": "buy"|"sell"|"hold", "confidence": 0.85, "reasoning": "..."}
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    t0 = time.time()
    try:
        prompt = build_signal_prompt(data)
        result = generate_signal(prompt)
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        result["symbol"] = data.get("symbol", "")
        return jsonify(result)
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 2)
        return jsonify({
            "action": "hold",
            "confidence": 0.0,
            "reasoning": f"Inference error: {e}",
            "latency_ms": latency_ms,
            "error": str(e),
        }), 200  # Return 200 with hold — don't break the trading loop


@app.route("/reload", methods=["POST"])
def reload():
    """Reload model with new adapter or configuration."""
    data = request.json or {}
    base = data.get("base_model", BASE_MODEL_NAME)
    adapter = data.get("adapter_path", ADAPTER_PATH)
    device = data.get("device", "auto")
    quantize = data.get("quantize_4bit", True)
    try:
        load_model(base, adapter, device, quantize)
        return jsonify({"status": "reloaded", "base_model": base, "adapter": adapter})
    except Exception as e:
        return jsonify({"error": f"Reload failed: {e}"}), 500


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Signal Server")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--port", type=int, default=18095,
                        help="HTTP port (default: 18095)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cpu, mps")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    args = parser.parse_args()

    GENERATION_CONFIG["max_new_tokens"] = args.max_tokens
    GENERATION_CONFIG["temperature"] = args.temperature

    # Auto-detect adapter: prefer DPO (trained with P&L preferences), then SFT
    adapter = args.adapter
    if adapter is None:
        for candidate in [
            "ml_models/llm_adapters/dpo/adapter",
            "ml_models/llm_adapters/sft/adapter",
        ]:
            if Path(candidate).exists():
                adapter = candidate
                print(f"[llm_signal_serve] Auto-detected adapter: {candidate}")
                break

    load_model(args.base_model, adapter, args.device, not args.no_quantize)

    print(f"[llm_signal_serve] Starting server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
