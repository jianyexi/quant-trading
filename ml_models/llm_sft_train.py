#!/usr/bin/env python3
"""
llm_sft_train.py — LoRA Supervised Fine-Tuning for quant-trading LLM.

Trains a LoRA adapter on chat history + sentiment data using HuggingFace PEFT.
Supports Qwen2.5, LLaMA, Mistral, and other causal LM architectures.

Usage:
  python ml_models/llm_sft_train.py --base-model Qwen/Qwen2.5-7B-Instruct
  python ml_models/llm_sft_train.py --base-model meta-llama/Llama-3.1-8B-Instruct --epochs 3
  python ml_models/llm_sft_train.py --data-dir data/llm_training --output-dir ml_models/llm_adapters/sft

Requirements (add to ml_models/requirements.txt):
  pip install transformers>=4.40 peft>=0.10 trl>=0.8 datasets bitsandbytes accelerate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def check_dependencies():
    """Verify required packages are available."""
    missing = []
    for pkg in ["transformers", "peft", "trl", "datasets", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("  Install: pip install transformers peft trl datasets bitsandbytes accelerate")
        sys.exit(1)


def load_sft_data(data_dir: Path) -> list[dict]:
    """Load SFT JSONL files and convert to Alpaca-format conversations."""
    records = []
    for fname in ["sft_chat.jsonl", "sft_sentiment.jsonl"]:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fpath} not found")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"  Loaded {fpath.name}: {sum(1 for _ in open(fpath, encoding='utf-8'))} records")
    return records


def format_for_training(records: list[dict]) -> list[dict]:
    """Convert Alpaca-format records to conversation format for SFTTrainer."""
    conversations = []
    for rec in records:
        system_msg = rec.get("system", "")
        instruction = rec.get("instruction", "")
        inp = rec.get("input", "")
        output = rec.get("output", "")

        user_content = instruction
        if inp:
            user_content += f"\n{inp}"

        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})

        conversations.append({"messages": messages})
    return conversations


def train(args):
    check_dependencies()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"\n[SFT Training] {datetime.now().isoformat()}")
    print(f"  Base model: {args.base_model}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  LoRA rank:  {args.lora_rank}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print()

    # Load data
    print("[1/4] Loading training data...")
    data_dir = Path(args.data_dir)
    records = load_sft_data(data_dir)
    if not records:
        print("[ERROR] No training data found. Run llm_dataset_export.py first.")
        sys.exit(1)

    conversations = format_for_training(records)
    print(f"  Total conversations: {len(conversations)}")

    # Train/eval split (90/10)
    split_idx = int(len(conversations) * 0.9)
    train_data = conversations[:split_idx]
    eval_data = conversations[split_idx:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    # Load tokenizer
    print(f"\n[2/4] Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = None
    if args.load_in_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("  Using 4-bit quantization (QLoRA)")
        except Exception:
            print("  [WARN] bitsandbytes not available, loading in full precision")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )
    model.config.use_cache = False

    # LoRA config
    print("\n[3/4] Configuring LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    print(f"  LoRA rank={args.lora_rank}, alpha={args.lora_rank * 2}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_grad_norm=1.0,
        save_total_limit=2,
    )

    # Train
    print(f"\n[4/4] Training... ({args.epochs} epochs)")
    start_time = time.time()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    peft_model = trainer.model
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    result = trainer.train()
    elapsed = time.time() - start_time

    # Save adapter
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  Adapter saved to: {adapter_path}")

    # Save training report
    report = {
        "type": "sft_lora",
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "train_loss": result.training_loss,
        "elapsed_seconds": round(elapsed, 1),
        "trainable_params": trainable,
        "total_params": total_params,
        "created_at": datetime.now().isoformat(),
    }
    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] SFT LoRA training complete in {elapsed:.0f}s")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Report:     {report_path}")
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="SFT LoRA training for quant-trading LLM")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--data-dir", default="data/llm_training",
                        help="Directory with JSONL training files")
    parser.add_argument("--output-dir", default="ml_models/llm_adapters/sft",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (8/16/32)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                        help="Disable 4-bit quantization")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
