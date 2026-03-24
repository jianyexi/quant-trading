#!/usr/bin/env python3
"""
llm_dpo_train.py — DPO (Direct Preference Optimization) training for quant-trading LLM.

Uses trade outcome preference pairs to align the model:
  - Chosen  = profitable trade analysis (positive P&L)
  - Rejected = losing trade analysis (negative P&L)

Can optionally start from an SFT adapter (recommended: SFT first, then DPO).

Usage:
  python ml_models/llm_dpo_train.py --base-model Qwen/Qwen2.5-7B-Instruct
  python ml_models/llm_dpo_train.py --sft-adapter ml_models/llm_adapters/sft/adapter

Requirements:
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


def load_dpo_data(data_dir: Path) -> list[dict]:
    """Load DPO preference pairs from JSONL."""
    fpath = data_dir / "dpo_trades.jsonl"
    if not fpath.exists():
        print(f"[ERROR] {fpath} not found. Run llm_dataset_export.py first.")
        return []

    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} DPO pairs from {fpath.name}")
    return records


def train(args):
    check_dependencies()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    print(f"\n[DPO Training] {datetime.now().isoformat()}")
    print(f"  Base model:   {args.base_model}")
    print(f"  SFT adapter:  {args.sft_adapter or '(none — training from base)'}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Beta:         {args.beta}")
    print()

    # Load data
    print("[1/4] Loading DPO preference data...")
    data_dir = Path(args.data_dir)
    records = load_dpo_data(data_dir)
    if len(records) < 2:
        print("[ERROR] Need at least 2 DPO pairs for training.")
        sys.exit(1)

    # Format for DPOTrainer: prompt, chosen, rejected
    dpo_records = []
    for rec in records:
        dpo_records.append({
            "prompt": rec["prompt"],
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
        })

    # Split 90/10
    split_idx = max(1, int(len(dpo_records) * 0.9))
    train_data = dpo_records[:split_idx]
    eval_data = dpo_records[split_idx:] if split_idx < len(dpo_records) else []
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    # Load tokenizer + model
    print(f"\n[2/4] Loading model: {args.base_model}")
    model_name = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if args.load_in_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("  Using 4-bit quantization")
        except Exception:
            print("  [WARN] bitsandbytes not available, full precision")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )

    # If SFT adapter provided, merge it first
    if args.sft_adapter and Path(args.sft_adapter).exists():
        from peft import PeftModel
        print(f"  Merging SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()

    # Reference model (frozen copy for DPO KL penalty)
    ref_model = None  # DPOTrainer creates implicit ref when using PEFT

    # LoRA config for DPO adapter
    print("\n[3/4] Configuring DPO LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_grad_norm=1.0,
        save_total_limit=2,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 4,
    )

    # Train
    print(f"\n[4/4] DPO Training... ({args.epochs} epochs, beta={args.beta})")
    start_time = time.time()

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    peft_model = trainer.model
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    result = trainer.train()
    elapsed = time.time() - start_time

    # Save adapter
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  Adapter saved to: {adapter_path}")

    # Training report
    report = {
        "type": "dpo_lora",
        "base_model": args.base_model,
        "sft_adapter": args.sft_adapter,
        "adapter_path": str(adapter_path),
        "lora_rank": args.lora_rank,
        "beta": args.beta,
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

    print(f"\n[DONE] DPO training complete in {elapsed:.0f}s")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="DPO training for quant-trading LLM")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-adapter", default=None,
                        help="Path to SFT adapter to merge before DPO (recommended)")
    parser.add_argument("--data-dir", default="data/llm_training")
    parser.add_argument("--output-dir", default="ml_models/llm_adapters/dpo")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL penalty strength, lower=stronger preference)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
