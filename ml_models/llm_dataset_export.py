#!/usr/bin/env python3
"""
llm_dataset_export.py — Export training datasets for LLM post-training.

Sources:
  1. Chat history   (PostgreSQL chat_sessions + chat_messages)  → SFT pairs
  2. Sentiment data (PostgreSQL / in-memory sentiment_items)     → SFT pairs
  3. Trade journal  (SQLite data/trade_journal.db)               → DPO pairs

Output:  data/llm_training/{sft_chat.jsonl, sft_sentiment.jsonl, dpo_trades.jsonl}
Usage:   python ml_models/llm_dataset_export.py [--db-url ...] [--output-dir ...]
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pg_connect(db_url: str):
    """Connect to PostgreSQL. Returns (conn, cursor) or (None, None)."""
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        return conn, conn.cursor()
    except ImportError:
        print("[WARN] psycopg2 not installed — trying psycopg2-binary or skipping PG.")
        try:
            import psycopg2cffi as psycopg2  # type: ignore
            conn = psycopg2.connect(db_url)
            return conn, conn.cursor()
        except Exception:
            return None, None
    except Exception as e:
        print(f"[WARN] PostgreSQL connection failed: {e}")
        return None, None


def _write_jsonl(path: Path, records: list[dict]):
    """Write list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# 1. Chat History → SFT (instruction / response pairs)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一位专业的量化交易助手，擅长中国A股市场分析。"
    "你掌握技术分析（MACD/KDJ/RSI/布林带/均线）、基本面分析（PE/PB/ROE）、"
    "因子模型（动量/价值/质量/波动率）以及回测评估（夏普比率/最大回撤/年化收益率）。"
)


def export_chat_sft(db_url: str, output_dir: Path) -> int:
    """Export chat sessions as SFT instruction/response pairs."""
    conn, cur = _pg_connect(db_url)
    if not conn:
        print("[SKIP] Chat SFT export — no PostgreSQL connection")
        return 0

    try:
        # Get all sessions with messages
        cur.execute("""
            SELECT s.id, s.title
            FROM chat_sessions s
            ORDER BY s.created_at
        """)
        sessions = cur.fetchall()

        records = []
        for session_id, title in sessions:
            cur.execute("""
                SELECT role, content, tool_calls
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at
            """, (session_id,))
            messages = cur.fetchall()

            # Build conversation turns: pair consecutive user→assistant messages
            i = 0
            while i < len(messages) - 1:
                role, content, tool_calls = messages[i]
                if role == "user" and content and content.strip():
                    # Find next assistant response (skip tool results)
                    for j in range(i + 1, len(messages)):
                        next_role, next_content, _ = messages[j]
                        if next_role == "assistant" and next_content and next_content.strip():
                            records.append({
                                "instruction": content.strip(),
                                "input": "",
                                "output": next_content.strip(),
                                "system": SYSTEM_PROMPT,
                                "source": "chat_history",
                                "session_id": str(session_id),
                            })
                            i = j + 1
                            break
                    else:
                        i += 1
                else:
                    i += 1

        _write_jsonl(output_dir / "sft_chat.jsonl", records)
        return len(records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. Sentiment Data → SFT (news → score + reasoning)
# ---------------------------------------------------------------------------

def _score_to_label(score: float) -> str:
    if score >= 0.5:
        return "利好"
    elif score >= 0.2:
        return "轻微利好"
    elif score > -0.2:
        return "中性"
    elif score > -0.5:
        return "轻微利空"
    else:
        return "利空"


def export_sentiment_sft(db_url: str, output_dir: Path) -> int:
    """Export sentiment analysis records as SFT pairs."""
    conn, cur = _pg_connect(db_url)
    if not conn:
        # Fallback: try reading from a sentiment JSON export if available
        print("[SKIP] Sentiment SFT export — no PostgreSQL connection")
        return 0

    try:
        # Check if sentiment table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'sentiment_items'
            )
        """)
        if not cur.fetchone()[0]:
            # Sentiment data is in-memory only (SentimentStore), not persisted to PG
            print("[SKIP] Sentiment SFT — no sentiment_items table in PostgreSQL")
            print("  (Sentiment data is stored in-memory; consider persisting to PG)")
            return 0

        cur.execute("""
            SELECT symbol, source, title, content, sentiment_score
            FROM sentiment_items
            WHERE content IS NOT NULL AND content != ''
            ORDER BY published_at
        """)
        rows = cur.fetchall()

        records = []
        for symbol, source, title, content, score in rows:
            instruction = (
                f"请分析以下关于{symbol}的{source}，给出情感评分(-1到+1)和分析理由。\n\n"
                f"标题：{title}\n内容：{content}"
            )
            label = _score_to_label(score)
            output = f"情感评分：{score:.2f}（{label}）"

            records.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "system": SYSTEM_PROMPT,
                "source": "sentiment_analysis",
                "symbol": symbol,
            })

        _write_jsonl(output_dir / "sft_sentiment.jsonl", records)
        return len(records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. Trade Journal → DPO preference pairs
# ---------------------------------------------------------------------------

def export_trade_dpo(journal_db: Path, output_dir: Path) -> int:
    """
    Export trade outcomes as DPO preference pairs.
    
    Chosen  = analysis for profitable trades (positive P&L)
    Rejected = analysis for losing trades (negative P&L)
    
    We pair each winning trade with a losing trade on the same symbol (or 
    random losing trade if no match) to create preference pairs.
    """
    if not journal_db.exists():
        print(f"[SKIP] DPO export — {journal_db} not found")
        return 0

    conn = sqlite3.connect(str(journal_db))
    cur = conn.cursor()

    try:
        # Check for entries table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'")
        if not cur.fetchone():
            print("[SKIP] DPO export — no 'entries' table in journal DB")
            return 0

        cur.execute("""
            SELECT timestamp, symbol, entry_type, side, price, quantity, pnl, notes
            FROM entries
            WHERE entry_type = 'order_filled' AND pnl IS NOT NULL
            ORDER BY timestamp
        """)
        trades = cur.fetchall()

        if not trades:
            print("[SKIP] DPO export — no filled trades with P&L found")
            return 0

        winners = []
        losers = []

        for ts, symbol, _, side, price, qty, pnl, notes in trades:
            analysis = (
                f"交易分析 — {symbol}\n"
                f"时间：{ts}\n"
                f"方向：{'买入' if side == 'Buy' else '卖出'}，价格：{price}，数量：{qty}\n"
                f"盈亏：{pnl:+.2f}\n"
            )
            if notes:
                analysis += f"备注：{notes}\n"

            if pnl > 0:
                analysis += f"结论：该交易盈利{pnl:.2f}元，交易决策正确。建议继续保持该策略的纪律性。"
                winners.append({"symbol": symbol, "text": analysis, "pnl": pnl})
            else:
                analysis += f"结论：该交易亏损{abs(pnl):.2f}元，需要反思入场时机和风险控制。"
                losers.append({"symbol": symbol, "text": analysis, "pnl": pnl})

        # Build DPO pairs: match winners with losers
        records = []
        prompt_template = "请评估以下交易决策的质量，并给出改进建议。"

        used_losers = set()
        for w in winners:
            # Prefer same-symbol loser
            best_loser = None
            for idx, l in enumerate(losers):
                if idx not in used_losers:
                    if l["symbol"] == w["symbol"]:
                        best_loser = idx
                        break
            if best_loser is None:
                # Use any unused loser
                for idx in range(len(losers)):
                    if idx not in used_losers:
                        best_loser = idx
                        break
            if best_loser is None:
                continue  # no more losers to pair with

            used_losers.add(best_loser)
            records.append({
                "prompt": prompt_template,
                "chosen": w["text"],
                "rejected": losers[best_loser]["text"],
                "source": "trade_journal",
                "winner_symbol": w["symbol"],
                "winner_pnl": w["pnl"],
                "loser_symbol": losers[best_loser]["symbol"],
                "loser_pnl": losers[best_loser]["pnl"],
            })

        _write_jsonl(output_dir / "dpo_trades.jsonl", records)
        return len(records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export LLM training datasets")
    parser.add_argument(
        "--db-url",
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@127.0.0.1:5432/quant_trading",
        ),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--journal-db",
        default="data/trade_journal.db",
        help="Path to trade journal SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        default="data/llm_training",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--format",
        choices=["alpaca", "sharegpt"],
        default="alpaca",
        help="SFT data format (alpaca=instruction/input/output, sharegpt=conversations)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LLM Dataset Export] {datetime.now().isoformat()}")
    print(f"  DB URL: {args.db_url[:30]}...")
    print(f"  Output: {output_dir}")
    print()

    total = 0

    # 1. Chat SFT
    print("[1/3] Exporting chat history → SFT pairs...")
    n = export_chat_sft(args.db_url, output_dir)
    total += n
    print(f"  Chat SFT: {n} pairs")
    print()

    # 2. Sentiment SFT
    print("[2/3] Exporting sentiment data → SFT pairs...")
    n = export_sentiment_sft(args.db_url, output_dir)
    total += n
    print(f"  Sentiment SFT: {n} pairs")
    print()

    # 3. Trade DPO
    print("[3/3] Exporting trade journal → DPO preference pairs...")
    n = export_trade_dpo(Path(args.journal_db), output_dir)
    total += n
    print(f"  Trade DPO: {n} pairs")
    print()

    # Write manifest
    manifest = {
        "exported_at": datetime.now().isoformat(),
        "total_records": total,
        "files": {
            "sft_chat": str(output_dir / "sft_chat.jsonl"),
            "sft_sentiment": str(output_dir / "sft_sentiment.jsonl"),
            "dpo_trades": str(output_dir / "dpo_trades.jsonl"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Total {total} records exported. Manifest: {manifest_path}")

    # Output JSON summary for API consumption
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
