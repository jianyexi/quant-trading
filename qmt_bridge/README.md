# QMT Bridge — Python Sidecar

HTTP bridge between the Rust quant-trading system and QMT (迅投量化) via `xtquant` SDK.

## Prerequisites

1. A broker account with QMT/miniQMT permission enabled
2. QMT client installed and running in **miniQMT** mode
3. Python 3.8+ with `xtquant` installed

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `xtquant` may need to be installed from the QMT client directory
> if not available on PyPI: copy the `xtquant` folder to your Python `site-packages`.

## Usage

```bash
# Start the bridge (connects to QMT automatically)
python qmt_bridge.py --qmt-path "C:/QMT/userdata_mini" --account "YOUR_ACCOUNT_ID"

# Start without connecting (for development/testing)
python qmt_bridge.py --qmt-path "." --account "test" --no-connect
```

## API Endpoints

| Method | Path                    | Description                |
|--------|-------------------------|----------------------------|
| GET    | `/health`               | Bridge status              |
| POST   | `/connect`              | (Re)connect to QMT        |
| POST   | `/order`                | Place stock order          |
| POST   | `/cancel`               | Cancel an order            |
| GET    | `/positions`            | Query positions            |
| GET    | `/account`              | Query account assets       |
| GET    | `/orders`               | Today's orders             |
| GET    | `/order_result/<id>`    | Async order callback data  |

### Place Order Example

```bash
curl -X POST http://127.0.0.1:18090/order \
  -H "Content-Type: application/json" \
  -d '{"stock_code":"000001.SZ","price":10.5,"amount":100,"side":"buy","price_type":"limit"}'
```

## Configuration

In the Rust system's `config/default.toml`:

```toml
[qmt]
bridge_url = "http://127.0.0.1:18090"
account = "YOUR_ACCOUNT_ID"
qmt_path = "C:/QMT/userdata_mini"
```
