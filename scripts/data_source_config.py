#!/usr/bin/env python3
"""
Configurable data source provider selection.

Reads provider priority from environment variables (set by Rust API handler)
or falls back to config/default.toml. Provides a unified get_providers(market)
API used by market_cache.py and market_data.py.

Env vars (set by Rust when calling Python subprocess):
    QUANT_CN_PROVIDERS=tushare,akshare
    QUANT_US_PROVIDERS=yfinance
    QUANT_HK_PROVIDERS=yfinance
    QUANT_CACHE_ONLY=false
"""

import os
from typing import List

# Valid provider names
VALID_PROVIDERS = {"tushare", "akshare", "yfinance"}

# Defaults (match config/default.toml)
_DEFAULTS = {
    "CN": ["tushare", "akshare"],
    "US": ["yfinance"],
    "HK": ["yfinance"],
}


def get_providers(market: str = "CN") -> List[str]:
    """Get ordered provider list for a market.

    Checks env vars first, then falls back to defaults.
    """
    market = market.upper()
    env_key = f"QUANT_{market}_PROVIDERS"
    env_val = os.environ.get(env_key, "").strip()

    if env_val:
        providers = [p.strip().lower() for p in env_val.split(",") if p.strip()]
        # Filter to valid providers only
        return [p for p in providers if p in VALID_PROVIDERS] or _DEFAULTS.get(market, [])

    # Try loading from TOML (fallback for CLI / standalone script usage)
    toml_providers = _load_from_toml(market)
    if toml_providers:
        return toml_providers

    return _DEFAULTS.get(market, _DEFAULTS["CN"])


def is_cache_only() -> bool:
    """Check if cache-only mode is enabled."""
    env_val = os.environ.get("QUANT_CACHE_ONLY", "").strip().lower()
    if env_val in ("true", "1", "yes"):
        return True
    if env_val in ("false", "0", "no", ""):
        # Fallback to TOML
        return _load_cache_only_from_toml()
    return False


def _load_from_toml(market: str) -> List[str]:
    """Try to load provider config from config/default.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # Python < 3.11
        except ImportError:
            return []

    toml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "default.toml"
    )
    if not os.path.exists(toml_path):
        return []

    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        ds = config.get("data_source", {})
        key = f"{market.lower()}_providers"
        providers = ds.get(key, [])
        return [p for p in providers if p in VALID_PROVIDERS]
    except Exception:
        return []


def _load_cache_only_from_toml() -> bool:
    """Try to load cache_only from config/default.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return False

    toml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "default.toml"
    )
    if not os.path.exists(toml_path):
        return False

    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        return bool(config.get("data_source", {}).get("cache_only", False))
    except Exception:
        return False


def try_provider(provider: str, fetch_fn_map: dict, *args, **kwargs):
    """Try fetching data using a specific provider.

    Args:
        provider: Provider name ("tushare", "akshare", "yfinance")
        fetch_fn_map: Dict mapping provider name to callable
        *args, **kwargs: Passed to the callable

    Returns:
        (result, None) on success, (None, error_msg) on failure
    """
    fn = fetch_fn_map.get(provider)
    if fn is None:
        return None, f"{provider}: not configured"
    try:
        result = fn(*args, **kwargs)
        return result, None
    except ImportError:
        return None, f"{provider}: not installed"
    except Exception as e:
        return None, f"{provider}: {e}"


def provider_status() -> dict:
    """Return status of all providers with configured priority."""
    status = {
        "configured": {
            "CN": get_providers("CN"),
            "US": get_providers("US"),
            "HK": get_providers("HK"),
        },
        "cache_only": is_cache_only(),
        "available": {},
    }

    # Check tushare
    try:
        from tushare_provider import is_available as ts_ok
        status["available"]["tushare"] = ts_ok()
    except ImportError:
        status["available"]["tushare"] = False

    # Check akshare
    try:
        import akshare
        status["available"]["akshare"] = True
    except ImportError:
        status["available"]["akshare"] = False

    # Check yfinance
    try:
        from yfinance_provider import is_available as yf_ok
        status["available"]["yfinance"] = yf_ok()
    except ImportError:
        status["available"]["yfinance"] = False

    return status
