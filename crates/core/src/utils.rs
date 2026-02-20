//! Shared utility functions used across multiple crates.

use chrono::NaiveDate;
use serde_json::{json, Value};
use std::process::Command;

// ── Python Discovery ────────────────────────────────────────────────

/// Find a working Python 3 interpreter.
///
/// Search order:
/// 1. `PYTHON_PATH` environment variable
/// 2. Platform-specific default (Windows: `%USERPROFILE%\...\Python312\python.exe`)
/// 3. `python3`
/// 4. `python`
pub fn find_python() -> Option<String> {
    let mut candidates = vec![
        std::env::var("PYTHON_PATH").unwrap_or_default(),
    ];

    #[cfg(windows)]
    candidates.push(format!(
        "{}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        std::env::var("USERPROFILE").unwrap_or_default(),
    ));

    candidates.push("python3".into());
    candidates.push("python".into());

    for path in &candidates {
        if path.is_empty() { continue; }
        if let Ok(output) = Command::new(path)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            if output.success() {
                return Some(path.clone());
            }
        }
    }
    None
}

/// Run a Python script and return structured JSON output.
pub fn run_python_script(python: &str, args: &[String]) -> Result<Value, String> {
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = Command::new(python)
        .args(&str_args)
        .env("PYTHONIOENCODING", "utf-8")
        .output()
        .map_err(|e| format!("Failed to start python: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if output.status.success() {
        Ok(json!({
            "status": "completed",
            "stdout": stdout,
            "stderr": stderr,
        }))
    } else {
        let detail = if stderr.is_empty() { &stdout } else { &stderr };
        Err(format!("Script exit {}: {}", output.status, detail))
    }
}

// ── Date Parsing ────────────────────────────────────────────────────

/// Parse a date string in `YYYY-MM-DD` format.
pub fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
}

/// Parse an optional date string, returning `None` for `None` or empty input.
pub fn parse_optional_date(s: Option<&str>) -> Option<NaiveDate> {
    s.filter(|v| !v.is_empty())
        .and_then(|v| parse_date(v))
}

/// Parse a date in compact `YYYYMMDD` format.
pub fn parse_date_compact(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y%m%d").ok()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_date() {
        assert!(parse_date("2024-01-15").is_some());
        assert!(parse_date("bad").is_none());
    }

    #[test]
    fn test_parse_optional_date() {
        assert!(parse_optional_date(Some("2024-01-15")).is_some());
        assert!(parse_optional_date(Some("")).is_none());
        assert!(parse_optional_date(None).is_none());
    }

    #[test]
    fn test_parse_date_compact() {
        let d = parse_date_compact("20240115").unwrap();
        assert_eq!(d.to_string(), "2024-01-15");
    }
}
