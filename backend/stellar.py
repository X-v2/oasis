from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Any
from dotenv import load_dotenv

load_dotenv()


HEX_64_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")


def compute_analysis_hash(payload: dict[str, Any]) -> str:
    def normalize(data):
        if isinstance(data, float):
            return round(data, 6)  # stabilize float precision

        if isinstance(data, dict):
            return {k: normalize(v) for k, v in sorted(data.items())}

        if isinstance(data, list):
            # sort list of dicts by itemId if present
            if all(isinstance(x, dict) and "itemId" in x for x in data):
                data = sorted(data, key=lambda x: x["itemId"])
            return [normalize(v) for v in data]

        return data

    normalized = normalize(payload)

    canonical = json.dumps(
        normalized,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()

def record_analysis_hash(analysis_id: str, data_hash: str, created_at: str) -> dict[str, Any]:
    result = {
        "status": "prototype_unconfigured",
        "network": "stellar-testnet",
        "contractId": os.getenv("STELLAR_ANALYSIS_CONTRACT_ID"),
        "txHash": None,
        "error": None,
    }

    if os.getenv("STELLAR_ENABLE_WRITE", "0") != "1":
        result["error"] = "Set STELLAR_ENABLE_WRITE=1"
        return result

    contract_id = os.getenv("STELLAR_ANALYSIS_CONTRACT_ID")
    source_secret = os.getenv("STELLAR_SOURCE_SECRET")

    cli_path = r"C:\Program Files (x86)\Stellar CLI\stellar.exe"

    if not contract_id or not source_secret:
        result["status"] = "config_error"
        result["error"] = "Missing contract id / source"
        return result

    command = [
        cli_path,
        "contract",
        "invoke",
        "--id",
        contract_id,
        "--source-account",
        source_secret,
        "--network",
        "testnet",
        "--send=yes",
        "--",
        "create",
        "--analysis_id",
        analysis_id,
        "--analysis_hash",
        data_hash,
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        stdout, stderr = process.communicate(timeout=45)

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    output = ((stdout or "") + (stderr or "")).strip()
    _safe_log("STELLAR OUTPUT", output)

    if process.returncode != 0:
        result["status"] = "failed"
        result["error"] = output
        return result

    tx_hash = _extract_tx_hash(output)

    result["status"] = "recorded"
    result["txHash"] = tx_hash or output
    result["error"] = None

    return result


def _extract_tx_hash(output: str) -> str | None:
    match = re.search(r"/tx/([a-f0-9]{64})", output)
    return match.group(1) if match else None


def get_onchain_hash(analysis_id: str) -> str | None:
    contract_id = os.getenv("STELLAR_ANALYSIS_CONTRACT_ID")
    source_secret = os.getenv("STELLAR_SOURCE_SECRET")

    cli_path = r"C:\Program Files (x86)\Stellar CLI\stellar.exe"

    if not contract_id or not source_secret:
        return None

    command = [
        cli_path,
        "contract",
        "invoke",
        "--id",
        contract_id,
        "--source-account",
        source_secret,
        "--network",
        "testnet",
        "--",
        "get",
        "--analysis_id",
        analysis_id,
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        stdout, stderr = process.communicate(timeout=30)

    except Exception as exc:
        _safe_log("GET HASH ERROR", str(exc))
        return None

    if process.returncode != 0:
        _safe_log("GET HASH STDERR", (stderr or "").strip())
        return None

    normalized = _parse_contract_hash(stdout or "")
    _safe_log("GET HASH STDOUT", stdout or "")
    if normalized is None:
        _safe_log("GET HASH PARSE MISS", stdout or "")
    return normalized


def _parse_contract_hash(stdout: str) -> str | None:
    text = stdout.strip()
    if not text:
        return None
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    match = HEX_64_RE.search(text)
    if match:
        return match.group(0).lower()
    return None


def _safe_log(label: str, value: str) -> None:
    safe_value = value.encode("utf-8", "backslashreplace").decode("utf-8", "ignore")
    print(f"{label}: {safe_value}")
