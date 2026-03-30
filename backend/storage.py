from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator


DB_PATH = "analysis.db"


@contextmanager
def connect_db() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with connect_db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                total_cost REAL NOT NULL,
                total_area REAL NOT NULL,
                cost_per_m2 REAL NOT NULL,
                model_json TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                stellar_status TEXT NOT NULL,
                stellar_network TEXT,
                stellar_contract_id TEXT,
                stellar_tx_hash TEXT,
                stellar_error TEXT,
                verification_status TEXT DEFAULT 'verified',
                last_verified_at TEXT
            );

            CREATE TABLE IF NOT EXISTS analysis_items (
                item_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                element_type TEXT NOT NULL,
                material TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT NOT NULL,
                unit_rate REAL NOT NULL,
                subtotal REAL NOT NULL,
                justification TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
            );
            """
        )


def insert_analysis(
    analysis: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                analysis_id,
                created_at,
                total_cost,
                total_area,
                cost_per_m2,
                model_json,
                data_hash,
                stellar_status,
                stellar_network,
                stellar_contract_id,
                stellar_tx_hash,
                stellar_error,
                verification_status,
                last_verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis["analysisId"],
                analysis["createdAt"],
                analysis["totalCost"],
                analysis["totalArea"],
                analysis["costPerM2"],
                json.dumps(analysis["modelJson"], separators=(",", ":"), sort_keys=True),
                analysis["dataHash"],
                analysis["stellar"]["status"],
                analysis["stellar"].get("network"),
                analysis["stellar"].get("contractId"),
                analysis["stellar"].get("txHash"),
                analysis["stellar"].get("error"),
                "verified",  # initial state
                analysis["createdAt"],  # initial verification time
            ),
        )

        connection.executemany(
            """
            INSERT INTO analysis_items (
                item_id,
                analysis_id,
                element_type,
                material,
                quantity,
                unit,
                unit_rate,
                subtotal,
                justification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["itemId"],
                    analysis["analysisId"],
                    item["elementType"],
                    item["material"],
                    item["quantity"],
                    item["unit"],
                    item["unitRate"],
                    item["subtotal"],
                    item["justification"],
                )
                for item in items
            ],
        )


def list_analyses() -> list[dict[str, Any]]:
    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM analyses
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [_hydrate_analysis(connection, row) for row in rows]


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    with connect_db() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM analyses
            WHERE analysis_id = ?
            """,
            (analysis_id,),
        ).fetchone()

        if row is None:
            return None

        return _hydrate_analysis(connection, row)


def update_verification_status(analysis_id: str, status: str, verified_at: str) -> None:
    with connect_db() as connection:
        connection.execute(
            """
            UPDATE analyses
            SET verification_status = ?, last_verified_at = ?
            WHERE analysis_id = ?
            """,
            (status, verified_at, analysis_id),
        )


def _hydrate_analysis(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    items = connection.execute(
        """
        SELECT *
        FROM analysis_items
        WHERE analysis_id = ?
        ORDER BY item_id ASC
        """,
        (row["analysis_id"],),
    ).fetchall()

    return {
        "analysisId": row["analysis_id"],
        "createdAt": row["created_at"],
        "totalCost": row["total_cost"],
        "totalArea": row["total_area"],
        "costPerM2": row["cost_per_m2"],
        "lineItems": [
            {
                "itemId": item["item_id"],
                "elementType": item["element_type"],
                "material": item["material"],
                "quantity": item["quantity"],
                "unit": item["unit"],
                "unitRate": item["unit_rate"],
                "subtotal": item["subtotal"],
                "justification": item["justification"],
            }
            for item in items
        ],
        "modelJson": json.loads(row["model_json"]),
        "dataHash": row["data_hash"],
        "verificationStatus": row["verification_status"],
        "lastVerifiedAt": row["last_verified_at"],  # 👈 NEW
        "stellar": {
            "status": row["stellar_status"],
            "network": row["stellar_network"],
            "contractId": row["stellar_contract_id"],
            "txHash": row["stellar_tx_hash"],
            "error": row["stellar_error"],
        },
    }