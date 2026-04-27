"""SQLite-backed store for employee face embeddings."""
from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmployeeRecord:
    employee_id: str
    name: str
    embedding: np.ndarray
    enrolled_at: str


class EmbeddingDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                enrolled_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, employee_id: str, name: str, embedding: np.ndarray) -> None:
        blob = pickle.dumps(embedding.astype(np.float32))
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            """
            INSERT INTO employees (employee_id, name, embedding, enrolled_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(employee_id) DO UPDATE SET
                name = excluded.name,
                embedding = excluded.embedding,
                enrolled_at = excluded.enrolled_at
            """,
            (employee_id, name, blob, ts),
        )
        self._conn.commit()
        logger.info("Upserted employee %s (%s)", employee_id, name)

    def delete(self, employee_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM employees WHERE employee_id = ?", (employee_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, employee_id: str) -> Optional[EmployeeRecord]:
        row = self._conn.execute(
            "SELECT employee_id, name, embedding, enrolled_at FROM employees WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self) -> List[EmployeeRecord]:
        rows = self._conn.execute(
            "SELECT employee_id, name, embedding, enrolled_at FROM employees"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row) -> EmployeeRecord:
        emb = pickle.loads(row[2])
        return EmployeeRecord(
            employee_id=row[0],
            name=row[1],
            embedding=np.asarray(emb, dtype=np.float32),
            enrolled_at=row[3],
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
