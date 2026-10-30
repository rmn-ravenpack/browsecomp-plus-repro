"""SQLite checkpoint for resumable BrowseComp-Plus uploads."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


class UploadState:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                docid TEXT PRIMARY KEY,
                url TEXT,
                file_name TEXT,
                language TEXT,
                is_english INTEGER,
                translation_requested INTEGER,
                published_ts TEXT,
                content_id TEXT,
                status TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_id ON documents(content_id)"
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()

    def uploaded_or_skipped_ids(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT docid FROM documents WHERE status IN ('uploaded', 'skipped')"
            ).fetchall()
        return {str(r["docid"]) for r in rows}

    def failed_ids(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT docid FROM documents WHERE status = 'failed'"
            ).fetchall()
        return {str(r["docid"]) for r in rows}

    def uploaded_file_names(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_name FROM documents WHERE status = 'uploaded' AND TRIM(file_name) != ''"
            ).fetchall()
        return [str(r["file_name"]) for r in rows]

    def export_mapping(
        self,
        tsv_path: Path,
        jsonl_path: Path | None = None,
        json_path: Path | None = None,
    ) -> int:
        import json

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT docid, content_id, file_name, url, published_ts, status
                FROM documents
                WHERE status = 'uploaded'
                ORDER BY length(docid), docid
                """
            ).fetchall()
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with tsv_path.open("w", encoding="utf-8") as fh:
            fh.write("docid\tcontent_id\tfile_name\turl\tpublished_ts\n")
            for r in rows:
                fh.write(
                    f"{r['docid']}\t{r['content_id']}\t{r['file_name']}\t{r['url']}\t{r['published_ts']}\n"
                )
        if jsonl_path is not None:
            with jsonl_path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(
                        json.dumps(
                            {
                                "docid": r["docid"],
                                "content_id": r["content_id"],
                                "file_name": r["file_name"],
                                "url": r["url"],
                                "published_ts": r["published_ts"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        if json_path is not None:
            docid_to_content_id = {str(r["docid"]): str(r["content_id"]) for r in rows}
            content_id_to_docid = {str(r["content_id"]): str(r["docid"]) for r in rows}
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(
                    {
                        "docid_to_content_id": docid_to_content_id,
                        "content_id_to_docid": content_id_to_docid,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return len(rows)

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
            ).fetchall()
            translation = self._conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE translation_requested = 1"
            ).fetchone()
        out = {str(r["status"]): int(r["n"]) for r in rows}
        out["translation_requested"] = int(translation["n"] if translation else 0)
        return out

    def record(self, row: dict[str, Any]) -> None:
        payload = {
            "docid": row["docid"],
            "url": row.get("url") or "",
            "file_name": row.get("file_name") or "",
            "language": row.get("language") or "",
            "is_english": 1 if row.get("is_english") else 0,
            "translation_requested": 1 if row.get("translation_requested") else 0,
            "published_ts": row.get("published_ts") or "",
            "content_id": (row.get("content_id") or "").strip().upper(),
            "status": row["status"],
            "error": row.get("error") or "",
            "updated_at": _utc_now(),
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO documents (
                    docid, url, file_name, language, is_english,
                    translation_requested, published_ts, content_id,
                    status, error, updated_at
                ) VALUES (
                    :docid, :url, :file_name, :language, :is_english,
                    :translation_requested, :published_ts, :content_id,
                    :status, :error, :updated_at
                )
                ON CONFLICT(docid) DO UPDATE SET
                    url=excluded.url,
                    file_name=excluded.file_name,
                    language=excluded.language,
                    is_english=excluded.is_english,
                    translation_requested=excluded.translation_requested,
                    published_ts=excluded.published_ts,
                    content_id=excluded.content_id,
                    status=excluded.status,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            self._conn.commit()
