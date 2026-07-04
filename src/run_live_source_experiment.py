from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import cocoindex as coco
import lancedb
import run_pipeline as pipeline


def _load_listing(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _listing_identity(raw: dict[str, Any]) -> tuple[str, str]:
    return raw["domain_name"], raw["item_id"]


def _table_snapshot() -> dict[str, Any]:
    conn = lancedb.connect(str(pipeline.LANCEDB_URI))
    table = conn.open_table(pipeline.TABLE_NAME)
    rows = table.to_arrow().to_pylist()
    return {
        "version": table.version,
        "rows": table.count_rows(),
        "columns": table.schema.names,
        "records": rows,
    }


def _find_row(domain_name: str, product_id: str) -> dict[str, Any] | None:
    for row in _table_snapshot()["records"]:
        if row["domain_name"] == domain_name and row["product_id"] == product_id:
            return row
    return None


async def _wait_for(
    label: str,
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                snapshot = _table_snapshot()
                return {
                    "label": label,
                    "version": snapshot["version"],
                    "rows": snapshot["rows"],
                }
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
        await asyncio.sleep(1)
    raise TimeoutError(f"Timed out waiting for {label}; last_error={last_error!r}")


async def _wait_until_initial_ready(handle: coco.UpdateHandle[None]) -> None:
    async for snapshot in handle.watch():
        if snapshot.status == coco.UpdateStatus.READY:
            return


def _pick_listing_files() -> tuple[Path, Path]:
    files = sorted(pipeline.LISTINGS_DIR.glob("*.json"))
    if len(files) < 2:
        raise RuntimeError(f"Expected at least two listing files in {pipeline.LISTINGS_DIR}")
    return files[0], files[1]


async def run_experiment(*, overwrite: bool, declare_indexes: bool) -> dict[str, Any]:
    if overwrite:
        shutil.rmtree(pipeline.LANCEDB_URI, ignore_errors=True)
        shutil.rmtree(pipeline.COCOINDEX_STATE_DB.parent, ignore_errors=True)

    pipeline.LIMIT = None
    pipeline.LIVE = True
    pipeline.DECLARE_INDEXES = declare_indexes

    delete_path, update_path = _pick_listing_files()
    delete_original = delete_path.read_bytes()
    update_raw = _load_listing(update_path)

    deleted_domain, deleted_product_id = _listing_identity(_load_listing(delete_path))
    updated_domain, updated_product_id = _listing_identity(update_raw)

    async with coco.runtime():
        handle = pipeline.app.update(live=True)
        await _wait_until_initial_ready(handle)
        results = [
            await _wait_for(
                "initial ingest",
                lambda: _table_snapshot()["rows"] == 1000,
                timeout_seconds=120,
            )
        ]

        delete_path.unlink()
        results.append(
            await _wait_for(
                "delete one source record",
                lambda: (
                    _table_snapshot()["rows"] == 999
                    and _find_row(deleted_domain, deleted_product_id) is None
                ),
                timeout_seconds=60,
            )
        )

        delete_path.write_bytes(delete_original)
        results.append(
            await _wait_for(
                "restore deleted source record",
                lambda: (
                    _table_snapshot()["rows"] == 1000
                    and _find_row(deleted_domain, deleted_product_id) is not None
                ),
                timeout_seconds=60,
            )
        )

        marker = f"cocoindex live edit {int(time.time())}"
        item_name = update_raw.get("item_name")
        if not isinstance(item_name, list) or not item_name:
            raise RuntimeError(f"Listing has no editable item_name: {update_path}")
        first_name = item_name[0]
        if not isinstance(first_name, dict) or not isinstance(first_name.get("value"), str):
            raise RuntimeError(f"Listing has no editable item_name value: {update_path}")
        first_name["value"] = f"{first_name['value']} [{marker}]"
        update_path.write_text(json.dumps(update_raw, indent=2, sort_keys=True) + "\n")
        results.append(
            await _wait_for(
                "update one source property",
                lambda: (
                    (row := _find_row(updated_domain, updated_product_id)) is not None
                    and marker in row["title"]
                ),
                timeout_seconds=60,
            )
        )

    return {
        "cocoindex_version": getattr(coco, "__version__", None),
        "cocoindex_path": getattr(coco, "__file__", None),
        "lancedb_uri": str(pipeline.LANCEDB_URI),
        "table": pipeline.TABLE_NAME,
        "deleted_record": {
            "source_file": str(delete_path),
            "domain_name": deleted_domain,
            "product_id": deleted_product_id,
        },
        "updated_record": {
            "source_file": str(update_path),
            "domain_name": updated_domain,
            "product_id": updated_product_id,
        },
        "results": results,
        "note": (
            "The updated source file is intentionally left edited so the final "
            "target state still shows the property change."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the blog live-source-change experiment."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe LanceDB and CocoIndex state before starting the live watcher.",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="skip vector and full-text index declarations for quick smoke tests.",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_experiment(overwrite=args.overwrite, declare_indexes=not args.no_indexes)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
