"""Schema- and logic-change experiment for the ABO -> LanceDB pipeline.

``run_pipeline.py`` is the single, base pipeline the blog post walks through. This
script owns the schema-evolution story end to end so that ``run_pipeline.py`` stays
a clean, edit-and-rerun example. It reuses the base building blocks (the parsing
and embedding functions, ``Product``/``ProductRow``, the shared resources) and adds:

- a nullable ``product_bucket`` column on the row model,
- a ``compute_product_bucket`` function that derives its value,

then drives one app through three phases against the same table:

- ``base``            -> the original ``ProductRow`` schema,
- ``schema_nullable`` -> add ``product_bucket`` with no value yet (all NULL),
- ``logic``           -> compute ``product_bucket`` for every row.

Per-function counters confirm the point: the embedding functions stay memoized
across all three phases, so only the genuinely new work runs in each one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from dataclasses import dataclass
from typing import Any, Literal

import cocoindex as coco
import lancedb
import run_pipeline as rp
from cocoindex.connectors import lancedb as coco_lancedb
from cocoindex.connectors import localfs
from cocoindex.resources.file import PatternFilePathMatcher

Phase = Literal["base", "schema_nullable", "logic"]

APP_NAME = "abo-shoes-lancedb-schema-exp"
PHASE: Phase = "base"
LIMIT: int | None = None
DECLARE_INDEXES = True

_stats_lock = threading.Lock()
_run_stats = {
    "text_embedding_calls": 0,
    "image_embedding_calls": 0,
    "product_bucket_calls": 0,
}


def reset_run_stats() -> None:
    with _stats_lock:
        for key in _run_stats:
            _run_stats[key] = 0


def get_run_stats() -> dict[str, int]:
    with _stats_lock:
        return dict(_run_stats)


def _record_call(key: str) -> None:
    with _stats_lock:
        _run_stats[key] += 1


@dataclass(frozen=True)
class ProductRowWithProductBucket(rp.ProductRow):
    """The base row plus one nullable, derived column."""

    product_bucket: str | None = None


@coco.fn(memo=True)
async def embed_text(text: str) -> rp.TextVector:
    """Same as the base embedder, but counts real (non-memoized) calls."""
    _record_call("text_embedding_calls")
    return await coco.use_context(rp.TEXT_EMBEDDER).embed(text)


@coco.fn(memo=True)
async def embed_image(image_bytes: bytes) -> rp.ImageVector:
    """Same as the base embedder, but counts real (non-memoized) calls."""
    _record_call("image_embedding_calls")
    return await coco.use_context(rp.IMAGE_EMBEDDER).embed(image_bytes)


@coco.fn(memo=True)
def compute_product_bucket(product: rp.Product) -> str:
    """A deliberately simple derived column for the logic-change phase."""
    _record_call("product_bucket_calls")
    brand = product.brand or "unknown-brand"
    product_type = product.product_type or "unknown-type"
    normalized_brand = "-".join(brand.lower().split())
    normalized_type = "-".join(product_type.lower().split())
    return f"{product.domain_name}:{normalized_type}:{normalized_brand}"


@coco.fn(memo=True)
def build_row_with_bucket(
    product: rp.Product,
    content: str,
    image_bytes: bytes,
    text_embedding: rp.TextVector,
    image_embedding: rp.ImageVector,
    product_bucket: str | None,
) -> ProductRowWithProductBucket:
    base = rp.build_lancedb_row(
        product=product,
        content=content,
        image_bytes=image_bytes,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
    )
    return ProductRowWithProductBucket(**base.__dict__, product_bucket=product_bucket)


@coco.fn
async def process_product(
    listing_file: localfs.File,
    target: coco_lancedb.TableTarget[Any],
) -> None:
    """The base component, extended to populate product_bucket per phase."""
    raw = await rp.load_listing(listing_file)
    product = rp.normalize_product(raw)
    content = rp.build_retrieval_text(product)
    image_file = localfs.File(localfs.FilePath(rp.IMAGES_DIR / product.main_image_path))
    image_bytes = await rp.read_image_bytes(image_file)
    text_embedding = await embed_text(content)
    image_embedding = await embed_image(image_bytes)
    if PHASE == "base":
        row: Any = rp.build_lancedb_row(
            product=product,
            content=content,
            image_bytes=image_bytes,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
        )
    else:
        product_bucket = compute_product_bucket(product) if PHASE == "logic" else None
        row = build_row_with_bucket(
            product=product,
            content=content,
            image_bytes=image_bytes,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            product_bucket=product_bucket,
        )
    target.declare_row(row=row)


@coco.fn
async def app_main() -> None:
    row_type: type[Any] = rp.ProductRow if PHASE == "base" else ProductRowWithProductBucket
    table_schema = await coco_lancedb.TableSchema.from_class(
        row_type,
        primary_key=["domain_name", "product_id"],
    )
    target = await coco_lancedb.mount_table_target(
        rp.LANCEDB,
        rp.TABLE_NAME,
        table_schema,
    )

    source = localfs.walk_dir(
        rp.LISTINGS_DIR,
        path_matcher=PatternFilePathMatcher(included_patterns=["*.json"]),
        live=True,
    )
    items = source.items() if LIMIT is None else rp._take(source.items(), LIMIT)
    products = await coco.mount_each(
        coco.component_subpath("products"),
        process_product,
        items,
        target,
    )
    await products.ready()

    if DECLARE_INDEXES:
        target.declare_vector_index(name="text_embedding", column="text_embedding")
        target.declare_vector_index(name="image_embedding", column="image_embedding")
        target.declare_fts_index(name="content", column="content")


app = coco.App(coco.AppConfig(name=APP_NAME), app_main)


def _inspect_table(phase: Phase) -> dict[str, Any]:
    conn = lancedb.connect(str(rp.LANCEDB_URI))
    table = conn.open_table(rp.TABLE_NAME)
    arrow_table = table.to_arrow()
    columns = table.schema.names

    product_bucket_nulls: int | None = None
    product_bucket_non_nulls: int | None = None
    product_bucket_sample: str | None = None
    if "product_bucket" in columns:
        values = arrow_table.column("product_bucket").to_pylist()
        non_null_values = [value for value in values if value is not None]
        product_bucket_nulls = len(values) - len(non_null_values)
        product_bucket_non_nulls = len(non_null_values)
        product_bucket_sample = non_null_values[0] if non_null_values else None

    return {
        "phase": phase,
        "version": table.version,
        "rows": table.count_rows(),
        "columns": columns,
        "product_bucket_nulls": product_bucket_nulls,
        "product_bucket_non_nulls": product_bucket_non_nulls,
        "product_bucket_sample": product_bucket_sample,
        "function_calls": get_run_stats(),
    }


def _run_phase(phase: Phase, *, limit: int | None, declare_indexes: bool) -> dict[str, Any]:
    global PHASE, LIMIT, DECLARE_INDEXES
    PHASE = phase
    LIMIT = limit
    DECLARE_INDEXES = declare_indexes
    reset_run_stats()
    app.update_blocking(live=False)
    return _inspect_table(phase)


def _assert_results(results: list[dict[str, Any]], expected_rows: int) -> None:
    by_phase = {result["phase"]: result for result in results}
    base = by_phase["base"]
    schema_nullable = by_phase["schema_nullable"]
    logic = by_phase["logic"]

    assert base["rows"] == expected_rows
    assert "product_bucket" not in base["columns"]
    assert base["function_calls"]["text_embedding_calls"] == expected_rows
    assert base["function_calls"]["image_embedding_calls"] == expected_rows

    assert schema_nullable["rows"] == expected_rows
    assert "product_bucket" in schema_nullable["columns"]
    assert schema_nullable["product_bucket_nulls"] == expected_rows
    assert schema_nullable["product_bucket_non_nulls"] == 0
    assert schema_nullable["function_calls"]["text_embedding_calls"] == 0
    assert schema_nullable["function_calls"]["image_embedding_calls"] == 0
    assert schema_nullable["function_calls"]["product_bucket_calls"] == 0

    assert logic["rows"] == expected_rows
    assert logic["product_bucket_nulls"] == 0
    assert logic["product_bucket_non_nulls"] == expected_rows
    assert logic["function_calls"]["text_embedding_calls"] == 0
    assert logic["function_calls"]["image_embedding_calls"] == 0
    assert logic["function_calls"]["product_bucket_calls"] == expected_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the blog schema-change experiment against the real ABO -> LanceDB "
            "pipeline."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="ingest at most N products; omit for the full 1000-row blog run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe LanceDB and CocoIndex state before running the phases.",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="skip vector and full-text index declarations for quick smoke tests.",
    )
    args = parser.parse_args()

    if args.overwrite:
        shutil.rmtree(rp.LANCEDB_URI, ignore_errors=True)
        shutil.rmtree(rp.COCOINDEX_STATE_DB.parent, ignore_errors=True)

    declare_indexes = not args.no_indexes
    results = [
        _run_phase("base", limit=args.limit, declare_indexes=declare_indexes),
        _run_phase("schema_nullable", limit=args.limit, declare_indexes=declare_indexes),
        _run_phase("logic", limit=args.limit, declare_indexes=declare_indexes),
    ]

    expected_rows = args.limit or 1000
    _assert_results(results, expected_rows)
    print(
        json.dumps(
            {
                "cocoindex_version": getattr(coco, "__version__", None),
                "cocoindex_path": getattr(coco, "__file__", None),
                "lancedb_uri": str(rp.LANCEDB_URI),
                "table": rp.TABLE_NAME,
                "limit": args.limit,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
