from __future__ import annotations

import argparse
import json
import shutil
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import cocoindex as coco
import numpy as np
from cocoindex.connectors import lancedb as coco_lancedb
from cocoindex.connectors import localfs
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.resources.file import PatternFilePathMatcher
from cocoindex.resources.schema import VectorSchema, VectorSchemaProvider
from lancedb.db import AsyncConnection as LanceAsyncConnection
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
ABO_DIR = DATA_DIR / "abo"
LISTINGS_DIR = ABO_DIR / "listings"
IMAGES_DIR = ABO_DIR / "images"
LANCEDB_URI = DATA_DIR / "lancedb"
COCOINDEX_STATE_DB = DATA_DIR / "cocoindex" / "state.lmdb"
TABLE_NAME = "abo_shoes"
APP_NAME = "abo-shoes-lancedb"
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"
LANCEDB_TRANSACTIONS_BEFORE_OPTIMIZE = 500

LIMIT: int | None = None
LIVE = False
DECLARE_INDEXES = True


class ClipImageEmbedder(VectorSchemaProvider):
    """A CLIP image encoder shaped like CocoIndex's own ``SentenceTransformerEmbedder``.

    It is a shared, picklable resource that lazily loads the model and reports its
    own vector schema, so the target column dimensions are inferred from the model
    rather than hard-coded. CocoIndex ships a text embedder but no image embedder,
    so this is the seam where we plug a new modality into the pipeline.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        return {"model_name": self._model_name}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._model_name = state["model_name"]
        self._model = None
        self._lock = threading.Lock()

    def __coco_memo_key__(self) -> object:
        """Swapping the model invalidates every memo that embedded with it."""
        return self._model_name

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    async def __coco_vector_schema__(self) -> VectorSchema:
        model = self._get_model()
        if hasattr(model, "get_embedding_dimension"):
            dim = model.get_embedding_dimension()
        else:
            dim = model.get_sentence_embedding_dimension()
        return VectorSchema(dtype=np.dtype(np.float32), size=int(dim))

    @coco.fn.as_async(batching=True, runner=coco.GPU, max_batch_size=64)
    def _embed(self, images: list[Image.Image]) -> list[NDArray[np.float32]]:
        """Per-batch body. The decorator gathers concurrent single-image calls
        from :meth:`embed` into one batch and runs them as a single CLIP pass."""
        embeddings = self._get_model().encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [np.asarray(vec, dtype=np.float32) for vec in embeddings]

    async def embed(self, image_bytes: bytes) -> NDArray[np.float32]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return await self._embed(image)


# Shared resources, bound once in the lifespan and reached through context.
LANCEDB = coco.ContextKey[LanceAsyncConnection]("lancedb")
TEXT_EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder]("text_embedder", detect_change=True)
IMAGE_EMBEDDER = coco.ContextKey[ClipImageEmbedder]("image_embedder", detect_change=True)

# Each vector column infers its dtype and size from the embedder behind its key.
TextVector = Annotated[NDArray[np.float32], TEXT_EMBEDDER]
ImageVector = Annotated[NDArray[np.float32], IMAGE_EMBEDDER]


@dataclass(frozen=True)
class Product:
    product_id: str
    domain_name: str
    title: str
    brand: str | None
    product_type: str | None
    bullets: list[str]
    color: str | None
    material: str | None
    style: str | None
    main_image_id: str
    main_image_path: str


@dataclass(frozen=True)
class ProductRow:
    product_id: str
    domain_name: str
    title: str
    content: str
    product_type: str | None
    brand: str | None
    image_path: str
    image_bytes: bytes
    text_embedding: TextVector
    image_embedding: ImageVector
    metadata_json: str


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    """Bind external resources once, then expose their handles through context."""
    COCOINDEX_STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    LANCEDB_URI.mkdir(parents=True, exist_ok=True)
    builder.settings.db_path = COCOINDEX_STATE_DB

    conn = await coco_lancedb.connect_async(str(LANCEDB_URI))
    builder.provide(LANCEDB, conn)
    builder.provide(TEXT_EMBEDDER, SentenceTransformerEmbedder(TEXT_EMBEDDING_MODEL))
    builder.provide(IMAGE_EMBEDDER, ClipImageEmbedder(IMAGE_EMBEDDING_MODEL))
    yield


def _first_value(values: Any, *, language: str = "en_US") -> str | None:
    if not isinstance(values, list):
        return None
    fallback: str | None = None
    for item in values:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if fallback is None:
            fallback = value
        if item.get("language_tag") == language:
            return value
    return fallback


def _all_values(values: Any, *, language: str = "en_US") -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        if item.get("language_tag") not in (language, None):
            continue
        value = item.get("value")
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _product_type(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    for item in values:
        if isinstance(item, dict) and isinstance(item.get("value"), str):
            return item["value"].strip()
    return None


@coco.fn(memo=True)
async def load_listing(file: localfs.File) -> dict[str, Any]:
    """A CocoIndex function: the file argument makes source changes observable."""
    return json.loads(await file.read_text())


@coco.fn(memo=True)
def normalize_product(raw: dict[str, Any]) -> Product:
    """Turn irregular ABO JSON into the object shape used by the pipeline."""
    return Product(
        product_id=raw["item_id"],
        domain_name=raw["domain_name"],
        title=_first_value(raw.get("item_name")) or raw["item_id"],
        brand=_first_value(raw.get("brand")),
        product_type=_product_type(raw.get("product_type")),
        bullets=_all_values(raw.get("bullet_point")),
        color=_first_value(raw.get("color")),
        material=_first_value(raw.get("material")),
        style=_first_value(raw.get("style")),
        main_image_id=raw["main_image_id"],
        main_image_path=raw["_main_image_path"],
    )


@coco.fn(memo=True)
def build_retrieval_text(product: Product) -> str:
    """Build the text representation that an agent or retriever should see."""
    sections = [
        ("Title", product.title),
        ("Brand", product.brand),
        ("Product type", product.product_type),
        ("Color", product.color),
        ("Material", product.material),
        ("Style", product.style),
    ]
    lines = [f"{label}: {value}" for label, value in sections if value]
    if product.bullets:
        lines.append("Details:")
        lines.extend(f"- {bullet}" for bullet in product.bullets)
    return "\n".join(lines)


@coco.fn(memo=True)
async def read_image_bytes(file: localfs.File) -> bytes:
    """Read image bytes through FileLike so image replacements invalidate work."""
    return await file.read()


@coco.fn(memo=True)
async def embed_text(text: str) -> TextVector:
    """Embed retrieval text with the shared sentence-transformer model."""
    return await coco.use_context(TEXT_EMBEDDER).embed(text)


@coco.fn(memo=True)
async def embed_image(image_bytes: bytes) -> ImageVector:
    """Embed the product image with the shared CLIP encoder."""
    return await coco.use_context(IMAGE_EMBEDDER).embed(image_bytes)


@coco.fn(memo=True)
def build_lancedb_row(
    product: Product,
    content: str,
    image_bytes: bytes,
    text_embedding: TextVector,
    image_embedding: ImageVector,
) -> ProductRow:
    metadata = {
        "main_image_id": product.main_image_id,
        "color": product.color,
        "material": product.material,
        "style": product.style,
    }
    return ProductRow(
        product_id=product.product_id,
        domain_name=product.domain_name,
        title=product.title,
        content=content,
        product_type=product.product_type,
        brand=product.brand,
        image_path=str(IMAGES_DIR / product.main_image_path),
        image_bytes=image_bytes,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


@coco.fn
async def process_product(
    listing_file: localfs.File,
    target: coco_lancedb.TableTarget[ProductRow],
) -> None:
    """One mounted processing component owns one product and its target row."""
    raw = await load_listing(listing_file)
    product = normalize_product(raw)
    content = build_retrieval_text(product)
    image_file = localfs.File(localfs.FilePath(IMAGES_DIR / product.main_image_path))
    image_bytes = await read_image_bytes(image_file)
    row = build_lancedb_row(
        product=product,
        content=content,
        image_bytes=image_bytes,
        text_embedding=await embed_text(content),
        image_embedding=await embed_image(image_bytes),
    )
    target.declare_row(row=row)


async def _take(items: Any, n: int) -> Any:
    """Yield at most ``n`` items from a source's async iterable (for test runs)."""
    count = 0
    async for item in items:
        if count >= n:
            return
        yield item
        count += 1


@coco.fn
async def app_main() -> None:
    """One live-capable pipeline from ABO listing files to the LanceDB target."""
    table_schema = await coco_lancedb.TableSchema.from_class(
        ProductRow,
        primary_key=["domain_name", "product_id"],
    )
    target = await coco_lancedb.mount_table_target(
        LANCEDB,
        TABLE_NAME,
        table_schema,
        num_transactions_before_optimize=LANCEDB_TRANSACTIONS_BEFORE_OPTIMIZE,
    )

    source = localfs.walk_dir(
        LISTINGS_DIR,
        path_matcher=PatternFilePathMatcher(included_patterns=["*.json"]),
        # Source is always live-capable; pass -L to `cocoindex update` (or
        # --live to this script) to actually watch the directory and run live.
        live=True,
    )
    items = source.items() if LIMIT is None else _take(source.items(), LIMIT)
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


def main() -> None:
    global LIMIT, LIVE, DECLARE_INDEXES
    parser = argparse.ArgumentParser(description="Run the ABO to LanceDB pipeline.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="ingest at most N products, for fast local test runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe the local LanceDB directory and pipeline state first.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="keep running and watch the source for changes continuously.",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="skip vector and full-text index declarations for quick smoke tests.",
    )
    args = parser.parse_args()
    LIMIT = args.limit
    LIVE = args.live
    DECLARE_INDEXES = not args.no_indexes
    if LIVE and LIMIT is not None:
        parser.error("--limit cannot be combined with --live; live mode watches the full source.")
    if args.overwrite:
        shutil.rmtree(LANCEDB_URI, ignore_errors=True)
        shutil.rmtree(COCOINDEX_STATE_DB.parent, ignore_errors=True)
    app.update_blocking(live=LIVE)
    if not LIVE:
        print(
            f"Completed run for table '{TABLE_NAME}' "
            f"with {LANCEDB_TRANSACTIONS_BEFORE_OPTIMIZE} transactions before optimize."
        )


if __name__ == "__main__":
    main()
