"""Multimodal query against the LanceDB table built by ``run_pipeline.py``.

CLIP maps text and images into the same vector space, so a natural-language
query can be matched directly against the image vectors. Every row already
holds the product's image bytes and metadata, so a single query returns the
ranked matches *and* everything we need to render them, with no second lookup.

Run it with::

    uv run scripts/query.py "white leather sneakers with a chunky sole"

It prints the top matches and writes a labeled grid of their images to
``data/query_results.png``.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import lancedb
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer

LANCEDB_URI = "data/lancedb"
TABLE_NAME = "abo_shoes"
IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"
OUTPUT_PATH = Path("data/query_results.png")

# Layout of the rendered result grid.
THUMB = 320
PADDING = 16
CAPTION_H = 96


def search(query: str, limit: int) -> list[dict]:
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(TABLE_NAME)

    # CLIP puts text and images in the same vector space, so a text query
    # can search the image vectors directly.
    clip = SentenceTransformer(IMAGE_EMBEDDING_MODEL)
    return (
        table.search(clip.encode(query), vector_column_name="image_embedding")
        .limit(limit)
        .select(["title", "brand", "product_type", "image_bytes", "metadata_json"])
        .to_list()
    )


def _caption_title(row: dict) -> str:
    """Prefer the product title, but fall back to the brand when the title is
    in a script the default font can't render (the ABO data is multilingual)."""
    title = row["title"]
    if all(ord(char) < 0x250 for char in title):  # Latin scripts the font covers
        return title
    return row["brand"] or title


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if font.getbbox(candidate)[2] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines[:2]


def render_grid(results: list[dict], query: str, output_path: Path) -> None:
    """Compose the matching product images into one labeled grid image."""
    try:
        font = ImageFont.truetype("Helvetica.ttc", 16)
        bold = ImageFont.truetype("Helvetica.ttc", 18)
    except OSError:
        font = bold = ImageFont.load_default()

    cell_w = THUMB + PADDING
    canvas = Image.new(
        "RGB",
        (cell_w * len(results) + PADDING, THUMB + CAPTION_H + 2 * PADDING),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for i, row in enumerate(results):
        x = PADDING + i * cell_w
        image = Image.open(BytesIO(row["image_bytes"])).convert("RGB")
        image.thumbnail((THUMB, THUMB))
        canvas.paste(image, (x + (THUMB - image.width) // 2, PADDING))

        caption_y = THUMB + 2 * PADDING
        for line in _wrap(_caption_title(row), bold, THUMB):
            draw.text((x, caption_y), line, fill="black", font=bold)
            caption_y += 22
        subtitle = " · ".join(v for v in (row["brand"], row["product_type"]) if v)
        draw.text((x, caption_y), subtitle, fill="#666666", font=font)

    canvas.save(output_path)
    print(f"\nWrote {len(results)} matches to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search over the ABO table.")
    parser.add_argument(
        "query",
        nargs="?",
        default="white leather sneakers with a chunky sole",
        help="natural-language description of the product to find.",
    )
    parser.add_argument("--limit", type=int, default=3, help="number of matches to return.")
    args = parser.parse_args()

    results = search(args.query, args.limit)
    print(f'Query: "{args.query}"\n')
    for i, row in enumerate(results, start=1):
        metadata = json.loads(row["metadata_json"])
        print(f"{i}. {row['title']} — {row['brand']} ({row['product_type']})")
        print(f"   color={metadata.get('color')} material={metadata.get('material')}")

    render_grid(results, args.query, OUTPUT_PATH)


if __name__ == "__main__":
    main()
