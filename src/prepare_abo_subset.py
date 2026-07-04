from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from pathlib import Path


BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
LISTINGS_ARCHIVE_URL = f"{BASE_URL}/archives/abo-listings.tar"
IMAGES_METADATA_URL = f"{BASE_URL}/images/metadata/images.csv.gz"
DATA_DIR = Path("data")
SUBSET_SIZE = 1000
PRODUCT_TYPE = "SHOES"
IMAGE_DOWNLOAD_WORKERS = 16


def download(url: str, path: Path, *, verbose: bool = True) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if verbose:
        print(f"Downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(path)


def listing_members(archive_path: Path) -> Iterable[tarfile.TarInfo]:
    with tarfile.open(archive_path) as tar:
        for member in tar:
            if member.isfile() and member.name.startswith("listings/metadata/") and member.name.endswith(".json.gz"):
                yield member


def iter_listings(archive_path: Path) -> Iterable[dict]:
    with tarfile.open(archive_path) as tar:
        members = [
            member
            for member in tar
            if member.isfile()
            and member.name.startswith("listings/metadata/")
            and member.name.endswith(".json.gz")
        ]
        for member in sorted(members, key=lambda item: item.name):
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            with gzip.GzipFile(fileobj=fileobj) as gz:
                for raw_line in gz:
                    yield json.loads(raw_line)


def english_value(values: object) -> str | None:
    if not isinstance(values, list):
        return None
    fallback: str | None = None
    for item in values:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        if fallback is None:
            fallback = value.strip()
        if item.get("language_tag") == "en_US":
            return value.strip()
    return fallback


def product_type_value(values: object) -> str | None:
    if not isinstance(values, list):
        return None
    for item in values:
        if isinstance(item, dict) and isinstance(item.get("value"), str):
            return item["value"].strip()
    return None


def has_teaching_shape(listing: dict) -> bool:
    return bool(
        listing.get("item_id")
        and listing.get("domain_name")
        and listing.get("main_image_id")
        and english_value(listing.get("item_name"))
        and product_type_value(listing.get("product_type"))
    )


def select_listings(archive_path: Path, limit: int, product_type: str | None) -> list[dict]:
    selected: list[dict] = []
    for listing in iter_listings(archive_path):
        if not has_teaching_shape(listing):
            continue
        if product_type is not None and product_type_value(listing.get("product_type")) != product_type:
            continue
        selected.append(listing)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"Only found {len(selected)} matching listings; requested {limit}."
        )
    return selected


def load_image_metadata(path: Path, image_ids: set[str]) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            if image_id in image_ids:
                found[image_id] = row
                if len(found) == len(image_ids):
                    break
    missing = image_ids - found.keys()
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"Missing image metadata for {len(missing)} ids: {preview}")
    return found


def image_url(image_id: str, path: str) -> str:
    extension = Path(path).suffix.lstrip(".")
    return f"https://m.media-amazon.com/image/I/{image_id}.{extension}"


def small_image_url(path: str) -> str:
    return f"{BASE_URL}/images/small/{path}"


def download_images(image_metadata: dict[str, dict[str, str]], images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, Path]] = []
    for image_id, row in sorted(image_metadata.items()):
        path = row["path"]
        out_path = images_dir / path
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((image_id, small_image_url(path), out_path))

    if not jobs:
        print("All selected product images are already present.")
        return

    print(f"Downloading {len(jobs)} missing product images with {IMAGE_DOWNLOAD_WORKERS} workers.")

    def _download_one(job: tuple[str, str, Path]) -> str:
        image_id, url, out_path = job
        try:
            download(url, out_path, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to download {image_id} from {url}") from exc
        return image_id

    completed = 0
    with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(_download_one, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(jobs):
                print(f"Downloaded {completed}/{len(jobs)} missing images.")


def write_subset(listings: list[dict], image_metadata: dict[str, dict[str, str]], abo_dir: Path) -> None:
    listings_dir = abo_dir / "listings"
    listings_dir.mkdir(parents=True, exist_ok=True)
    subset_path = abo_dir / "subset.jsonl"
    image_manifest_path = abo_dir / "image_metadata.jsonl"

    with subset_path.open("w", encoding="utf-8") as subset_file:
        for listing in listings:
            image_id = listing["main_image_id"]
            listing = dict(listing)
            listing["_main_image_path"] = image_metadata[image_id]["path"]
            subset_file.write(json.dumps(listing, ensure_ascii=False) + "\n")
            item_id = listing["item_id"].replace("/", "_")
            domain_name = listing["domain_name"].replace("/", "_")
            listing_path = listings_dir / f"{domain_name}__{item_id}.json"
            listing_path.write_text(
                json.dumps(listing, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    with image_manifest_path.open("w", encoding="utf-8") as metadata_file:
        for image_id, row in sorted(image_metadata.items()):
            enriched = dict(row)
            enriched["image_url"] = image_url(image_id, row["path"])
            enriched["local_path"] = f"images/{row['path']}"
            metadata_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def main() -> None:
    raw_dir = DATA_DIR / "raw"
    abo_dir = DATA_DIR / "abo"
    listings_archive = raw_dir / "abo-listings.tar"
    images_metadata = raw_dir / "images.csv.gz"

    download(LISTINGS_ARCHIVE_URL, listings_archive)
    download(IMAGES_METADATA_URL, images_metadata)

    listings = select_listings(listings_archive, SUBSET_SIZE, PRODUCT_TYPE)
    image_ids = {listing["main_image_id"] for listing in listings}
    metadata = load_image_metadata(images_metadata, image_ids)

    write_subset(listings, metadata, abo_dir)
    download_images(metadata, abo_dir / "images")

    print(
        json.dumps(
            {
                "records": len(listings),
                "images": len(metadata),
                "product_type": PRODUCT_TYPE,
                "subset": str(abo_dir / "subset.jsonl"),
                "listings_dir": str(abo_dir / "listings"),
                "images_dir": str(abo_dir / "images"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
