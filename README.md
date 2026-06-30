# cocoindex-v1-intro

Code for blog post introducing CocoIndex's v1 API in action on a multimodal indexing pipeline with [LanceDB](https://docs.lancedb.com).

The pipeline and experiment scripts here were written with the help of CocoIndex's
[AI coding agent skill](https://cocoindex.io/docs/getting_started/ai_coding_agents/).

## Recreate the ABO Sample Data

This repo uses a small, reproducible slice of the [Amazon Berkeley Objects
dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) for the
blog examples. The data is not committed to the repo.

From the repo root, run:

```bash
uv run scripts/prepare_abo_subset.py
```

The script creates the same local sample used for the post:

- downloads the official `abo-listings.tar` archive
- downloads the official `images/metadata/images.csv.gz` manifest
- selects the first 1000 usable `SHOES` listings in deterministic archive order
- keeps listings with `item_id`, `domain_name`, `main_image_id`, title, and product type
- downloads only each selected product's small main image from `images/small/<path>`

It writes:

```text
data/
├── raw/
│   ├── abo-listings.tar
│   └── images.csv.gz
└── abo/
    ├── subset.jsonl
    ├── image_metadata.jsonl
    ├── listings/
    │   └── amazon.com__<item_id>.json
    └── images/
        └── <two-hex-chars>/<image-file>.jpg
```

The constants that define the sample are at the top of
`scripts/prepare_abo_subset.py`:

```python
DATA_DIR = Path("data")
SUBSET_SIZE = 1000
PRODUCT_TYPE = "SHOES"
```

ABO is licensed under CC BY 4.0. Credit for the data, including images and 3D
models, must be given to Amazon.com. If you publish derived work using this
dataset, cite the ABO CVPR 2022 paper referenced on the official dataset page.

## Run the pipeline

[`scripts/run_pipeline.py`](scripts/run_pipeline.py) is the single, base pipeline.
It loads the ABO subset (above) into a local LanceDB table, computing a text
embedding (`all-MiniLM-L6-v2`) and an image embedding (CLIP `clip-ViT-B-32`) per
product. The file defines a CocoIndex `app`, so the CocoIndex CLI runs it directly:

```bash
# catch-up: scan, sync, exit
cocoindex update scripts/run_pipeline.py

 # live: keep watching for changes
cocoindex update -L scripts/run_pipeline.py
```

`-L` keeps one engine watching `data/abo/listings/` and applies each change to the
same LanceDB table. The pipeline mounts one LanceDB target with
`num_transactions_before_optimize=500`, so CocoIndex batches target maintenance
during the initial 1,000-row ingest and during later source changes.

The file also stays runnable as a plain script, which adds a few convenience flags
the CocoIndex CLI doesn't expose:

```bash
uv run scripts/run_pipeline.py --overwrite
```

- `--limit N` — ingest only the first N products, for fast iteration.
- `--overwrite` — wipe the local LanceDB directory and pipeline state first.
- `--live` — keep running and watch the source (same as `cocoindex update -L`).
- `--no-indexes` — skip vector and full-text index declarations for quick smoke
  tests.

> [!NOTE]
> `--limit` is intentionally incompatible with `--live`, because live mode should
> watch the full source directory. The CocoIndex CLI has no `--overwrite`; delete
> `data/lancedb` and `data/cocoindex` (or use the script's `--overwrite`) to start
> from a clean slate.

## Blog experiments

### Live source changes

```bash
uv run scripts/run_live_source_experiment.py --overwrite
```

This drives the base pipeline (the same app as `cocoindex update -L`) through one
scripted sequence: it starts a live update, waits for the initial 1,000-row ingest,
then mutates the source files and waits for the target to settle after each step:

- delete one listing and wait for 999 target rows
- add the listing back and wait for 1,000 target rows
- edit another listing's title and wait for that target row to change

You can reproduce this by hand too: run `cocoindex update -L scripts/run_pipeline.py`
in one terminal and add, remove, or edit JSON files under `data/abo/listings/` in
another.

### Schema and logic changes

A schema or logic change to the base pipeline is just a code edit: change the row
model or a transformation function in `run_pipeline.py` and re-run
`cocoindex update`, the same as always. To exercise that end to end (and assert
that embeddings stay memoized), this experiment is self-contained:
[`scripts/run_lancedb_schema_experiment.py`](scripts/run_lancedb_schema_experiment.py)
reuses the base building blocks from `run_pipeline.py`, adds a nullable
`product_bucket` column plus a function that derives it, and drives one app through
three phases against the same table:

```bash
uv run scripts/run_lancedb_schema_experiment.py --overwrite
```

- `base` — original `ProductRow` schema; every embedding computed once
- `schema_nullable` — add nullable `product_bucket` (all NULL); nothing recomputes
- `logic` — compute `product_bucket` for every row; embeddings stay memoized

It keeps per-function call counters and asserts the memoization behavior (zero
embedding calls in the last two phases). For a quick smoke test:

```bash
uv run scripts/run_lancedb_schema_experiment.py --limit 5 --overwrite --no-indexes
```
