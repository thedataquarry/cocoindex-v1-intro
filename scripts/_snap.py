import sys, lancedb
sys.path.insert(0, "scripts")
import run_pipeline as rp
t = lancedb.connect(str(rp.LANCEDB_URI)).open_table(rp.TABLE_NAME)
cols = [c for c in t.schema.names]
rows = [r for r in t.to_arrow().to_pylist() if r["product_id"] == "B01N2AU9HU"]
r = rows[0]
print("SCHEMA:", cols)
print("CONTENT:", repr(r["content"][:90]))
print("TEXT_VEC_HEAD:", [round(float(x), 4) for x in r["text_embedding"][:4]])
print("IMG_VEC_HEAD:", [round(float(x), 4) for x in r["image_embedding"][:4]])
