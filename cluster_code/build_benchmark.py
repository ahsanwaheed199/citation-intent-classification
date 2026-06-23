import json
from pathlib import Path
import pandas as pd

SRC     = "soft/dataset/ACLARC_SOFT.tsv"
OUT_DIR = "benchmark"
N       = 400          # target size, inside the 200-500 range from the slides
SEED    = 42

# SOFT 7-class intent -> SciCite 3-class (same mapping we trained on)
SOFT_TO_SCICITE = {
    "Contextualize":         "background",
    "Signal Gap":            "background",
    "Use":                   "method",
    "Modify":                "method",
    "Justify Design Choice": "method",
    "Evaluate Against":      "result",
    "Highlight Limitation":  "result",
}

df = pd.read_csv(SRC, sep="\t")

TEXT_COL = next(c for c in ["citation_context", "cite_context", "context", "text"]
                if c in df.columns)

# keep rows that have every label we need
cols = ["unique_id", TEXT_COL, "citation_function", "citation_object", "aclarc_label"]
df = df[cols].dropna()
for c in ["citation_function", "citation_object", "aclarc_label"]:
    df[c] = df[c].astype(str).str.strip()

# derive SciCite from the SOFT intent
df["scicite_label"] = df["citation_function"].map(SOFT_TO_SCICITE)
df = df.dropna(subset=["scicite_label"])

# proportional stratified sample on the finest taxonomy (SOFT intent),
# so every class is represented and total is ~N
frac = min(1.0, N / len(df))
sample = df.groupby("citation_function", group_keys=False).sample(
    frac=frac, random_state=SEED)

records = []
for _, r in sample.iterrows():
    records.append({
        "unique_id":    r["unique_id"],
        "text":         r[TEXT_COL],
        "soft_intent":  r["citation_function"],   # SOFT taxonomy (7-class intent)
        "soft_content": r["citation_object"],     # SOFT taxonomy (3-class content)
        "aclarc":       r["aclarc_label"],         # ACL-ARC taxonomy (6-class)
        "scicite":      r["scicite_label"],        # SciCite taxonomy (3-class, mapped)
    })

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
with open(out / "benchmark.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
pd.DataFrame(records).to_csv(out / "benchmark.csv", index=False)

print(f"wrote {len(records)} triply-labeled contexts to {OUT_DIR}/")
b = pd.DataFrame(records)
print("\nSOFT intent counts:\n", b["soft_intent"].value_counts())
print("\nACL-ARC counts:\n", b["aclarc"].value_counts())
print("\nSciCite counts:\n", b["scicite"].value_counts())
