import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SRC       = "soft/dataset/ACLARC_SOFT.tsv"
SRC_LABEL = "citation_function"          # SOFT's 7-class label, which we map down
OUT_DIR   = "data_lora/soft_scicite"
TEST_SIZE = 0.2
SEED      = 42

# SOFT 7-class -> SciCite 3-class mapping (the grouping we agreed on)
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
print("columns:", list(df.columns))

TEXT_COL = next(c for c in ["cite_context", "citation_context", "context", "text"]
                if c in df.columns)
print("using text column:", TEXT_COL)

df = df[[TEXT_COL, SRC_LABEL]].dropna()
df[SRC_LABEL] = df[SRC_LABEL].astype(str).str.strip()

# apply the mapping
df["scicite"] = df[SRC_LABEL].map(SOFT_TO_SCICITE)
unmapped = df[df["scicite"].isna()][SRC_LABEL].unique()
if len(unmapped):
    print("WARNING - these SOFT labels did not map:", unmapped)
df = df.dropna(subset=["scicite"])

LABEL_COL = "scicite"
print("\nclass counts:\n", df[LABEL_COL].value_counts())

labels   = sorted(df[LABEL_COL].unique())
label2id = {l: i for i, l in enumerate(labels)}

train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df[LABEL_COL])

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
for name, part in [("train", train_df), ("test", test_df)]:
    with open(out / f"{name}.jsonl", "w") as f:
        for _, r in part.iterrows():
            f.write(json.dumps({"text": r[TEXT_COL],
                                "label": label2id[r[LABEL_COL]]}) + "\n")
json.dump(label2id, open(out / "label2id.json", "w"), indent=2)
print(f"\nwrote {len(train_df)} train / {len(test_df)} test")
print("label2id:", label2id)
