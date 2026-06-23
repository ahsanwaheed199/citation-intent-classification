import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SRC       = "soft/dataset/ACLARC_SOFT.tsv"   # full 1930-row SOFT file
LABEL_COL = "citation_function"              # SOFT's own 7-class intent label
OUT_DIR   = "data_lora/soft_function"
TEST_SIZE = 0.2
SEED      = 42

df = pd.read_csv(SRC, sep="\t")
print("columns:", list(df.columns))

# the text column name differs between files, so detect it
TEXT_COL = next(c for c in ["cite_context","citation_context","context","text"]
                if c in df.columns)
print("using text column:", TEXT_COL)

df = df[[TEXT_COL, LABEL_COL]].dropna()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
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
