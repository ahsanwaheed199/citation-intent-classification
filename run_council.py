"""
LLM council for the citation-intent benchmark.

Runs THREE proxy models over each benchmark row, asks each to label the
SOFT 7-class intent, then:
  - where all three AGREE  -> a confident "council" label
  - where they DISAGREE    -> goes to the manual-review file (this is the
                              ~set Tom described: humans check the hard cases)

Reuses the same .env + OpenAI(base_url=...) pattern as scicite_llm_classifier.
Runs on the proxy (API), so no GPU / no cluster needed. Must be on the
university network (eduroam / HiWi pool / Cisco VPN).
"""
import os, json, time
from collections import Counter
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---- config ----
BENCHMARK = "benchmark/benchmark.jsonl"        # the file you scp'd down
OUT_DIR   = "benchmark"
MODELS = [                              # three different model families = a real "council"
    "hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8",
    "hosted_vllm/gpt-oss-120b",
    "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
]
SOFT_LABELS = ["Contextualize", "Use", "Justify Design Choice",
               "Signal Gap", "Highlight Limitation",
               "Evaluate Against", "Modify"]
# ----------------

load_dotenv()
api_key = os.getenv("LITELLM_API_KEY")
if not api_key:
    raise SystemExit("No LITELLM_API_KEY found in .env")
client = OpenAI(api_key=api_key, base_url="https://litellm.professor-x.de/v1")

LABEL_LIST = "\n".join(f"- {l}" for l in SOFT_LABELS)

def classify(model, text):
    """Ask one model for one label. Returns a clean label string or 'PARSE_FAIL'."""
    prompt = f"""Classify why this citation was made, using exactly one of these SOFT intent labels:
{LABEL_LIST}

Citation: "{text}"

Answer with ONLY the exact label text, nothing else."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        out = r.choices[0].message.content.strip()
        # match back to a known label (case-insensitive, tolerant of extra words)
        for lab in SOFT_LABELS:
            if lab.lower() in out.lower():
                return lab
        return "PARSE_FAIL"
    except Exception as e:
        print("   API error:", str(e)[:80])
        return "PARSE_FAIL"

# load benchmark
rows = [json.loads(l) for l in open(BENCHMARK)]
print(f"loaded {len(rows)} benchmark rows; querying {len(MODELS)} models each\n")

results = []
for i, row in enumerate(rows):
    votes = [classify(m, row["text"]) for m in MODELS]
    counts = Counter(v for v in votes if v != "PARSE_FAIL")
    if counts:
        top_label, top_n = counts.most_common(1)[0]
    else:
        top_label, top_n = "PARSE_FAIL", 0
    unanimous = (top_n == len(MODELS))                       # all 3 agree
    agree_with_mapping = (top_label == row["soft_intent"])   # council vs our auto-label

    results.append({
        "unique_id": row["unique_id"],
        "text": row["text"],
        "mapping_label": row["soft_intent"],   # what build_benchmark assigned
        "votes": votes,
        "council_label": top_label,
        "unanimous": unanimous,
        "agree_with_mapping": agree_with_mapping,
    })
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(rows)} done")
    time.sleep(0.2)   # gentle on the proxy

df = pd.DataFrame(results)
out = os.path.join(OUT_DIR, "council_all.csv")
df.to_csv(out, index=False)

# the disagreement set = rows to review by hand
review = df[(~df["unanimous"]) | (~df["agree_with_mapping"])]
review.to_csv(os.path.join(OUT_DIR, "council_to_review.csv"), index=False)

print("\n=== council summary ===")
print(f"total rows:                  {len(df)}")
print(f"all 3 models unanimous:      {df['unanimous'].sum()}")
print(f"council agrees with mapping: {df['agree_with_mapping'].sum()}")
print(f"-> rows needing manual review: {len(review)}")
print(f"\nwrote: {out}")
print(f"wrote: {os.path.join(OUT_DIR, 'council_to_review.csv')}  <- review these by hand")