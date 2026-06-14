# Citation Intent Classification

Comparing citation intent classification across three taxonomies (ACL-ARC,
SciCite, SOFT), and analysing whether AI-authored papers cite differently from
human-authored ones.

**ML/NLP Praktikum SS 2026 — Universität Würzburg** · Supervisor: Tom Völker

## Research Questions
1. Do AI-authored papers cite differently than human-authored papers?
2. Does the choice of citation taxonomy change the answer?

## Setup
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Create a `.env` file (gitignored) with your API key and base URL.

## Datasets (not in repo — download separately)
- SciCite: https://github.com/allenai/scicite
- ACL-ARC: https://github.com/davidjurgens/citation-function
- SOFT: https://github.com/zhiyintan/SOFT
- Project APE (AI): https://ape.socialcatalystlab.org
- unarXive (human): https://unarxive.org

## Results so far (June 1, SciCite)
| Approach             | Accuracy | Macro F1 |
|----------------------|----------|----------|
| SciBERT + LR         | 82.4%    | 0.797    |
| TF-IDF + Naive Bayes | 74.5%    | 0.664    |
| LLM few-shot         | 73.3%    | 0.696    |
| LLM zero-shot        | 67.5%    | 0.658    |

## Authors
- Ahsan Waheed
- Ali Anwar