# May 18 Milestone - Literature Review & Data Exploration

## Datasets Downloaded
- ✅ ACL-ARC (6 classes, complex format - XML/ANN files)
- ✅ SciCite (3 classes, simple JSONL format) 
- ✅ SOFT (not explored yet)

## SciCite Dataset Details
- **Format:** JSONL (one JSON object per line)
- **Training examples:** 8,243 citations
- **Classes:** 
  - Background (58.7%) - citing context/existing work
  - Method (27.8%) - using someone's method
  - Result (13.5%) - comparing results
- **Key fields:** string (citation sentence), label (class), sectionName

## Decision for June 1
- **Choose SciCite** (simplest, most examples)
- **Build 2 classifiers:**
  1. Fine-tuned encoder (SciBERT)
  2. LLM prompting (using LiteLLM API)
- **Compare results** - which is better?

## Next Steps (June 1 Milestone)
1. Build baseline classifier with simple approach
2. Test fine-tuned encoder approach
3. Test LLM prompting approach
4. Compare accuracy, F1, speed
5. Submit baseline results