# Label/ y-lookup VLM downstream eval — run instructions

The agent built **all** the code but could not execute it in-session
(sandbox blocked Python execution). Run locally:

```bash
cd .

# Pilot (20 imgs, ~$3, ~10 min) — sanity check before full run
python \
  experiments/iter/run_label_vlm_downstream.py --pilot 20

# Inspect: experiments/results/vlm_downstream/label_summary.md
# If parse_fail rate > 15% on B, abort and re-examine prompt/serializer.
# If pilot Δ < -0.05, abort and re-examine extraction / axis mapping.

# Full run (100 imgs, 1600 calls, ~$15-25, ~1-2h)
python \
  experiments/iter/run_label_vlm_downstream.py --stage all

# Or stage-by-stage:
python experiments/iter/qa_generators/gen_label_qa.py
python experiments/iter/extract_label_curves.py --workers 2
python experiments/iter/vlm_qa_label.py --workers 8
python experiments/iter/score_label.py
```

## File index

| File | Purpose |
|---|---|
| `experiments/iter/curve_serializer.py` | mask -> resampled real-coord JSON curves |
| `experiments/iter/qa_generators/gen_label_qa.py` | sample 100 imgs (stratified by K), 8 Qs each, GT y from polylines |
| `experiments/iter/extract_label_curves.py` | run pipeline.run_pipeline() and cache curves to `curves_cache/<id>.json` |
| `experiments/iter/vlm_qa_label.py` | run gemini Conditions A/B, cache responses to `llm_cache/<qid>__{A,B}.json`, 8 threads, 3-retry exp backoff |
| `experiments/iter/score_label.py` | per-Q correctness + 10 000 cluster-bootstrap CI on Δ + per-K + worst regressions |
| `experiments/iter/run_label_vlm_downstream.py` | one-shot driver |

All scripts are **idempotent**: re-running picks up where it left off via on-disk caching.

## Cost guard

Every API call hits `llm_cache/<qid>__<cond>.json` first; existing files are not re-paid for. Pilot (20 imgs × 8 Q × 2 cond = 320 calls) → ~$3. Full (1600 calls) → ~$15-25.

## Stopping rule (from protocol §5.3)

- If parse-fail rate in B > 15%, stop — long context is breaking the model.
- If Δ < -5pp, stop — extraction is hurting more than helping.
