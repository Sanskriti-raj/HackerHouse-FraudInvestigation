# fraud_agent.py — 3 targeted fixes

## Fix 1: `_gather_evidence` — actually compute historical_baseline_ratio

Replace the `if deep:` block with:

```python
extra = {}
if deep:
    try:
        wide_win = self.engine.query_card_window(card, ts, 24 * 14, 24 * 14)
        extra["extended_14d_window_count"] = len(wide_win)
        if wide_win:
            wide_mean = sum(float(t.get("amount") or 0) for t in wide_win) / len(wide_win)
            hist_mean = float(base.get("mean_amount") or 0)
            extra["historical_baseline_ratio"] = round(wide_mean / hist_mean, 2) if hist_mean > 0 else 1.0
        else:
            extra["historical_baseline_ratio"] = 1.0
    except Exception as e:
        extra["extended_window_error"] = str(e)
        extra["historical_baseline_ratio"] = 1.0
    extra["linked_case_detail"] = devn.get("linked_closed_cases", [])[:10]
```

## Fix 2: `_llm_stage` — track llm_used based on actual success, not availability

Change the trace initialization and the except block:

```python
trace = {"llm_used": False, "pre_evidence_decision": None, "post_evidence_decision": None}
...
if _llm_available:
    try:
        ...
        trace["llm_used"] = True   # <-- set ONLY on confirmed success, inside the try
        ...
        return pass1, trace   # or pass2, trace
    except Exception as e:
        logger.warning(f"[{cid}] LLM call error ({e}); switching to Graph Analytical Synthesis.")
        trace["llm_used"] = False   # <-- explicit, don't rely on the initial value
```

This makes `llm_used` an honest record of what happened on that specific case, not a
blanket statement about whether an API key was configured.

## Fix 3: verdict-aware pre-evidence action in the fallback path

Replace:
```python
action_p1 = "FLAG_FOR_REVIEW"
```
with:
```python
action_p1 = self._action(prelim_verdict, prelim_conf)
```
so the pre-evidence recommendation reflects the actual preliminary signal direction
instead of a fixed placeholder.

## One more thing worth doing before you submit
Rename the comment "Zero-hallucination agentic fallback" to something like
"deterministic template fallback (used when LLM is unavailable or fails)" —
and don't carry the "zero-hallucination agentic" phrasing into your blog post
or report. It's a fine engineering fallback; it just isn't LLM reasoning, and
claiming otherwise is the kind of thing a judge will notice if they read two
case files side by side (one real LLM output, one templated) and compare tone.
