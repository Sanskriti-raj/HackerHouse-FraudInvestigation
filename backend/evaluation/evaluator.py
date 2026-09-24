"""
Evaluation Module — HackerHouse Fraud Investigation Challenge
=============================================================
Scores agent verdicts against benchmark ground truth.
Scoring: FRAUD match = +2, NOT FRAUD match = +1, wrong = -1
"""
import json, os, sys
from typing import List, Dict, Any

GROUND_TRUTH = {
    "HHG-001": "NOT FRAUD",  "HHG-002": "FRAUD",
    "HHG-003": "FRAUD",      "HHG-004": "NOT FRAUD",
    "HHG-005": "FRAUD",      "HHG-006": "FRAUD",
    "HHG-007": "NOT FRAUD",  "HHG-008": "FRAUD",
    "HHG-009": "NOT FRAUD",  "HHG-010": "FRAUD",
    "HHG-011": "FRAUD",      "HHG-012": "NOT FRAUD",
    "HHG-013": "FRAUD",      "HHG-014": "FRAUD",
    "HHG-015": "NOT FRAUD",  "HHG-016": "FRAUD",
    "HHG-017": "FRAUD",      "HHG-018": "NOT FRAUD",
    "HHG-019": "FRAUD",      "HHG-020": "FRAUD",
}

SCORE_MAP = {("FRAUD","FRAUD"): 2, ("NOT FRAUD","NOT FRAUD"): 1,
             ("FRAUD","NOT FRAUD"): -1, ("NOT FRAUD","FRAUD"): -1}

def evaluate(results_dir: str) -> Dict[str, Any]:
    total_score = 0
    case_scores = []
    for case_id, gt in GROUND_TRUTH.items():
        fpath = os.path.join(results_dir, f"{case_id}.json")
        if not os.path.exists(fpath):
            case_scores.append({"case_id": case_id, "ground_truth": gt,
                                 "verdict": "MISSING", "score": -1, "correct": False})
            total_score -= 1
            continue
        with open(fpath) as f:
            result = json.load(f)
        verdict = result.get("verdict", "ERROR")
        score = SCORE_MAP.get((gt, verdict), -1)
        total_score += score
        case_scores.append({
            "case_id": case_id,
            "ground_truth": gt,
            "verdict": verdict,
            "confidence": result.get("confidence", 0),
            "score": score,
            "correct": score > 0,
            "signals": result.get("signal_count", 0),
            "action": result.get("recommended_action", ""),
        })
    
    correct = sum(1 for c in case_scores if c["correct"])
    accuracy = correct / len(GROUND_TRUTH)
    max_score = sum(2 if gt == "FRAUD" else 1 for gt in GROUND_TRUTH.values())
    
    return {
        "total_score": total_score,
        "max_possible_score": max_score,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total_cases": len(GROUND_TRUTH),
        "case_scores": case_scores,
        "estimated_rank_percentile": _estimate_rank(total_score, max_score),
    }

def _estimate_rank(score, max_score):
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.90: return "Top 5 (Elite)"
    if ratio >= 0.80: return "Top 10%"
    if ratio >= 0.70: return "Top 20%"
    if ratio >= 0.60: return "Top 40%"
    return "Bottom 60%"

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cases_dir = os.path.join(base_dir, "cases")
    report = evaluate(cases_dir)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"  Score:    {report['total_score']} / {report['max_possible_score']}")
    print(f"  Accuracy: {report['accuracy']:.1%} ({report['correct']}/{report['total_cases']})")
    print(f"  Rank:     {report['estimated_rank_percentile']}")
    print(f"{'='*60}")
    for c in report["case_scores"]:
        marker = "OK" if c["correct"] else "XX"
        print(f"  [{marker}] {c['case_id']:10s} GT={c['ground_truth']:10s} "
              f"PRED={c['verdict']:10s} conf={c.get('confidence',0):.1%} score={c['score']:+d}")
    
    out_path = os.path.join(cases_dir, "_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull evaluation saved to: {out_path}")
    return report

if __name__ == "__main__":
    main()
