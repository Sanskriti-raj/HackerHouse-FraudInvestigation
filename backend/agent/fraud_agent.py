"""
HackerHouse Fraud Investigation Agent
======================================
Processes all 20 benchmark cases using the TigerGraph in-memory engine.
9-signal rule engine fused with model risk scores for fraud verdicts.
"""
import sys, os, json, logging
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.graph.engine import TigerGraphEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraud_agent")

POLICY = {
    "risk_score_high": 0.70, "risk_score_medium": 0.50,
    "velocity_count": 6, "region_ratio": 0.5,
    "card_test_micro": 3, "amount_spike": 3.0,
    "device_shared": 3,
}


class Sig:
    def __init__(self, code, desc, w, ev=None):
        self.code, self.desc, self.w, self.ev = code, desc, w, ev
    def to_dict(self):
        return {"code": self.code, "description": self.desc, "weight": self.w, "evidence": self.ev}


class FraudInvestigationAgent:
    def __init__(self, data_dir):
        logger.info("Loading TigerGraph engine...")
        self.engine = TigerGraphEngine(data_dir)
        self.data_dir = data_dir
        logger.info("Engine ready.")

    def investigate(self, case):
        cid = case["case_id"]
        cust = case["customer_id"]
        card = case["card_id"]
        txn_id = case["flagged_txn_id"]
        risk = float(case.get("model_risk_score") or 0.0)
        amt = float(case.get("amount") or 0.0)
        ts_str = case.get("ts", "")
        region = case.get("addr1")
        dev = case.get("device_profile")
        proxy = case.get("proxy")

        logger.info(f"[{cid}] txn={txn_id} cust={cust} risk={risk:.2f}")

        try:
            ts = datetime.strptime(str(ts_str), "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = datetime.now()

        sigs = []
        steps = []

        win = self.engine.query_card_window(card, ts, 48, 48)
        steps.append({"step": "card_window_48h", "txns_found": len(win)})

        base = self.engine.query_customer_baseline(cust, ts, 90)
        steps.append({"step": "customer_baseline", "result": base})

        if dev and str(dev) not in ("nan", "None", ""):
            devn = self.engine.query_device_neighbors(str(dev))
        else:
            devn = {"shared_txn_count": 0, "linked_closed_cases": []}
        steps.append({"step": "device_neighbors", "result": devn})

        ct = self.engine.detect_card_testing(card, ts)
        steps.append({"step": "card_testing", "result": ct})

        # R1 High risk score
        if risk >= POLICY["risk_score_high"]:
            sigs.append(Sig("R1_HIGH_RISK", f"Risk={risk:.2f} >= {POLICY['risk_score_high']}", 2.5, {"risk": risk}))
        elif risk >= POLICY["risk_score_medium"]:
            sigs.append(Sig("R2_ELEVATED_RISK", f"Risk={risk:.2f} elevated", 1.2, {"risk": risk}))

        # R3 Velocity
        if len(win) > POLICY["velocity_count"]:
            sigs.append(Sig("R3_VELOCITY", f"{len(win)} txns in 48h", 1.5, {"count": len(win)}))

        # R4 Geographic anomaly
        typical = set(str(r) for r in base.get("typical_regions", []))
        reg_str = str(region) if region else None
        if reg_str and typical and reg_str not in typical:
            unf = sum(1 for t in win if str(t.get("addr1","")) not in typical)
            if unf / max(len(win), 1) >= POLICY["region_ratio"]:
                sigs.append(Sig("R4_GEO_ANOMALY", f"Region {reg_str} not in typical {list(typical)[:3]}", 1.8,
                                {"flagged": reg_str, "typical": list(typical)[:3]}))

        # R5 Card testing
        if ct.get("is_card_testing"):
            sigs.append(Sig("R5_CARD_TESTING",
                            f"{ct['micro_auth_count']} micro-auths + larger purchase", 3.0, ct))

        # R6 Amount spike
        hmean = float(base.get("mean_amount") or 0.0)
        if hmean > 0 and amt > POLICY["amount_spike"] * hmean:
            sigs.append(Sig("R6_AMOUNT_SPIKE", f"${amt:.2f} = {amt/hmean:.1f}x mean ${hmean:.2f}", 1.5,
                            {"amt": amt, "mean": hmean}))

        # R7 Shared device
        shared = devn.get("shared_txn_count", 0)
        if shared >= POLICY["device_shared"]:
            sigs.append(Sig("R7_SHARED_DEVICE", f"Device shared by {shared} txns", 1.2,
                            {"dev": dev, "shared": shared}))

        # R8 Proxy/VPN
        if proxy and str(proxy) not in ("nan", "None", ""):
            sigs.append(Sig("R8_PROXY", f"Proxy/VPN: {proxy}", 2.0, {"proxy": proxy}))

        # R9 Prior linked fraud cases
        linked = devn.get("linked_closed_cases", [])
        if linked:
            sigs.append(Sig("R9_PRIOR_LINK", f"Linked to {len(linked)} prior fraud cases", 1.5,
                            {"cases": linked[:5]}))

        verdict, conf, fscore = self._decide(sigs, risk)
        narr = self._narrative(cid, cust, txn_id, amt, ts_str, sigs, verdict, conf, base, win, ct)

        result = {
            "case_id": cid, "customer_id": cust, "flagged_txn_id": txn_id,
            "investigated_at": datetime.now().isoformat() + "Z",
            "verdict": verdict, "confidence": round(conf, 3),
            "fraud_score": round(fscore, 3), "model_risk_score": risk,
            "signals": [s.to_dict() for s in sigs],
            "signal_count": len(sigs),
            "investigation_steps": steps,
            "analyst_narrative": narr,
            "recommended_action": self._action(verdict, conf),
        }
        self.engine.write_case_to_graph(result)
        return result

    def _decide(self, sigs, risk):
        if not sigs:
            fs = risk
            if risk >= 0.70:
                return "FRAUD", 0.72, fs
            return "NOT FRAUD", 0.80, fs

        # Signal-based scoring with boosted sensitivity
        top_w = max(s.w for s in sigs)
        total_w = sum(s.w for s in sigs)
        n_sigs = len(sigs)

        # Strong individual signal (card testing, high risk, proxy) = guaranteed fraud
        if top_w >= 2.5:
            sig_score = min(0.68 + (total_w - top_w) * 0.04, 0.95)
        elif top_w >= 2.0:
            sig_score = min(0.60 + (total_w - top_w) * 0.04, 0.90)
        elif n_sigs >= 3:
            sig_score = min(0.58 + total_w * 0.03, 0.85)
        elif n_sigs >= 2:
            sig_score = min(0.53 + total_w * 0.025, 0.78)
        else:
            sig_score = min(0.42 + total_w * 0.05, 0.65)

        fraud_score = 0.65 * sig_score + 0.35 * risk
        conf = min(0.50 + abs(fraud_score - 0.50), 0.97)
        verdict = "FRAUD" if fraud_score >= 0.50 else "NOT FRAUD"
        return verdict, conf, fraud_score

    def _action(self, verdict, conf):
        if verdict == "FRAUD":
            if conf >= 0.85:
                return "BLOCK_AND_REPORT"
            if conf >= 0.70:
                return "BLOCK_AND_CALL_CUSTOMER"
            return "FLAG_FOR_REVIEW"
        return "APPROVE" if conf >= 0.85 else "MONITOR_24H"

    def _narrative(self, cid, cust, txn_id, amt, ts_str, sigs, verdict, conf, base, win, ct):
        lines = [
            f"=== Investigation Report: {cid} ===",
            f"Customer: {cust} | Flagged Transaction: {txn_id}",
            f"Amount: ${amt:.2f} | Timestamp: {ts_str}",
            "", f"VERDICT: {verdict} (Confidence: {conf:.1%})", "",
            "--- Historical Baseline ---",
            f"  Prior transactions (90d): {base.get('total_prior_txns', 0)}",
            f"  Historical mean: ${base.get('mean_amount', 0):.2f}",
            f"  Typical regions: {base.get('typical_regions', [])}",
            "", "--- Fraud Signals ---",
        ]
        for s in sigs:
            lines.append(f"  [{s.code}] {s.desc}")
        if not sigs:
            lines.append("  No signals detected.")
        lines += ["", f"--- 48h Window: {len(win)} transactions ---",
                  f"  Action: {self._action(verdict, conf)}"]
        return "\n".join(lines)

    def run_batch(self, cases, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for case in cases:
            try:
                r = self.investigate(case)
                results.append(r)
                with open(os.path.join(out_dir, f"{case['case_id']}.json"), "w") as f:
                    json.dump(r, f, indent=2, default=str)
                logger.info(f"[{case['case_id']}] -> {r['verdict']} ({r['confidence']:.1%})")
            except Exception as e:
                logger.error(f"[{case['case_id']}] FAILED: {e}")
                results.append({"case_id": case["case_id"], "verdict": "ERROR", "error": str(e)})
        return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(base_dir, "deep_case_analysis.json")) as f:
        raw = json.load(f)

    def fix(obj):
        if isinstance(obj, dict):
            return {k: fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix(v) for v in obj]
        if str(obj) in ("nan", "NaN"):
            return None
        return obj

    cases = [fix(c) for c in raw]
    agent = FraudInvestigationAgent(data_dir=base_dir)
    results = agent.run_batch(cases, os.path.join(base_dir, "cases"))

    fraud = sum(1 for r in results if r.get("verdict") == "FRAUD")
    not_fraud = sum(1 for r in results if r.get("verdict") == "NOT FRAUD")
    print(f"\nINVESTIGATION COMPLETE: FRAUD={fraud}, NOT_FRAUD={not_fraud}")
    for r in results:
        print(f"  {r['case_id']:10s} -> {r.get('verdict','ERROR'):10s} ({r.get('confidence',0):.1%})")

    summary = {
        "total": len(results), "fraud": fraud, "not_fraud": not_fraud,
        "results": [{"case_id": r["case_id"], "verdict": r.get("verdict"),
                     "confidence": r.get("confidence"), "action": r.get("recommended_action")}
                    for r in results]
    }
    with open(os.path.join(base_dir, "cases", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
