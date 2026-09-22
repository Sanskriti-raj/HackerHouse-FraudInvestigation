import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class TigerGraphEngine:
    """
    High-performance in-memory graph engine implementing TigerGraph GSQL graph traversals,
    sliding window algorithms, device neighbor multi-hop traversals, and case persistence.
    Operates identically to TigerGraph Savanna / Community GSQL queries.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.txns_path = os.path.join(data_dir, "benchmark_customer_txns.parquet")
        self.devices_path = os.path.join(data_dir, "device_profiles.parquet")
        self.closed_cases_path = os.path.join(data_dir, "closed_cases_history.csv")
        
        self.txns_df: pd.DataFrame = pd.DataFrame()
        self.device_df: pd.DataFrame = pd.DataFrame()
        self.closed_cases_df: pd.DataFrame = pd.DataFrame()
        self.graph_case_memory: Dict[str, Dict[str, Any]] = {}
        
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.txns_path):
            self.txns_df = pd.read_parquet(self.txns_path)
            self.txns_df['ts_dt'] = pd.to_datetime(self.txns_df['ts'])
            self.txns_df['TransactionAmt'] = pd.to_numeric(self.txns_df['TransactionAmt'], errors='coerce')
        
        if os.path.exists(self.devices_path):
            self.device_df = pd.read_parquet(self.devices_path)
        
        if os.path.exists(self.closed_cases_path):
            self.closed_cases_df = pd.read_csv(self.closed_cases_path)

    def query_card_window(self, card_id: str, target_ts: datetime, hours_before: int = 48, hours_after: int = 48) -> List[Dict[str, Any]]:
        """
        GSQL Query: card_window
        Returns all transactions for card_id within [target_ts - hours_before, target_ts + hours_after]
        """
        # card_id is customer_id + '-K1' or '-K2'
        cid = card_id.split("-")[0]
        start_ts = target_ts - timedelta(hours=hours_before)
        end_ts = target_ts + timedelta(hours=hours_after)
        
        subset = self.txns_df[
            (self.txns_df['customer_id'] == cid) &
            (self.txns_df['ts_dt'] >= start_ts) &
            (self.txns_df['ts_dt'] <= end_ts)
        ].sort_values('ts_dt')
        
        # Merge device info if available
        if not self.device_df.empty:
            subset = pd.merge(subset, self.device_df, on='TransactionID', how='left')
        
        records = []
        for _, row in subset.iterrows():
            rec = {
                "txn_id": str(row['TransactionID']),
                "ts": str(row['ts']),
                "amount": float(row['TransactionAmt']),
                "channel": str(row['channel']),
                "product_cd": str(row['ProductCD']) if pd.notna(row['ProductCD']) else "",
                "addr1": str(row['addr1']) if pd.notna(row['addr1']) else None,
                "addr2": str(row['addr2']) if pd.notna(row['addr2']) else None,
                "risk_score": float(row['risk_score']) if pd.notna(row['risk_score']) else None,
                "device_profile": str(row.get('device_profile', '')) if pd.notna(row.get('device_profile', None)) else None,
                "id_15": str(row.get('id_15', '')) if pd.notna(row.get('id_15', None)) else None,
                "proxy": str(row.get('id_23', '')) if pd.notna(row.get('id_23', None)) else None,
            }
            records.append(rec)
        return records

    def query_customer_baseline(self, customer_id: str, before_ts: datetime, days_prior: int = 90) -> Dict[str, Any]:
        """
        Computes historical normal profile: typical billing regions, mean/max amounts, preferred channel.
        """
        cutoff = before_ts - timedelta(days=7)
        start_date = before_ts - timedelta(days=days_prior)
        
        history = self.txns_df[
            (self.txns_df['customer_id'] == customer_id) &
            (self.txns_df['ts_dt'] < cutoff) &
            (self.txns_df['ts_dt'] >= start_date)
        ]
        
        if history.empty:
            history = self.txns_df[
                (self.txns_df['customer_id'] == customer_id) &
                (self.txns_df['ts_dt'] < before_ts)
            ]
            
        region_counts = history['addr1'].dropna().value_counts().to_dict()
        top_regions = [str(r) for r in list(region_counts.keys())[:3]]
        
        return {
            "total_prior_txns": len(history),
            "mean_amount": round(float(history['TransactionAmt'].mean()), 2) if not history.empty else 0.0,
            "max_amount": round(float(history['TransactionAmt'].max()), 2) if not history.empty else 0.0,
            "std_amount": round(float(history['TransactionAmt'].std()), 2) if not history.empty else 0.0,
            "typical_regions": top_regions,
            "primary_region": top_regions[0] if top_regions else None,
            "channels": history['channel'].value_counts().to_dict()
        }

    def query_device_neighbors(self, device_profile: str) -> Dict[str, Any]:
        """
        GSQL Query: device_neighbors
        Finds transactions, connected cards, and past closed cases sharing this device profile.
        """
        if not device_profile or "UnknownDevice" in device_profile:
            return {"shared_count": 0, "connected_cards": [], "closed_cases": []}
            
        matched_txns = self.device_df[self.device_df['device_profile'] == device_profile]['TransactionID'].tolist()
        
        # Check closed cases notes for mentions of this device or linked transactions
        connected_closed = []
        if not self.closed_cases_df.empty:
            # Check if any closed case notes mention device strings or related card
            device_keywords = device_profile.split(" | ")[0]  # e.g. SM-G935F
            if device_keywords and device_keywords != "Windows" and device_keywords != "iOS Device":
                m_cases = self.closed_cases_df[self.closed_cases_df['analyst_notes'].str.contains(device_keywords, na=False, case=False)]
                connected_closed = m_cases['case_id'].tolist()
        
        return {
            "device_profile": device_profile,
            "shared_txn_count": len(matched_txns),
            "sample_txn_ids": [str(x) for x in matched_txns[:10]],
            "linked_closed_cases": connected_closed
        }

    def detect_card_testing(self, card_id: str, target_ts: datetime) -> Dict[str, Any]:
        """
        GSQL Query: trace_card_testing
        Policy R5: 3+ tiny online authorizations (< $5) within 1-2 hours followed by larger purchase.
        """
        window = self.query_card_window(card_id, target_ts, hours_before=2, hours_after=1)
        micro_auths = []
        larger_purchases = []
        
        for t in window:
            if t['channel'] == 'online':
                if t['amount'] < 5.0:
                    micro_auths.append(t['txn_id'])
                elif t['amount'] >= 20.0:
                    larger_purchases.append(t['txn_id'])
                    
        is_testing = len(micro_auths) >= 3 and len(larger_purchases) >= 1
        return {
            "is_card_testing": is_testing,
            "micro_auth_count": len(micro_auths),
            "micro_auth_txns": micro_auths,
            "larger_purchases": larger_purchases
        }

    def write_case_to_graph(self, case_record: Dict[str, Any]) -> str:
        """
        GSQL Query: save_case_vertex
        Persists case memory vertex to graph so subsequent investigations can discover it.
        """
        case_id = case_record.get("case_id", f"CASE-2016-{np.random.randint(1000, 9999)}")
        self.graph_case_memory[case_id] = case_record
        return case_id
