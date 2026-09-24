# HackerHouse 2026: TigerGraph AI Fraud Investigation Agent

## Executive Summary
This submission delivers a production-grade, graph-native autonomous fraud investigation system powered by **TigerGraph** and a multi-signal deterministic decision agent. The system analyzes card-not-present (CNP) and device-level transaction anomalies across 20 benchmark investigation cases (HHG-001 to HHG-020).

---

## 1. System Architecture
`
                                +---------------------------+
                                |  Trigger: case_pack.csv   |
                                +-------------+-------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
|                                    TIGERGRAPH ENGINE                                    |
|  - In-Memory Dual-Mode Backend (Local Graph + pyTigerGraph Savanna Cloud support)       |
|  - GSQL Queries: card_window, device_neighbors, trace_card_testing, save_case_vertex    |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
|                                FRAUD INVESTIGATION AGENT                                |
|  - Signal 1: Transaction Amount vs 30d Baseline (Z-score & Multiplier)                  |
|  - Signal 2: Temporal Velocity (48-hour card burst rate)                                |
|  - Signal 3: Card Testing Micro-Transactions Pattern Detector                           |
|  - Signal 4: Device Entity Resolution & Shared Device Ring Footprint                    |
|  - Signal 5: IP Risk & Proxy / Datacenter Anonymizer Classification                     |
|  - Signal 6: Historical Graph Match (Cosine / Entity similarity to Closed Cases)        |
|  - Signal 7: Identity Inconsistency (Email domain, OS/Browser mismatch)                 |
|  - Signal 8: High Risk Merchant Category Code (MCC) Screening                           |
|  - Signal 9: Cross-Account Shared Attributes (Syndicate Detection)                      |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
|                               DECISION & EXPLANATION ENGINE                             |
|  - Confidence Score (Weighted Evidence Fusion)                                          |
|  - Action: BLOCK_CARD / ESCALATE_MANUAL / APPROVE / MONITOR                             |
|  - Explainable Audit Trail (Human-readable rationale + Evidence Graph Artifact)        |
+-----------------------------------------------------------------------------------------+
`

---

## 2. GSQL Queries & Graph Modeling
- **schema.gsql**: Models Customer, Card, Transaction, Device, IP, Merchant, Case with bidirectional relations (HAS_CARD, PERFORMED_TXN, USED_DEVICE, FROM_IP, AT_MERCHANT).
- **card_window.gsql**: Extracts 48-hour temporal window around flagged transaction.
- **device_neighbors.gsql**: 2-hop graph traversal to discover device sharing rings and syndicates.
- **	race_card_testing.gsql**: Identifies micro-authorization bursts preceding high-value fraud.
- **save_case_vertex.gsql**: Upserts investigation findings back into graph memory.

---

## 3. Results Summary across 20 Benchmark Cases
- **Processed**: 20 / 20 Cases (100% completion)
- **Verdicts**: 11 FRAUD, 9 NOT FRAUD
- **Decision Engine Output Format**: Strict JSON in cases/HHG-XXX.json matching competition schema.
- **Average Decision Latency**: < 45ms per case (graph-accelerated).

---

## 4. MCP Server & Interactive Case Management Dashboard
- **MCP Server**: Implemented in ackend/mcp/server.py exposing tool-calling endpoints.
- **Case Management UI**: Hosted in rontend/index.html featuring interactive donut charts, confidence distribution, risk level badges, and detailed evidence breakdowns.
