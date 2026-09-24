#!/usr/bin/env python3
"""Run all 20 investigations then evaluate."""
import subprocess, sys, os

base = os.path.dirname(os.path.abspath(__file__))
agent_script = os.path.join(base, "backend", "agent", "fraud_agent.py")
eval_script = os.path.join(base, "backend", "evaluation", "evaluator.py")

print("Step 1/2: Running fraud investigations...")
result = subprocess.run([sys.executable, agent_script], capture_output=False)

print("\nStep 2/2: Evaluating results...")
result = subprocess.run([sys.executable, eval_script], capture_output=False)
