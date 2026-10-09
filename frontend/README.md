# Frontend

Carry over v1's index.html (analyst dashboard) and realtime.html (scanner)
largely unchanged. Additions for v2 (4-feature scope):

- Analyst dashboard: add a "Drift Status" panel (green/amber/red) reading
  from GET /drift/status.
- Analyst dashboard: show both threshold candidates (F2-optimal vs
  cost-optimal) from GET /model/info for comparison — no toggle/adaptive
  logic needed, just transparency about which is active and why.
- Explanation panel: display the SHAP top-5 features alongside the LLM
  narrative, not just the narrative — makes the "grounded, not hallucinated"
  claim visible and demoable.
