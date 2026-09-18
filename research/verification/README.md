# Appendix D verification

`evaluated_candidates.csv` lists the 115 evaluated designs from the final report's task-prioritised search. `additional_verification.json` preserves the analytical and direct-constraint observations made on 16 September 2026.

Run `python research/verification/verification_check.py` from the repository root to repeat the archived-source checks. It writes a new result to `results/verification/additional_verification.json`, leaving the historical evidence unchanged. The checks cover straight-element references, 15 single-arc cases spanning the straight/curved transition, and 27 decoded states per original/proposed geometry. They check those properties, not physical accuracy.
