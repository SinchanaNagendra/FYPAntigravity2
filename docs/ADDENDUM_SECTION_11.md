<USER_REQUEST>
# ADDENDUM TO THE FINAL PROMPT: DASHBOARD DEFECTS AND RESULT BUGS (SECTION 11)

This addendum is **part of the final prompt** (`final_completion_prompt.md`). The same Anti-Bluff Contract applies: nothing is done without raw evidence committed under `evidence/`, failures are reported as failures, and no number or claim may appear on the dashboard unless a script produced it from a file in `results/` or `evidence/`.

The dashboard at `localhost:8765` was reviewed tab by tab. The defects below are observed facts. Fix each one, show the before/after, and save evidence under `evidence/dashboard/`.

---

## A. CLAIMS THAT ARE NOT SUPPORTED (remove, or earn with evidence)

| On screen | Problem | Required action |
|---|---|---|
| Header badge: "103 Tests Passed · N=5 Banks · Real IEEE-CIS Data"; Overview "103 / 103" | 13 tests were skipped; badge hides it; "N=5 Banks" implies a 5-bank network exists | Generate from `evidence/tests/pytest_full.txt`: "X passed, Y skipped, Z failed". Remove "N=5 banks" wording unless a live N=5 network ran. |
| "Privacy Invariant Proven", "Raw Data Leaked: 0 bytes (100% verified by privacy audit)", "Cryptographic boundary audit proving zero raw cardholder records…", "0 Bytes (BLOCKED)" rows | No traffic capture, ledger dump, or IPFS dump has been analysed. Values are typed text | Show these only from `evidence/privacy_audit/` live results (bytes captured, canaries searched, matches found). Until then display "NOT YET VERIFIED". |
| Overview "Ledger: Hyperledger Fabric 2.5 (12 Go unit tests pass)" | Go tests use an in-memory mock stub; no ledger exists | Replace with live block height, chaincode version, org count from the real peer. If no network is up, show "Fabric network not running". |
| Convergence tab status "Finalized" on every round | Implies on-chain `FinalizeRound`; no ledger was used | Show status from the ledger (tx ID, block number). Without it, show "completed (off-chain)". |
| "Publication-Grade Report Figures" | Two figures are broken (below) | Remove the label. |
| Baselines: "empirical upper bounds" | Federated exceeds "upper bounds" on 3 of 5 models | Fix baselines until they are fair, then reword honestly. |
| Federated tab: "WHY FEDERATED BEATS CENTRALIZED" (ensemble diversity, gradient averaging reduces overfitting) | Post-hoc story with no experiment behind it | Delete, or support each reason with an ablation that you run. |
| Non-IID tab: "Addressed by FedProx (µ=0.01)" | No FedProx result exists for S3; the S3 row is Random Forest, which does not use FedProx | Run LR/CNN with FedAvg vs FedProx on S3 and show both, or remove the sentence. |
| DP tab: "Key Takeaway: tabular fraud detection is vulnerable to Gaussian noise due to sparse 3.5% minority signals" | Asserted without investigation; the data look like a bug (see B3) | Remove until the root cause is proven. |
| DP tab "Privacy level" labels: ε = 24.78 "Moderate", ε = 1.41 "High" | ε of about 25 gives little meaningful protection | Report ε and δ only, with a neutral note; no marketing labels. |
| Baselines: "+0.02% gain", "-19.5% PR-AUC drop", "SMOTE/ADASYN create false-positive noise" | Not computed from displayed data; SMOTE/ADASYN/ROS/Hybrid rows are absent from the table | Generate every sentence from the data; include all balancing rows or remove the claim. |

---

## B. BUGS VISIBLE IN THE RESULTS (find root cause, fix, re-run, explain in `docs/RESULTS_NOTES.md`)

1. **Convergence chart and Figure 4 plot a single dot at round 1** (x-axis 0.96 to 1.04) although ten rounds exist. Fix the plotting data path. Also the LR PR-AUC is flat at about 0.37 to 0.386 across 10 rounds: either it converged immediately (explain) or the rounds are not actually learning (check). Show the 20-round curve.
2. **Gaussian NB is broken.** `none` gives ROC-AUC 0.4974, accuracy 0.1144 (predicts nearly everything as fraud); `rus` gives 0.7502. Federated NB ROC-AUC 0.4479. Fix priors/variance/scaling; the federated aggregate must equal the pooled fit.
3. **DP sweep is invalid.** Labels `eps_8/3/1` show ε = 24.78/6.12/2.51; PR-AUC is flat 0.0812 to 0.0815 across σ = 0.5 to 1.5 and far below the no-DP 0.2930. Verify noise is applied at the stated σ (log σ, clip norm, sample rate, steps), re-run with target-ε mode, run seeds, fill the **CNN** panel (currently empty), and plot ε on a log axis. The x-axis currently plots "∞" as 100, which fakes a straight line.
4. **Class weighting does nothing for several models.** `weighted` is identical to `none` to 4 decimals for Decision Tree, Random Forest, Gaussian NB and KNN. Either implement it (class_weight, priors) or drop those rows and say it is not applicable.
5. **Missing balancing rows.** Baselines table has only none/weighted/RUS. ROS, SMOTE, ADASYN and Hybrid appear only in Figure 5 as an aggregate over models and banks. Add per-model rows for all seven methods from real runs; confirm resampling touches training data only.
6. **Federated beats centralized on LR (0.2787 vs 0.1330), DT, RF.** Fix the centralized baselines (same features, scaling, split, enough iterations) and add a LightGBM/XGBoost reference. Re-assess the gaps afterwards.
7. **Attack table is inconsistent.** Label-flip with m=1 and m=2 gives PR-AUC 0.2837/0.2963, **higher** than clean 0.2787. "Recall collapses (−84%)" compares m=1 (0.0886) with m=2 (0.0142), not against clean. Mode B trimmed-mean (0.3533) beats clean. Add clean recall, compute deltas against the clean baseline, run multiple seeds, and explain or fix. Scaled 10× poisoning barely hurts (0.2569): show whether clipping absorbed it.
8. **Non-IID S3 α=0.1 table is not valid evidence.** Bank 4 has 5 samples with 40% fraud and PR-AUC 1.0000; Bank 2 has 11 samples; Bank 1 has 0.00% fraud and PR-AUC 0.0000. Metrics on 5 rows are meaningless, and a bank with that few rows cannot satisfy `min_samples_leaf ≥ 20`. Enforce a minimum bank size in the Dirichlet partitioner (resample until every bank has enough train/val/test rows and at least a minimum fraud count), state the rule, and re-run. Report std across seeds, not across degenerate banks. The RF S3 result "0.4109 ± 0.4424" (std larger than the mean) must be replaced.
9. **S3 gap is computed against the S1 centralized baseline** (+0.0110 = 0.4109 − 0.3999). Compare with a centralized baseline trained on the S3 pooled data.
10. **Rounds are too few:** DT 1 round, GNB 3, RF 5, LR 10. Use ≥ 20 rounds for iterative models; document why one-shot models (DT/RF/NB) need a single aggregation.
11. **Validation vs test confusion.** Convergence tab shows validation PR-AUC (about 0.38); other tabs show test PR-AUC (0.2787 for LR). Label the split on every number; headline numbers are test.
12. **Per-round time 70 to 91 s** for LR on about 400K rows is slow; report where time goes (training vs masking vs I/O) once real Fabric/IPFS overhead exists.

---

## C. MISSING DASHBOARD CONTENT (it currently has no blockchain presence at all)

Add tabs populated from the live system or `results/`; empty-state text must say "not running", never invent data:
1. **Blockchain / Ledger:** channel height, last N blocks with tx IDs, per-round state transitions, endorsements per tx, security events, model-hash chain, reputation per bank.
2. **IPFS / Artifacts:** CIDs per round, sizes, hash-verified flag, tamper test result.
3. **Framework comparison:** PyTorch vs TFF table (PR-AUC, F1, time, memory) from `evidence/tff/`.
4. **CNN:** centralized, local, federated rows (none exist today).
5. **Privacy audits (live):** canary search results, boundary schema validation, isolation tests, secure-aggregation test, with capture sizes and timestamps.
6. **Membership inference:** attack AUC with and without DP.
7. **Communication and latency:** bytes per round per bank, tx latency, throughput, ledger growth.
8. **Provenance footer on every tab:** source file path, SHA-256 of that file, timestamp, git commit.

---

## D. ACCEPTANCE CHECKS (add to `make verify`)

- A script scans `dashboard/frontend/public/*.html|js|json` for the banned phrases in section A (e.g. "Proven", "100% verified", "Publication-Grade", "0 bytes", "Finalized" when not ledger-sourced, "upper bounds") and fails if they appear without a linked evidence file.
- A script regenerates `data.json`/`data.js` from `results/` + `evidence/` and fails if the committed copy differs.
- Every number in a table has a hover/“source” link to its result file.
- Screenshots of **every tab** taken during a live blockchain-backed run are saved to `evidence/dashboard/` and committed.

Report each item in sections A, B, C as DONE / PARTIAL / NOT DONE with its evidence path, using the final report format from the main prompt.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-01T17:43:04+05:30.
</ADDITIONAL_METADATA>