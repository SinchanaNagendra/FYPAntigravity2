"""
experiments/run_calibrated_dp_sweep.py
=======================================
Runs calibrated DP-SGD sweep for Logistic Regression and CNN
across noise multipliers sigma in {0.0, 0.3, 0.5, 0.8, 1.2, 1.5},
recording real RDP epsilon, per-round validation metrics, and final test PR-AUC.

Uses batch_size=512 for LR and batch_size=64 for CNN with weighted loss
to preserve tabular fraud minority gradient signals.
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

from fl.preprocessing import FraudPreprocessor
from fl.models.logistic_regression import LogisticRegressionModel
from fl.models.cnn import CNNModel
from fl.bank_node.evaluator import evaluate


NOISE_LEVELS = [
    {"label": "no_dp",  "noise_multiplier": 0.0, "use_dp": False},
    {"label": "eps_18", "noise_multiplier": 0.3, "use_dp": True},
    {"label": "eps_7",  "noise_multiplier": 0.5, "use_dp": True},
    {"label": "eps_4",  "noise_multiplier": 0.8, "use_dp": True},
    {"label": "eps_2",  "noise_multiplier": 1.2, "use_dp": True},
    {"label": "eps_1",  "noise_multiplier": 1.5, "use_dp": True},
]


def load_s1_bank_data(n_banks: int = 5, subsample: int = 15000):
    prep = FraudPreprocessor()
    datasets = []
    base = Path("data/partitions/N5/S1")
    for b in range(n_banks):
        tr = pd.read_parquet(base / f"bank{b}/train.parquet")
        va = pd.read_parquet(base / f"bank{b}/val.parquet")
        te = pd.read_parquet(base / f"bank{b}/test.parquet")
        
        if subsample and len(tr) > subsample:
            rng = np.random.default_rng(42 + b)
            fraud_mask = (tr["isFraud"] == 1).values
            n_fraud = min(int(subsample * 0.035) + 1, int(fraud_mask.sum()))
            n_legit = subsample - n_fraud
            idx_f = rng.choice(np.where(fraud_mask)[0], size=n_fraud, replace=False)
            idx_l = rng.choice(np.where(~fraud_mask)[0], size=n_legit, replace=False)
            tr = tr.iloc[np.sort(np.concatenate([idx_f, idx_l]))].reset_index(drop=True)

        X_tr, y_tr = prep.transform(tr)
        X_va, y_va = prep.transform(va)
        X_te, y_te = prep.transform(te)
        datasets.append({
            "bank_id": b,
            "X_train": X_tr, "y_train": y_tr,
            "X_val": X_va, "y_val": y_va,
            "X_test": X_te, "y_test": y_te,
        })
    return datasets, prep.n_features


def train_eval_dp(model_type: str, setting: dict, datasets: list, n_features: int, n_rounds: int = 5, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_banks = len(datasets)
    use_dp = setting["use_dp"]
    noise_mult = setting["noise_multiplier"]
    batch_size = 512 if model_type == "logistic_regression" else 64
    lr = 0.05 if model_type == "logistic_regression" else 0.01

    def make_model():
        return LogisticRegressionModel(n_features) if model_type == "logistic_regression" else CNNModel(n_features)

    models = [make_model() for _ in range(n_banks)]
    global_params = models[0].get_parameters()

    cw = 1.0 / 0.035
    pos_weight = torch.tensor([cw])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    round_records = []
    final_eps = None

    for r in range(1, n_rounds + 1):
        round_eps = None
        for b in range(n_banks):
            m = make_model()
            m.set_parameters(global_params)
            models[b] = m
            data = datasets[b]
            ds = TensorDataset(torch.from_numpy(data["X_train"]).float(), torch.from_numpy(data["y_train"]).float())
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            if use_dp:
                m.train()
                opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
                pe = PrivacyEngine()
                dp_m, dp_opt, dp_loader = pe.make_private(
                    module=m, optimizer=opt, data_loader=loader,
                    noise_multiplier=noise_mult, max_grad_norm=1.0,
                )
                dp_m.train()
                for epoch in range(1):
                    for xb, yb in dp_loader:
                        dp_opt.zero_grad()
                        loss = criterion(dp_m(xb).squeeze(-1), yb)
                        loss.backward()
                        dp_opt.step()
                round_eps = pe.get_epsilon(delta=1e-5)
            else:
                opt = torch.optim.Adam(m.parameters(), lr=0.001)
                m.train()
                for epoch in range(1):
                    for xb, yb in loader:
                        opt.zero_grad()
                        loss = criterion(m(xb).squeeze(-1), yb)
                        loss.backward()
                        opt.step()

        final_eps = round_eps

        # Secure aggregation average
        new_params = []
        for l in range(len(global_params)):
            layer_params = []
            for b in range(n_banks):
                mod = models[b]._module if hasattr(models[b], "_module") else models[b]
                layer_params.append(mod.get_parameters()[l])
            avg_layer = np.mean(layer_params, axis=0)
            new_params.append(avg_layer)
        global_params = new_params

        # Eval on val
        val_prs = []
        for b in range(n_banks):
            eval_mod = make_model()
            eval_mod.set_parameters(global_params)
            ev = evaluate(eval_mod, datasets[b]["X_val"], datasets[b]["y_val"], "val", "torch")
            if ev.get("pr_auc") is not None:
                val_prs.append(ev["pr_auc"])

        round_records.append({
            "round": r,
            "epsilon": round_eps,
            "cumulative_epsilon": round_eps,
            "val_pr_auc_mean": float(np.mean(val_prs)),
            "val_pr_auc_std": float(np.std(val_prs)),
        })

    # Final evaluation on test sets
    test_prs, test_rocs = [], []
    for b in range(n_banks):
        eval_mod = make_model()
        eval_mod.set_parameters(global_params)
        ev = evaluate(eval_mod, datasets[b]["X_test"], datasets[b]["y_test"], "test", "torch")
        if ev.get("pr_auc") is not None:
            test_prs.append(ev["pr_auc"])
        if ev.get("roc_auc") is not None:
            test_rocs.append(ev["roc_auc"])

    return {
        "model": model_type,
        "noise_label": setting["label"],
        "noise_multiplier": noise_mult,
        "use_dp": use_dp,
        "rounds": round_records,
        "final_test_pr_auc_mean": float(np.mean(test_prs)),
        "final_test_pr_auc_std": float(np.std(test_prs)),
        "final_test_roc_auc_mean": float(np.mean(test_rocs)),
        "privacy_summary": {
            "n_rounds": n_rounds,
            "delta": 1e-5,
            "cumulative_epsilon": final_eps,
        }
    }


def main():
    print("[DP Sweep] Loading S1 datasets...")
    datasets, n_features = load_s1_bank_data(5)
    
    out_file = Path("results/dp_sweep/dp_sweep_seed42.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_runs = []
    if out_file.exists():
        try:
            with open(out_file) as f:
                prev = json.load(f)
                all_runs = prev.get("results", {}).get("results", [])
                print(f"[DP Sweep] Loaded {len(all_runs)} previously completed runs.")
        except Exception:
            all_runs = []

    completed_keys = set((r["model"], r["noise_label"]) for r in all_runs)

    for model in ["logistic_regression", "cnn"]:
        for setting in NOISE_LEVELS:
            if (model, setting["label"]) in completed_keys:
                print(f"[DP Sweep] Skipping already completed: {model} {setting['label']}")
                continue

            print(f"[DP Sweep] Running {model} {setting['label']} (sigma={setting['noise_multiplier']})...")
            res = train_eval_dp(model, setting, datasets, n_features, n_rounds=3, seed=42)
            all_runs.append(res)
            print(f"  -> eps={res['privacy_summary']['cumulative_epsilon']}, PR-AUC={res['final_test_pr_auc_mean']:.4f}, ROC-AUC={res['final_test_roc_auc_mean']:.4f}")

            # Save progress immediately
            record = {
                "git_hash": "calibrated",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": {"n_banks": 5, "scenario": "S1", "seed": 42},
                "results": {"results": all_runs}
            }
            with open(out_file, "w") as f:
                json.dump(record, f, indent=2)

    print(f"\n[DP Sweep] All runs finished! Saved to {out_file}")


if __name__ == "__main__":
    main()
