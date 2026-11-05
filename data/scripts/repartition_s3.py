"""
data/scripts/repartition_s3.py
==============================
Repartition IEEE-CIS into S3 Dirichlet scenarios (alpha=0.1, 0.5, 1.0)
using the clean partition files in S1 as source, with a strict minimum
bank size guard (min_prop >= 0.04) to eliminate degenerate banks.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    return float(np.sqrt(np.clip(js, 0, None)))


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    return (
        df_sorted.iloc[:n_train].copy(),
        df_sorted.iloc[n_train:n_val].copy(),
        df_sorted.iloc[n_val:].copy(),
    )


def compute_stats(bank_splits: dict, global_fraud_rate: float, n_global: int) -> dict:
    global_dist = np.array([1.0 - global_fraud_rate, global_fraud_rate])
    stats = {
        "global": {
            "n_transactions": n_global,
            "fraud_rate": global_fraud_rate,
        },
        "banks": {},
    }

    for bank_id, splits in bank_splits.items():
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]
        total_len = len(train) + len(val) + len(test)

        # Reconstruct fraud count
        fraud_count = train["isFraud"].sum() + val["isFraud"].sum() + test["isFraud"].sum()
        fraud_rate = float(fraud_count / total_len) if total_len > 0 else 0.0
        train_fraud_rate = float(train["isFraud"].mean()) if len(train) > 0 else 0.0

        bank_dist = np.array([1.0 - fraud_rate, fraud_rate])
        js_dist = js_distance(bank_dist + 1e-10, global_dist + 1e-10)

        min_dt = int(train["TransactionDT"].min()) if len(train) > 0 else None
        max_dt = int(test["TransactionDT"].max()) if len(test) > 0 else None

        stats["banks"][str(bank_id)] = {
            "n_total": total_len,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "fraud_rate_total": fraud_rate,
            "fraud_rate_train": train_fraud_rate,
            "js_distance_from_global": round(js_dist, 6),
            "min_TransactionDT": min_dt,
            "max_TransactionDT": max_dt,
        }

    return stats


def repartition_s3(alpha: float, seed: int = 42, min_prop: float = 0.04):
    print(f"\n[repartition] Running S3 Dirichlet with alpha={alpha}, seed={seed}, min_prop={min_prop}")
    s1_base = Path("data/partitions/N5/S1")
    out_base = Path(f"data/partitions/N5/S3_a{alpha}")
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Read metadata (uid, isFraud, TransactionDT) from S1
    meta_dfs = []
    for b in range(5):
        for split in ["train", "val", "test"]:
            p = s1_base / f"bank{b}" / f"{split}.parquet"
            meta_dfs.append(pd.read_parquet(p, columns=["uid", "isFraud", "TransactionDT"]))
    meta = pd.concat(meta_dfs, ignore_index=True)
    n_global = len(meta)
    global_fraud_rate = float(meta["isFraud"].mean())

    # 2. Assign UIDs
    rng = np.random.default_rng(seed)
    uid_fraud = meta.groupby("uid")["isFraud"].mean().reset_index()
    uid_fraud["binary_fraud"] = (uid_fraud["isFraud"] > 0).astype(int)

    n_banks = 5
    uid_to_bank = {}
    for label in [0, 1]:
        label_uids = uid_fraud.loc[uid_fraud["binary_fraud"] == label, "uid"].tolist()
        found = False
        for attempt in range(5000):
            proportions = rng.dirichlet(np.full(n_banks, alpha))
            if np.all(proportions >= min_prop):
                found = True
                break
        if not found:
            proportions = np.maximum(proportions, min_prop)
            proportions /= proportions.sum()

        counts = (proportions * len(label_uids)).astype(int)
        diff = len(label_uids) - counts.sum()
        counts[np.argmax(proportions)] += diff

        shuffled = rng.permutation(label_uids)
        idx = 0
        for bank_id, count in enumerate(counts):
            for uid in shuffled[idx: idx + count]:
                uid_to_bank[uid] = int(bank_id)
            idx += count

    # 3. Read full S1 files bank by bank and distribute rows to S3 banks
    bank_data = {b: [] for b in range(n_banks)}
    for b in range(5):
        for split in ["train", "val", "test"]:
            p = s1_base / f"bank{b}" / f"{split}.parquet"
            df = pd.read_parquet(p)
            df["target_bank"] = df["uid"].map(uid_to_bank)
            for target_b in range(n_banks):
                sub = df[df["target_bank"] == target_b].drop(columns=["target_bank"])
                if len(sub) > 0:
                    bank_data[target_b].append(sub)

    # 4. For each S3 bank, concatenate, temporal split, and save
    bank_splits = {}
    for b in range(n_banks):
        b_df = pd.concat(bank_data[b], ignore_index=True)
        train, val, test = temporal_split(b_df)
        bank_splits[b] = {"train": train, "val": val, "test": test}

        b_dir = out_base / f"bank{b}"
        b_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(b_dir / "train.parquet", index=False)
        val.to_parquet(b_dir / "val.parquet", index=False)
        test.to_parquet(b_dir / "test.parquet", index=False)

        print(
            f"  bank{b}: total={len(b_df):6,}  "
            f"train={len(train):6,} (fraud={train['isFraud'].sum():4,})  "
            f"val={len(val):5,} (fraud={val['isFraud'].sum():3,})  "
            f"test={len(test):5,} (fraud={test['isFraud'].sum():3,})  "
            f"fraud_rate={b_df['isFraud'].mean():.3%}"
        )

    # 5. Write stats.json
    stats = compute_stats(bank_splits, global_fraud_rate, n_global)
    stats["config"] = {
        "n_banks": n_banks,
        "scenario": f"S3_a{alpha}",
        "alpha": alpha,
        "seed": seed,
        "rule": "Guarded Dirichlet with min_prop >= 0.04 to prevent degenerate banks",
    }
    with open(out_base / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote {out_base / 'stats.json'}")


if __name__ == "__main__":
    for a in [0.1, 0.5, 1.0]:
        repartition_s3(alpha=a, seed=42)
