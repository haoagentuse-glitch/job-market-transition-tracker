"""產生一份「假的對照快照」，用來在真爬蟲跑完之前把整條線走通（walking skeleton）。

做法：從舊快照抽樣＋刻意扭曲分布（某些產業砍半、某些留全部），
這樣分群與遷移分析會看到真實的結構差異，不是兩份一模一樣的資料。
正式資料到位後直接覆蓋掉，不影響任何正式流程。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from jobshift import ingest
from jobshift import settings as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="2026-08-02-dev")
    ap.add_argument("--n", type=int, default=6000, help="假快照筆數")
    ap.add_argument("--base-n", type=int, default=6000, help="基準期也只取這麼多（加速）")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    df = ingest.load_legacy_csv(S.LEGACY_CSV, S.REFERENCE_SNAPSHOT)
    # 依 job_id 排序後取前 N —— 跟 `embed --limit` 的 ORDER BY job_id 對齊，
    # 這樣骨架測試時基準期與對照期的存活職缺才會真的重疊。
    df = df.sort_values("job_id", key=lambda s: s.astype(str)).reset_index(drop=True)

    base = df.head(min(args.base_n, len(df)))

    # 對照期：從基準期抽 55% 當「存活」，其餘從剩下的池子抽當「新進」，並扭曲產業權重
    survivors = base.sample(frac=0.55, random_state=2)
    pool = df.iloc[len(base):]
    weights = {b: rng.uniform(0.3, 2.0) for b in df.industry_bucket.unique()}
    w = pool.industry_bucket.map(weights).fillna(1.0).to_numpy()
    fresh = pool.sample(n=min(args.n - len(survivors), len(pool)),
                        weights=w / w.sum(), random_state=3)
    new = pd.concat([survivors, fresh], ignore_index=True)

    out = S.raw_jsonl(args.snapshot)
    with open(out, "w", encoding="utf-8") as f:
        for rec in new.to_dict(orient="records"):
            rec["snapshot_date"] = args.snapshot
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[devdata] 基準期取 {len(base)} 筆（配 embed --limit {args.base_n}）")
    print(f"[devdata] 對照期 {len(new)} 筆（存活 {len(survivors)} / 新進 {len(fresh)}）→ {out}")


if __name__ == "__main__":
    main()
