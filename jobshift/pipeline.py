"""一個指令跑完一次快照更新：爬取 → 清洗入庫 → 向量化 → 重算分析。

設計成「每隔一兩個月跑一次」的作業：

- **可重入**：每個階段先看產物在不在，在就跳過。中途斷掉（爬到一半、額度、當機）
  直接把同一行再跑一次，會從斷掉的地方接上，不會重來。
- **階段隔離**：每個階段開獨立 subprocess。爬取要跑好幾小時、torch 佔的顯存要在
  階段結束時完整還回去 —— 混在同一個行程裡，前者的記憶體會一路帶到後者。
- **參考座標系不動**：analyze 會沿用既有的群集定義，只把新快照指派進去，
  所以每次跑完都是在同一條時間軸上多一個點，而不是重新發明一套分類。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

from jobshift import db
from jobshift import settings as S

STAGES = ("crawl", "ingest", "embed", "analyze")


def run(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 70}\n▶ {label}\n  $ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, "-m", *cmd], check=False)
    if result.returncode != 0:
        raise SystemExit(f"\n✗ 「{label}」失敗（exit {result.returncode}）。"
                         f"修好之後把同一行 pipeline 指令再跑一次，前面的階段會自動跳過。")
    print(f"✓ {label} 完成，耗時 {time.time() - t0:.0f}s", flush=True)


def crawl_state(snapshot: str) -> tuple[int, bool]:
    """回傳（已落地筆數, 是否整場跑完）。

    只看「JSONL 有沒有內容」是不夠的：被擋在半路時檔案裡有資料但資料不完整，
    那樣會直接跳過爬取、拿半份快照去做分析。完成與否要看爬蟲自己寫的標記。
    """
    path = S.raw_jsonl(snapshot)
    n = 0
    if path.exists():
        with open(path, encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
    prog = S.crawl_progress(snapshot)
    complete = (json.loads(prog.read_text(encoding="utf-8")).get("complete", False)
                if prog.exists() else False)
    return n, complete


def already_ingested(snapshot: str) -> int:
    with db.connect(read_only=True) as con:
        if not db.table_exists(con, "jobs"):
            return 0
        return con.execute("SELECT count(*) FROM jobs WHERE snapshot_date = ?",
                           [snapshot]).fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="跑一次完整的快照更新（爬取 → 清洗 → 向量化 → 分析）")
    ap.add_argument("--snapshot", default=S.SNAPSHOT_DATE,
                    help="這次快照的日期標籤，預設今天")
    ap.add_argument("--from-stage", choices=STAGES, default="crawl",
                    help="從哪個階段開始（前面的一律跳過）")
    ap.add_argument("--force", action="store_true",
                    help="忽略既有產物，每個階段都重跑")
    args = ap.parse_args()

    snap = args.snapshot
    start_at = STAGES.index(args.from_stage)
    t0 = datetime.now()

    print(f"快照標籤：{snap}")
    print(f"參考座標系：{S.REFERENCE_SNAPSHOT}（不會被這次執行改動）")
    print(f"資料目錄：{S.DATA_DIR}")

    # ① 爬取
    if start_at <= 0:
        n, complete = crawl_state(snap)
        if complete and not args.force:
            print(f"\n⏭  crawl 跳過：{S.raw_jsonl(snap).name} 已完整抓完 {n:,} 筆。"
                  "要重爬請刪掉該檔與同名的 _progress.json，或加 --force。")
        else:
            if n:
                print(f"\n↻ 上次抓到一半（{n:,} 筆），這次從斷點續抓")
            run("① 爬取 1111 職缺", ["jobshift.crawler"])

    # ② 清洗入庫（參考快照只在第一次需要）
    if start_at <= 1:
        if not already_ingested(S.REFERENCE_SNAPSHOT) or args.force:
            run("② 清洗入庫（參考快照）", ["jobshift.ingest", "--legacy"])
        n = already_ingested(snap)
        if n and not args.force:
            print(f"\n⏭  ingest 跳過：jobs 表已有 {snap} 的 {n:,} 筆")
        else:
            run("② 清洗入庫（本次快照）", ["jobshift.ingest", "--snapshot", snap])

    # ③ 向量化（embed 自己會判斷 .npy 在不在）
    if start_at <= 2:
        cmd = ["jobshift.embed", "--all", "--snapshot", snap]
        run("③ bge-m3 向量化", cmd + (["--force"] if args.force else []))

    # ④ 分析（一定重跑：多一個時間點，整條序列都要重算）
    if start_at <= 3:
        run("④ 分群與遷移分析", ["jobshift.analyze"])

    with db.connect(read_only=True) as con:
        rows = con.execute(
            "SELECT snapshot_date, count(*) n FROM jobs GROUP BY 1 ORDER BY 1").fetchall()
    print(f"\n{'=' * 70}\n全部完成，總耗時 {datetime.now() - t0}")
    print("目前時間軸：")
    for snapshot_date, n in rows:
        mark = " ← 本次" if snapshot_date == snap else ""
        print(f"  {snapshot_date}  {n:>7,} 筆{mark}")
    print("\n看結果： docker compose up -d api dashboard  →  http://localhost:8501")
    print("=" * 70)


if __name__ == "__main__":
    main()
