"""1111 職缺爬蟲。請求行為完全照抄前一版專案 —— 那是唯一有證據不被封鎖的設定。

## 為什麼不是 Scrapy

本專案原本用 Scrapy 重寫並提高併發，實測結果：併發 4 ＋ 1 秒延遲，約 70 個請求後
整站封鎖 IP（連首頁 HTML 都回 403）；降到序列 ＋ 3 秒延遲，14 個請求後仍被擋。
而前一版 requests 序列版本抓完 6.5 萬筆從未被擋。差別不在單純的「快慢」，在節流的形狀：

前一版是**三層巢狀節流**，而且每個類別開獨立行程：

    launcher.py     每個類別之間睡 120–180 秒，每類別一個 subprocess
      crawl_single  類別內最多 2 輪，輪與輪之間睡 45–90 秒
        crawler.py  每頁之間睡 2.2–3.0 秒

我的 Scrapy 版只有最內層，而且從頭到尾不間斷 —— 連續兩個多小時的穩定流量，
比「慢但有大量停頓」更像機器。這支把三層全部還原。

## 保留的改良（都不改變送出去的請求）

- append-only JSONL，不再每 500 筆重寫整份 55MB CSV（原版是 O(n²) I/O）
- 頁級續跑：中斷後只抓沒抓過的頁
- 爬取階段零過濾、零評分，只忠實落地
- 連續被擋就中止，不繼續加深封鎖
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime

import requests

from jobshift import settings as S

# 照抄前一版的標頭，一個都不減。sec-fetch-* 這組少了，請求就不像是從站方
# 自己的頁面發出的 XHR。
HEADERS = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-TW,zh,en-US,en",
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "pragma": "no-cache",
    "referer": "https://www.1111.com.tw/search/job",
    "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
}


class Blocked(Exception):
    """連續被來源站擋下。繼續打只會延長封鎖，直接收工。"""


class Crawler:
    def __init__(self, snapshot: str):
        self.snapshot = snapshot
        self.out = S.raw_jsonl(snapshot)
        self.prog_path = S.crawl_progress(snapshot)
        self.seen: set[str] = set()
        self.done: set[str] = set()
        self.blocked = 0
        self.per_bucket: dict[str, int] = {}
        self._load_progress()

    # ── 續跑 ────────────────────────────────────────────────────────
    def _load_progress(self) -> None:
        if self.prog_path.exists():
            self.done = set(json.loads(
                self.prog_path.read_text(encoding="utf-8")).get("pages", []))
        if self.out.exists():
            with open(self.out, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.seen.add(str(row["job_id"]))
                        b = row.get("industry_bucket")
                        self.per_bucket[b] = self.per_bucket.get(b, 0) + 1
        if self.done or self.seen:
            print(f"[續跑] 已完成 {len(self.done)} 頁、已有 {len(self.seen):,} 筆，"
                  "只抓沒抓過的頁")

    def _save_progress(self, complete: bool = False) -> None:
        self.prog_path.write_text(
            json.dumps({"pages": sorted(self.done), "complete": complete},
                       ensure_ascii=False), encoding="utf-8")

    # ── 單頁請求 ────────────────────────────────────────────────────
    def fetch(self, codes: list[str], page: int) -> dict | None:
        params = [*S.FETCH_PARAMS_BASE.items(), ("page", str(page))]
        params += [("jobPositions", c) for c in codes]

        for attempt in range(1, S.CRAWL_MAX_RETRIES + 1):
            try:
                # 刻意不用 Session：前一版每個請求都是新連線，照抄。
                resp = requests.get(S.BASE_URL, headers=HEADERS, params=params, timeout=30)
            except requests.RequestException as exc:
                print(f"    [WARN] 請求失敗 {attempt}/{S.CRAWL_MAX_RETRIES}｜{exc}")
                time.sleep(2)
                continue

            if resp.status_code in (403, 429):
                self.blocked += 1
                print(f"    [BLOCK] HTTP {resp.status_code}"
                      f"（連續 {self.blocked}/{S.CRAWL_ABORT_AFTER_403}）")
                if self.blocked >= S.CRAWL_ABORT_AFTER_403:
                    raise Blocked(f"連續 {self.blocked} 次 HTTP {resp.status_code}")
                # 指數退避：30 → 60 → 120…。原版根本沒有 403 分支，403 會掉進
                # 「非 200」的路徑用 2 秒間隔猛重試 —— 它沒出事只是因為從沒觸發過。
                back = 30 * (2 ** (self.blocked - 1))
                print(f"    等 {back} 秒再試")
                time.sleep(back)
                continue

            if resp.status_code != 200:
                print(f"    [WARN] HTTP {resp.status_code} {attempt}/{S.CRAWL_MAX_RETRIES}")
                time.sleep(2)
                continue

            self.blocked = 0
            try:
                result = resp.json().get("result")
            except ValueError as exc:
                print(f"    [WARN] JSON 解析失敗｜{exc}")
                time.sleep(2)
                continue
            if isinstance(result, dict):
                return result
            print("    [WARN] 回應沒有 result")
            time.sleep(2)

        print(f"    [ERROR] 第 {page} 頁重試全部失敗，跳過")
        return None

    # ── 落地 ────────────────────────────────────────────────────────
    def write(self, result: dict, bucket: str) -> int:
        new = 0
        with open(self.out, "a", encoding="utf-8") as f:
            for job in result.get("hits") or []:
                jid = job.get("jobId")
                if not jid or str(jid) in self.seen:
                    continue
                self.seen.add(str(jid))
                f.write(json.dumps(self._row(job, bucket), ensure_ascii=False) + "\n")
                new += 1
        self.per_bucket[bucket] = self.per_bucket.get(bucket, 0) + new
        return new

    def _row(self, job: dict, bucket: str) -> dict:
        """只做欄位萃取，不做任何清洗／過濾／評分 —— 那些都在 ingest 階段。"""
        industry = job.get("industry") or {}
        city = job.get("workCity") or {}
        require = job.get("require") or {}
        jid = str(job.get("jobId"))
        return {
            "snapshot_date": self.snapshot,
            "job_id": jid,
            "job_title": job.get("title"),
            "company_id": job.get("companyId"),
            "company_name": job.get("companyName"),
            "updated_at": job.get("updateAt"),
            "industry_id": industry.get("id"),
            "industry_name": industry.get("name"),
            "industry_bucket": bucket,
            "exp_code": require.get("experience"),
            "edu_codes": require.get("grades", []),
            "major_codes": require.get("majors", []),
            "city_name": city.get("name"),
            "salary_desc": job.get("salary"),
            "description": job.get("description", ""),
            "job_url": f"https://www.1111.com.tw/job/{jid}/",
            "is_happiness": job.get("isHappiness", False),
            "recruit_range": job.get("recruitCountString"),
        }

    # ── 一輪：從第 1 頁翻到上限或翻完 ────────────────────────────────
    def run_once(self, bucket: str, codes: list[str]) -> int:
        added = 0
        prev_ids: set | None = None

        for page in range(1, S.MAX_PAGES + 1):
            if f"{bucket}|{page}" in self.done:
                continue
            if self.per_bucket.get(bucket, 0) >= S.TARGET_PER_CATEGORY:
                print(f"  [{bucket}] 已達 {S.TARGET_PER_CATEGORY} 筆上限，停止")
                break

            result = self.fetch(codes, page)
            if result is None:
                # 失敗也要等。原版這裡是直接 continue，等於「對方剛出錯，我們馬上
                # 再打一次」—— 正是最不該加壓的時候。
                time.sleep(random.uniform(S.SLEEP_PAGE_MIN, S.SLEEP_PAGE_MAX))
                continue

            hits = result.get("hits") or []
            if not hits:
                print(f"  [{bucket}] p{page} 無資料，本輪結束")
                break

            ids = {h.get("jobId") for h in hits}
            if ids == prev_ids:
                print(f"  [{bucket}] p{page} 與上一頁重複，本輪結束")
                break
            prev_ids = ids

            new = self.write(result, bucket)
            added += new
            self.done.add(f"{bucket}|{page}")
            total_page = int((result.get("pagination") or {}).get("totalPage") or 0)
            print(f"  [{bucket}] p{page}/{min(total_page, S.MAX_PAGES)}｜"
                  f"{len(hits)} 筆，新增 {new}｜累計 {self.per_bucket.get(bucket, 0)}")

            if page % 20 == 0:
                self._save_progress()
            if total_page and page >= min(total_page, S.MAX_PAGES):
                break

            # 第一層節流：每頁之間
            time.sleep(random.uniform(S.SLEEP_PAGE_MIN, S.SLEEP_PAGE_MAX))

        return added

    # ── 一個類別：最多 N 輪，輪之間長休 ──────────────────────────────
    def crawl_bucket(self, bucket: str, codes: list[str]) -> None:
        print(f"\n{'=' * 70}\n類別 {bucket}\n{'=' * 70}")
        for run in range(1, S.MAX_RUNS_PER_CATEGORY + 1):
            have = self.per_bucket.get(bucket, 0)
            if have >= S.TARGET_PER_CATEGORY:
                print(f"  已達目標 {S.TARGET_PER_CATEGORY} 筆，跳過")
                return
            print(f"  【第 {run} 輪】目前 {have} / 目標 {S.TARGET_PER_CATEGORY}")

            added = self.run_once(bucket, codes)
            self._save_progress()
            print(f"  【第 {run} 輪結束】新增 {added} 筆｜累計 {self.per_bucket.get(bucket, 0)}")

            if added == 0:
                print("  本輪無新增，資料已耗盡，不再重跑")
                return
            if self.per_bucket.get(bucket, 0) >= S.TARGET_PER_CATEGORY:
                return
            # 頁級續跑會讓「所有頁都抓過」的第二輪變成空轉，只是白等 45–90 秒。
            # 真的還有沒抓過的頁才值得再跑一輪。
            if all(f"{bucket}|{p}" in self.done for p in range(1, S.MAX_PAGES + 1)):
                print("  這個類別的頁都抓過了，不再重跑")
                return
            if run < S.MAX_RUNS_PER_CATEGORY:
                # 第二層節流：輪與輪之間
                nap = random.uniform(S.SLEEP_RUN_MIN, S.SLEEP_RUN_MAX)
                print(f"  還沒達標，{nap:.0f} 秒後開始下一輪")
                time.sleep(nap)


def main() -> None:
    snapshot = S.SNAPSHOT_DATE
    c = Crawler(snapshot)
    started = datetime.now()

    buckets = list(S.INDUSTRY_BUCKETS.items())
    est = (len(buckets) - 1) * (S.SLEEP_CATEGORY_MIN + S.SLEEP_CATEGORY_MAX) / 2 / 60
    print(f"快照 {snapshot}｜輸出 {c.out}")
    print(f"節流：每頁 {S.SLEEP_PAGE_MIN}–{S.SLEEP_PAGE_MAX}s、"
          f"每輪之間 {S.SLEEP_RUN_MIN}–{S.SLEEP_RUN_MAX}s、"
          f"每類別之間 {S.SLEEP_CATEGORY_MIN}–{S.SLEEP_CATEGORY_MAX}s")
    print(f"光是類別間的等待就有 {est:.0f} 分鐘 —— 這是刻意的，前一版就是這樣才沒被擋\n")

    aborted = None
    try:
        for i, (bucket, codes) in enumerate(buckets):
            c.crawl_bucket(bucket, codes)
            if i < len(buckets) - 1:
                # 第三層節流：類別與類別之間
                nap = random.uniform(S.SLEEP_CATEGORY_MIN, S.SLEEP_CATEGORY_MAX)
                print(f"\n… {nap:.0f} 秒後換下一個類別 "
                      f"（{i + 1}/{len(buckets)} 完成，累計 {len(c.seen):,} 筆）")
                time.sleep(nap)
    except Blocked as exc:
        aborted = str(exc)
    except KeyboardInterrupt:
        aborted = "手動中止"

    complete = aborted is None
    c._save_progress(complete=complete)

    print(f"\n{'=' * 70}")
    print(f"總耗時 {datetime.now() - started}｜落地 {len(c.seen):,} 筆")
    for b in S.INDUSTRY_BUCKETS:
        print(f"  {b}: {c.per_bucket.get(b, 0)}")
    print("=" * 70)

    if aborted:
        raise SystemExit(
            f"\n✗ 中止：{aborted}\n"
            f"   已落地 {len(c.seen):,} 筆，保留在 {c.out.name}，進度在 "
            f"{c.prog_path.name}。\n"
            "   確認解封後把同一行 pipeline 再跑一次，會從斷點續抓：\n"
            "   curl -s -o /dev/null -w '%{http_code}\\n' https://www.1111.com.tw/"
        )
    if not c.seen:
        raise SystemExit("\n✗ 跑完但一筆都沒抓到，不當作成功。先手動 curl 確認來源狀態。")


if __name__ == "__main__":
    main()
