"""量測爬蟲實際送出請求的節奏。對著本機模擬伺服器跑，不碰任何外部站台。

驗證三件事：
  1. 每頁之間的實際間隔落在 SLEEP_PAGE_MIN–MAX
  2. 每個類別之間的實際間隔落在 SLEEP_CATEGORY_MIN–MAX
  3. 沒有任何一對相鄰請求的間隔短到可疑（＝burst）
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8899
HITS_PER_PAGE = 30
TOTAL_PAGE = 999          # 讓爬蟲一路翻到 MAX_PAGES 才停
LOG: list[tuple[float, str, int]] = []
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        page = int(qs.get("page", ["1"])[0])
        codes = ",".join(qs.get("jobPositions", []))
        with _lock:
            LOG.append((time.monotonic(), codes, page))
        base = len(LOG) * 1000
        body = json.dumps({"result": {
            "pagination": {"page": page, "limit": 30,
                           "totalCount": TOTAL_PAGE * 30, "totalPage": TOTAL_PAGE},
            "hits": [{
                "jobId": base + i, "title": f"職缺{base + i}",
                "companyId": 1, "companyName": "測試公司",
                "updateAt": "2026/08/02 00:00:00",
                "industry": {"id": "1", "name": "測試業"},
                "workCity": {"name": "台北市"},
                "require": {"experience": "0", "grades": [], "majors": []},
                "salary": "月薪 40,000元以上",
                "description": "這是一段夠長的測試職缺描述內容。" * 3,
                "isHappiness": False, "recruitCountString": "1人",
            } for i in range(HITS_PER_PAGE)],
        }}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def report() -> int:
    from jobshift import settings as S

    if len(LOG) < 3:
        print(f"✗ 只收到 {len(LOG)} 個請求，樣本不足")
        return 1

    gaps_page, gaps_cat, all_gaps = [], [], []
    for (t0, c0, p0), (t1, c1, _p1) in itertools.pairwise(LOG):
        gap = t1 - t0
        all_gaps.append((gap, f"{c0[:6]}p{p0} → {c1[:6]}"))
        (gaps_cat if c1 != c0 else gaps_page).append(gap)

    ok = True
    print(f"\n{'=' * 66}\n總請求 {len(LOG)} 個\n{'=' * 66}")

    print(f"\n【每頁之間】規格 {S.SLEEP_PAGE_MIN}–{S.SLEEP_PAGE_MAX} 秒，實測 {len(gaps_page)} 筆")
    if gaps_page:
        print(f"  最小 {min(gaps_page):.2f}s｜中位 {statistics.median(gaps_page):.2f}s"
              f"｜最大 {max(gaps_page):.2f}s")
        if min(gaps_page) < S.SLEEP_PAGE_MIN:
            print(f"  ✗ 有間隔短於下限 {S.SLEEP_PAGE_MIN}s")
            ok = False
        else:
            print("  ✓ 全部不低於下限")

    print(f"\n【類別之間】規格 {S.SLEEP_CATEGORY_MIN}–{S.SLEEP_CATEGORY_MAX} 秒，"
          f"實測 {len(gaps_cat)} 筆")
    for g in gaps_cat:
        mark = "✓" if g >= S.SLEEP_CATEGORY_MIN else "✗"
        print(f"  {mark} {g:.1f}s")
        if g < S.SLEEP_CATEGORY_MIN:
            ok = False

    worst = sorted(all_gaps)[:5]
    print("\n【最短的 5 個間隔】任何一個低於 1 秒都代表有 burst 路徑")
    for g, label in worst:
        print(f"  {g:6.2f}s  {label}")
    if worst[0][0] < 1.0:
        print("  ✗ 出現次秒級連發")
        ok = False
    else:
        print("  ✓ 沒有次秒級連發")

    span = LOG[-1][0] - LOG[0][0]
    print(f"\n平均速率 {len(LOG) / span:.3f} req/s（{len(LOG)} 個請求 / {span:.0f} 秒）")
    print("=" * 66)
    print("結果：" + ("通過" if ok else "有問題"))
    return 0 if ok else 1


def main() -> int:
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    os.environ["JOBSHIFT_BASE_URL"] = f"http://127.0.0.1:{PORT}/api"
    os.environ["CRAWL_MAX_PAGES"] = os.environ.get("TEST_PAGES", "4")

    from jobshift import settings as S

    S.BASE_URL = f"http://127.0.0.1:{PORT}/api"
    S.MAX_PAGES = int(os.environ["CRAWL_MAX_PAGES"])
    S.TARGET_PER_CATEGORY = 10**9          # 別讓上限提早中斷
    keep = int(os.environ.get("TEST_BUCKETS", "3"))
    S.INDUSTRY_BUCKETS = dict(list(S.INDUSTRY_BUCKETS.items())[:keep])

    from jobshift import crawler

    crawler.S = S
    print(f"模擬伺服器 :{PORT}｜{keep} 個類別 × {S.MAX_PAGES} 頁"
          f"｜預估 {(keep - 1) * 150 / 60:.0f} 分鐘（類別間等待是主要成本）\n")
    try:
        crawler.main()
    except SystemExit as exc:
        print(f"(爬蟲結束：{exc})")
    srv.shutdown()
    return report()


if __name__ == "__main__":
    sys.exit(main())
