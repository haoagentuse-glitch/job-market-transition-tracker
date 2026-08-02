"""Scrapy 爬蟲：1111 職缺搜尋 API。

跟舊版（requests 序列迴圈）的差別只有兩件事，但這兩件就是慢的全部原因：

1. 舊版必須「拿到第 N 頁才知道要不要爬第 N+1 頁」，所以只能序列。
   這裡先抓每桶第 1 頁、讀 pagination.totalPage，再把 2..N 頁一次全部排進佇列，
   由 Scrapy 的排程器在併發上限內流水線化。
2. 舊版每 500 筆把整份 55MB CSV 重寫一次（O(n²) I/O）。
   這裡走 Scrapy FEEDS append-only JSONL，斷了就是斷在最後一行，前面全在。

爬取階段刻意「零過濾、零評分」——只負責忠實落地。所有清洗都在 ingest 階段，
這樣改清洗標準不必重爬。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import ClassVar
from urllib.parse import urlencode

import scrapy
from scrapy.crawler import CrawlerProcess

from jobshift import settings as S


class JobsSpider(scrapy.Spider):
    name = "jobs"

    custom_settings: ClassVar[dict] = {
        "CONCURRENT_REQUESTS": S.CRAWL_CONCURRENCY,
        "CONCURRENT_REQUESTS_PER_DOMAIN": S.CRAWL_CONCURRENCY,
        "DOWNLOAD_DELAY": S.CRAWL_DELAY,
        "RANDOMIZE_DOWNLOAD_DELAY": False,
        "DOWNLOAD_TIMEOUT": 30,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 522, 524],
        # 403／429 要讓它進到 callback 才數得到，否則 httperror 中介層會直接吞掉，
        # 結果就是「跑了半小時、零產出、零錯誤」。
        "HTTPERROR_ALLOWED_CODES": [403, 429],
        # 對方變慢就自動退讓，把固定延遲當下限而不是目標
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": S.CRAWL_DELAY,
        "AUTOTHROTTLE_MAX_DELAY": 60.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        # 實查 robots.txt（164 行、只有一組 User-agent: *）：沒有任何規則涵蓋
        # /api/v1/search/jobs/，也沒有 Crawl-delay。既然沒被禁止就遵守它，
        # 對方之後加規則我們會自動跟上。
        "ROBOTSTXT_OBEY": True,
        "COOKIES_ENABLED": False,
        "LOG_LEVEL": "INFO",
        # 完整照抄瀏覽器實際送出的標頭。少了 sec-fetch-* 這組，請求看起來就不像
        # 從站方自己的頁面發出的 XHR。舊版爬蟲送的是完整組，重寫時不該精簡掉。
        "DEFAULT_REQUEST_HEADERS": {
            "accept": "*/*",
            "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "cache-control": "no-cache",
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
        },
    }

    def __init__(self, snapshot: str | None = None, *a, **kw):
        super().__init__(*a, **kw)
        self.snapshot = snapshot or S.SNAPSHOT_DATE
        # 全域去重，first-seen wins —— 與舊爬蟲同語意（API 會跨桶塞推薦職缺）
        self.seen: set[str] = set()
        self.stats_per_bucket: dict[str, int] = {}
        self.blocked = 0          # 連續 403/429 計數，滿門檻就中止

        # 續跑：已抓完的 (桶, 頁) 與已見過的 job_id，都從上一次的產出讀回來。
        # 序列爬一整場要幾小時，被擋或手動中斷之後不該從第一頁重來。
        self.done: set[str] = set()
        prog = S.crawl_progress(self.snapshot)
        if prog.exists():
            self.done = set(json.loads(prog.read_text(encoding="utf-8")).get("pages", []))
        jsonl = S.raw_jsonl(self.snapshot)
        if jsonl.exists():
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.seen.add(str(json.loads(line)["job_id"]))
        if self.done or self.seen:
            print(f"[續跑] 已完成 {len(self.done)} 頁、已有 {len(self.seen)} 筆職缺，跳過那些頁")

    # ── 請求組裝 ────────────────────────────────────────────────────
    def _request(self, bucket: str, codes: list[str], page: int, is_probe: bool):
        # searchUrl 的值本身含 ? 與 &，一定要走 urlencode，手工字串拼接會炸掉查詢字串
        params = [*S.FETCH_PARAMS_BASE.items(), ("page", str(page))]
        params += [("jobPositions", c) for c in codes]
        return scrapy.Request(
            url=f"{S.BASE_URL}?{urlencode(params)}",
            method="GET",
            callback=self.parse_probe if is_probe else self.parse_page,
            cb_kwargs={"bucket": bucket, "codes": codes, "page": page},
            dont_filter=True,
        )

    def _key(self, bucket: str, page: int) -> str:
        return f"{bucket}|{page}"

    def _save_progress(self, complete: bool = False) -> None:
        S.crawl_progress(self.snapshot).write_text(
            json.dumps({"pages": sorted(self.done), "complete": complete},
                       ensure_ascii=False), encoding="utf-8")

    def _seed(self):
        for bucket, codes in S.INDUSTRY_BUCKETS.items():
            # 第 1 頁完成過，代表這個桶的總頁數已知；重排時仍要靠它取得 totalPage，
            # 所以照樣送，只是回來時不會重複計入（job_id 已在 seen 裡）。
            yield self._request(bucket, codes, page=1, is_probe=True)

    # Scrapy 2.13 起入口是 async def start()，2.17 已經把 start_requests 從
    # Spider 基底類別移除。只定義 start_requests 的話它不會被呼叫，基底的 start()
    # 會去讀空的 start_urls —— 零請求、零錯誤、finish_reason=finished，全靜默。
    # 兩個都定義，新舊版都能跑。
    async def start(self):
        for req in self._seed():
            yield req

    def start_requests(self):          # Scrapy < 2.13 走這個
        yield from self._seed()

    # ── 第 1 頁：讀總頁數，把其餘頁一次全部排進佇列 ──────────────────
    def parse_probe(self, response, bucket: str, codes: list[str], page: int):
        result = self._result(response, bucket, page)
        if result is None:
            return

        pg = result.get("pagination") or {}
        total_page = int(pg.get("totalPage") or 1)
        total_count = int(pg.get("totalCount") or 0)
        last = min(total_page, S.MAX_PAGES)
        todo = [p for p in range(2, last + 1) if self._key(bucket, p) not in self.done]
        self.logger.info(
            f"[{bucket}] totalCount={total_count} totalPage={total_page} → "
            f"需抓 {len(todo)}/{last - 1} 頁"
            + (f"（續跑跳過 {last - 1 - len(todo)} 頁）" if len(todo) < last - 1 else "")
        )

        self.done.add(self._key(bucket, 1))
        yield from self._emit(result, bucket)
        for p in todo:
            yield self._request(bucket, codes, page=p, is_probe=False)

    def parse_page(self, response, bucket: str, codes: list[str], page: int):
        result = self._result(response, bucket, page)
        if result is None:
            return
        self.done.add(self._key(bucket, page))
        if len(self.done) % 50 == 0:      # 定期落盤，當機也只損失最近幾頁
            self._save_progress()
        yield from self._emit(result, bucket)

    # ── 工具 ────────────────────────────────────────────────────────
    def _result(self, response, bucket: str, page: int) -> dict | None:
        if response.status in (403, 429):
            self.blocked += 1
            self.logger.warning(
                f"[{bucket}] p{page} HTTP {response.status}"
                f"（連續 {self.blocked}/{S.CRAWL_ABORT_AFTER_403}）"
            )
            if self.blocked >= S.CRAWL_ABORT_AFTER_403:
                self.logger.error(
                    "連續被擋，中止爬取。被封鎖後繼續請求只會延長封鎖時間。\n"
                    "  先用 curl 確認解封了沒："
                    "curl -s -o /dev/null -w '%{http_code}\\n' https://www.1111.com.tw/\n"
                    "  回到 200 之後再重跑；已抓到的部分留在 JSONL 裡，不會重來。"
                )
                self.crawler.engine.close_spider(self, "blocked_by_source")
            return None
        self.blocked = 0
        try:
            data = json.loads(response.text)
        except Exception as exc:
            self.logger.warning(f"[{bucket}] p{page} JSON decode 失敗: {exc}")
            return None
        result = data.get("result")
        if not isinstance(result, dict):
            self.logger.warning(f"[{bucket}] p{page} 回應無 result")
            return None
        return result

    def _emit(self, result: dict, bucket: str):
        for job in result.get("hits") or []:
            job_id = job.get("jobId")
            if not job_id:
                continue
            job_id = str(job_id)
            if job_id in self.seen:
                continue
            self.seen.add(job_id)
            self.stats_per_bucket[bucket] = self.stats_per_bucket.get(bucket, 0) + 1
            yield self._parse_job(job, bucket)

    def _parse_job(self, job: dict, bucket: str) -> dict:
        """只做欄位萃取，不做任何清洗／過濾／評分。"""
        industry = job.get("industry") or {}
        work_city = job.get("workCity") or {}
        require = job.get("require") or {}
        job_id = str(job.get("jobId"))
        return {
            "snapshot_date":   self.snapshot,
            "job_id":          job_id,
            "job_title":       job.get("title"),
            "company_id":      job.get("companyId"),
            "company_name":    job.get("companyName"),
            "updated_at":      job.get("updateAt"),
            "industry_id":     industry.get("id"),
            "industry_name":   industry.get("name"),
            "industry_bucket": bucket,
            "exp_code":        require.get("experience"),
            "edu_codes":       require.get("grades", []),
            "major_codes":     require.get("majors", []),
            "city_name":       work_city.get("name"),
            "salary_desc":     job.get("salary"),
            "description":     job.get("description", ""),
            "job_url":         f"https://www.1111.com.tw/job/{job_id}/",
            "is_happiness":    job.get("isHappiness", False),
            "recruit_range":   job.get("recruitCountString"),
        }

    def closed(self, reason):
        self._save_progress(complete=(reason == "finished"))
        total = len(self.seen)
        self.logger.info("=" * 60)
        self.logger.info(f"爬取結束（{reason}）：unique job_id = {total}")
        for b in S.INDUSTRY_BUCKETS:
            self.logger.info(f"  {b}: {self.stats_per_bucket.get(b, 0)}")
        self.logger.info("=" * 60)


def main() -> None:
    snapshot = S.SNAPSHOT_DATE
    out = S.raw_jsonl(snapshot)
    print(f"快照日期 = {snapshot}")
    print(f"輸出     = {out}")
    print(f"併發 {S.CRAWL_CONCURRENCY}／延遲 {S.CRAWL_DELAY}s／每桶上限 {S.MAX_PAGES} 頁")
    started = datetime.now()

    process = CrawlerProcess(settings={
        "FEEDS": {str(out): {
            "format": "jsonlines",
            "encoding": "utf8",
            "overwrite": False,          # append-only：中斷可續，不覆蓋既有進度
            "store_empty": False,
        }},
        "TELNETCONSOLE_ENABLED": False,
    })
    crawler = process.create_crawler(JobsSpider)
    process.crawl(crawler, snapshot=snapshot)
    process.start()

    stats = crawler.stats.get_stats()
    scraped = stats.get("item_scraped_count", 0)
    responses = stats.get("downloader/response_count", 0)
    print(f"總耗時：{datetime.now() - started}")
    print(f"回應數 {responses}｜落地職缺 {scraped}")

    if stats.get("finish_reason") == "blocked_by_source":
        raise SystemExit(
            f"\n✗ 被來源站封鎖而中止（已落地 {scraped} 筆，保留在 {out.name}）。\n"
            "   等解封後把同一行 pipeline 再跑一次，會從既有進度接續。\n"
            "   確認解封：curl -s -o /dev/null -w '%{http_code}\\n' https://www.1111.com.tw/"
        )

    # 「跑完了但什麼都沒抓到」必須是失敗。上一次 Scrapy 換 API 時就是這樣悄悄
    # 掛掉的：零請求、零錯誤、exit 0，管線繼續往下跑去清洗一個空檔案。
    if scraped == 0:
        raise SystemExit(
            "\n✗ 爬蟲跑完但一筆都沒抓到。常見原因：\n"
            "   - Scrapy 換版導致 spider 進入點沒被呼叫（檢查回應數是不是 0）\n"
            "   - 來源 API 改了路徑或參數（先手動 curl 一次確認）\n"
            "   - 被限流／擋 IP（回應數 > 0 但都是 4xx/5xx）\n"
            f"   統計：{ {k: v for k, v in stats.items() if 'response_status' in k or 'exception' in k} }"
        )


if __name__ == "__main__":
    main()
