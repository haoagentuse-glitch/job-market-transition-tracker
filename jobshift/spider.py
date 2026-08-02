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
        "ROBOTSTXT_OBEY": False,   # 打的是站方前端自己在呼叫的 JSON API
        "COOKIES_ENABLED": False,
        "LOG_LEVEL": "INFO",
        "DEFAULT_REQUEST_HEADERS": {
            "accept": "*/*",
            "accept-language": "zh-TW,zh,en-US,en",
            "referer": "https://www.1111.com.tw/search/job",
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

    def start_requests(self):
        for bucket, codes in S.INDUSTRY_BUCKETS.items():
            yield self._request(bucket, codes, page=1, is_probe=True)

    # ── 第 1 頁：讀總頁數，把其餘頁一次全部排進佇列 ──────────────────
    def parse_probe(self, response, bucket: str, codes: list[str], page: int):
        result = self._result(response, bucket, page)
        if result is None:
            return

        pg = result.get("pagination") or {}
        total_page = int(pg.get("totalPage") or 1)
        total_count = int(pg.get("totalCount") or 0)
        last = min(total_page, S.MAX_PAGES)
        self.logger.info(
            f"[{bucket}] totalCount={total_count} totalPage={total_page} "
            f"→ 排程 1..{last} 頁（上限 {S.MAX_PAGES}）"
        )

        yield from self._emit(result, bucket)
        for p in range(2, last + 1):
            yield self._request(bucket, codes, page=p, is_probe=False)

    def parse_page(self, response, bucket: str, codes: list[str], page: int):
        result = self._result(response, bucket, page)
        if result is None:
            return
        yield from self._emit(result, bucket)

    # ── 工具 ────────────────────────────────────────────────────────
    def _result(self, response, bucket: str, page: int) -> dict | None:
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
    process.crawl(JobsSpider, snapshot=snapshot)
    process.start()
    print(f"總耗時：{datetime.now() - started}")


if __name__ == "__main__":
    main()
