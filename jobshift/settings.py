"""集中設定。所有會變的東西都在這，其他模組不寫死任何路徑或參數。"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# ── 路徑 ────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("JOBSHIFT_DATA_DIR", "./data")).resolve()
RAW_DIR = DATA_DIR / "raw"
VEC_DIR = DATA_DIR / "vectors"
DB_PATH = DATA_DIR / "jobshift.duckdb"

for _d in (RAW_DIR, VEC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 快照 ────────────────────────────────────────────────────────────
# 參考快照：語意座標系建在它上面，之後每一份新快照都指派進這個座標系。
# 換掉它＝所有群集編號作廢、時間序列從頭來過，所以基本上不該換。
# 用 `or` 而不是 getenv 的 default：compose 會把未設定的變數展開成空字串，
# 那樣 getenv 拿到的是 ""（有設定但為空），default 不會生效。
REFERENCE_SNAPSHOT = os.getenv("JOBSHIFT_REFERENCE_SNAPSHOT") or "2026-05-19"
LEGACY_CSV = RAW_DIR / f"snapshot_{REFERENCE_SNAPSHOT}_raw.csv"

# 這次要跑的快照。預設今天。
SNAPSHOT_DATE = os.getenv("JOBSHIFT_SNAPSHOT_DATE") or date.today().isoformat()


def raw_jsonl(snapshot: str) -> Path:
    return RAW_DIR / f"snapshot_{snapshot}_raw.jsonl"


def crawl_progress(snapshot: str) -> Path:
    """已經抓完的 (桶, 頁)。序列爬完整場要幾小時，中斷後不該從頭再來。"""
    return RAW_DIR / f"snapshot_{snapshot}_progress.json"


def vec_npy(snapshot: str) -> Path:
    return VEC_DIR / f"{snapshot}.npy"


def vec_ids(snapshot: str) -> Path:
    return VEC_DIR / f"{snapshot}_ids.parquet"


# ── 1111 API ───────────────────────────────────────────────────────
# 參數與舊快照「一字不改」，否則兩期不可比。
# 可覆寫是為了能對著本機模擬伺服器量測節流行為 —— 驗證爬蟲的間隔對不對，
# 不該拿對方的站台當測試靶。
BASE_URL = os.getenv("JOBSHIFT_BASE_URL") or "https://www.1111.com.tw/api/v1/search/jobs/"
FETCH_PARAMS_BASE = {
    "sortBy": "da",
    "sortOrder": "desc",
    "searchUrl": "/search/job?page=1&col=da&sort=desc&da=14&tt=1&gr=16",
    "isSyncedRecommendJobs": "true",
}
PAGE_SIZE = 30                                   # API pagination.limit
MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "150"))   # 150 × 30 = 4500／類，同舊上限
TARGET_PER_CATEGORY = 4500

# ── 節流：完全照抄前一版專案，那是唯一有證據不被封鎖的設定 ──────────
#
# 前一版是三層巢狀節流（launcher → crawl_single → crawler），總時長約 2.5–3 小時，
# 抓完 6.5 萬筆從未被擋。關鍵不只是「慢」，是**大量停頓**：類別之間停 2–3 分鐘，
# 流量是一陣一陣的，不是持續穩定的。
#
# 實測反例（2026-08-02）：Scrapy 併發 4 ＋ 1 秒延遲，約 70 個請求後整站封鎖 IP，
# 連首頁 HTML 都回 403；降到序列 ＋ 3 秒連續請求，14 個請求後仍被擋。
# 這三組數字不要調小。
# 每頁之間。前一版是 2.2–3.0 秒，2026-08-02 起放寬到 3–5 秒：那天在 2.2–3.0
# 之下續抓仍然立刻被擋，代表對方的門檻已經比前一版執行時嚴，照抄舊值不夠。
SLEEP_PAGE_MIN = float(os.getenv("CRAWL_SLEEP_PAGE_MIN", "3.0"))
SLEEP_PAGE_MAX = float(os.getenv("CRAWL_SLEEP_PAGE_MAX", "5.0"))
SLEEP_RUN_MIN = float(os.getenv("CRAWL_SLEEP_RUN_MIN", "45"))          # 每輪之間
SLEEP_RUN_MAX = float(os.getenv("CRAWL_SLEEP_RUN_MAX", "90"))
SLEEP_CATEGORY_MIN = float(os.getenv("CRAWL_SLEEP_CAT_MIN", "120"))    # 每類別之間
SLEEP_CATEGORY_MAX = float(os.getenv("CRAWL_SLEEP_CAT_MAX", "180"))

MAX_RUNS_PER_CATEGORY = int(os.getenv("CRAWL_MAX_RUNS", "2"))
# 只跑單一類別。換網路或改參數後拿它探路，用最少的請求判斷會不會被擋，
# 不要一上來就跑全場把新 IP 也賠進去。
ONLY_BUCKET = os.getenv("CRAWL_ONLY_BUCKET") or ""
CRAWL_MAX_RETRIES = int(os.getenv("CRAWL_MAX_RETRIES", "3"))
# 連續這麼多個 403/429 就中止整場：被擋之後繼續打只會延長封鎖，而且一筆都拿不到。
CRAWL_ABORT_AFTER_403 = int(os.getenv("CRAWL_ABORT_AFTER_403", "5"))

# ── 產業分類（20 桶，沿用舊定義，保留為分析維度）────────────────────
INDUSTRY_BUCKETS: dict[str, list[str]] = {
    "01_management_hr_admin":       ["100100", "100200", "100300"],
    "02_finance_accounting_audit":  ["110100", "110200", "110300"],
    "03_sales_customer_service":    ["120100", "120200", "120300", "120400"],
    "04_marketing_pm_planning":     ["130100", "130200", "130300"],
    "05_it_software_data_cloud":    ["140100", "140200", "140300"],
    "06_electronics_semiconductor": ["150100", "150200"],
    "07_mechanical_manufacturing":  ["160100", "160200", "160300"],
    "08_technician_maintenance":    ["170100", "170200", "170300"],
    "09_supplychain_logistics_qa":  ["180100", "180200", "180300"],
    "10_construction_engineering":  ["190100", "190200"],
    "11_medical_healthcare":        ["200100", "200200", "200300", "200400"],
    "12_biotech_chemistry":         ["210100", "210200"],
    "13_media_translation":         ["220100", "220200", "220300", "220400"],
    "14_design":                    ["230100"],
    "15_legal_consulting":          ["240100", "240200"],
    "16_research_education":        ["250100", "250200", "250300", "250400", "250500"],
    "17_child_education_training":  ["260100", "260200", "260300"],
    "18_beauty_food_tourism":       ["270100", "270200", "270300"],
    "19_life_service_agriculture":  ["280100", "280200"],
    "20_security_military":         ["290100", "290200"],
}

# ── 嵌入 ────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("JOBSHIFT_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
EMBED_BATCH = int(os.getenv("JOBSHIFT_EMBED_BATCH", "64"))
EMBED_MAX_LEN = 512
DESC_TRUNCATE = 1000        # 描述截斷字數（p95 是 642 字，1000 涵蓋絕大多數）

# ── 分群與跨期指派 ──────────────────────────────────────────────────
PCA_DIM = 50
UMAP_DIM = 10
UMAP_NEIGHBORS = 15
HDBSCAN_MIN_FRAC = 0.0015   # min_cluster_size = 樣本數 × 此比例（下限 25）
NOISE_FALLBACK = 0.40       # 噪點比超過此值就退回 KMeans
KMEANS_K_GRID = [30, 40, 50, 60, 80]

KNN_K = 15                  # 跨期指派：對舊快照取 15 個最近鄰
KNN_VOTE_RATIO = 0.60       # 多數決門檻
KNN_SIM_FLOOR = 0.62        # 平均 cosine 下限，低於此視為「未歸類 → 新型態候選」
