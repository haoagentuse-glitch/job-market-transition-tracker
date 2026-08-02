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


def vec_npy(snapshot: str) -> Path:
    return VEC_DIR / f"{snapshot}.npy"


def vec_ids(snapshot: str) -> Path:
    return VEC_DIR / f"{snapshot}_ids.parquet"


# ── 1111 API ───────────────────────────────────────────────────────
# 參數與舊快照「一字不改」，否則兩期不可比。
BASE_URL = "https://www.1111.com.tw/api/v1/search/jobs/"
FETCH_PARAMS_BASE = {
    "sortBy": "da",
    "sortOrder": "desc",
    "searchUrl": "/search/job?page=1&col=da&sort=desc&da=14&tt=1&gr=16",
    "isSyncedRecommendJobs": "true",
}
PAGE_SIZE = 30                                   # API pagination.limit
MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "150"))   # 150 × 30 = 4500／類，同舊上限
TARGET_PER_CATEGORY = 4500

CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "4"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))

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
