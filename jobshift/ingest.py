"""清洗與入庫。舊快照 CSV 與新快照 JSONL 走「同一條」管線，否則兩期不可比。

清洗順序（相對舊版的關鍵調換）：
  舊版 = 爬取時就用關鍵字評分把資料丟掉 → 去重 → 斷詞 → 入庫
         過濾發生在正規化之前、且與爬蟲綁死，改標準得重爬，資訊不可逆損失。
  新版 = ① 型別正規化 → ② 去重 → ③ 只擋硬缺陷 → ④ 文字正規化 → ⑤ 薪資解析
         → ⑥ 組嵌入文字。任何語意判斷都推遲到向量階段，這裡不做任何主觀篩選。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata

import pandas as pd

from jobshift import db
from jobshift import settings as S

CANONICAL = [
    "snapshot_date", "job_id", "job_title", "company_id", "company_name",
    "updated_at", "industry_id", "industry_name", "industry_bucket",
    "exp_code", "edu_codes", "major_codes", "city_name",
    "salary_desc", "salary_type", "salary_min", "salary_max", "salary_month_min",
    "description", "description_len", "text_for_embed", "job_url",
    "is_happiness", "recruit_range",
]

# ── ④ 文字正規化 ────────────────────────────────────────────────────
_TAG = re.compile(r"<[^>]{1,200}>")
_DECOR = re.compile(r"[※★☆◆◇■□●○▲△▼▽*=＝_－—–\-~～·・]{3,}")
_BULLET = re.compile(r"^[\s\d]{0,4}[.、)）:：]\s*", re.M)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f​-‏﻿]")
_WS = re.compile(r"\s+")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿]"
)


def normalize_text(text) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = html.unescape(text)
    t = _TAG.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)      # 全形→半形、相容字元統一
    t = _CTRL.sub("", t)
    t = _EMOJI.sub(" ", t)
    t = _DECOR.sub(" ", t)                    # 「※※※※」這種分隔裝飾對語意是噪音
    t = _BULLET.sub("", t)                    # 條列編號
    t = _WS.sub(" ", t)
    return t.strip()


# ── ⑤ 薪資解析 ─────────────────────────────────────────────────────
_SAL_KIND = re.compile(r"(月薪|年薪|時薪|日薪|論件計酬|面議|待遇面議)")
_SAL_NUM = re.compile(r"([\d,]{3,})")
_MONTHLY_FACTOR = {"月薪": 1.0, "年薪": 1 / 12, "時薪": 176.0, "日薪": 22.0}


def parse_salary(raw) -> tuple[str | None, float | None, float | None, float | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, None, None, None
    s = unicodedata.normalize("NFKC", raw)
    kind_m = _SAL_KIND.search(s)
    kind = kind_m.group(1) if kind_m else None
    if kind in ("面議", "待遇面議"):
        return "面議", None, None, None
    nums = [float(n.replace(",", "")) for n in _SAL_NUM.findall(s)]
    nums = [n for n in nums if n >= 100]
    if not nums:
        return kind, None, None, None
    lo, hi = min(nums), (max(nums) if len(nums) > 1 else None)
    factor = _MONTHLY_FACTOR.get(kind or "")
    month_min = round(lo * factor) if factor else None
    return kind, lo, hi, month_min


# ── 載入 ───────────────────────────────────────────────────────────
def load_legacy_csv(path, snapshot: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["snapshot_date"] = snapshot
    return df


def load_jsonl(path, snapshot: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"[ingest] {path} 是空的，沒有東西可以清洗")
    df = pd.DataFrame(rows)
    df["snapshot_date"] = snapshot
    return df


# ── 主流程 ─────────────────────────────────────────────────────────
SOURCE_COLS = [
    "job_id", "job_title", "company_id", "company_name", "updated_at",
    "industry_id", "industry_name", "industry_bucket", "exp_code",
    "edu_codes", "major_codes", "city_name", "salary_desc", "description",
    "job_url", "is_happiness", "recruit_range",
]


def clean(df: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    n0 = len(df)
    for col in SOURCE_COLS:            # 兩個來源的欄位補齊，後面才能走同一條路
        if col not in df.columns:
            df[col] = None

    # ① 型別正規化
    for col in ("job_id", "company_id", "industry_id", "exp_code",
                "edu_codes", "major_codes", "industry_name", "city_name",
                "salary_desc", "recruit_range", "job_url", "industry_bucket",
                "company_name", "job_title", "description"):
        df[col] = df[col].astype(str).replace({"None": "", "nan": ""}).str.strip()
    df["updated_at"] = pd.to_datetime(df["updated_at"], format="mixed", errors="coerce")
    df["is_happiness"] = df["is_happiness"].astype(str).str.lower().isin(["true", "1"])

    # ② 去重：同 job_id 保留 updated_at 最新的一筆
    df = (df.sort_values("updated_at", ascending=False, na_position="last")
            .drop_duplicates(subset=["job_id"], keep="first"))
    n_dedup = n0 - len(df)

    # ④ 文字正規化（放在過濾之前：先把「看起來空但其實是 &nbsp;」的還原成真的空）
    df["job_title"] = df["job_title"].map(normalize_text)
    df["description"] = df["description"].map(normalize_text)
    df["description_len"] = df["description"].str.len()

    # ③ 硬缺陷過濾：只擋「結構上不可用」的，不做任何語意／關鍵字判斷
    bad_title = (df["job_title"].str.fullmatch(r"(?i)\s*(測試|test|測試職缺)?\s*", na=True)
                 .fillna(True).astype(bool))          # NA 混進布林遮罩會讓索引直接拋錯
    too_short = df["description_len"] < 10
    df = df[~(bad_title | too_short)].copy()
    n_dropped = n0 - n_dedup - len(df)

    # ⑤ 薪資
    parsed = df["salary_desc"].map(parse_salary)
    df["salary_type"] = [p[0] for p in parsed]
    df["salary_min"] = [p[1] for p in parsed]
    df["salary_max"] = [p[2] for p in parsed]
    df["salary_month_min"] = [p[3] for p in parsed]

    # ⑥ 嵌入文字：職稱 + 產業 + 描述。職稱權重靠重複一次拉高（短文本主導語意）
    df["text_for_embed"] = (
        df["job_title"].fillna("") + "。"
        + df["job_title"].fillna("") + "。"
        + df["industry_name"].fillna("") + "。"
        + df["description"].str.slice(0, S.DESC_TRUNCATE)
    ).str.strip()

    df["snapshot_date"] = snapshot
    for col in CANONICAL:
        if col not in df.columns:
            df[col] = None
    df = df[CANONICAL].sort_values("job_id").reset_index(drop=True)

    # 兩期的欄位型別必須一致，否則第二次 INSERT 進 DuckDB 會炸
    for col in ("salary_min", "salary_max", "salary_month_min"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["description_len"] = df["description_len"].astype("int64")
    df["salary_type"] = df["salary_type"].astype("string").fillna("")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    print(f"[ingest] {snapshot}: 讀入 {n0} → 去重 -{n_dedup} → 硬缺陷 -{n_dropped} "
          f"→ 保留 {len(df)}（保留率 {len(df) / n0:.1%}）")
    return df


def ingest_one(source: str, snapshot: str) -> pd.DataFrame:
    if source == "legacy":
        if not S.LEGACY_CSV.exists():
            raise SystemExit(f"[ingest] 找不到舊快照 {S.LEGACY_CSV}")
        df = load_legacy_csv(S.LEGACY_CSV, snapshot)
    else:
        path = S.raw_jsonl(snapshot)
        if not path.exists():
            raise SystemExit(f"[ingest] 找不到 {path}，請先跑 crawl")
        df = load_jsonl(path, snapshot)

    out = clean(df, snapshot)
    out.to_parquet(S.RAW_DIR / f"snapshot_{snapshot}_clean.parquet", index=False)
    with db.connect() as con:
        db.upsert_snapshot(con, "jobs", out, snapshot)
        total = con.execute("SELECT count(*) FROM jobs").fetchone()[0]
    print(f"[ingest] 寫入 DuckDB jobs 表，全表共 {total} 筆")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="舊快照 + 新快照都跑")
    ap.add_argument("--legacy", action="store_true", help="只跑舊快照 CSV")
    ap.add_argument("--snapshot", default=S.SNAPSHOT_DATE, help="新快照日期")
    args = ap.parse_args()

    if args.all or args.legacy:
        ingest_one("legacy", S.BASE_SNAPSHOT)
    if args.all or not args.legacy:
        ingest_one("jsonl", args.snapshot)


if __name__ == "__main__":
    main()
