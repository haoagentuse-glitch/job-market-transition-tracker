# jobshift — 職缺市場的結構遷移

用兩個時間點的職缺快照，看**市場的語意結構**怎麼變：哪類工作在長、哪類在縮、
有沒有出現舊快照裡根本不存在的新型態、哪些公司想找的人換了種類。

- 基準期：`2026-05-19`（舊專案留下的 65,519 筆快照）
- 對照期：現在爬的這一批
- 全部跑在 Docker 裡，資料只落在 `./data`

---

## 一分鐘看懂這條線

```
spider.py    1111 搜尋 API → JSONL（零過濾，只忠實落地）
   ↓
ingest.py    型別正規化 → 去重 → 只擋硬缺陷 → 文字正規化 → 薪資解析 → DuckDB
   ↓
embed.py     bge-m3（GPU）→ 已 L2 正規化的 .npy
   ↓
analyze.py   舊快照分群當座標系 → 新快照 kNN 指派 → 三層遷移 → DuckDB
   ↓
api.py       FastAPI（唯一資料出口） →  dashboard.py  Streamlit
```

## 為什麼是這樣設計（三個會被問到的決定）

**參考座標系只建在舊快照上。** 分群 fit 一次，新快照是被「指派」進來的。
若兩期各自分群再對齊，群集編號會漂移，遷移矩陣就變成對齊演算法的產物而不是市場的變化。

**跨期指派用 kNN 多數決，不用群心距離。** HDBSCAN 的群在降維空間常是非凸的，
用群心指派會錯得很難察覺。kNN 直接在原始 1024 維 cosine 空間投票，
非凸也不怕，票數與平均相似度天然就是信心指標（`KNN_VOTE_RATIO` / `KNN_SIM_FLOOR`）。

**「遷移」拆成三層。** 同一則職缺的描述兩期不會變，硬做 job 層級的群間移動是假的：

| 層 | 問的問題 | 表 |
|---|---|---|
| ① 產業 × 群集 | 某產業的職缺分散到哪些語意群，兩期怎麼重組 | `flow_industry_cluster` |
| ② 群集消長 | 哪些語意群在擴張／萎縮／新生／消亡 | `clusters`、`cluster_flow` |
| ③ 公司位移 | 同一家公司的招募重心向量偏移了多少 | `company_shift` |

## 相對舊版改了什麼

| | 舊版 | 這版 |
|---|---|---|
| 爬取 | requests 序列迴圈，拿到第 N 頁才知道要不要爬 N+1 | Scrapy 先探第 1 頁讀 `totalPage`，其餘頁一次排進佇列流水線化 |
| 存檔 | 每 500 筆重寫整份 55MB CSV（O(n²) I/O） | append-only JSONL，斷了只斷最後一行 |
| 過濾時機 | **爬取當下**就用關鍵字評分丟資料，改標準要重爬 | 爬取零過濾；清洗階段只擋硬缺陷，語意判斷全部推遲到向量階段 |
| 語意 | jieba 斷詞 + 人工關鍵字權重表 | bge-m3 向量；**完全沒有 jieba**，連群集命名都靠 embedding + 字元 n-gram |
| 儲存 | PostgreSQL（要 server、要密碼） | DuckDB 單檔 |
| 依賴 | `requirements.txt` + pip | `pyproject.toml` + `uv.lock`（本機與容器版本完全一致）、ruff |

## 已知限制（別過度解讀結果）

- **只有兩個觀測點，不是趨勢。** 任何「持續上升／下降」的說法在這份資料上都不成立。
- **兩期都套用每產業 4,500 筆的相同上限**（沿用舊快照的截斷條件以維持可比）。
  熱門產業的絕對數是被截斷的 —— **看佔比比看總數可靠**。
- 舊快照是既成事實，無法重爬，所以清洗管線必須對它與新資料一視同仁。
- API 參數與舊快照一字不改（含 `isSyncedRecommendJobs=true`，它會跨桶塞推薦職缺）。
  去重是全域 first-seen wins，與舊爬蟲同語意。

---

## 怎麼跑

### 0. 建 image（第一次，約 5–10 分鐘）

```bash
cd ~/jobshift && docker compose build
```

### 1. 【長任務・你來啟動】爬現在這個時刻的快照

保守設定：併發 4、固定 1 秒延遲，約 **45–55 分鐘**。中斷可直接重跑，JSONL 是 append-only。

```bash
cd ~/jobshift && docker compose run --rm crawl
```

想快一點（風險自負，約 15 分鐘）：

```bash
cd ~/jobshift && CRAWL_CONCURRENCY=8 CRAWL_DELAY=0.35 docker compose run --rm crawl
```

### 2. 清洗入庫（兩期一起，約 1–2 分鐘）

```bash
cd ~/jobshift && docker compose run --rm ingest
```

### 3. 【長任務・你來啟動】向量化（GPU，首次會下載 bge-m3 約 2.3GB）

13 萬筆在 RTX 4060 上約 **15–25 分鐘**。模型快取在 `./data/hf_cache`，只下載一次。

```bash
cd ~/jobshift && docker compose run --rm embed
```

### 4. 分析（約 5–10 分鐘，UMAP 是大宗）

```bash
cd ~/jobshift && docker compose run --rm analyze
```

### 5. 起服務

```bash
cd ~/jobshift && docker compose up -d api dashboard
```

- 儀表板 <http://localhost:8501>
- API 文件 <http://localhost:8000/docs>

---

---

## 在本機開發（不進容器）

依賴由 uv 管，`uv.lock` 鎖死版本，容器裡裝的跟你本機裝的是同一組。

```bash
cd ~/jobshift && uv sync
```

```bash
cd ~/jobshift && uv run ruff check . && uv run ruff format --check .
```

單獨跑某個階段（例：只重跑分析，不重爬也不重新向量化）：

```bash
cd ~/jobshift && JOBSHIFT_DATA_DIR=./data uv run python -m jobshift.analyze
```

### 真資料還沒到位時先把整條線走通

`devdata.py` 會從舊快照抽樣、刻意扭曲產業分布，造一份假的對照期，
讓分群與遷移分析看得到真實的結構差異：

```bash
cd ~/jobshift && docker compose run --rm ingest python -m jobshift.devdata --n 6000
```

---

## 調參數

全部在 `jobshift/settings.py`，或用環境變數蓋掉：

| 變數 | 預設 | 意思 |
|---|---|---|
| `CRAWL_CONCURRENCY` | 4 | Scrapy 併發 |
| `CRAWL_DELAY` | 1.0 | 每次請求間隔（秒） |
| `CRAWL_MAX_PAGES` | 150 | 每產業頁數上限（150 × 30 = 4500，同舊快照） |
| `JOBSHIFT_SNAPSHOT_DATE` | 今天 | 新快照日期 |
| `JOBSHIFT_EMBED_BATCH` | 64 | 顯存不夠就調小 |

分群品質相關的 `KNN_SIM_FLOOR`（未歸類門檻）、`HDBSCAN_MIN_FRAC`（群的最小規模）、
`NOISE_FALLBACK`（噪點超過就退回 KMeans）也都在 `settings.py`，改完重跑 `analyze` 即可，
不必重爬也不必重新向量化。
