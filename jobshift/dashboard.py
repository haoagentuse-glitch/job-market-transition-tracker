"""Streamlit 儀表板。只呼叫 API，不直接接 DuckDB —— 查詢邏輯只有一份。"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API = os.getenv("JOBSHIFT_API", "http://localhost:8000")

st.set_page_config(page_title="台灣職缺語意結構分析", page_icon="◧", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(ttl=300)
def get(path: str, **params) -> list | dict:
    r = httpx.get(f"{API}{path}",
                  params={k: v for k, v in params.items() if v is not None}, timeout=60)
    r.raise_for_status()
    return r.json()


def df(path: str, **params) -> pd.DataFrame:
    return pd.DataFrame(get(path, **params))


def money(v) -> str:
    return f"{int(v):,}" if pd.notna(v) else "—"


try:
    META = get("/meta")
    CMETA = get("/concepts/meta")
except Exception as exc:
    st.error(f"API 無回應（{API}）：{exc}")
    st.stop()

SNAP = META.get("reference_snapshot", "?")
N_JOBS = int(CMETA.get("n_jobs", 0))
N_AI = int(CMETA.get("n_ai_jobs", 0))

with st.sidebar:
    st.subheader("資料")
    st.write(f"快照　{SNAP}")
    st.write(f"職缺　{N_JOBS:,}")
    st.write("來源　1111 人力銀行")
    st.divider()
    st.subheader("方法")
    st.write("嵌入　bge-m3 · 1024 維")
    st.write(f"分群　{META.get('cluster_method')}")
    st.write(f"群集　{META.get('n_reference_clusters')}")
    st.write(f"噪點　{float(META.get('noise_ratio', 0)):.1%}")
    st.divider()
    DIM = st.radio("分類維度", ["行業大類", "職務類別"], index=0)

st.title("台灣職缺的語意結構")
st.caption(f"{SNAP} · {N_JOBS:,} 筆 · 以職缺描述的語意向量建立分類，不採用平台既有標籤")

tabs = st.tabs(["市場全景", "數位職缺", "AI 技能", "薪資", "方法與限制"])

# ── 市場全景 ───────────────────────────────────────────────────────
with tabs[0]:
    cl = df("/clusters")
    if cl.empty:
        st.warning("尚未執行分群")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("語意群集", len(cl))
        c2.metric("最大群集佔比", f"{cl.share_to.max():.1%}")
        c3.metric("前十群集合計", f"{cl.nlargest(10, 'share_to').share_to.sum():.1%}")

        top = cl.nlargest(25, "n_to")
        fig = px.bar(top.sort_values("n_to"), x="n_to", y="label", orientation="h",
                     hover_data=["top_titles"], labels={"n_to": "職缺數", "label": ""})
        fig.update_layout(height=680, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            cl[["cluster_id", "label", "top_titles", "top_terms", "n_to", "share_to"]]
            .sort_values("n_to", ascending=False),
            column_config={
                "cluster_id": "群集", "label": "代表職稱", "top_titles": "鄰近職稱",
                "top_terms": "鑑別詞", "n_to": st.column_config.NumberColumn("職缺數", format="%d"),
                "share_to": st.column_config.NumberColumn("佔比", format="%.2f%%"),
            },
            use_container_width=True, hide_index=True, height=420)

        ids = cl.set_index("cluster_id").label.to_dict()
        cid = st.selectbox("檢視群集內職缺", cl.cluster_id.tolist(),
                           format_func=lambda i: f"{i}　{ids.get(i, '')}")
        st.dataframe(df(f"/clusters/{cid}/jobs", limit=50),
                     use_container_width=True, hide_index=True)

# ── 數位職缺 ───────────────────────────────────────────────────────
with tabs[1]:
    cs = df("/concepts")
    if cs.empty:
        st.warning("尚未執行技能分析")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("數位技能職缺", f"{int(CMETA.get('n_digital_jobs', 0)):,}")
    c2.metric("佔全體", f"{int(CMETA.get('n_digital_jobs', 0)) / N_JOBS:.2%}")

    st.dataframe(
        cs[["concept", "n", "share", "median_salary", "salary_premium",
            "median_desc_len"]],
        column_config={
            "concept": "技能概念",
            "n": st.column_config.NumberColumn("職缺數", format="%d"),
            "share": st.column_config.NumberColumn("佔全體", format="%.2f%%"),
            "median_salary": st.column_config.NumberColumn("月薪中位", format="%d"),
            "salary_premium": st.column_config.NumberColumn("對全體倍數", format="%.2f"),
            "median_desc_len": st.column_config.NumberColumn("描述字數中位", format="%d"),
        },
        use_container_width=True, hide_index=True)

    cross = df("/concepts/cross", dimension=DIM, min_n=5)
    if not cross.empty:
        st.subheader(f"技能概念在{DIM}中的分布")
        piv = cross.pivot_table(index="category", columns="concept",
                                values="share_in_category", fill_value=0)
        fig = px.imshow(piv * 100, aspect="auto", color_continuous_scale="Blues",
                        labels={"color": "%"})
        fig.update_layout(height=520, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                          xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("數值為該類別中具備此技能的職缺比例。技能不受職務分類限制，"
                   "同一概念會橫跨多個類別。")

        pick = st.selectbox("概念集中於哪些類別", sorted(cross.concept.unique()))
        sub = cross[cross.concept == pick].nlargest(15, "n")
        fig = px.bar(sub.sort_values("n"), x="n", y="category", orientation="h",
                     labels={"n": "職缺數", "category": ""})
        fig.update_layout(height=420, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

# ── AI 技能 ────────────────────────────────────────────────────────
with tabs[2]:
    ai = cs[cs.is_ai]
    c1, c2, c3 = st.columns(3)
    c1.metric("AI 技能職缺", f"{N_AI:,}")
    c2.metric("佔全體", f"{N_AI / N_JOBS:.2%}")
    prem = ai.salary_premium.mean()
    c3.metric("月薪相對全體", f"{prem:.2f}×" if pd.notna(prem) else "—")

    st.dataframe(
        ai[["concept", "n", "share", "median_salary", "salary_premium",
            "median_desc_len"]],
        column_config={
            "concept": "概念", "n": st.column_config.NumberColumn("職缺數", format="%d"),
            "share": st.column_config.NumberColumn("佔全體", format="%.2f%%"),
            "median_salary": st.column_config.NumberColumn("月薪中位", format="%d"),
            "salary_premium": st.column_config.NumberColumn("對全體倍數", format="%.2f"),
            "median_desc_len": st.column_config.NumberColumn("描述字數中位", format="%d"),
        },
        use_container_width=True, hide_index=True)

    aic = cross[cross.concept.isin(ai.concept)] if not cross.empty else pd.DataFrame()
    if not aic.empty:
        st.subheader(f"AI 技能職缺的{DIM}分布")
        fig = px.bar(aic.nlargest(20, "n").sort_values("n"), x="n", y="category",
                     color="concept", orientation="h",
                     labels={"n": "職缺數", "category": "", "concept": ""})
        fig.update_layout(height=520, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                          legend={"orientation": "h", "y": -0.12})
        st.plotly_chart(fig, use_container_width=True)

    pick_ai = st.selectbox("檢視職缺", ai.concept.tolist())
    view = st.radio("", ["已認定", "待驗證候選"], horizontal=True,
                    label_visibility="collapsed")
    if view == "已認定":
        st.dataframe(df("/concepts/jobs", concept=pick_ai, limit=200),
                     use_container_width=True, hide_index=True, height=420)
    else:
        st.caption("關鍵字未命中、但語意錨點相似度最高者。錨點相似度的精確率不足以直接採信，"
                   "此處僅供人工檢視，不計入任何統計。")
        st.dataframe(df("/concepts/candidates", concept=pick_ai),
                     use_container_width=True, hide_index=True, height=380)

# ── 薪資 ───────────────────────────────────────────────────────────
with tabs[3]:
    sal = df("/concepts/salary")
    tl = df("/timeline")
    if not tl.empty:
        base = tl.iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("全體月薪中位", money(base.median_salary))
        c2.metric("可換算月薪比例", f"{base.salary_coverage:.0%}")
        st.caption("面議與論件計酬無可比數字，不計入薪資統計。")

    if not sal.empty:
        fig = px.bar(sal.sort_values("p50"), x="p50", y="concept", orientation="h",
                     error_x=sal.sort_values("p50").p75 - sal.sort_values("p50").p50,
                     error_x_minus=sal.sort_values("p50").p50 - sal.sort_values("p50").p25,
                     labels={"p50": "月薪中位數", "concept": ""})
        if not tl.empty and pd.notna(tl.iloc[0].median_salary):
            fig.add_vline(x=float(tl.iloc[0].median_salary), line_dash="dot")
        fig.update_layout(height=460, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("誤差線為四分位距。虛線為全體中位數。")
        st.dataframe(sal, use_container_width=True, hide_index=True,
                     column_config={"concept": "概念", "n": "樣本數",
                                    "p25": "P25", "p50": "中位", "p75": "P75"})

# ── 方法與限制 ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("方法")
    st.markdown("""
職缺結構以非監督方式建立：描述文字經 bge-m3 編碼為 1024 維向量，
PCA 降至 50 維後以 UMAP 投影，再由 HDBSCAN 分群。群集數量未事先指定，
標籤取自最接近群集中心的實際職稱。

技能偵測採用關鍵字樣式比對，樣式經誤判檢查後定版。
""")

    st.subheader("被否決的方案：向量錨點分類")
    st.markdown("""
原設計為每個技能概念撰寫描述工作內容的錨句，編碼後與職缺向量計算餘弦相似度，
以相似度認定技能。在全量資料上測量後否決，指標如下：
""")
    sep = cs[["concept", "anchor_sim_median", "anchor_sim_p99_all", "separation_sigma"]]
    st.dataframe(sep, use_container_width=True, hide_index=True,
                 column_config={
                     "concept": "概念",
                     "anchor_sim_median": st.column_config.NumberColumn(
                         "正例相似度中位", format="%.3f"),
                     "anchor_sim_p99_all": st.column_config.NumberColumn(
                         "全體相似度 P99", format="%.3f"),
                     "separation_sigma": st.column_config.NumberColumn(
                         "分離度 (σ)", format="%.2f"),
                 })
    st.markdown("""
多數概念的全體 P99 高於正例中位數，代表無關職缺的高分尾端與真正例重疊；
取相似度前 500 名僅召回兩成正例。原因為職缺描述中位數僅 128 字，
且含大量「工作內容」「福利」等模板語，單一技能詞在整段語意中的比重過低。

向量因此僅用於其有效之處：非監督分群、相似職缺檢索，以及待驗證候選的排序。
""")

    st.subheader("限制")
    st.markdown(f"""
- 單一時間點（{SNAP}），描述當期結構，不構成趨勢。
- 來源為單一商業求職平台，雇主組成有其特性，不代表整體勞動市場。
- 採集時每個職務類別設有 4,500 筆上限，熱門類別的絕對數為截斷值，佔比較絕對數可靠。
- 群集邊界由語意模型與分群參數決定，非官方職業分類。
- 技能偵測依賴雇主是否於描述中寫出該技能，未寫出者無法偵測。
- 薪資統計僅涵蓋可換算為月薪者。
""")
