"""Streamlit 儀表板。只呼叫 API，不碰 DuckDB —— 查詢邏輯只有一份，在 api.py。"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("JOBSHIFT_API", "http://localhost:8000")

st.set_page_config(page_title="職缺結構遷移", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def get(path: str, **params) -> list | dict:
    r = httpx.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def df(path: str, **params) -> pd.DataFrame:
    return pd.DataFrame(get(path, **params))


try:
    META = get("/meta")
except Exception as exc:
    st.error(f"連不上 API（{API}）或分析尚未跑完：{exc}")
    st.stop()

OLD, NEW = META.get("base_snapshot", "?"), META.get("new_snapshot", "?")

st.title("職缺市場的結構遷移")
st.caption(
    f"基準快照 **{OLD}** → 對照快照 **{NEW}**　|　"
    f"分群方法 `{META.get('cluster_method')}`，參考群集 {META.get('n_reference_clusters')} 個，"
    f"噪點率 {float(META.get('noise_ratio', 0)):.1%}"
)
st.info(
    "**這是兩個時間點的對照，不是趨勢線。** 只有兩個觀測點，任何「持續上升／下降」的說法都不成立。"
    "另外兩期都套用每產業 4,500 筆的相同上限，所以熱門產業的絕對數是被截斷的，看佔比比看總數可靠。",
    icon="⚠️",
)

tabs = st.tabs(["總覽", "語意群集消長", "遷移矩陣", "產業×群集結構", "公司招募位移", "相似職缺"])

# ── 總覽 ────────────────────────────────────────────────────────────
with tabs[0]:
    ov = get("/overview")
    totals = pd.DataFrame(ov["totals"])
    c = st.columns(4)
    for i, row in totals.iterrows():
        c[i].metric(f"{row.snapshot_date} 職缺數", f"{int(row.n):,}")
        c[i + 2].metric(f"{row.snapshot_date} 月薪中位數",
                        f"{int(row.median_salary):,}" if pd.notna(row.median_salary) else "—")

    surv = df("/survival")
    if not surv.empty:
        st.subheader("產業汰換率")
        st.caption("汰換率＝基準期的職缺到對照期已經不在了的比例；更新率＝對照期裡是新出現的比例。")
        fig = px.bar(surv.sort_values("churn_rate"), x="churn_rate", y="industry_bucket",
                     orientation="h", text=surv.sort_values("churn_rate")["churn_rate"]
                     .map(lambda v: f"{v:.0%}"), labels={"churn_rate": "汰換率"})
        fig.update_layout(height=650, yaxis_title=None, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(surv, use_container_width=True, hide_index=True)

    ind = pd.DataFrame(ov["industry"])
    if not ind.empty:
        piv = ind.pivot(index="industry_bucket", columns="snapshot_date", values="share")
        if piv.shape[1] == 2:
            piv["變化(百分點)"] = (piv[NEW] - piv[OLD]) * 100
            piv = piv.sort_values("變化(百分點)")
            st.subheader("產業佔比變化（百分點）")
            fig = px.bar(piv.reset_index(), x="變化(百分點)", y="industry_bucket",
                         orientation="h", color="變化(百分點)",
                         color_continuous_scale="RdBu", color_continuous_midpoint=0)
            fig.update_layout(height=650, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

# ── 語意群集消長 ────────────────────────────────────────────────────
with tabs[1]:
    cl = df("/clusters")
    if cl.empty:
        st.warning("clusters 表是空的，請先跑 analyze。")
    else:
        pick = st.multiselect("狀態篩選", sorted(cl.status.unique()),
                              default=sorted(cl.status.unique()))
        view = cl[cl.status.isin(pick)]
        st.subheader("群集地圖")
        st.caption("點的大小＝對照期職缺數；顏色＝佔比變化；對角線以上代表這個語意群在變大。")
        fig = px.scatter(
            view, x="share_old", y="share_new", size="n_new", color="share_delta_pp",
            hover_name="label", hover_data=["top_titles", "n_old", "n_new", "status"],
            color_continuous_scale="RdBu", color_continuous_midpoint=0, size_max=45,
            labels={"share_old": f"{OLD} 佔比", "share_new": f"{NEW} 佔比"})
        lim = float(max(view.share_old.max(), view.share_new.max())) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                      line=dict(dash="dot", width=1))
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("消長排行")
        cols = ["cluster_id", "status", "label", "top_titles", "top_terms",
                "n_old", "n_new", "delta", "share_delta_pp", "growth_pct",
                "salary_old", "salary_new"]
        st.dataframe(view[[c for c in cols if c in view.columns]]
                     .sort_values("share_delta_pp", ascending=False),
                     use_container_width=True, hide_index=True, height=460)

        cid = st.selectbox("看某個群集的實際職缺", view.cluster_id.tolist(),
                           format_func=lambda i: f"{i} — "
                           f"{view.set_index('cluster_id').label.get(i, '')}")
        if cid is not None:
            st.dataframe(df(f"/clusters/{cid}/jobs", limit=50),
                         use_container_width=True, hide_index=True)

# ── 遷移矩陣 ────────────────────────────────────────────────────────
with tabs[2]:
    st.caption(
        "存活職缺的「基準期群集 → 對照期群集」流向。同一則職缺描述不變，所以絕大多數是自環；"
        "**離開自環的那些才是真的被重新定位**。另外接上「下架」與「新進」兩個端點，流量才守恆。")
    only_moved = st.checkbox("只看換了群集的（隱藏自環）", value=True)
    fl = df("/cluster-flow", min_n=5, include_self=not only_moved)
    if fl.empty:
        st.warning("cluster_flow 是空的。")
    else:
        fl["from_label"] = fl.apply(
            lambda r: "【新進】" if r.from_cluster == -998
            else (r.from_label or f"未歸類/群{int(r.from_cluster)}"), axis=1)
        fl["to_label"] = fl.apply(
            lambda r: "【下架】" if r.to_cluster == -999
            else (r.to_label or f"未歸類/群{int(r.to_cluster)}"), axis=1)
        top = fl.nlargest(40, "n")
        srcs = [f"◀ {s}" for s in top.from_label]
        dsts = [f"{d} ▶" for d in top.to_label]
        nodes = list(dict.fromkeys(srcs + dsts))
        nidx = {n: i for i, n in enumerate(nodes)}
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, pad=12, thickness=14),
            link=dict(source=[nidx[s] for s in srcs], target=[nidx[d] for d in dsts],
                      value=top.n.tolist(),
                      label=[f"{k}" for k in top.kind])))
        fig.update_layout(height=760, font_size=11)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fl.sort_values("n", ascending=False),
                     use_container_width=True, hide_index=True, height=340)

# ── 產業 × 群集結構 ─────────────────────────────────────────────────
with tabs[3]:
    st.caption("每個產業的職缺分散到哪些語意群集，以及兩期之間這個分布怎麼重組。")
    fic = df("/flows", min_n=20)
    if fic.empty:
        st.warning("flow_industry_cluster 是空的。")
    else:
        piv = fic.pivot_table(index="industry_bucket", columns="cluster_id",
                              values="share_in_industry", aggfunc="first",
                              fill_value=0)
        a = fic[fic.snapshot_date == OLD].pivot_table(
            index="industry_bucket", columns="label", values="share_in_industry",
            aggfunc="sum", fill_value=0)
        b = fic[fic.snapshot_date == NEW].pivot_table(
            index="industry_bucket", columns="label", values="share_in_industry",
            aggfunc="sum", fill_value=0)
        delta = (b.reindex_like(a.reindex(columns=a.columns.union(b.columns)))
                 .fillna(0) - a.reindex(columns=a.columns.union(b.columns)).fillna(0))
        delta = delta.loc[:, delta.abs().max().nlargest(30).index]
        st.subheader("產業內部組成的變化（百分點）")
        fig = px.imshow(delta * 100, color_continuous_scale="RdBu",
                        color_continuous_midpoint=0, aspect="auto",
                        labels=dict(color="百分點"))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        snap = st.radio("Sankey 快照", [OLD, NEW], horizontal=True)
        s = fic[fic.snapshot_date == snap].nlargest(60, "n")
        nodes = list(dict.fromkeys(
            [f"◀ {x}" for x in s.industry_bucket] + [f"{x} ▶" for x in s.label.fillna("未命名")]))
        nidx = {n: i for i, n in enumerate(nodes)}
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, pad=12, thickness=14),
            link=dict(source=[nidx[f"◀ {x}"] for x in s.industry_bucket],
                      target=[nidx[f"{x} ▶"] for x in s.label.fillna("未命名")],
                      value=s.n.tolist())))
        fig.update_layout(height=800, font_size=11)
        st.plotly_chart(fig, use_container_width=True)

# ── 公司招募位移 ────────────────────────────────────────────────────
with tabs[4]:
    st.caption(
        "同一家公司在兩期各自招募 ≥3 個職缺時，把它兩期的職缺向量各取平均當「招募重心」，"
        "算兩個重心的 cosine。位移分數高＝這家公司想找的人明顯換了種類。")
    mj = st.slider("兩期各自至少幾個職缺", 3, 20, 3)
    cs = df("/company-shift", limit=300, min_jobs=mj)
    if cs.empty:
        st.warning("沒有符合條件的公司。")
    else:
        st.metric("符合條件的公司數", len(cs))
        fig = px.histogram(cs, x="shift_score", nbins=40,
                           labels={"shift_score": "位移分數 (1 - cosine)"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            cs[["company_name", "industry_bucket", "n_old", "n_new",
                "shift_score", "from_label", "to_label"]],
            use_container_width=True, hide_index=True, height=520,
            column_config={"shift_score": st.column_config.ProgressColumn(
                "位移分數", min_value=0.0, max_value=float(cs.shift_score.max()),
                format="%.3f")})

# ── 相似職缺 ────────────────────────────────────────────────────────
with tabs[5]:
    st.caption("直接用 bge-m3 向量做最近鄰。把目標快照切到基準期，就是在問"
               "「這個職缺兩個月前對應到什麼」。")
    jid = st.text_input("job_id", placeholder="例如 85186395")
    target = st.radio("在哪一期找", [NEW, OLD], horizontal=True)
    if jid.strip():
        try:
            res = get(f"/similar/{jid.strip()}", snapshot=target, k=15)
            st.write(f"來源快照：`{res['source']['snapshot_date']}` → "
                     f"目標快照：`{res['target_snapshot']}`")
            st.dataframe(pd.DataFrame(res["results"]), use_container_width=True,
                         hide_index=True)
        except Exception as exc:
            st.error(f"查詢失敗：{exc}")
