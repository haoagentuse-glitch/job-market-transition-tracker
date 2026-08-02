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
    r = httpx.get(f"{API}{path}", params={k: v for k, v in params.items() if v is not None},
                  timeout=60)
    r.raise_for_status()
    return r.json()


def df(path: str, **params) -> pd.DataFrame:
    return pd.DataFrame(get(path, **params))


try:
    META = get("/meta")
except Exception as exc:
    st.error(f"連不上 API（{API}）或分析尚未執行：{exc}")
    st.info("跑一次 `docker compose run --rm pipeline` 之後再回來。")
    st.stop()

SNAPS: list[str] = META.get("snapshots", [])

# ── 側邊欄：時間軸與比較區間 ────────────────────────────────────────
with st.sidebar:
    st.header("比較區間")
    if len(SNAPS) < 2:
        st.warning("目前只有一份快照，跨期分析要等下一次 pipeline 跑完。")
        FRM = TO = SNAPS[0] if SNAPS else "?"
    else:
        TO = st.selectbox("後期", SNAPS[::-1], index=0)
        earlier = [s for s in SNAPS if s < TO] or [SNAPS[0]]
        FRM = st.selectbox("前期", earlier[::-1], index=len(earlier) - 1)
    st.divider()
    st.caption("**時間軸**")
    for s in SNAPS:
        st.caption(f"{'🔹' if s in (FRM, TO) else '▫️'} {s}"
                   + ("　←參考座標系" if s == META.get("reference_snapshot") else ""))
    st.divider()
    st.header("分類維度")
    DIMS = get("/dimensions")
    DIM = st.radio("依什麼切分", [d["key"] for d in DIMS], index=1,
                   help="兩套分類量的是不同東西，切換看看結論穩不穩。")
    _d = next(d for d in DIMS if d["key"] == DIM)
    st.caption(f"來源：{_d['source']}\n\n⚠️ {_d['caveat']}")

    st.divider()
    st.caption(
        f"分群 `{META.get('cluster_method')}`\n\n"
        f"參考群集 {META.get('n_reference_clusters')} 個，"
        f"含新生共 {META.get('n_clusters_total')} 個\n\n"
        f"噪點率 {float(META.get('noise_ratio', 0)):.1%}"
    )

st.title("職缺市場的結構遷移")
st.caption(f"比較區間：**{FRM}** → **{TO}**　|　語意座標系建在 "
           f"**{META.get('reference_snapshot')}**，之後每份快照都指派進同一組群集，"
           "所以編號跨期一致。")

if len(SNAPS) == 2:
    st.info(
        "**目前只有兩個觀測點，這是對照不是趨勢。** 任何「持續上升／下降」的說法都還不成立，"
        "要第三份快照之後才談得上趨勢。另外各期都套用每產業 4,500 筆的相同上限，"
        "熱門產業的絕對數是被截斷的 —— 看佔比比看總數可靠。",
        icon="⚠️",
    )

tabs = st.tabs(["時間軸", "語意群集", "遷移矩陣", "產業組成", "公司招募位移", "相似職缺"])

# ── 時間軸 ─────────────────────────────────────────────────────────
with tabs[0]:
    tl = df("/timeline")
    if not tl.empty:
        cols = st.columns(min(len(tl), 4))
        for i, row in tl.iterrows():
            with cols[i % len(cols)]:
                st.metric(f"{row.snapshot_date}", f"{int(row.n):,} 筆")
                st.caption(
                    "月薪中位數 "
                    + (f"{int(row.median_salary):,}" if pd.notna(row.median_salary) else "—")
                    + f"（涵蓋 {row.salary_coverage:.0%}）\n\n"
                    f"{int(row.n_companies):,} 家公司"
                )
        st.caption("月薪中位數只涵蓋能換算成月薪的職缺；面議與論件計酬沒有可比數字，不計入。")

    st.subheader(f"{DIM}佔比隨時間")
    ind = df("/industry-series", dimension=DIM)
    if not ind.empty:
        fig = px.line(ind, x="snapshot_date", y="share", color="category",
                      markers=True, labels={"share": "佔比", "snapshot_date": "",
                                            "category": DIM})
        # 快照是離散事件，不是連續時間軸；不鎖成類別軸，Plotly 會把日期字串
        # 當時間戳解析並在兩點之間生出無意義的毫秒刻度。
        fig.update_xaxes(type="category")
        fig.update_layout(height=520, yaxis_tickformat=".1%", legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

    surv = df("/survival", from_snapshot=FRM, to_snapshot=TO, dimension=DIM)
    if not surv.empty:
        st.subheader(f"{DIM}汰換率（{FRM} → {TO}）")
        st.caption("汰換率＝前期職缺到後期已經不在的比例；更新率＝後期裡屬於新出現的比例。")
        s = surv.sort_values("churn_rate")
        fig = px.bar(s, x="churn_rate", y="category", orientation="h",
                     text=s.churn_rate.map("{:.0%}".format),
                     labels={"churn_rate": "汰換率", "category": DIM})
        fig.update_layout(height=640, yaxis_title=None, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(surv, use_container_width=True, hide_index=True)

# ── 語意群集 ───────────────────────────────────────────────────────
with tabs[1]:
    cl = df("/clusters", from_snapshot=FRM, to_snapshot=TO)
    if cl.empty:
        st.warning("clusters 是空的，請先跑 pipeline。")
    else:
        pick = st.multiselect("狀態", sorted(cl.status.unique()),
                              default=sorted(cl.status.unique()))
        view = cl[cl.status.isin(pick)]

        st.subheader("群集地圖")
        st.caption("點大小＝後期職缺數；顏色＝佔比變化；對角線以上代表這個語意群在變大。")
        fig = px.scatter(
            view, x="share_from", y="share_to", size="n_to", color="share_delta_pp",
            hover_name="label", hover_data=["top_titles", "n_from", "n_to", "status"],
            color_continuous_scale="RdBu", color_continuous_midpoint=0, size_max=45,
            labels={"share_from": f"{FRM} 佔比", "share_to": f"{TO} 佔比"})
        lim = float(max(view.share_from.max(), view.share_to.max(), 1e-6)) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line={"dash": "dot", "width": 1})
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)

        if len(SNAPS) > 2:
            st.subheader("變化最大的群集，完整時間序列")
            ser = df("/clusters/series", top=12)
            if not ser.empty:
                fig = px.line(ser, x="snapshot_date", y="share", color="label", markers=True,
                              labels={"share": "佔比", "snapshot_date": ""})
                fig.update_xaxes(type="category")
                fig.update_layout(height=460, yaxis_tickformat=".2%", legend_title=None)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("消長排行")
        show = ["cluster_id", "status", "label", "top_titles", "top_terms", "first_seen",
                "n_from", "n_to", "delta", "share_delta_pp", "growth_pct"]
        st.dataframe(view[[c for c in show if c in view.columns]]
                     .sort_values("share_delta_pp", ascending=False),
                     use_container_width=True, hide_index=True, height=440)

        labels = view.set_index("cluster_id").label.to_dict()
        cid = st.selectbox("看某個群集的實際職缺", view.cluster_id.tolist(),
                           format_func=lambda i: f"{i} — {labels.get(i, '')}")
        if cid is not None:
            st.dataframe(df(f"/clusters/{cid}/jobs", snapshot=TO, limit=50),
                         use_container_width=True, hide_index=True)

# ── 遷移矩陣 ───────────────────────────────────────────────────────
with tabs[2]:
    st.caption(
        "職缺在兩期之間的流量。**存活職缺直接繼承前期的群集編號**（描述沒變、向量也沒變，"
        "重新指派只會引入抖動），所以真正有訊號的是「哪些群集在流失」與「新職缺流進哪裡」。")

    stab = get("/stability", from_snapshot=FRM, to_snapshot=TO)
    if stab:
        c1, c2 = st.columns([1, 3])
        c1.metric("指派雜訊底線", f"{stab.get('noise_floor', 0):.1%}")
        c2.caption(
            f"把 {int(stab.get('n_comparable', 0)):,} 筆存活職缺重新指派一次，"
            f"與繼承結果的一致率是 {stab.get('agreement', 0):.1%}。"
            "這些職缺的描述兩期完全沒變，所以不一致的部分純粹是方法的量測誤差 —— "
            "下面任何變化要大過這個數字才值得解讀。")

    hide_self = st.checkbox("隱藏存活的自環，只看下架與新進", value=True)
    fl = df("/cluster-flow", from_snapshot=FRM, to_snapshot=TO,
            min_n=5, include_self=not hide_self)
    if fl.empty:
        st.warning("這個區間沒有符合條件的流向。")
    else:
        fl["from_label"] = [
            "【新進】" if f == -998 else (lb or f"未歸類/群{int(f)}")
            for f, lb in zip(fl.from_cluster, fl.from_label, strict=False)]
        fl["to_label"] = [
            "【下架】" if t == -999 else (lb or f"未歸類/群{int(t)}")
            for t, lb in zip(fl.to_cluster, fl.to_label, strict=False)]
        top = fl.nlargest(40, "n")
        srcs = [f"◀ {s}" for s in top.from_label]
        dsts = [f"{d} ▶" for d in top.to_label]
        nodes = list(dict.fromkeys(srcs + dsts))
        nidx = {n: i for i, n in enumerate(nodes)}
        fig = go.Figure(go.Sankey(
            node={"label": nodes, "pad": 12, "thickness": 14},
            link={"source": [nidx[s] for s in srcs], "target": [nidx[d] for d in dsts],
                  "value": top.n.tolist(), "label": top.kind.tolist()}))
        fig.update_layout(height=760, font_size=11)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fl.sort_values("n", ascending=False),
                     use_container_width=True, hide_index=True, height=320)

# ── 產業組成 ───────────────────────────────────────────────────────
with tabs[3]:
    st.caption(f"每個{DIM}的職缺分散到哪些語意群集，以及兩期之間這個分布怎麼重組。")
    comp = df("/industry-composition", min_n=20, dimension=DIM)
    if comp.empty:
        st.warning("flow_industry_cluster 是空的。")
    else:
        comp["label"] = comp.label.fillna("未命名")
        a = comp[comp.snapshot_date == FRM].pivot_table(
            index="category", columns="label", values="share_in_category",
            aggfunc="sum", fill_value=0)
        b = comp[comp.snapshot_date == TO].pivot_table(
            index="category", columns="label", values="share_in_category",
            aggfunc="sum", fill_value=0)
        idx = a.index.union(b.index)
        col = a.columns.union(b.columns)
        delta = (b.reindex(index=idx, columns=col).fillna(0)
                 - a.reindex(index=idx, columns=col).fillna(0))
        keep = delta.abs().max().nlargest(30).index
        st.subheader(f"{DIM}內部組成的變化（百分點）")
        fig = px.imshow(delta[keep] * 100, color_continuous_scale="RdBu",
                        color_continuous_midpoint=0, aspect="auto",
                        labels={"color": "百分點"})
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        snap = st.radio("Sankey 快照", SNAPS, horizontal=True, index=len(SNAPS) - 1)
        s = comp[comp.snapshot_date == snap].nlargest(60, "n")
        nodes = list(dict.fromkeys([f"◀ {x}" for x in s.category]
                                   + [f"{x} ▶" for x in s.label]))
        nidx = {n: i for i, n in enumerate(nodes)}
        fig = go.Figure(go.Sankey(
            node={"label": nodes, "pad": 12, "thickness": 14},
            link={"source": [nidx[f"◀ {x}"] for x in s.category],
                  "target": [nidx[f"{x} ▶"] for x in s.label],
                  "value": s.n.tolist()}))
        fig.update_layout(height=800, font_size=11)
        st.plotly_chart(fig, use_container_width=True)

# ── 公司招募位移 ───────────────────────────────────────────────────
with tabs[4]:
    st.caption(
        "同一家公司在兩期各自招募 ≥N 個職缺時，把它兩期的職缺向量各取平均當「招募重心」，"
        "算兩個重心的 cosine。位移分數高＝這家公司想找的人明顯換了種類。")
    mj = st.slider("兩期各自至少幾個職缺", 3, 20, 3)
    cs = df("/company-shift", from_snapshot=FRM, to_snapshot=TO, limit=300, min_jobs=mj)
    if cs.empty:
        st.warning("沒有符合條件的公司。把門檻調低試試。")
    else:
        st.metric("符合條件的公司數", len(cs))
        fig = px.histogram(cs, x="shift_score", nbins=40,
                           labels={"shift_score": "位移分數 (1 − cosine)"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            cs[["company_name", "industry_class", "bucket_label", "n_from", "n_to",
                "shift_score", "from_label", "to_label"]],
            use_container_width=True, hide_index=True, height=520,
            column_config={"shift_score": st.column_config.ProgressColumn(
                "位移分數", min_value=0.0, max_value=float(cs.shift_score.max()),
                format="%.3f")})

# ── 相似職缺 ───────────────────────────────────────────────────────
with tabs[5]:
    st.caption("直接用 bge-m3 向量做最近鄰。把目標快照切到前一期，"
               "就是在問「這個職缺在前一期對應到什麼」。")
    jid = st.text_input("job_id", placeholder="例如 85186395")
    target = st.radio("在哪一期找", SNAPS[::-1], horizontal=True)
    if jid.strip():
        try:
            res = get(f"/similar/{jid.strip()}", snapshot=target, k=15)
            st.write(f"來源快照 `{res['source']['snapshot_date']}` → "
                     f"目標快照 `{res['target_snapshot']}`")
            st.dataframe(pd.DataFrame(res["results"]), use_container_width=True,
                         hide_index=True)
        except Exception as exc:
            st.error(f"查詢失敗：{exc}")
