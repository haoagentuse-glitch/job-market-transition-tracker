/* 靜態版儀表板。資料由 jobshift/export_static.py 預先算好，無伺服器。 */
"use strict";

const PLOT_FONT = {
  family: '"Source Sans Pro", -apple-system, "Noto Sans TC", "PingFang TC", sans-serif',
  color: "#31333f",
  size: 12,
};
const LAYOUT = {
  font: PLOT_FONT,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 0, r: 0, t: 10, b: 40 },
  xaxis: { gridcolor: "rgba(49,51,63,0.1)", zerolinecolor: "rgba(49,51,63,0.2)" },
  yaxis: { gridcolor: "rgba(49,51,63,0.1)", automargin: true },
  hoverlabel: { font: PLOT_FONT },
};
const CONFIG = { displayModeBar: false, responsive: true };
const RED = "#ff4b4b";

const num = (v, d = 0) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : Number(v).toLocaleString("zh-TW", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : (v * 100).toFixed(d) + "%";
const el = (id) => document.getElementById(id);

let CORE, JOBS, DIM;

/* ── 表格 ─────────────────────────────────────────────────── */
function table(target, cols, rows) {
  const t = typeof target === "string" ? el(target) : target;
  const head = cols.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows
    .map((r) => {
      const tds = cols
        .map((c) => {
          const raw = r[c.key];
          const v = c.fmt ? c.fmt(raw, r) : raw ?? "—";
          const cls = [c.num ? "num" : "", c.wide ? "wide" : ""].filter(Boolean).join(" ");
          if (c.bar) {
            const w = c.bar(raw, r);
            return `<td class="bar-cell ${cls}"><div class="fill" style="width:${w}%"></div><span>${v}</span></td>`;
          }
          return `<td${cls ? ` class="${cls}"` : ""}>${v}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");
  t.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
}

const linkTitle = (v, r) =>
  r.job_url ? `<a href="${r.job_url}" target="_blank" rel="noopener">${v ?? "—"}</a>` : v ?? "—";

/* ── 分頁 ─────────────────────────────────────────────────── */
function initTabs() {
  const panels = [...document.querySelectorAll(".panel")];
  const bar = el("tabs");
  panels.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === 0 ? " active" : "");
    b.textContent = p.dataset.tab;
    b.onclick = () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      panels.forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      p.classList.add("active");
      window.dispatchEvent(new Event("resize"));
    };
    bar.appendChild(b);
  });
  panels[0].classList.add("active");
}

/* ── 市場全景 ─────────────────────────────────────────────── */
function renderClusters() {
  const cl = [...CORE.clusters].sort((a, b) => b.n_to - a.n_to);
  el("m-clusters").textContent = cl.length;
  el("m-top1").textContent = pct(cl[0].share_to, 1);
  el("m-top10").textContent = pct(
    cl.slice(0, 10).reduce((s, c) => s + c.share_to, 0), 1);

  const top = cl.slice(0, 25).reverse();
  Plotly.newPlot("c-clusters", [{
    type: "bar", orientation: "h",
    x: top.map((c) => c.n_to), y: top.map((c) => c.label),
    marker: { color: RED, opacity: 0.85 },
    customdata: top.map((c) => c.top_titles),
    hovertemplate: "<b>%{y}</b><br>職缺數 %{x:,}<br>%{customdata}<extra></extra>",
  }], { ...LAYOUT, height: 680, xaxis: { ...LAYOUT.xaxis, title: "職缺數" } }, CONFIG);

  table("t-clusters", [
    { key: "cluster_id", label: "群集", num: true },
    { key: "label", label: "代表職稱" },
    { key: "top_titles", label: "鄰近職稱", wide: true },
    { key: "top_terms", label: "鑑別詞" },
    { key: "n_to", label: "職缺數", num: true, fmt: (v) => num(v) },
    { key: "share_to", label: "佔比", num: true, fmt: (v) => pct(v) },
  ], cl);

  const sel = el("sel-cluster");
  sel.innerHTML = cl
    .map((c) => `<option value="${c.cluster_id}">${c.cluster_id}　${c.label}</option>`)
    .join("");
  sel.onchange = () => renderClusterJobs(sel.value);
  renderClusterJobs(cl[0].cluster_id);
}

function renderClusterJobs(cid) {
  table("t-cluster-jobs", [
    { key: "job_title", label: "職稱", wide: true, fmt: linkTitle },
    { key: "company_name", label: "公司" },
    { key: "bucket_label", label: "職務類別" },
    { key: "industry_class", label: "行業大類" },
    { key: "city_name", label: "地點" },
    { key: "salary_desc", label: "薪資" },
    { key: "sim", label: "相似度", num: true, fmt: (v) => (v == null ? "—" : v.toFixed(3)) },
  ], JOBS.by_cluster[String(cid)] || []);
}

/* ── 數位職缺 ─────────────────────────────────────────────── */
const CONCEPT_COLS = [
  { key: "concept", label: "技能概念" },
  { key: "n", label: "職缺數", num: true, fmt: (v) => num(v) },
  { key: "share", label: "佔全體", num: true, fmt: (v) => pct(v) },
  { key: "median_salary", label: "月薪中位", num: true, fmt: (v) => num(v) },
  { key: "salary_premium", label: "對全體倍數", num: true,
    fmt: (v) => (v == null ? "—" : v.toFixed(2) + "×") },
  { key: "median_desc_len", label: "描述字數中位", num: true, fmt: (v) => num(v) },
];

function renderDigital() {
  const cm = CORE.concept_meta;
  const nDigital = Number(cm.n_digital_jobs || 0);
  const nJobs = Number(cm.n_jobs || 1);
  el("m-digital").textContent = num(nDigital);
  el("m-digital-share").textContent = pct(nDigital / nJobs);
  table("t-concepts", CONCEPT_COLS, CORE.concepts);

  const sel = el("sel-concept-cross");
  sel.innerHTML = CORE.concepts
    .map((c) => `<option value="${c.concept}">${c.concept}</option>`)
    .join("");
  sel.onchange = () => renderConceptByCategory(sel.value);
  renderCrossHeatmap();
  renderConceptByCategory(CORE.concepts[0].concept);
}

function renderCrossHeatmap() {
  const rows = CORE.cross[DIM] || [];
  el("h-cross").textContent = `技能概念在${DIM}中的分布`;
  const cats = [...new Set(rows.map((r) => r.category))].sort();
  const cons = CORE.concepts.map((c) => c.concept);
  const lookup = new Map(rows.map((r) => [r.concept + "|" + r.category, r.share_in_category]));
  const z = cats.map((cat) => cons.map((c) => (lookup.get(c + "|" + cat) || 0) * 100));

  Plotly.newPlot("c-heatmap", [{
    type: "heatmap", x: cons, y: cats, z,
    colorscale: "Blues", colorbar: { title: "%", thickness: 12 },
    hovertemplate: "%{y}<br>%{x}<br>%{z:.2f}%<extra></extra>",
  }], { ...LAYOUT, height: 520, margin: { l: 0, r: 0, t: 10, b: 90 },
        xaxis: { ...LAYOUT.xaxis, tickangle: -30, automargin: true } }, CONFIG);
}

function renderConceptByCategory(concept) {
  const rows = (CORE.cross[DIM] || [])
    .filter((r) => r.concept === concept)
    .sort((a, b) => b.n - a.n)
    .slice(0, 15)
    .reverse();
  Plotly.newPlot("c-concept-cat", [{
    type: "bar", orientation: "h",
    x: rows.map((r) => r.n), y: rows.map((r) => r.category),
    marker: { color: RED, opacity: 0.85 },
    hovertemplate: "%{y}<br>職缺數 %{x:,}<extra></extra>",
  }], { ...LAYOUT, height: 420, xaxis: { ...LAYOUT.xaxis, title: "職缺數" } }, CONFIG);
}

/* ── AI 技能 ──────────────────────────────────────────────── */
function renderAI() {
  const ai = CORE.concepts.filter((c) => c.is_ai);
  const cm = CORE.concept_meta;
  const nAI = Number(cm.n_ai_jobs || 0);
  const nJobs = Number(cm.n_jobs || 1);
  el("m-ai").textContent = num(nAI);
  el("m-ai-share").textContent = pct(nAI / nJobs);
  const prem = ai.reduce((s, c) => s + (c.salary_premium || 0), 0) / (ai.length || 1);
  el("m-ai-prem").textContent = prem ? prem.toFixed(2) + "×" : "—";

  table("t-ai", CONCEPT_COLS, ai);

  const names = new Set(ai.map((c) => c.concept));
  const rows = (CORE.cross[DIM] || [])
    .filter((r) => names.has(r.concept))
    .sort((a, b) => b.n - a.n)
    .slice(0, 20);
  el("h-ai-dist").textContent = `AI 技能職缺的${DIM}分布`;
  const cats = [...new Set(rows.map((r) => r.category))].reverse();
  const traces = [...names].map((c, i) => ({
    type: "bar", orientation: "h", name: c,
    y: cats,
    x: cats.map((cat) => {
      const hit = rows.find((r) => r.concept === c && r.category === cat);
      return hit ? hit.n : 0;
    }),
    marker: { color: ["#ff4b4b", "#4b7bff", "#00b894"][i % 3] },
  }));
  Plotly.newPlot("c-ai-dist", traces, {
    ...LAYOUT, height: 520, barmode: "stack",
    legend: { orientation: "h", y: -0.14, font: PLOT_FONT },
    xaxis: { ...LAYOUT.xaxis, title: "職缺數" },
  }, CONFIG);

  const sel = el("sel-ai-concept");
  sel.innerHTML = ai.map((c) => `<option value="${c.concept}">${c.concept}</option>`).join("");
  const pills = [...document.querySelectorAll("#ai-view .pill")];
  const draw = () => {
    const mode = pills.find((p) => p.classList.contains("active")).dataset.v;
    el("cand-note").style.display = mode === "cand" ? "" : "none";
    if (mode === "hit") {
      table("t-ai-jobs", [
        { key: "job_title", label: "職稱", wide: true, fmt: linkTitle },
        { key: "company_name", label: "公司" },
        { key: "bucket_label", label: "職務類別" },
        { key: "industry_class", label: "行業大類" },
        { key: "city_name", label: "地點" },
        { key: "salary_month_min", label: "月薪下限", num: true, fmt: (v) => num(v) },
        { key: "anchor_similarity", label: "錨點相似度", num: true,
          fmt: (v) => (v == null ? "—" : v.toFixed(3)) },
      ], JOBS.by_concept[sel.value] || []);
    } else {
      table("t-ai-jobs", [
        { key: "job_title", label: "職稱", wide: true, fmt: linkTitle },
        { key: "company_name", label: "公司" },
        { key: "bucket_label", label: "職務類別" },
        { key: "anchor_similarity", label: "錨點相似度", num: true,
          fmt: (v) => (v == null ? "—" : v.toFixed(3)) },
      ], CORE.concept_candidates.filter((r) => r.concept === sel.value));
    }
  };
  sel.onchange = draw;
  pills.forEach((p) => (p.onclick = () => {
    pills.forEach((x) => x.classList.remove("active"));
    p.classList.add("active");
    draw();
  }));
  draw();
}

/* ── 薪資 ─────────────────────────────────────────────────── */
function renderSalary() {
  const tl = CORE.timeline[0] || {};
  el("m-sal").textContent = num(tl.median_salary);
  el("m-sal-cov").textContent = pct(tl.salary_coverage, 0);

  const s = [...CORE.concept_salary].sort((a, b) => a.p50 - b.p50);
  Plotly.newPlot("c-salary", [{
    type: "bar", orientation: "h",
    x: s.map((r) => r.p50), y: s.map((r) => r.concept),
    marker: { color: RED, opacity: 0.85 },
    error_x: {
      type: "data", symmetric: false,
      array: s.map((r) => r.p75 - r.p50),
      arrayminus: s.map((r) => r.p50 - r.p25),
      color: "rgba(49,51,63,0.45)", thickness: 1.2, width: 5,
    },
    hovertemplate: "%{y}<br>中位 %{x:,}<extra></extra>",
  }], {
    ...LAYOUT, height: 460,
    xaxis: { ...LAYOUT.xaxis, title: "月薪中位數" },
    shapes: tl.median_salary ? [{
      type: "line", x0: tl.median_salary, x1: tl.median_salary,
      y0: -0.5, y1: s.length - 0.5,
      line: { dash: "dot", width: 1, color: "rgba(49,51,63,0.5)" },
    }] : [],
  }, CONFIG);

  table("t-salary", [
    { key: "concept", label: "概念" },
    { key: "n", label: "樣本數", num: true, fmt: (v) => num(v) },
    { key: "p25", label: "P25", num: true, fmt: (v) => num(v) },
    { key: "p50", label: "中位", num: true, fmt: (v) => num(v) },
    { key: "p75", label: "P75", num: true, fmt: (v) => num(v) },
  ], CORE.concept_salary);
}

/* ── 方法與限制 ───────────────────────────────────────────── */
function renderMethod() {
  table("t-sep", [
    { key: "concept", label: "概念" },
    { key: "anchor_sim_median", label: "正例相似度中位", num: true,
      fmt: (v) => (v == null ? "—" : v.toFixed(3)) },
    { key: "anchor_sim_p99_all", label: "全體相似度 P99", num: true,
      fmt: (v) => (v == null ? "—" : v.toFixed(3)) },
    { key: "separation_sigma", label: "分離度 (σ)", num: true,
      fmt: (v) => (v == null ? "—" : v.toFixed(2)) },
  ], CORE.concepts);

  const snap = CORE.meta.reference_snapshot;
  const noise = pct(Number(CORE.meta.noise_ratio || 0), 1);
  el("limits").innerHTML = [
    `單一時間點（${snap}），描述當期結構，不構成趨勢。`,
    "來源為單一商業求職平台，雇主組成有其特性，不代表整體勞動市場。",
    "採集時每個職務類別設有 4,500 筆上限，熱門類別的絕對數為截斷值，佔比較絕對數可靠。",
    `群集邊界由語意模型與分群參數決定，非官方職業分類；噪點比例 ${noise}。`,
    "技能偵測依賴雇主是否於描述中寫出該技能，未寫出者無法偵測。",
    "薪資統計僅涵蓋可換算為月薪者，面議與論件計酬不計入。",
  ].map((t) => `<li>${t}</li>`).join("");
}

/* ── 啟動 ─────────────────────────────────────────────────── */
function renderDimDependent() {
  renderCrossHeatmap();
  renderConceptByCategory(el("sel-concept-cross").value);
  renderAI();
}

async function main() {
  const [core, jobs] = await Promise.all([
    fetch("data/core.json").then((r) => r.json()),
    fetch("data/jobs.json").then((r) => r.json()),
  ]);
  CORE = core;
  JOBS = jobs;
  DIM = CORE.dimensions[1] ? "行業大類" : CORE.dimensions[0].key;

  const m = CORE.meta;
  el("s-snap").textContent = m.reference_snapshot;
  el("s-n").textContent = num(Number(CORE.concept_meta.n_jobs));
  el("s-method").textContent = m.cluster_method;
  el("s-clusters").textContent = m.n_reference_clusters;
  el("s-noise").textContent = pct(Number(m.noise_ratio || 0), 1);
  el("subtitle").textContent =
    `${m.reference_snapshot} · ${num(Number(CORE.concept_meta.n_jobs))} 筆 · ` +
    "以職缺描述的語意向量建立分類，不採用平台既有標籤";

  const radios = el("dim-radios");
  radios.innerHTML = ["行業大類", "職務類別"]
    .map((d) => `<label class="radio"><input type="radio" name="dim" value="${d}"${
      d === DIM ? " checked" : ""}>${d}</label>`)
    .join("");
  radios.onchange = (e) => {
    DIM = e.target.value;
    renderDimDependent();
  };

  initTabs();
  el("loading").style.display = "none";
  renderClusters();
  renderDigital();
  renderAI();
  renderSalary();
  renderMethod();
}

main().catch((e) => {
  el("loading").textContent = "資料載入失敗：" + e.message;
});
