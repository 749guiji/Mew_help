const overviewEl = document.getElementById("overview");
const materialsEl = document.getElementById("materials");
const previewEl = document.getElementById("preview");
const materialJobLog = document.getElementById("material-job-log");
const entryForm = document.getElementById("entry-form");
const entryResult = document.getElementById("entry-result");
const stagingEl = document.getElementById("staging");
const mineJobLog = document.getElementById("mine-job-log");
const vectorizeBtn = document.getElementById("vectorize-btn");
const resetBtn = document.getElementById("reset-btn");
const vectorizeResult = document.getElementById("vectorize-result");
const vectorizeJobLog = document.getElementById("vectorize-job-log");
const searchStrategy = document.getElementById("search-strategy");
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchResult = document.getElementById("search-result");
const recentEl = document.getElementById("recent");

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]
  ));
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function loadOverview() {
  const res = await fetch("/api/kb/overview");
  const o = await res.json();
  overviewEl.innerHTML = "";
  const items = [
    ["块数", o.chunks],
    ["待向量化", o.pending],
    ["已向量化", o.done],
    ["Milvus", o.milvus ?? "读不到"],
    ["关键条款", o.key_clauses],
    ["双写一致", o.consistent === null ? "读不到" : (o.consistent ? "一致" : "不一致")],
  ];
  for (const [label, val] of items) {
    const div = document.createElement("div");
    div.className = "kb-stat";
    div.innerHTML = `${label}：<b>${esc(val)}</b>`;
    overviewEl.appendChild(div);
  }
}

async function loadMaterials() {
  const res = await fetch("/api/kb/materials");
  const items = await res.json();
  materialsEl.innerHTML = "";
  for (const item of items) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "kb-material";
    btn.textContent = `${item.name}（${item.content_type} · ${item.chars} 字）`;
    btn.addEventListener("click", () => loadPreview(item.name));
    materialsEl.appendChild(btn);
  }
}

async function loadPreview(name) {
  const res = await fetch(`/api/kb/preview?path=${encodeURIComponent(name)}`);
  if (!res.ok) {
    previewEl.textContent = "预览失败：" + res.status;
    return;
  }
  const chunks = await res.json();
  previewEl.innerHTML = `<h3>切块预览：${esc(name)}（${chunks.length} 块）</h3>`;
  for (const c of chunks) {
    const div = document.createElement("div");
    div.className = "kb-chunk";
    div.innerHTML = `<div class="kb-chunk-head">${esc(c.category || "未分类")} — ${esc(c.section_path || "")}</div>` +
      `<div class="kb-chunk-q">Q: ${esc(c.questions)}</div>` +
      `<div class="kb-chunk-a">A: ${esc(c.answer)}</div>`;
    previewEl.appendChild(div);
  }
}

async function runJob(name, logEl) {
  const start = await fetch(`/api/jobs/${name}/start`, { method: "POST" });
  if (start.status === 404) { logEl.hidden = false; logEl.textContent = "未知作业"; return; }
  if (start.status === 409) { logEl.hidden = false; logEl.textContent = "作业已在运行"; return; }
  logEl.hidden = false;
  logEl.textContent = "运行中…";
  let done = false;
  while (!done) {
    await sleep(1000);
    const st = await (await fetch(`/api/jobs/${name}/status`)).json();
    if (!st.running) {
      done = true;
      if (st.error) logEl.textContent = "失败：" + st.error;
    }
  }
  const log = await (await fetch(`/api/jobs/${name}/log`)).text();
  logEl.textContent = log || "（无日志）";
  loadOverview();
  loadRecent();
}

async function runHeavyJob(name, logEl) {
  if (!confirm(`确定要运行 ${name}？该操作较重，可能耗时较长。`)) return;
  await runJob(name, logEl);
}

entryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(entryForm);
  const payload = {
    content_type: fd.get("content_type"),
    category: fd.get("category").trim(),
    questions: fd.get("questions").trim(),
    answer: fd.get("answer").trim(),
  };
  const res = await fetch("/api/kb/chunks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    const { id } = await res.json();
    entryResult.textContent = `已录入 chunk #${id}（pending，待补向量化）`;
    entryForm.reset();
    loadOverview();
    loadRecent();
  } else if (res.status === 409) {
    const body = await res.json();
    entryResult.textContent = `重复：库里已有 chunk #${body.id}，跳过`;
  } else {
    entryResult.textContent = "录入失败：" + res.status;
  }
});

document.getElementById("preview-btn").addEventListener("click", () => runJob("kb-preview", materialJobLog));
document.getElementById("build-btn").addEventListener("click", () => runJob("kb-build", materialJobLog));
document.getElementById("mine-btn").addEventListener("click", () => runHeavyJob("kb-mine", mineJobLog));
vectorizeBtn.addEventListener("click", () => runJob("kb-vectorize", vectorizeJobLog));
resetBtn.addEventListener("click", () => runHeavyJob("kb-reset", vectorizeJobLog));

async function loadStaging() {
  const res = await fetch("/api/kb/staging");
  const s = await res.json();
  stagingEl.innerHTML = "";
  for (const [label, val] of [["抽出", s.extracted], ["保留", s.kept], ["丢弃", s.discarded]]) {
    const div = document.createElement("div");
    div.className = "kb-stat";
    div.innerHTML = `${label}：<b>${esc(val)}</b>`;
    stagingEl.appendChild(div);
  }
}

async function doSearch() {
  const query = searchInput.value.trim();
  if (!query) return;
  const strategy = searchStrategy.value;
  searchResult.textContent = "检索中…";
  const res = await fetch("/api/kb/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, strategy }),
  });
  const chunks = await res.json();
  searchResult.innerHTML = "";
  if (!chunks.length) {
    searchResult.textContent = "未检索到结果";
    return;
  }
  for (const c of chunks) {
    const div = document.createElement("div");
    div.className = "kb-hit";
    div.innerHTML = `<div class="kb-hit-head">#${c.chunk_id} · score ${c.score.toFixed(4)} · ${esc(c.section_path || c.category)}</div>` +
      `<div class="kb-hit-a">${esc(c.answer)}</div>`;
    searchResult.appendChild(div);
  }
}

searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); doSearch(); }
});

async function loadRecent() {
  const res = await fetch("/api/kb/recent?limit=20");
  const rows = await res.json();
  recentEl.innerHTML = "";
  if (!rows.length) {
    recentEl.textContent = "（暂无）";
    return;
  }
  for (const r of rows) {
    const div = document.createElement("div");
    div.className = "kb-chunk";
    div.innerHTML = `<div class="kb-chunk-head">#${r.id} · ${esc(r.content_type || "")} · ${esc(r.vectorize_status)}</div>` +
      `<div class="kb-chunk-q">Q: ${esc(r.questions)}</div>` +
      `<div class="kb-chunk-a">A: ${esc(r.answer)}</div>`;
    recentEl.appendChild(div);
  }
}

loadOverview();
loadMaterials();
loadStaging();
loadRecent();
