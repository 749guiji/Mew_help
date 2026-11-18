const materialsEl = document.getElementById("materials");
const previewEl = document.getElementById("preview");
const entryForm = document.getElementById("entry-form");
const entryResult = document.getElementById("entry-result");
const vectorizeBtn = document.getElementById("vectorize-btn");
const vectorizeResult = document.getElementById("vectorize-result");
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchResult = document.getElementById("search-result");

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
  previewEl.innerHTML = `<h3>切块预览：${name}（${chunks.length} 块）</h3>`;
  for (const c of chunks) {
    const div = document.createElement("div");
    div.className = "kb-chunk";
    div.innerHTML = `<div class="kb-chunk-head">${esc(c.category || "未分类")} — ${esc(c.section_path || "")}</div>` +
      `<div class="kb-chunk-q">Q: ${esc(c.questions)}</div>` +
      `<div class="kb-chunk-a">A: ${esc(c.answer)}</div>`;
    previewEl.appendChild(div);
  }
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
  } else {
    entryResult.textContent = "录入失败：" + res.status + " " + (await res.text());
  }
});

vectorizeBtn.addEventListener("click", async () => {
  vectorizeBtn.disabled = true;
  vectorizeResult.textContent = "向量化中…";
  const res = await fetch("/api/kb/vectorize", { method: "POST" });
  if (res.ok) {
    const { vectorized } = await res.json();
    vectorizeResult.textContent = `已补 ${vectorized} 个 pending chunk`;
  } else {
    vectorizeResult.textContent = "向量化失败：" + res.status;
  }
  vectorizeBtn.disabled = false;
});

async function doSearch() {
  const query = searchInput.value.trim();
  if (!query) return;
  searchResult.textContent = "检索中…";
  const res = await fetch("/api/kb/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
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

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]
  ));
}

loadMaterials();
