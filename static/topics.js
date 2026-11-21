const distEl = document.getElementById("dist");
const qlistEl = document.getElementById("qlist");

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function loadDist() {
  const res = await fetch("/api/topics/distribution");
  const { items } = await res.json();
  distEl.innerHTML = "";
  if (!items.length) {
    distEl.textContent = "还没有归类数据。跑一次「classify-topics」后这里就有分布了。";
    return;
  }
  const max = Math.max(...items.map((x) => x.count));
  for (const it of items) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:10px;margin:6px 0";
    const label = document.createElement("button");
    label.type = "button";
    label.style.cssText = "width:88px;text-align:right;border:none;background:none;color:var(--brand);cursor:pointer;font-size:13px";
    label.textContent = it.label;
    label.addEventListener("click", () => loadQuestions(it.label));
    const bar = document.createElement("div");
    bar.style.cssText = `height:16px;width:${Math.max(8, (it.count / max) * 320)}px;background:var(--brand);border-radius:4px`;
    const cnt = document.createElement("span");
    cnt.style.cssText = "font-size:13px;color:var(--muted)";
    cnt.textContent = it.count;
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(cnt);
    distEl.appendChild(row);
  }
}

async function loadQuestions(label) {
  qlistEl.textContent = "加载中…";
  const res = await fetch("/api/topics/questions?label=" + encodeURIComponent(label));
  const { items } = await res.json();
  qlistEl.innerHTML = `<div class="kb-chunk-head">「${esc(label)}」的问题(${items.length})</div>`;
  if (!items.length) {
    qlistEl.innerHTML += `<div class="kb-chunk-a">（无）</div>`;
    return;
  }
  for (const it of items) {
    const d = document.createElement("div");
    d.className = "kb-chunk-a";
    d.textContent = it.text + "  →  [" + (it.labels || []).join(", ") + "]";
    qlistEl.appendChild(d);
  }
}

loadDist();
