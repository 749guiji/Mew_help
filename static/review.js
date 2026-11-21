// 待审队列后台页(ch09):列表 / 详情 / 通过(写回知识库) / 驳回。

const listEl = document.getElementById("review-list");

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
  ));
}

const STATUS_ICON = { "待审": "⏳", "通过": "✅", "驳回": "❌" };

async function load() {
  let res;
  try {
    res = await fetch("/api/review");
  } catch (err) {
    listEl.textContent = "加载失败";
    return;
  }
  if (!res.ok) {
    listEl.textContent = "加载失败";
    return;
  }
  const { items } = await res.json();
  listEl.innerHTML = "";
  if (!items.length) {
    listEl.textContent = "暂无待审项。";
    return;
  }
  for (const it of items) {
    const div = document.createElement("div");
    div.className = "kb-chunk";
    const icon = STATUS_ICON[it.review_status] || "⏳";
    div.innerHTML =
      `<div class="kb-chunk-head">${icon} ${esc(it.normalized_question)}` +
      ` <span style="color:var(--muted)">×${it.occurrence_count}</span></div>` +
      `<div class="kb-chunk-a">示例答案:${esc(it.ai_suggested_answer || "（无）")}</div>` +
      `<div class="kb-chunk-a">` +
      `<button class="fb-btn" data-act="detail">详情</button> ` +
      `<button class="fb-btn" data-act="approve">通过</button> ` +
      `<button class="fb-btn" data-act="reject">驳回</button>` +
      `</div>`;
    div.querySelector('[data-act="detail"]').addEventListener("click", () => showDetail(it.id));
    div.querySelector('[data-act="approve"]').addEventListener("click", () => approve(it.id));
    div.querySelector('[data-act="reject"]').addEventListener("click", () => reject(it.id));
    listEl.appendChild(div);
  }
}

async function showDetail(id) {
  const res = await fetch("/api/review/" + id);
  if (!res.ok) return;
  const d = await res.json();
  let html =
    `<div class="kb-chunk-head">${esc(d.item.normalized_question)}</div>` +
    `<div class="kb-chunk-a">示例答案:${esc(d.item.ai_suggested_answer || "无")}</div>` +
    `<div class="kb-chunk-a">出现 ${d.item.occurrence_count} 次 · 状态 ${esc(d.item.review_status)}</div>` +
    `<hr style="border:none;border-top:1px solid var(--border)">`;
  for (const o of d.originals) {
    html += `<div class="kb-chunk-a"><b>原话:</b> ${esc(o.raw_question)} <span style="color:var(--muted)">(${esc(o.source)})</span></div>`;
    const chunks = o.retrieved_chunks || [];
    if (chunks.length) {
      for (const c of chunks) {
        html += `<div class="kb-chunk-a" style="padding-left:14px">· [${c.score}] ${esc(c.section_path || "")}: ${esc(c.answer || "")}</div>`;
      }
    } else {
      html += `<div class="kb-chunk-a" style="padding-left:14px">· （无召回片段）</div>`;
    }
  }
  // 追加到列表顶部作详情面板
  const panel = document.createElement("div");
  panel.className = "kb-chunk";
  panel.innerHTML = html;
  const close = document.createElement("button");
  close.className = "fb-btn";
  close.textContent = "收起详情";
  close.addEventListener("click", () => panel.remove());
  panel.appendChild(close);
  listEl.prepend(panel);
}

async function approve(id) {
  const answer = prompt("核准答案(将写入知识库):");
  if (answer === null) return;
  if (!answer.trim()) return;
  const res = await fetch(`/api/review/${id}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved_answer: answer.trim() }),
  });
  if (res.ok) load();
}

async function reject(id) {
  if (!confirm("驳回这个缺口?")) return;
  const res = await fetch(`/api/review/${id}/reject`, { method: "POST" });
  if (res.ok) load();
}

load();
