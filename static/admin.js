const cardsEl = document.getElementById("cards");

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]
  ));
}

async function loadCards() {
  const res = await fetch("/api/admin/overview");
  const { cards } = await res.json();
  cardsEl.innerHTML = "";
  for (const card of cards) {
    const div = document.createElement("div");
    div.className = "kb-chunk";
    if (card.error) {
      div.innerHTML = `<div class="kb-chunk-head">${esc(card.title)}</div>` +
        `<div class="kb-chunk-a">读取失败：${esc(card.error)}</div>`;
    } else {
      const m = card.metrics || {};
      const parts = Object.entries(m).map(([k, v]) => `${k} ${v}`).join(" · ");
      div.innerHTML = `<div class="kb-chunk-head">${esc(card.title)}</div>` +
        `<div class="kb-chunk-a">${parts || "-"}</div>`;
    }
    cardsEl.appendChild(div);
  }
}

// 快捷操作:跑飞轮 / 跑评估(触发后台 job)
async function runJob(name, btn, label) {
  btn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${name}/start`, { method: "POST" });
    btn.textContent = res.ok ? "已触发 ✓" : "触发失败";
  } catch (err) {
    btn.textContent = "触发失败";
  }
  setTimeout(() => { btn.textContent = label; btn.disabled = false; }, 2500);
}

document.getElementById("run-flywheel").addEventListener("click", function () {
  runJob("flywheel", this, "🔄 跑飞轮");
});
document.getElementById("run-eval").addEventListener("click", function () {
  runJob("eval-run", this, "📊 跑评估");
});
document.getElementById("run-classify").addEventListener("click", function () {
  runJob("classify-topics", this, "🏷 跑主题归类");
});

loadCards();
