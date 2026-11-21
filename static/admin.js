const cardsEl = document.getElementById("cards");
const navEl = document.getElementById("admin-nav");

const LINKS = [
  ["知识库", "/kb"],
];

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]
  ));
}

function renderNav() {
  navEl.innerHTML = "";
  for (const [label, href] of LINKS) {
    const a = document.createElement("a");
    a.href = href;
    a.textContent = label;
    navEl.appendChild(a);
  }
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

renderNav();
loadCards();
