const state = { sessionId: null, streaming: false };
let sseBuffer = "";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const citeModal = document.getElementById("cite-modal");
const citePathEl = document.getElementById("cite-path");
const citeBodyEl = document.getElementById("cite-body");
const citeCloseEl = document.getElementById("cite-close");

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble " + role;
  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.textContent = "🤖";
    bubble.appendChild(avatar);
  }
  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;
  bubble.appendChild(content);
  messagesEl.appendChild(bubble);
  scrollToBottom();
  return content;
}

function setStreaming(on) {
  state.streaming = on;
  sendBtn.disabled = on;
}

const TOOL_LABELS = {
  query_order: "查询订单",
  query_product: "查询商品",
  query_logistics: "查询物流",
  query_faq: "查询常见问题",
  create_ticket: "创建工单",
};

function toolLabel(name) {
  return TOOL_LABELS[name] || name;
}

function renderToolBadge(badgeEl, evt) {
  const label = toolLabel(evt.name);
  if (evt.status === "running") {
    badgeEl.className = "tool-badge visible running";
    badgeEl.textContent = "🔧 正在" + label + "…";
  } else if (evt.status === "done") {
    badgeEl.className = "tool-badge visible done";
    badgeEl.textContent = "🔧 " + label + " ✓";
  } else if (evt.status === "error") {
    badgeEl.className = "tool-badge visible error";
    badgeEl.textContent = "🔧 " + label + " 出错";
  }
}

// 转人工 / 建工单两个独立按钮(后端只下建议,不自动执行)。
function renderHandoff(suggestions, contentEl) {
  if (!suggestions.length) return;
  const wrap = document.createElement("div");
  wrap.className = "handoff";
  for (const s of suggestions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "handoff-btn";
    if (s === "转人工") {
      btn.textContent = "转人工";
      btn.addEventListener("click", () => {
        wrap.innerHTML = "";
        const done = document.createElement("div");
        done.className = "handoff-done";
        done.textContent = "已转接人工客服\n您好，我是客服小猫，请问有什么可以帮您的？";
        wrap.appendChild(done);
      });
    } else if (s === "建工单") {
      btn.textContent = "建工单";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          const res = await fetch("/api/tickets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: state.sessionId,
              description: "用户投诉工单",
              ticket_type: "投诉",
            }),
          });
          if (res.ok) {
            const { ticket_no } = await res.json();
            btn.textContent = "已建工单 " + ticket_no;
          } else {
            btn.textContent = "建工单失败";
          }
        } catch (err) {
          btn.textContent = "建工单失败";
        }
      });
    }
    wrap.appendChild(btn);
  }
  contentEl.appendChild(wrap);
  scrollToBottom();
}

// 把答案文本里的 [n] / ［n］ 转成可点的引用编号。
function linkifyCitations(textSpan, citations) {
  const text = textSpan.textContent;
  if (!text) return;
  const map = {};
  for (const c of citations) map[c.n] = c;
  if (!/[\[［][0-9]+[\]］]/.test(text)) return;

  textSpan.textContent = "";
  const parts = text.split(/([\[［][0-9]+[\]］])/g);
  for (const part of parts) {
    const m = part.match(/^[\[［]([0-9]+)[\]］]$/);
    if (m && map[Number(m[1])]) {
      const n = Number(m[1]);
      const btn = document.createElement("button");
      btn.className = "cite";
      btn.textContent = "[" + n + "]";
      btn.type = "button";
      btn.addEventListener("click", () => showCitation(n, map));
      textSpan.appendChild(btn);
    } else {
      textSpan.appendChild(document.createTextNode(part));
    }
  }
}

function showCitation(n, map) {
  const c = map[n];
  if (!c) return;
  citePathEl.textContent = c.section_path || "（未标注章节）";
  citeBodyEl.textContent = c.answer_snippet || "（无正文）";
  citeModal.hidden = false;
}

function closeCitation() {
  citeModal.hidden = true;
}

// 每段助手回答左下角的 👍👎，点一次点亮所选并锁定。
function addFeedback(contentEl) {
  const fb = document.createElement("div");
  fb.className = "feedback";

  const up = document.createElement("button");
  up.className = "fb-btn";
  up.type = "button";
  up.textContent = "👍";
  up.setAttribute("aria-label", "满意");

  const down = document.createElement("button");
  down.className = "fb-btn";
  down.type = "button";
  down.textContent = "👎";
  down.setAttribute("aria-label", "不满意");

  const label = document.createElement("span");
  label.className = "fb-done";
  label.textContent = "已反馈";
  label.hidden = true;

  let locked = false;
  function lock(chosen) {
    if (locked) return;
    locked = true;
    chosen.classList.add("selected");
    up.disabled = true;
    down.disabled = true;
    label.hidden = false;
  }
  up.addEventListener("click", () => lock(up));
  down.addEventListener("click", () => lock(down));

  fb.appendChild(up);
  fb.appendChild(down);
  fb.appendChild(label);
  contentEl.appendChild(fb);
}

function dispatch(evt, ctx) {
  if (evt.type === "meta") {
    state.sessionId = evt.session_id;
  } else if (evt.type === "token") {
    ctx.textSpan.textContent += evt.content;
    scrollToBottom();
  } else if (evt.type === "tool") {
    renderToolBadge(ctx.badgeEl, evt);
    scrollToBottom();
  } else if (evt.type === "citations") {
    ctx.citations = evt.citations || [];
  } else if (evt.type === "handoff") {
    renderHandoff(evt.suggestions || [], ctx.contentEl);
  } else if (evt.type === "error") {
    ctx.textSpan.textContent += "\n[出错了] " + evt.message;
    scrollToBottom();
  }
}

function handleSSE(chunk, ctx) {
  sseBuffer += chunk;
  const parts = sseBuffer.split("\n\n");
  sseBuffer = parts.pop();
  for (const part of parts) {
    for (const line of part.split("\n")) {
      if (line.startsWith("data: ")) {
        try {
          dispatch(JSON.parse(line.slice(6)), ctx);
        } catch (err) {
          // 忽略无法解析的帧
        }
      }
    }
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || state.streaming) return;
  inputEl.value = "";

  appendBubble("user", text);
  const contentEl = appendBubble("assistant", "");

  const badgeEl = document.createElement("div");
  badgeEl.className = "tool-badge";
  contentEl.appendChild(badgeEl);

  const textSpan = document.createElement("span");
  textSpan.className = "text";
  contentEl.appendChild(textSpan);

  const cursor = document.createElement("span");
  cursor.className = "cursor";
  contentEl.appendChild(cursor);

  const ctx = { textSpan, badgeEl, citations: [], contentEl };

  setStreaming(true);
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: state.sessionId }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      handleSSE(decoder.decode(value, { stream: true }), ctx);
    }
  } catch (err) {
    textSpan.textContent += "\n[出错了] " + err.message;
  } finally {
    cursor.remove();
    setStreaming(false);
    if (ctx.citations.length) linkifyCitations(textSpan, ctx.citations);
    addFeedback(contentEl);
    scrollToBottom();
  }
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.isComposing || e.keyCode === 229) return;
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

citeCloseEl.addEventListener("click", closeCitation);
citeModal.addEventListener("click", (e) => {
  if (e.target === citeModal) closeCitation();
});
