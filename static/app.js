const state = { userId: localStorage.getItem("mew_user_id"), conversationId: null, streaming: false };
if (!state.userId) {
  state.userId = "u-" + Math.random().toString(36).slice(2, 10);
  localStorage.setItem("mew_user_id", state.userId);
}
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
    avatar.textContent = "🐱";
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
    state.conversationId = parseInt(evt.session_id, 10);
  } else if (evt.type === "token") {
    if (ctx.typing) {
      ctx.typing.remove();
      ctx.typing = null;
    }
    ctx.textSpan.textContent += evt.content;
    scrollToBottom();
  } else if (evt.type === "tool") {
    renderToolBadge(ctx.badgeEl, evt);
    scrollToBottom();
  } else if (evt.type === "citations") {
    ctx.citations = evt.citations || [];
  } else if (evt.type === "handoff") {
    renderHandoff(evt.suggestions || [], ctx.contentEl);
  } else if (evt.type === "interrupt") {
    if (evt.kind === "ticket_preview") {
      renderTicketPreview(evt.draft || {}, ctx.contentEl);
    } else {
      renderOrderCards(evt.orders || [], ctx.contentEl);
    }
  } else if (evt.type === "actions") {
    renderActions(evt.items || [], ctx.contentEl);
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

// 通用 SSE 流式:fetch 后把事件 dispatch 进给定的助手气泡。
async function streamTo(fetchUrl, body, contentEl) {
  const badgeEl = document.createElement("div");
  badgeEl.className = "tool-badge";
  contentEl.appendChild(badgeEl);

  const textSpan = document.createElement("span");
  textSpan.className = "text";
  contentEl.appendChild(textSpan);

  const typing = document.createElement("span");
  typing.className = "typing";
  for (let i = 0; i < 3; i++) {
    const d = document.createElement("span");
    d.className = "typing-dot";
    typing.appendChild(d);
  }
  contentEl.appendChild(typing);

  const cursor = document.createElement("span");
  cursor.className = "cursor";
  contentEl.appendChild(cursor);

  const ctx = { textSpan, badgeEl, citations: [], contentEl, typing };

  setStreaming(true);
  try {
    const res = await fetch(fetchUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
    if (ctx.typing) {
      ctx.typing.remove();
      ctx.typing = null;
    }
    setStreaming(false);
    if (ctx.citations.length) linkifyCitations(textSpan, ctx.citations);
    addFeedback(contentEl);
    scrollToBottom();
  }
}

function send() {
  const text = inputEl.value.trim();
  if (!text || state.streaming) return;
  inputEl.value = "";
  appendBubble("user", text);
  const contentEl = appendBubble("assistant", "");
  streamTo(
    "/chat",
    { message: text, user_id: state.userId, conversation_id: state.conversationId },
    contentEl
  );
}

// 订单选择卡片:interrupt 事件渲染,点选后 /api/actions/resume 续跑子流程。
function renderOrderCards(orders, contentEl) {
  if (!orders || !orders.length) return;
  const wrap = document.createElement("div");
  wrap.className = "order-cards";
  for (const o of orders) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "order-card";

    const main = document.createElement("div");
    main.className = "order-card-main";
    const id = document.createElement("span");
    id.className = "order-card-id";
    id.textContent = "订单 " + o.order_id;
    const status = document.createElement("span");
    status.className = "order-card-status";
    status.textContent = o.status;
    main.appendChild(id);
    main.appendChild(status);

    const sub = document.createElement("div");
    sub.className = "order-card-sub";
    sub.textContent = o.product + " · ¥" + o.amount;

    card.appendChild(main);
    card.appendChild(sub);
    card.addEventListener("click", () => selectOrder(o));
    wrap.appendChild(card);
  }
  contentEl.appendChild(wrap);
  scrollToBottom();
}

function removeOrderCards() {
  document.querySelectorAll(".order-cards").forEach((el) => el.remove());
}

async function selectOrder(order) {
  if (state.streaming) return;
  removeOrderCards();
  const contentEl = appendBubble("assistant", "");
  await streamTo(
    "/api/actions/resume",
    { conversation_id: state.conversationId, order_id: order.order_id },
    contentEl
  );
}

// 工单预览卡片:interrupt 事件(kind=ticket_preview)渲染,确认/取消走 resume。
function renderTicketPreview(draft, contentEl) {
  const wrap = document.createElement("div");
  wrap.className = "ticket-preview";

  const title = document.createElement("div");
  title.className = "ticket-preview-title";
  title.textContent = "工单确认";
  wrap.appendChild(title);

  const typeRow = document.createElement("div");
  typeRow.className = "ticket-preview-row";
  const typeLabel = document.createElement("span");
  typeLabel.className = "ticket-label";
  typeLabel.textContent = "工单类型";
  const typeVal = document.createElement("span");
  typeVal.className = "ticket-value";
  typeVal.textContent = draft.ticket_type || "售后";
  typeRow.appendChild(typeLabel);
  typeRow.appendChild(typeVal);
  wrap.appendChild(typeRow);

  const descRow = document.createElement("div");
  descRow.className = "ticket-preview-row";
  const descLabel = document.createElement("span");
  descLabel.className = "ticket-label";
  descLabel.textContent = "问题描述";
  const descVal = document.createElement("span");
  descVal.className = "ticket-desc";
  descVal.textContent = draft.description || "";
  descRow.appendChild(descLabel);
  descRow.appendChild(descVal);
  wrap.appendChild(descRow);

  const btnRow = document.createElement("div");
  btnRow.className = "ticket-preview-actions";
  const confirmBtn = document.createElement("button");
  confirmBtn.type = "button";
  confirmBtn.className = "ticket-confirm";
  confirmBtn.textContent = "确认提交";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "ticket-cancel";
  cancelBtn.textContent = "取消";
  confirmBtn.addEventListener("click", () => confirmTicket(true));
  cancelBtn.addEventListener("click", () => confirmTicket(false));
  btnRow.appendChild(confirmBtn);
  btnRow.appendChild(cancelBtn);
  wrap.appendChild(btnRow);

  contentEl.appendChild(wrap);
  scrollToBottom();
}

function removeTicketPreview() {
  document.querySelectorAll(".ticket-preview").forEach((el) => el.remove());
}

async function confirmTicket(confirm) {
  if (state.streaming) return;
  removeTicketPreview();
  const contentEl = appendBubble("assistant", "");
  await streamTo(
    "/api/actions/resume",
    { conversation_id: state.conversationId, confirm: confirm },
    contentEl
  );
}

// 动作帧:refund_form → 渲染退款表单(submit_refund 工具拦截的产物)。
function renderActions(items, contentEl) {
  for (const a of items) {
    if (a.type === "refund_form") {
      renderRefundForm(contentEl, a.draft.order_id);
    }
  }
}

const REFUND_REASONS = ["七天无理由", "质量问题", "发错货", "不想要了", "其他"];

function renderRefundForm(contentEl, orderId) {
  const form = document.createElement("div");
  form.className = "refund-form";

  const title = document.createElement("div");
  title.className = "refund-form-title";
  title.textContent = "申请退款";

  const orderRow = document.createElement("div");
  orderRow.className = "refund-form-row";
  const orderLabel = document.createElement("span");
  orderLabel.className = "refund-label";
  orderLabel.textContent = "订单";
  const orderVal = document.createElement("span");
  orderVal.className = "refund-order";
  orderVal.textContent = orderId;
  orderRow.appendChild(orderLabel);
  orderRow.appendChild(orderVal);

  const reasonRow = document.createElement("div");
  reasonRow.className = "refund-form-row";
  const reasonLabel = document.createElement("span");
  reasonLabel.className = "refund-label";
  reasonLabel.textContent = "退款原因";
  const select = document.createElement("select");
  select.className = "refund-select";
  for (const r of REFUND_REASONS) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    select.appendChild(opt);
  }
  reasonRow.appendChild(reasonLabel);
  reasonRow.appendChild(select);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "refund-submit";
  btn.textContent = "提交退款申请";

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const res = await fetch("/api/actions/create-refund", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sessionId,
          order_id: orderId,
          reason: select.value,
        }),
      });
      if (res.ok) {
        const { ticket_no } = await res.json();
        btn.textContent = "已提交退款单 " + ticket_no;
      } else {
        btn.textContent = "提交失败";
      }
    } catch (err) {
      btn.textContent = "提交失败";
    }
  });

  form.appendChild(title);
  form.appendChild(orderRow);
  form.appendChild(reasonRow);
  form.appendChild(btn);
  contentEl.appendChild(form);
  scrollToBottom();
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

// 会话侧栏:列用户会话、切换回载、新对话;加载失败静默降级不影响聊天。
async function loadConversations() {
  try {
    const res = await fetch("/api/conversations?user_id=" + encodeURIComponent(state.userId));
    if (!res.ok) return;
    const { conversations } = await res.json();
    const el = document.getElementById("conversations");
    el.innerHTML = "";
    for (const c of conversations) {
      const item = document.createElement("button");
      item.className = "conv-item" + (c.id === state.conversationId ? " active" : "");
      item.type = "button";
      item.textContent = (c.summarized ? "📄 " : "") + (c.preview || "（新对话）");
      item.addEventListener("click", () => switchConversation(c.id));
      el.appendChild(item);
    }
  } catch (err) {
    /* 静默降级 */
  }
}

async function switchConversation(id) {
  state.conversationId = id;
  document.getElementById("messages").innerHTML = "";
  try {
    const res = await fetch("/api/conversations/" + id + "/messages");
    if (res.ok) {
      const { messages } = await res.json();
      for (const m of messages) {
        if (m.role === "user") appendBubble("user", m.content);
        else if (m.role === "assistant") appendBubble("assistant", m.content);
      }
    }
  } catch (err) {
    /* 静默降级 */
  }
  loadConversations();
}

function newConversation() {
  state.conversationId = null;
  document.getElementById("messages").innerHTML = "";
  loadConversations();
}

document.getElementById("new-chat").addEventListener("click", newConversation);
loadConversations();
