const state = { sessionId: null, streaming: false };
let sseBuffer = "";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

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

function dispatch(evt, textSpan) {
  if (evt.type === "meta") {
    state.sessionId = evt.session_id;
  } else if (evt.type === "token") {
    textSpan.textContent += evt.content;
    scrollToBottom();
  } else if (evt.type === "error") {
    textSpan.textContent += "\n[出错了] " + evt.message;
    scrollToBottom();
  }
}

function handleSSE(chunk, textSpan) {
  sseBuffer += chunk;
  const parts = sseBuffer.split("\n\n");
  sseBuffer = parts.pop();
  for (const part of parts) {
    for (const line of part.split("\n")) {
      if (line.startsWith("data: ")) {
        try {
          dispatch(JSON.parse(line.slice(6)), textSpan);
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

  const textSpan = document.createElement("span");
  textSpan.className = "text";
  contentEl.appendChild(textSpan);

  const cursor = document.createElement("span");
  cursor.className = "cursor";
  contentEl.appendChild(cursor);

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
      handleSSE(decoder.decode(value, { stream: true }), textSpan);
    }
  } catch (err) {
    textSpan.textContent += "\n[出错了] " + err.message;
  } finally {
    cursor.remove();
    setStreaming(false);
    scrollToBottom();
  }
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
