# 第二章「聊天前端」— 设计

日期:2026-11-16

## Intent / Why

第一章「纯对话」后端已完成(`/chat` SSE 流式 + `/extract` 结构化),但没有界面,只能 curl 演示。本章给它做一个客服聊天网页:浏览器打开就能和智能客服对话,回复逐字流式渲染,支持连续多轮——把「纯对话」从命令行变成可实际打开的体验。

## Goals

- `GET /` 打开聊天页(暖橙风格)。
- 发消息 → SSE 回复逐字渲染进气泡,流式期间带闪烁光标。
- 连续多轮:前端保存 `session_id`,追问时带上,上下文接得住。
- 桌面 + 移动端可用。

## Non-goals

- 不做登录/账号,不做历史会话持久化(刷新即清空;后端本就是内存会话)。
- 不做 `/extract` 的前端界面。
- 不做 Agent / 工具调用界面。

## Anti-goals

- 不引入 npm / 构建工具 / 框架(纯静态 HTML+CSS+JS,零依赖)。
- 不改后端 `/chat`、`/extract` 的业务逻辑,只加静态托管。

## Constraints

- `/chat` 是 POST,SSE 解析必须用 `fetch` + `ReadableStream`(`EventSource` 只支持 GET)。
- 后端目前无 CORS 中间件,前端必须同源(由 FastAPI 托管),否则浏览器会拦跨域。

## Approaches

1. **静态 HTML/CSS/JS 由 FastAPI 托管(选中)**:`app/main.py` 末尾挂 `StaticFiles`,同源零 CORS、零构建。学习项目最省事、最贴合「打开即聊」。用户拍板选此。
2. **Vite + React/Vue SPA**:更工程化,但要加 CORS、npm、双服务,超出本章需要。落选。

## How(已定)

- 文件:`static/index.html`、`static/style.css`、`static/app.js`;`app/main.py` 的 `create_app` 里先 include 两个 API 路由、再 `mount("/", StaticFiles(directory="static", html=True))` 兜底——`/chat`、`/extract`、`/docs` 先注册优先命中,`/` 及其它路径落到静态文件。
- SSE 解析:逐块 decode → 按 `\n\n` 分帧 → 剥 `data: ` 前缀 → JSON → 按 `type` 分发:`meta` 存 session_id、`token` 追加当前客服气泡、`done` 收尾移除光标、`error` 显示错误。
- 风格(暖橙):`--bg #faf5ef`(暖米白)、`--brand #f97316`(橙,用户气泡 + 发送按钮)、`--brand-hover #ea580c`、`--assistant-bubble #fff7ed`(橙淡染)、`--text #2f2a26`、`--muted #8a8178`、`--online #22c55e`。顶栏「🤖 在线客服 + 绿点」;用户右橙白字、客服左淡橙深字、带小尾巴;底部圆角输入框 + 橙发送按钮,Enter 发送;流式期间禁发;自动滚底。

## 测试 / 验收

- 自动化(pytest):新增「GET `/` 返回 200 且 `text/html`、GET 静态资源返回 200」的路由测试;既有 21 个测试不回归。
- 手动验收(本章验收方式):浏览器打开 `http://127.0.0.1:8000/`,发一句看逐字蹦出来,再带 session 追问一句看上下文接得住(即用户原话)。

## 留给实现者

- 气泡小尾巴 / 淡入动画 / 响应式断点的具体 CSS。
- `StaticFiles` 挂载与 `html=True` 兜底的具体写法。
- JS 里 SSE 分帧、追加渲染、自动滚底、流式期间禁发的细节。
