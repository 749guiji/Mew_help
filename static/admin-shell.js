/* 管理端统一侧栏外壳 —— 单一导航源。
 * 读取 <body data-page="…"> 注入左侧导航，激活态按 href === "/" + data-page 判定。
 * 各管理页只声明 data-page，不再各自手抄导航。
 */
(function () {
  "use strict";

  var NAV_HTML = [
    '<div class="admin-badge">管理端</div>',
    '<a class="admin-back" href="/">← 返回聊天</a>',
    '<a href="/admin">后台总览</a>',
    '<a href="/kb">知识库</a>',
    '<a href="/review">待审队列</a>',
    '<a href="/topics">主题分布</a>',
    '<a href="/acceptance">验收</a>',
  ].join("\n");

  document.addEventListener("DOMContentLoaded", function () {
    var page = document.body.dataset.page || "";
    var aside = document.createElement("aside");
    aside.className = "admin-side";
    aside.innerHTML = NAV_HTML;

    var activeHref = "/" + page;
    aside.querySelectorAll("a").forEach(function (a) {
      if (a.getAttribute("href") === activeHref) {
        a.classList.add("active");
      }
    });

    document.body.prepend(aside);
  });
})();
