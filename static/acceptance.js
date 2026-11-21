async function main() {
  const root = document.getElementById("root");
  const res = await fetch("/api/acceptance/overview");
  const d = await res.json();
  const rep = d.eval_report || {};
  const macro = rep.macro || {};
  root.innerHTML =
    `<div class="kb-chunk"><div class="kb-chunk-head">评测 macro</div>` +
    `<div class="kb-chunk-a">Precision ${macro.precision ?? "-"} · Recall ${macro.recall ?? "-"} · F1 ${macro.f1 ?? "-"}</div></div>` +
    `<div class="kb-chunk"><div class="kb-chunk-head">各页入口</div>` +
    `<div class="kb-chunk-a"><a href="/acceptance/eval">评测详情</a> · <a href="/acceptance/data">数据产物</a> · <a href="/acceptance/errors">错例复核</a></div></div>` +
    (macro.f1 == null ? `<div class="kb-chunk"><div class="kb-chunk-a">还没跑评测,先训练出 best.pt 再跑 ml/eval.py。</div></div>` : "");
}
main();
