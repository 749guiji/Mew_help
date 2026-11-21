function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
async function main(){
  const root=document.getElementById("err-root");
  const res=await fetch("/api/acceptance/errors");
  const d=await res.json();
  const items=d.items||[];
  if(!items.length){root.textContent="无错例(或还没跑评测)。";return;}
  let html=`<div class="kb-chunk"><div class="kb-chunk-head">错例(${items.length})</div>`;
  for(const it of items){
    html+=`<div class="kb-chunk-a"><b>${esc(it.text)}</b><br>标注: ${esc((it.gold||[]).join(","))} · 预测: ${esc((it.pred||[]).join(","))}</div>`;
  }
  html+=`</div>`;
  root.innerHTML=html;
}
main();
