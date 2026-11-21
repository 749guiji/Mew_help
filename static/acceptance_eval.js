function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
async function main(){
  const root=document.getElementById("eval-root");
  const res=await fetch("/api/acceptance/eval-report");
  const rep=await res.json();
  if(!rep.per_class){root.textContent="还没评测报告。";return;}
  let html=`<div class="kb-chunk"><div class="kb-chunk-head">每类 F1</div>`;
  for(const c of rep.per_class){
    html+=`<div class="kb-chunk-a">${esc(c.name)}: P ${c.precision} · R ${c.recall} · F1 ${c.f1}</div>`;
  }
  html+=`</div><div class="kb-chunk"><div class="kb-chunk-head">混淆矩阵(每类 2×2)</div>`;
  for(const c of rep.confusion||[]){
    html+=`<div class="kb-chunk-a">${esc(c.name)}: TN ${c.tn} FP ${c.fp} FN ${c.fn} TP ${c.tp}</div>`;
  }
  html+=`</div>`;
  root.innerHTML=html;
}
main();
