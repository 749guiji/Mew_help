async function main(){
  const root=document.getElementById("data-root");
  const res=await fetch("/api/acceptance/data");
  const d=await res.json();
  let html=`<div class="kb-chunk"><div class="kb-chunk-head">数据产物</div>`;
  for(const [k,v] of Object.entries(d.files||{})){
    html+=`<div class="kb-chunk-a">${k}: ${v} 条</div>`;
  }
  html+=`</div>`;
  root.innerHTML=html;
}
main();
