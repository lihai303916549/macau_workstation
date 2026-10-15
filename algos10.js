'use strict';
/**
 * 六子 —— 前10种算法加权融合
 * 成员（2026-10-16 全量613期 walk-forward 回测前10）：频率/专家/蒙特卡洛/马尔可夫/
 *   ARIMA/GBDT/GA/MLP/VAE/Stacking。已剔除：KNN、LSTM。
 *
 * ⚠️ 2026-10-16 重大修正（用户决策「消除」）：聚合步骤（49维分数→按生肖汇总→取top）
 *   除以各生肖号码数（马÷5、其余÷4），12生肖按“平均分”公平竞争，
 *   消除“马5个号总和自动胜出”的号数偏置。
 *   消除偏置后重跑回测（平特肖7球口径，随机基线 47.86%）：
 *   Stacking 51.06% | 专家 50.90% | VAE 50.57% | GBDT 49.92% | GA 49.43%
 *   蒙特卡洛 48.94% | 频率 48.78% | MLP 48.78% | ARIMA 48.12% | 马尔可夫 44.54%
 *   → 马尔可夫跌至基线以下 3.32 点，其原超额全靠「82.4%时间推马」的偏置红利，已剔除出融合。
 *
 * 融合权重 = 消除偏置后各算法超额（命中率 - 0.4786）。
 */
const STD = {1:'马',2:'蛇',3:'龙',4:'兔',5:'虎',6:'牛',7:'鼠',8:'猪',9:'狗',10:'鸡',11:'猴',12:'羊'};
const ZODIACS = Object.values(STD);
const ZI = {}; ZODIACS.forEach((z,i)=>ZI[z]=i);
function zOf(n){ const k = n<=12 ? n : ((n-1)%12)+1; return ZI[STD[k]]; }

// 固定种子随机数（保证同样历史数据 → 同样预测结果，可复现、不随天数漂移）
let _rngState = 0;
function rngReset(seed){ _rngState = seed >>> 0; }
function rng(){
  _rngState = (_rngState + 0x6D2B79F5) >>> 0;
  let t = _rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const WEIGHTS = {
  stacking: 0.0320, expert: 0.0304, vae: 0.0271, gbdt: 0.0206, ga: 0.0157,
  montecarlo: 0.0108, freq: 0.0092, mlp: 0.0092, arima: 0.0026,
  // ⚠️ 马尔可夫已剔除出融合：消除号数偏置后它仅 44.54%（低于随机基线 3.32 点），
  // 之前 54.81% 的超额几乎全部来自「82.4%时间推马」的号数偏置红利，被证伪。
};

function normalize(sc){
  let mn=Infinity,mx=-Infinity;
  for(const v of sc){ if(v<mn)mn=v; if(v>mx)mx=v; }
  const r=(mx-mn)||1, o=new Float64Array(49);
  for(let i=0;i<49;i++) o[i]=(sc[i]-mn)/r;
  return o;
}

// ================= 算法实现 =================

// 1) 频率统计：近30期7球频率
function algoFreq(hist, W=30){
  const win=hist.slice(-W), c=new Array(49).fill(0);
  for(const d of win) for(const n of [...d.numbers,d.special]) c[n-1]++;
  return c;
}

// 2) 专家指标：频率+生肖频率+遗漏-重号惩罚
function algoExpert(hist){
  const W=30, win=hist.slice(-W);
  const f=new Array(49).fill(0), zf=new Array(12).fill(0), mis=new Array(49).fill(0);
  for(const d of win) for(const n of [...d.numbers,d.special]){ f[n-1]++; zf[zOf(n)]++; }
  for(let n=1;n<=49;n++){ let k=win.length-1; while(k>=0&&![...win[k].numbers,win[k].special].includes(n))k--; mis[n-1]=k<0?W:W-1-k; }
  const prev=new Set([...hist[hist.length-1].numbers,hist[hist.length-1].special]);
  const sc=new Array(49);
  for(let n=1;n<=49;n++) sc[n-1]=f[n-1]*1.0+zf[zOf(n)]*0.4+(mis[n-1]<8?0.3:0)-(prev.has(n)?0.4:0);
  return sc;
}

// 3) 蒙特卡洛：近50期频率分布采样2万次
function algoMonteCarlo(hist, W=50){
  const win=hist.slice(-W), c=new Array(49).fill(0);
  for(const d of win) for(const n of [...d.numbers,d.special]) c[n-1]++;
  const tot=c.reduce((a,b)=>a+b,0)||1, hit=new Array(49).fill(0);
  for(let t=0;t<20000;t++){
    const r=rng()*tot; let acc=0,pick=48;
    for(let n=0;n<49;n++){ acc+=c[n]; if(r<=acc){pick=n;break;} }
    hit[pick]++;
  }
  return hit;
}

// 4) 马尔可夫：特码一阶(0.7)+二阶(0.3)转移
function algoMarkov(hist, W=150){
  const win=hist.slice(-W);
  const M1=Array.from({length:49},()=>new Array(49).fill(1));
  for(let t=1;t<win.length;t++) M1[win[t-1].special-1][win[t].special-1]++;
  const M2={};
  for(let t=2;t<win.length;t++){
    const k=win[t-2].special+'-'+win[t-1].special;
    if(!M2[k]) M2[k]=new Array(49).fill(1);
    M2[k][win[t].special-1]++;
  }
  const last=hist[hist.length-1].special;
  const last2=hist.length>1?hist[hist.length-2].special:last;
  const p1=M1[last-1], s1=p1.reduce((a,b)=>a+b,0);
  const k2=last2+'-'+last, p2=M2[k2]||new Array(49).fill(1), s2=p2.reduce((a,b)=>a+b,0);
  const sc=new Array(49);
  for(let n=0;n<49;n++) sc[n]=0.7*p1[n]/s1+0.3*p2[n]/s2;
  return sc;
}

// 5) ARIMA：特码序列 AR(2) → 高斯核打分
function algoARIMA(hist, W=60){
  const y=hist.slice(-W).map(d=>d.special);
  let S11=0,S12=0,S13=0,S22=0,S23=0,S33=0,T1=0,T2=0,T3=0;
  for(let t=2;t<y.length;t++){
    const x1=y[t-1],x2=y[t-2];
    S11+=x1*x1;S12+=x1*x2;S13+=x1;S22+=x2*x2;S23+=x2;S33+=1;
    T1+=x1*y[t];T2+=x2*y[t];T3+=y[t];
  }
  const A=[[S11,S12,S13],[S12,S22,S23],[S13,S23,S33]],b=[T1,T2,T3];
  for(let i=0;i<3;i++){
    let piv=i; for(let r=i+1;r<3;r++) if(Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
    [A[i],A[piv]]=[A[piv],A[i]]; [b[i],b[piv]]=[b[piv],b[i]];
    if(Math.abs(A[i][i])<1e-9) continue;
    for(let r=0;r<3;r++){ if(r===i)continue; const f=A[r][i]/A[i][i]; for(let c=i;c<3;c++)A[r][c]-=f*A[i][c]; b[r]-=f*b[i]; }
  }
  const a1=b[0]/(A[0][0]||1), a2=b[1]/(A[1][1]||1), c=b[2]/(A[2][2]||0);
  let pred=a1*y[y.length-1]+a2*y[y.length-2]+c;
  pred=Math.min(49,Math.max(1,pred));
  const sc=new Array(49), sigma=7;
  for(let n=1;n<=49;n++) sc[n-1]=Math.exp(-((n-pred)*(n-pred))/(2*sigma*sigma));
  return sc;
}

// 6) GBDT（代表 XGBoost/LightGBM/CatBoost）：每号4维特征 + 20棵depth3提升树
function gbdtBuildTree(X,g,h,depth,idx){
  function build(idxs,d){
    const node={};
    if(d>=depth||idxs.length<4){
      let s=0; for(const i of idxs) s+=g[i]/Math.max(h[i],1e-6);
      node.leaf=s/idxs.length; return node;
    }
    let best=null;
    for(let f=0;f<X[0].length;f++){
      const vals=[...new Set(idxs.map(i=>X[i][f]))].sort((a,b)=>a-b);
      for(let vi=0;vi<vals.length-1;vi++){
        const thr=(vals[vi]+vals[vi+1])/2;
        const l=idxs.filter(i=>X[i][f]<=thr), r=idxs.filter(i=>X[i][f]>thr);
        if(l.length<2||r.length<2) continue;
        const gl=l.reduce((s,i)=>s+g[i],0), gr=r.reduce((s,i)=>s+g[i],0);
        const hl=l.reduce((s,i)=>s+Math.max(h[i],1e-6),0), hr=r.reduce((s,i)=>s+Math.max(h[i],1e-6),0);
        const gain=gl*gl/hl+gr*gr/hr;
        if(!best||gain>best.gain) best={gain,f,thr,l,r};
      }
    }
    if(!best){ let s=0; for(const i of idxs) s+=g[i]/Math.max(h[i],1e-6); node.leaf=s/idxs.length; return node; }
    node.f=best.f; node.thr=best.thr; node.l=build(best.l,d+1); node.r=build(best.r,d+1);
    return node;
  }
  return build(idx,0);
}
function gbdtPredictTree(node,x){ while(node.leaf===undefined){ node=x[node.f]<=node.thr?node.l:node.r; } return node.leaf; }
function algoGBDT(hist){
  const win=hist.slice(-40);
  const feats=[], labels=[];
  for(let t=1;t<win.length;t++){
    const past=hist.slice(0,hist.length-win.length+t);
    if(past.length<20){ feats.push(null); continue; }
    const p10=past.slice(-10),p30=past.slice(-30);
    const f1=new Array(49).fill(0),f3=new Array(49).fill(0),mis=new Array(49).fill(0);
    for(const d of p10) for(const n of [...d.numbers,d.special]) f1[n-1]++;
    for(const d of p30) for(const n of [...d.numbers,d.special]) f3[n-1]++;
    for(let n=1;n<=49;n++){ let k=p30.length-1; while(k>=0&&![...p30[k].numbers,p30[k].special].includes(n))k--; mis[n-1]=k<0?p30.length:p30.length-1-k; }
    const prev=new Set([...past[past.length-1].numbers,past[past.length-1].special]);
    const ft=[]; for(let n=1;n<=49;n++) ft.push([f1[n-1]/10,f3[n-1]/30,mis[n-1]/30,prev.has(n)?1:0]);
    feats.push(ft);
    const y=new Array(49).fill(0); for(const n of win[t].numbers.concat([win[t].special])) y[n-1]=1;
    labels.push(y);
  }
  const p10=hist.slice(-10),p30=hist.slice(-30);
  const f1=new Array(49).fill(0),f3=new Array(49).fill(0),mis=new Array(49).fill(0);
  for(const d of p10) for(const n of [...d.numbers,d.special]) f1[n-1]++;
  for(const d of p30) for(const n of [...d.numbers,d.special]) f3[n-1]++;
  for(let n=1;n<=49;n++){ let k=p30.length-1; while(k>=0&&![...p30[k].numbers,p30[k].special].includes(n))k--; mis[n-1]=k<0?p30.length:p30.length-1-k; }
  const prev=new Set([...hist[hist.length-1].numbers,hist[hist.length-1].special]);
  const latest=[]; for(let n=1;n<=49;n++) latest.push([f1[n-1]/10,f3[n-1]/30,mis[n-1]/30,prev.has(n)?1:0]);
  const scores=new Array(49).fill(0.2);
  for(let n=0;n<49;n++){
    const X=[],Y=[];
    for(let t=1;t<win.length;t++){ if(feats[t]){ X.push(feats[t][n]); Y.push(labels[t][n]); } }
    if(X.length<5){ scores[n]=0.2; continue; }
    let p=new Array(X.length).fill(0.2);
    const trees=[];
    for(let it=0;it<20;it++){
      const g=new Array(X.length),h=new Array(X.length);
      for(let i=0;i<X.length;i++){ g[i]=p[i]-Y[i]; h[i]=p[i]*(1-p[i])+1e-6; }
      const tree=gbdtBuildTree(X,g,h,3,[...Array(X.length).keys()]);
      trees.push(tree);
      for(let i=0;i<X.length;i++){ const st=gbdtPredictTree(tree,X[i]); p[i]=Math.min(0.95,Math.max(0.05,p[i]+0.1*st)); }
    }
    let v=0.2; for(const tr of trees) v+=0.1*gbdtPredictTree(tr,latest[n]);
    scores[n]=Math.min(0.95,Math.max(0.05,v));
  }
  return scores;
}

// 7) 遗传算法：搜23个号，适应度=频率和+尾数多样性
function algoGA(hist){
  const W=30, win=hist.slice(-W);
  const freq=new Array(49).fill(0);
  for(const d of win) for(const n of [...d.numbers,d.special]) freq[n-1]++;
  const fitness=(mask)=>{
    let s=0,cnt=0; const tails=new Set();
    for(let n=0;n<49;n++) if(mask&(1<<n)){ s+=freq[n]; cnt++; tails.add((n+1)%10); }
    if(cnt!==23) return -1e9;
    return s+0.5*tails.size;
  };
  let pop=[];
  for(let i=0;i<60;i++){
    const idx=[...Array(49).keys()].sort(()=>rng()-0.5).slice(0,23);
    let mask=0; for(const j of idx) mask|=(1<<j);
    pop.push(mask);
  }
  for(let gen=0;gen<40;gen++){
    pop.sort((a,b)=>fitness(b)-fitness(a));
    const next=pop.slice(0,12);
    while(next.length<60){
      const a=pop[Math.floor(rng()*30)], b=pop[Math.floor(rng()*30)];
      let child=0;
      for(let n=0;n<49;n++){ const bit=(rng()<0.5?a:b)&(1<<n); child|=bit; }
      for(let k=0;k<2;k++) child^=(1<<Math.floor(rng()*49));
      let bits=[]; for(let n=0;n<49;n++) if(child&(1<<n)) bits.push(n);
      while(bits.length>23) bits.splice(Math.floor(rng()*bits.length),1);
      while(bits.length<23){ const cand=Math.floor(rng()*49); if(!bits.includes(cand)) bits.push(cand); }
      let m=0; for(const bt of bits) m|=(1<<bt);
      next.push(m);
    }
    pop=next;
  }
  pop.sort((a,b)=>fitness(b)-fitness(a));
  const sc=new Array(49).fill(0);
  for(let n=0;n<49;n++) if(pop[0]&(1<<n)) sc[n]=freq[n]+1;
  return sc;
}

// 8) MLP：98维特征 → 49维 sigmoid（全历史训练）
function algoMLP(hist, epochs=15){
  const W1=Array.from({length:32},()=>Array.from({length:98},()=>(rng()*2-1)*0.1));
  const b1=new Array(32).fill(0);
  const W2=Array.from({length:49},()=>Array.from({length:32},()=>(rng()*2-1)*0.1));
  const b2=new Array(49).fill(0);
  const feat=(h)=>{
    const p20=h.slice(-20);
    const f=new Array(49).fill(0),m=new Array(49).fill(0);
    for(const d of p20) for(const n of [...d.numbers,d.special]) f[n-1]++;
    for(let n=1;n<=49;n++){ let k=p20.length-1; while(k>=0&&![...p20[k].numbers,p20[k].special].includes(n))k--; m[n-1]=k<0?p20.length:p20.length-1-k; }
    return [...f.map(v=>v/20),...m.map(v=>v/20)];
  };
  const lr=0.05;
  for(let ep=0;ep<epochs;ep++){
    for(let t=20;t<hist.length;t++){
      const past=hist.slice(0,t);
      const x=feat(past);
      const y=new Array(49).fill(0);
      for(const n of hist[t].numbers.concat([hist[t].special])) y[n-1]=1;
      const h=new Array(32).fill(0);
      for(let j=0;j<32;j++){ let s=b1[j]; for(let k=0;k<98;k++) s+=W1[j][k]*x[k]; h[j]=Math.max(0,s); }
      const o=new Array(49);
      for(let yy=0;yy<49;yy++){ let s=b2[yy]; for(let j=0;j<32;j++) s+=W2[yy][j]*h[j]; o[yy]=1/(1+Math.exp(-s)); }
      const d2=new Array(49); for(let yy=0;yy<49;yy++) d2[yy]=o[yy]-y[yy];
      const dh=new Array(32).fill(0);
      for(let yy=0;yy<49;yy++) for(let j=0;j<32;j++) dh[j]+=W2[yy][j]*d2[yy];
      for(let yy=0;yy<49;yy++){ for(let j=0;j<32;j++) W2[yy][j]-=lr*d2[yy]*h[j]; b2[yy]-=lr*d2[yy]; }
      for(let j=0;j<32;j++){ if(h[j]<=0){dh[j]=0;continue;} for(let k=0;k<98;k++) W1[j][k]-=lr*dh[j]*x[k]; b1[j]-=lr*dh[j]; }
    }
  }
  const x=feat(hist);
  const h=new Array(32).fill(0);
  for(let j=0;j<32;j++){ let s=b1[j]; for(let k=0;k<98;k++) s+=W1[j][k]*x[k]; h[j]=Math.max(0,s); }
  const o=new Array(49);
  for(let yy=0;yy<49;yy++){ let s=b2[yy]; for(let j=0;j<32;j++) s+=W2[yy][j]*h[j]; o[yy]=1/(1+Math.exp(-s)); }
  return o;
}

// 9) VAE（生成式）：频率分布→latent6→重建49维（全历史训练）
function algoVAE(hist, epochs=15){
  const r=()=>(rng()*2-1)*0.1;
  const e1=Array.from({length:16},()=>Array.from({length:49},r)), eb1=new Array(16).fill(0);
  const mu=Array.from({length:6},()=>Array.from({length:16},r)), lv=Array.from({length:6},()=>Array.from({length:16},r));
  const bmu=new Array(6).fill(0), blv=new Array(6).fill(0);
  const d1=Array.from({length:16},()=>Array.from({length:6},r)), db1=new Array(16).fill(0);
  const d2=Array.from({length:49},()=>Array.from({length:16},r)), db2=new Array(49).fill(0);
  const inp=(h)=>{ const w=h.slice(-30); const f=new Array(49).fill(0); for(const d of w) for(const n of [...d.numbers,d.special]) f[n-1]++; return f.map(v=>v/30); };
  const tgt=(d)=>{ const y=new Array(49).fill(0); for(const n of [...d.numbers,d.special]) y[n-1]=1; return y; };
  const lr=0.02;
  for(let ep=0;ep<epochs;ep++){
    for(let t=30;t<hist.length;t++){
      const x=inp(hist.slice(0,t)), y=tgt(hist[t-1]);
      const h=new Array(16).fill(0);
      for(let j=0;j<16;j++){ let s=eb1[j]; for(let k=0;k<49;k++) s+=e1[j][k]*x[k]; h[j]=Math.max(0,s); }
      const m=new Array(6),l=new Array(6);
      for(let j=0;j<6;j++){ let sm=bmu[j],sl=blv[j]; for(let k=0;k<16;k++){ sm+=mu[j][k]*h[k]; sl+=lv[j][k]*h[k]; } m[j]=sm; l[j]=sl; }
      const z=new Array(6);
      for(let j=0;j<6;j++) z[j]=m[j]+(rng()*2-1)*Math.exp(l[j]/2);
      const g=new Array(16).fill(0);
      for(let j=0;j<16;j++){ let s=db1[j]; for(let k=0;k<6;k++) s+=d1[j][k]*z[k]; g[j]=Math.max(0,s); }
      const o=new Array(49);
      for(let yy=0;yy<49;yy++){ let s=db2[yy]; for(let j=0;j<16;j++) s+=d2[yy][j]*g[j]; o[yy]=1/(1+Math.exp(-s)); }
      const e2=new Array(49); for(let yy=0;yy<49;yy++) e2[yy]=o[yy]-y[yy];
      const dg=new Array(16).fill(0);
      for(let yy=0;yy<49;yy++) for(let j=0;j<16;j++){ d2[yy][j]-=lr*e2[yy]*g[j]; dg[j]+=d2[yy][j]*e2[yy]; }
      for(let yy=0;yy<49;yy++) db2[yy]-=lr*e2[yy];
      for(let j=0;j<16;j++){ if(g[j]<=0){dg[j]=0;continue;} for(let k=0;k<6;k++) d1[j][k]-=lr*dg[j]*z[k]; db1[j]-=lr*dg[j]; }
    }
  }
  const x=inp(hist);
  const h=new Array(16).fill(0);
  for(let j=0;j<16;j++){ let s=eb1[j]; for(let k=0;k<49;k++) s+=e1[j][k]*x[k]; h[j]=Math.max(0,s); }
  const m=new Array(6);
  for(let j=0;j<6;j++){ let sm=bmu[j]; for(let k=0;k<16;k++) sm+=mu[j][k]*h[k]; m[j]=sm; }
  const g=new Array(16).fill(0);
  for(let j=0;j<16;j++){ let s=db1[j]; for(let k=0;k<6;k++) s+=d1[j][k]*m[k]; g[j]=Math.max(0,s); }
  const o=new Array(49);
  for(let yy=0;yy<49;yy++){ let s=db2[yy]; for(let j=0;j<16;j++) s+=d2[yy][j]*g[j]; o[yy]=1/(1+Math.exp(-s)); }
  return o;
}

// ================= 融合与派生 =================
// 各生肖号码数（2026年：马5个，其余11生肖各4个）——用于消除聚合号数偏置
const ZODIAC_NUMS = ZODIACS.map(z => { let c = 0; for (let n = 1; n <= 49; n++) if (zOf(n) === ZI[z]) c++; return c; });

function derive(sc){
  const zs=new Array(12).fill(0);
  for(let n=1;n<=49;n++) zs[zOf(n)]+=sc[n-1];
  // 消除号数偏置（2026-10-16 用户决策）：除以各生肖号码数（马÷5、其余÷4），
  // 让12生肖按“平均分”公平竞争，而非马靠5个号的总和自动胜出
  for(let z=0;z<12;z++) zs[z]/=ZODIAC_NUMS[z];
  const zr=ZODIACS.map((z,i)=>({z,s:zs[i]})).sort((a,b)=>b.s-a.s);
  const top3=zr.slice(0,3).map(x=>x.z);
  const sum3=zr.slice(0,3).reduce((s,x)=>s+x.s,0)||1;
  const probs={}; top3.forEach(x=>{ probs[x]=zr.find(y=>y.z===x).s/sum3; });
  const order=[...Array(49).keys()].sort((a,b)=>sc[b]-sc[a]);
  const codes=order.slice(0,23).map(x=>x+1).sort((a,b)=>a-b);
  const six=zr.slice(0,6).map(x=>x.z);
  return { zodiac:zr[0].z, top3, probs, codes, six };
}

function predictAll(hist){
  rngReset(20261016);
  const raw = {
    freq: algoFreq(hist),
    expert: algoExpert(hist),
    montecarlo: algoMonteCarlo(hist),
    markov: algoMarkov(hist),
    arima: algoARIMA(hist),
    gbdt: algoGBDT(hist),
    ga: algoGA(hist),
    mlp: algoMLP(hist),
    vae: algoVAE(hist),
  };
  // Stacking = 独立算法归一化等权融合（第10种，作参照；markov 因偏置证伪已排除）
  const stack=new Float64Array(49);
  const keys=Object.keys(raw).filter(k=>k!=='markov');
  for(const k of keys){ const s=normalize(raw[k]); for(let n=0;n<49;n++) stack[n]+=s[n]; }
  for(let n=0;n<49;n++) stack[n]/=keys.length;
  raw.stacking=Array.from(stack);

  // 加权融合（10种，权重=回测超额）
  const fused=new Float64Array(49);
  let wsum=0;
  for(const [k,w] of Object.entries(WEIGHTS)){
    const s=normalize(raw[k]);
    for(let n=0;n<49;n++) fused[n]+=w*s[n];
    wsum+=w;
  }
  for(let n=0;n<49;n++) fused[n]/=wsum;

  const details={};
  for(const [k,sc] of Object.entries(raw)) details[k]=derive(sc);
  return { fused: derive(fused), details, weights: WEIGHTS };
}

module.exports = { predictAll, derive, WEIGHTS, ZODIACS, zOf };
