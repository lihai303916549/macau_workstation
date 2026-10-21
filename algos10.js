/**
 * 六子 —— 预测引擎 v8（2026-10-21 用户定版：删 TCN + 权重窗口 60→200 期）
 * ==================== 模型清单（8个，全部基于历史数据学习）====================
 * 【现代时序 / 卷积 2 个】
 *   1) N-HiTS          —— 多尺度块线性(8/16/32期) + 残差 + 加性聚合，层级线性
 *   2) LSTM            —— 标准i/f/o/g门，隐层16，滑窗20步BPTT
 *   3) CNN             —— T=20×49时序图，2个kernel=3一维卷积 + FC49
 * 【统计 / 机器学习 3 个】
 *   4) 卡尔曼滤波/状态空间 —— 局部水平模型(随机游走)，逐号贝叶斯平滑+一步外推
 *   5) LightGBM/GBDT    —— 提升树×15深度3，每号独立模型
 *   6) CatBoost        —— 对称树(oblivious)depth4×20轮 + 因果目标统计特征
 * 【优化 / 进化 2 个】（优化"12维生肖权重向量"，适应度=近30期平特肖命中）
 *   7)  遗传算法 GA     —— 锦标赛选择+交叉+变异
 *   8)  粒子群 PSO      —— 速度/位置更新
 * 【框架层】
 *   1) Ensemble 动态融合 —— 8模型归一化后按 Walk-forward 超额权重加权融合
 *   2) Walk-forward 优化 —— 近200期影子回测（v8：60→200期，修短视）
 *   3) 多属性命中判定    —— judgeDraw：23码/平特肖/6肖三层判定
 *   4) 共识投票层       —— 8模型主推生肖投票；一致度≥60% 正常下注，分歧大则【跳过不下注】
 *
 * ⚠️ v8 变更（基于300期正式回测，2025-359~2026-293，显著性检验）：
 *   ① 删除 TCN：300期平特肖命中率仅 32.3%（随机基线46.3%，p<0.001 显著反向），
 *      但近60期运气好骗过了旧权重机制（60期窗口把"短期运气"当"真实能力"，LCG 当年同款）。
 *   ② 权重回测窗口 60→200 期：让长期有效的模型拿权重（CatBoost 平特肖51.7% 升为主导 33.9%）。
 *   ③ 实测排名（300期）：CatBoost 51.7% > LightGBM 50.3%(23码56.0% 显著) > LSTM/CNN 46.3%
 *      > PSO 43.0% > 卡尔曼 42.7% > GA 41.7% > N-HiTS 40.7% > TCN 32.3%（已删）
 *      仅用 lgbm+catboost 融合：平特肖 48.0% / 23码 54.0%（23码显著优于随机47%）
 * ⚠️ 成本控制：慢模型(lstm/cnn/lgbm/catboost)预训练到回测起点后只前向；
 *   优化类(ga/pso)每期重跑。训练范围：全量历史。固定种子20261019 → 可复现。
 * 事前预测铁律：只做事前预测并存档；开奖后按存档判定记录，严禁事后重算。
 */
const STD = {1:'马',2:'蛇',3:'龙',4:'兔',5:'虎',6:'牛',7:'鼠',8:'猪',9:'狗',10:'鸡',11:'猴',12:'羊'};
const ZODIACS = Object.values(STD);
const ZI = {}; ZODIACS.forEach((z,i)=>ZI[z]=i);
function zOf(n){ const k = n<=12 ? n : ((n-1)%12)+1; return ZI[STD[k]]; }

let _rngState = 0;
function rngReset(seed){ _rngState = seed >>> 0; }
function rng(){
  _rngState = (_rngState + 0x6D2B79F5) >>> 0;
  let t = _rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function randn(){ let u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

function normalize(sc){
  let mn=Infinity,mx=-Infinity;
  for(const v of sc){ if(v<mn)mn=v; if(v>mx)mx=v; }
  const r=(mx-mn)||1, o=new Float64Array(49);
  for(let i=0;i<49;i++) o[i]=(sc[i]-mn)/r;
  return o;
}
function balls(d){ return [...d.numbers, d.special]; }
function feats49(d){ const x=new Float64Array(49); for(const n of balls(d)) x[n-1]=1; return x; }
function clamp(x,a,b){ return Math.max(a, Math.min(b,x)); }
const sigmoid = x => 1/(1+Math.exp(-x));
const TANH = Math.tanh;
const clip = (g, C=5) => Math.max(-C, Math.min(C, g));

// 各生肖号码数（2026：马5个，其余4个）——聚合除号数消除偏置
const ZODIAC_NUMS = ZODIACS.map(z => { let c = 0; for (let n = 1; n <= 49; n++) if (zOf(n) === ZI[z]) c++; return c; });

function derive(sc){
  const zs=new Array(12).fill(0);
  for(let n=1;n<=49;n++) zs[zOf(n)]+=sc[n-1];
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
function top1Z(sc){
  const zs=new Array(12).fill(0),cnt=new Array(12).fill(0);
  for(let n=1;n<=49;n++){ zs[zOf(n)]+=sc[n-1]; cnt[zOf(n)]++; }
  for(let z=0;z<12;z++) zs[z]/=cnt[z];
  let t=0; for(let z=1;z<12;z++) if(zs[z]>zs[t]) t=z;
  return t;
}
// 49维号码分 → 12生肖聚合分（除号数）
function zodiacScores(sc){
  const zs=new Array(12).fill(0);
  for(let n=1;n<=49;n++) zs[zOf(n)]+=sc[n-1];
  for(let z=0;z<12;z++) zs[z]/=ZODIAC_NUMS[z];
  return zs;
}

// ================= 2) LSTM =================
const LSTM_H = 16, LSTM_WIN = 20, LSTM_EPOCHS = 5;
function lstmTrain(hist){
  const data = hist;
  const p = (r,c)=>Array.from({length:r},()=>Array.from({length:c},()=>randn()*0.1));
  const Wxi=p(LSTM_H,49), Whi=p(LSTM_H,LSTM_H), Wxf=p(LSTM_H,49), Whf=p(LSTM_H,LSTM_H);
  const Wxo=p(LSTM_H,49), Who_=p(LSTM_H,LSTM_H), Wxg=p(LSTM_H,49), Whg=p(LSTM_H,LSTM_H);
  const bi=new Array(LSTM_H).fill(0), bf=new Array(LSTM_H).fill(1), bo_=new Array(LSTM_H).fill(0), bg=new Array(LSTM_H).fill(0);
  const Wy=p(49,LSTM_H), by=new Array(49).fill(0);
  const lr=0.1;
  const X=data.map(feats49);
  const samples=Math.max(1, data.length-LSTM_WIN);
  for(let ep=0;ep<LSTM_EPOCHS;ep++){
    for(let s=0;s<samples;s++){
      const cs=[new Array(LSTM_H).fill(0)], hs=[new Array(LSTM_H).fill(0)];
      const cache=[];
      for(let t=0;t<LSTM_WIN;t++){
        const x=X[s+t], hPrev=hs[t], cPrev=cs[t];
        const i=new Array(LSTM_H),f=new Array(LSTM_H),o=new Array(LSTM_H),g=new Array(LSTM_H);
        for(let j=0;j<LSTM_H;j++){
          let ai=bi[j],af=bf[j],ao=bo_[j],ag=bg[j];
          for(let k=0;k<49;k++){ ai+=Wxi[j][k]*x[k]; af+=Wxf[j][k]*x[k]; ao+=Wxo[j][k]*x[k]; ag+=Wxg[j][k]*x[k]; }
          for(let k=0;k<LSTM_H;k++){ ai+=Whi[j][k]*hPrev[k]; af+=Whf[j][k]*hPrev[k]; ao+=Who_[j][k]*hPrev[k]; ag+=Whg[j][k]*hPrev[k]; }
          i[j]=sigmoid(ai); f[j]=sigmoid(af); o[j]=sigmoid(ao); g[j]=TANH(ag);
        }
        const c=new Array(LSTM_H), h=new Array(LSTM_H);
        for(let j=0;j<LSTM_H;j++){ c[j]=f[j]*cPrev[j]+i[j]*g[j]; h[j]=o[j]*TANH(c[j]); }
        cs.push(c); hs.push(h); cache.push({x,hPrev,cPrev,i,f,o,g,c,h});
      }
      const hLast=hs[LSTM_WIN];
      const y=X[s+LSTM_WIN] || X[s+LSTM_WIN-1];
      const oOut=new Array(49);
      for(let k=0;k<49;k++){ let acc=by[k]; for(let j=0;j<LSTM_H;j++) acc+=Wy[k][j]*hLast[j]; oOut[k]=sigmoid(acc); }
      const dOut=new Array(49); for(let k=0;k<49;k++) dOut[k]=oOut[k]-y[k];
      const dW=new Array(49).fill(0).map(()=>new Array(LSTM_H).fill(0));
      const dby=new Array(49).fill(0);
      for(let k=0;k<49;k++){ const gk=dOut[k]*oOut[k]*(1-oOut[k]); dby[k]=gk;
        for(let j=0;j<LSTM_H;j++) dW[k][j]=gk*hLast[j]; }
      for(let k=0;k<49;k++){ for(let j=0;j<LSTM_H;j++) Wy[k][j]-=lr*clip(dW[k][j]); by[k]-=lr*clip(dby[k]); }
      const t=LSTM_WIN-1, cc=cache[t];
      const dh=new Array(LSTM_H).fill(0);
      for(let j=0;j<LSTM_H;j++){ let g=0; for(let k=0;k<49;k++) g+=dOut[k]*oOut[k]*(1-oOut[k])*Wy[k][j]; dh[j]=g; }
      const dC=new Array(LSTM_H).fill(0);
      for(let j=0;j<LSTM_H;j++){ dC[j]=dh[j]*cc.o[j]*(1-cc.c[j]*cc.c[j]); }
      for(let j=0;j<LSTM_H;j++){
        const dg=dC[j]*cc.i[j], di_=dC[j]*cc.g[j], df_=dC[j]*cc.cPrev[j], do_=dh[j]*TANH(cc.c[j]);
        for(let k=0;k<49;k++){
          Wxi[j][k]-=lr*clip(di_*cc.i[j]*(1-cc.i[j])*cc.x[k]);
          Wxf[j][k]-=lr*clip(df_*cc.f[j]*(1-cc.f[j])*cc.x[k]);
          Wxo[j][k]-=lr*clip(do_*cc.o[j]*(1-cc.o[j])*cc.x[k]);
          Wxg[j][k]-=lr*clip(dg*(1-cc.g[j]*cc.g[j])*cc.x[k]);
        }
        for(let k=0;k<LSTM_H;k++){
          Whi[j][k]-=lr*clip(di_*cc.i[j]*(1-cc.i[j])*cc.hPrev[k]);
          Whf[j][k]-=lr*clip(df_*cc.f[j]*(1-cc.f[j])*cc.hPrev[k]);
          Who_[j][k]-=lr*clip(do_*cc.o[j]*(1-cc.o[j])*cc.hPrev[k]);
          Whg[j][k]-=lr*clip(dg*(1-cc.g[j]*cc.g[j])*cc.hPrev[k]);
        }
        bi[j]-=lr*clip(di_*cc.i[j]*(1-cc.i[j]));
        bf[j]-=lr*clip(df_*cc.f[j]*(1-cc.f[j]));
        bo_[j]-=lr*clip(do_*cc.o[j]*(1-cc.o[j]));
        bg[j]-=lr*clip(dg*(1-cc.g[j]*cc.g[j]));
      }
    }
  }
  return {Wxi,Whi,Wxf,Whf,Wxo,Who_,Wxg,Whg,bi,bf,bo_,bg,Wy,by};
}
function lstmFwd(m, hist){
  const data=hist.slice(-LSTM_WIN);
  let h=new Array(LSTM_H).fill(0), c=new Array(LSTM_H).fill(0);
  for(let t=0;t<data.length;t++){
    const x=feats49(data[t]); const nh=new Array(LSTM_H), nc=new Array(LSTM_H);
    for(let j=0;j<LSTM_H;j++){
      let ai=m.bi[j],af=m.bf[j],ao=m.bo_[j],ag=m.bg[j];
      for(let k=0;k<49;k++){ ai+=m.Wxi[j][k]*x[k]; af+=m.Wxf[j][k]*x[k]; ao+=m.Wxo[j][k]*x[k]; ag+=m.Wxg[j][k]*x[k]; }
      for(let k=0;k<LSTM_H;k++){ ai+=m.Whi[j][k]*h[k]; af+=m.Whf[j][k]*h[k]; ao+=m.Who_[j][k]*h[k]; ag+=m.Whg[j][k]*h[k]; }
      nc[j]=sigmoid(af)*c[j]+sigmoid(ai)*TANH(ag);
      nh[j]=sigmoid(ao)*TANH(nc[j]);
    }
    h=nh; c=nc;
  }
  const sc=new Array(49);
  for(let k=0;k<49;k++){ let acc=m.by[k]; for(let j=0;j<LSTM_H;j++) acc+=m.Wy[k][j]*h[j]; sc[k]=sigmoid(acc); }
  return sc;
}

// ================= 4) CNN =================
const CNN_T = 20, CNN_K = 3, CNN_C = 2, CNN_EPOCHS = 5;
function cnnTrain(hist){
  const data=hist.map(feats49);
  const conv=Array.from({length:CNN_C},()=>Array.from({length:CNN_K},()=>Array.from({length:49},()=>randn()*0.1)));
  const bconv=new Array(CNN_C).fill(0);
  const flatN=(CNN_T-CNN_K+1)*CNN_C;
  const Wfc=Array.from({length:49},()=>Array.from({length:flatN},()=>randn()*0.1));
  const bfc=new Array(49).fill(0);
  const lr=0.1;
  const convOut=(win)=>{ const out=[];
    for(let t=0;t<=CNN_T-CNN_K;t++){ const row=new Array(CNN_C).fill(0);
      for(let c=0;c<CNN_C;c++){ let s=bconv[c];
        for(let k=0;k<CNN_K;k++) for(let d=0;d<49;d++) s+=conv[c][k][d]*win[t+k][d];
        row[c]=Math.max(0,s); }
      out.push(row); }
    return out; };
  const samples=Math.max(1, data.length-CNN_T);
  for(let ep=0;ep<CNN_EPOCHS;ep++){
    for(let s=0;s<samples;s++){
      const win=data.slice(s,s+CNN_T);
      const y=data[s+CNN_T] || data[s+CNN_T-1];
      const co=convOut(win);
      const flat=[]; for(let t=0;t<co.length;t++) for(let c=0;c<CNN_C;c++) flat.push(co[t][c]);
      const out=new Array(49);
      for(let k=0;k<49;k++){ let acc=bfc[k]; for(let f=0;f<flatN;f++) acc+=Wfc[k][f]*flat[f]; out[k]=sigmoid(acc); }
      const dflat=new Array(flatN).fill(0);
      for(let k=0;k<49;k++){ const g=out[k]-y[k];
        for(let f=0;f<flatN;f++){ dflat[f]+=g*Wfc[k][f]; Wfc[k][f]-=lr*clip(g); }
        bfc[k]-=lr*clip(g); }
      let fi=0;
      for(let t=0;t<co.length;t++) for(let c=0;c<CNN_C;c++){
        const g=dflat[fi++]*(co[t][c]>0?1:0);
        if(g!==0){ bconv[c]-=lr*clip(g);
          for(let k=0;k<CNN_K;k++) for(let d=0;d<49;d++) conv[c][k][d]-=lr*clip(g*win[t+k][d]); } }
    }
  }
  return {conv,bconv,Wfc,bfc};
}
function cnnFwd(m, hist){
  const data=hist.slice(-CNN_T).map(feats49);
  const win=data.length<CNN_T ? [...new Array(CNN_T-data.length).fill(null).map(()=>new Float64Array(49)), ...data] : data;
  const co=[];
  for(let t=0;t<=CNN_T-CNN_K;t++){ const row=new Array(CNN_C).fill(0);
    for(let c=0;c<CNN_C;c++){ let s=m.bconv[c];
      for(let k=0;k<CNN_K;k++) for(let d=0;d<49;d++) s+=m.conv[c][k][d]*win[t+k][d];
      row[c]=Math.max(0,s); }
    co.push(row); }
  const flat=[]; for(let t=0;t<co.length;t++) for(let c=0;c<CNN_C;c++) flat.push(co[t][c]);
  const sc=new Array(49);
  for(let k=0;k<49;k++){ let acc=m.bfc[k]; for(let f=0;f<flat.length;f++) acc+=m.Wfc[k][f]*flat[f]; sc[k]=sigmoid(acc); }
  return sc;
}

// ================= 5) 卡尔曼滤波 / 状态空间 =================
// 局部水平模型（随机游走状态空间）：x_t = x_{t-1}+w, y_t = x_t+v，逐号贝叶斯平滑
function algoKalman(hist, W=60, Q=0.01, R=0.06){
  const win = hist.slice(-W);
  const sc = new Array(49);
  for (let n=1;n<=49;n++){
    let x = 0.5, P = 1;
    for (const d of win){
      const y = balls(d).includes(n) ? 1 : 0;
      const xPred = x, pPred = P + Q;
      const K = pPred/(pPred+R);
      x = xPred + K*(y-xPred);
      P = (1-K)*pPred;
    }
    sc[n-1] = x;   // 一步预测（状态水平）
  }
  return sc;
}

// ================= 7) LightGBM / GBDT =================
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
function lgbmTrain(hist){
  const win = hist.slice(-60);
  const feats=[], labels=[];
  for(let t=1;t<win.length;t++){
    const past = hist.slice(0, hist.length - win.length + t);
    if(past.length<20){ feats.push(null); continue; }
    const p10=past.slice(-10), p30=past.slice(-30);
    const f1=new Array(49).fill(0), f3=new Array(49).fill(0), mis=new Array(49).fill(0);
    for(const d of p10) for(const n of balls(d)) f1[n-1]++;
    for(const d of p30) for(const n of balls(d)) f3[n-1]++;
    for(let n=1;n<=49;n++){ let k=p30.length-1; while(k>=0 && !balls(p30[k]).includes(n)) k--; mis[n-1]=k<0?p30.length:p30.length-1-k; }
    const prev=new Set(balls(past[past.length-1]));
    const ft=[]; for(let n=1;n<=49;n++) ft.push([f1[n-1]/10, f3[n-1]/30, mis[n-1]/30, prev.has(n)?1:0]);
    feats.push(ft);
    const y=new Array(49).fill(0); for(const n of balls(win[t])) y[n-1]=1;
    labels.push(y);
  }
  const p10=hist.slice(-10), p30=hist.slice(-30);
  const f1=new Array(49).fill(0), f3=new Array(49).fill(0), mis=new Array(49).fill(0);
  for(const d of p10) for(const n of balls(d)) f1[n-1]++;
  for(const d of p30) for(const n of balls(d)) f3[n-1]++;
  for(let n=1;n<=49;n++){ let k=p30.length-1; while(k>=0 && !balls(p30[k]).includes(n)) k--; mis[n-1]=k<0?p30.length:p30.length-1-k; }
  const prev=new Set(balls(hist[hist.length-1]));
  const latest=[]; for(let n=1;n<=49;n++) latest.push([f1[n-1]/10, f3[n-1]/30, mis[n-1]/30, prev.has(n)?1:0]);
  const models=[];
  for(let n=0;n<49;n++){
    const X=[],Y=[];
    for(let t=1;t<win.length;t++){ if(feats[t]){ X.push(feats[t][n]); Y.push(labels[t][n]); } }
    if(X.length<5){ models.push(null); continue; }
    let p=new Array(X.length).fill(0.2);
    const trees=[];
    for(let it=0;it<15;it++){
      const g=new Array(X.length), h=new Array(X.length);
      for(let i=0;i<X.length;i++){ g[i]=p[i]-Y[i]; h[i]=p[i]*(1-p[i])+1e-6; }
      const tree=gbdtBuildTree(X,g,h,3,[...Array(X.length).keys()]);
      trees.push(tree);
      for(let i=0;i<X.length;i++){ const st=gbdtPredictTree(tree,X[i]); p[i]=clamp(p[i]+0.1*st,0.05,0.95); }
    }
    models.push({trees, n});
  }
  return {models, latest};
}
function lgbmFwd(m, hist){
  const sc = new Array(49).fill(0.2);
  // 用最新特征
  const p10=hist.slice(-10), p30=hist.slice(-30);
  const f1=new Array(49).fill(0), f3=new Array(49).fill(0), mis=new Array(49).fill(0);
  for(const d of p10) for(const n of balls(d)) f1[n-1]++;
  for(const d of p30) for(const n of balls(d)) f3[n-1]++;
  for(let n=1;n<=49;n++){ let k=p30.length-1; while(k>=0 && !balls(p30[k]).includes(n)) k--; mis[n-1]=k<0?p30.length:p30.length-1-k; }
  const prev=new Set(balls(hist[hist.length-1]));
  for(let n=0;n<49;n++){
    if(!m.models[n]) continue;
    const x=[f1[n]/10, f3[n]/30, mis[n]/30, prev.has(n+1)?1:0];
    let v=0.2;
    for(const tr of m.models[n].trees) v+=0.1*gbdtPredictTree(tr,x);
    sc[n]=clamp(v,0.05,0.95);
  }
  return sc;
}

// ================= 8) N-HiTS（多尺度层级线性块 + 残差，DLinear升级版） =================
// 核心思想：多分辨率分块 × 每块独立线性投影 + 残差连接 + 加性聚合（层级时间序列）
function algoNHiTS(hist, W=90){
  const win = hist.slice(-W);
  const sc = new Array(49);
  // 三个尺度（block长度）：短-中-长，捕捉不同周期的线性趋势
  const BLOCKS = [8, 16, 32];
  for (let n=1;n<=49;n++){
    const y = win.map(d => balls(d).includes(n) ? 1 : 0);
    let agg = 0, wsum = 0;
    for (const B of BLOCKS){
      if (y.length < B+2) continue;
      const seg = y.slice(-B);
      // 块内线性外推（最小二乘 y_t = a*t + b，t=0..B-1）
      const m = seg.length;
      let St=0,Sy=0,Stt=0,Sty=0;
      for (let i=0;i<m;i++){ St+=i; Sy+=seg[i]; Stt+=i*i; Sty+=i*seg[i]; }
      const den = m*Stt - St*St;
      const a = Math.abs(den)>1e-9 ? (m*Sty - St*Sy)/den : 0;
      const b = (Sy - a*St)/m;
      const linPred = a*m + b;                     // 线性外推值（t=m）
      const pool = seg.reduce((s,v)=>s+v,0)/m;      // 块内池化（残差基线）
      // N-HiTS 的残差结构：外推增量 + 块内均值基线
      agg += 0.5*pool + 0.5*linPred;
      wsum += 1;
    }
    sc[n-1] = wsum ? agg/wsum : 0.5;
  }
  return sc;
}

// ================= 10) CatBoost（对称树 oblivious boosting + 因果目标统计特征） =================
function catBuildTree(X, g, h, depth, idx){
  // 对称树：每一层所有节点使用同一特征、同一阈值（CatBoost 的 oblivious tree）
  function build(idxs, d){
    const node = {};
    if (d>=depth || idxs.length < 8){
      let s=0; for(const i of idxs) s += g[i]/Math.max(h[i],1e-6);
      node.leaf = s/idxs.length; return node;
    }
    let bestF=-1, bestThr=0, bestGain=-Infinity;
    for (let f=0; f<X[0].length; f++){
      const vals=[...new Set(idxs.map(i=>X[i][f]))].sort((a,b)=>a-b);
      for (let vi=0; vi<vals.length-1; vi++){
        const thr=(vals[vi]+vals[vi+1])/2;
        const l=idxs.filter(i=>X[i][f]<=thr), r=idxs.filter(i=>X[i][f]>thr);
        if (l.length<4 || r.length<4) continue;
        const gl=l.reduce((s,i)=>s+g[i],0), gr=r.reduce((s,i)=>s+g[i],0);
        const hl=l.reduce((s,i)=>s+Math.max(h[i],1e-6),0), hr=r.reduce((s,i)=>s+Math.max(h[i],1e-6),0);
        const gain = gl*gl/hl + gr*gr/hr;
        if (gain>bestGain){ bestGain=gain; bestF=f; bestThr=thr; }
      }
    }
    if (bestF<0){ let s=0; for(const i of idxs) s+=g[i]/Math.max(h[i],1e-6); node.leaf=s/idxs.length; return node; }
    node.f=bestF; node.thr=bestThr;
    const L=idxs.filter(i=>X[i][bestF]<=bestThr), R=idxs.filter(i=>X[i][bestF]>bestThr);
    node.l=build(L,d+1); node.r=build(R,d+1);
    return node;
  }
  return build(idx,0);
}
function catPredict(node,x){ while(node.leaf===undefined){ node = x[node.f] <= node.thr ? node.l : node.r; } return node.leaf; }
// 因果特征（严格只用第 t 期之前的数据，防泄漏）：CatBoost ordered statistics 的思想
function catFeatures(past){
  const p10=past.slice(-10), p30=past.slice(-30), p50=past.slice(-50);
  const f1=new Array(49).fill(0), f3=new Array(49).fill(0), f5=new Array(49).fill(0), mis=new Array(49).fill(0);
  for(const d of p10) for(const n of balls(d)) f1[n-1]++;
  for(const d of p30) for(const n of balls(d)) f3[n-1]++;
  for(const d of p50) for(const n of balls(d)) f5[n-1]++;
  for(let n=1;n<=49;n++){ let k=p30.length-1; while(k>=0 && !balls(p30[k]).includes(n)) k--; mis[n-1]= k<0 ? p30.length : p30.length-1-k; }
  const prevSet=new Set(balls(past[past.length-1]));
  const ft=[];
  for(let n=1;n<=49;n++) ft.push([f1[n-1]/10, f3[n-1]/30, f5[n-1]/50, mis[n-1]/30, prevSet.has(n)?1:0, past.length>1 ? (balls(past[past.length-2]).includes(n)?1:0) : 0]);
  return ft;
}
function catTrain(hist){
  const win = hist.slice(-80);
  const feats=[], labels=[];
  for(let t=1;t<win.length;t++){
    const past = hist.slice(0, hist.length - win.length + t);
    if (past.length < 25){ feats.push(null); continue; }
    feats.push(catFeatures(past));
    const y=new Array(49).fill(0); for(const n of balls(win[t])) y[n-1]=1;
    labels.push(y);
  }
  const latest = catFeatures(hist);
  const models=[];
  for(let n=0;n<49;n++){
    const X=[],Y=[];
    for(let t=1;t<win.length;t++){ if(feats[t]){ X.push(feats[t][n]); Y.push(labels[t][n]); } }
    if (X.length < 10){ models.push(null); continue; }
    let p=new Array(X.length).fill(0.25);
    const trees=[];
    for(let it=0; it<20; it++){
      const g=new Array(X.length), h=new Array(X.length);
      for(let i=0;i<X.length;i++){ g[i]=p[i]-Y[i]; h[i]=p[i]*(1-p[i])+1e-6; }
      const tree=catBuildTree(X,g,h,4,[...Array(X.length).keys()]);
      trees.push(tree);
      for(let i=0;i<X.length;i++){ p[i]=clamp(p[i]+0.12*catPredict(tree,X[i]), 0.03, 0.97); }
    }
    models.push({trees});
  }
  return {models, latest};
}
function catFwd(m, hist){
  const sc = new Array(49).fill(0.25);
  const feat = catFeatures(hist);   // 用最新历史重算因果特征（与训练口径一致）
  for(let n=0;n<49;n++){
    if(!m.models[n]) continue;
    let v=0.25;
    for(const tr of m.models[n].trees) v += 0.12*catPredict(tr, feat[n]);
    sc[n] = clamp(v, 0.03, 0.97);
  }
  return sc;
}

// ================= 优化类共用：生肖权重向量 =================
// 目标：找 12 维生肖权重 w，使「近W期用 w 加权基础分取 top1 生肖」的命中数最大
function baseZodiacScores(hist, W=30){
  const win = hist.slice(-W);
  const c = new Array(12).fill(0);
  for (const d of win) for (const n of balls(d)) c[zOf(n)]++;
  const tot = c.reduce((a,b)=>a+b,0)||1;
  return c.map(v => v/tot);
}
function evalZodiacWeights(w, hist, base, W=30){
  const win = hist.slice(-W);
  let hits=0, n=0;
  for (let i=0;i<win.length;i++){
    const t = hist.length - win.length + i;   // win[i] 对应 hist[t]
    if (t<=0) continue;
    const seven = balls(hist[t]);
    let bz=0, bv=-Infinity;
    for (let z=0;z<12;z++){ const v = w[z]*base[z]; if (v>bv){ bv=v; bz=z; } }
    const hit = seven.some(x => zOf(x)===bz) ? 1 : 0;
    hits+=hit; n++;
  }
  return n ? hits/n : 0;
}
function weightsToScores(w, base){
  const sc = new Array(49);
  for (let n=1;n<=49;n++) sc[n-1] = w[zOf(n)] * base[zOf(n)];
  return sc;
}

// 8) 遗传算法 GA
function algoGA(hist, W=30, POP=30, GEN=15){
  const base = baseZodiacScores(hist, W);
  const evalW = w => evalZodiacWeights(w, hist, base, W);
  let pop = Array.from({length:POP},()=>Array.from({length:12},()=>rng()));
  let fit = pop.map(evalW);
  for (let g=0; g<GEN; g++){
    const idx = pop.map((_,i)=>i).sort((a,b)=>fit[b]-fit[a]);
    const next=[];
    for (let i=0;i<POP;i+=2){
      const pa = pop[idx[Math.floor(rng()*Math.max(1,POP/2))]];
      const pb = pop[idx[Math.floor(rng()*Math.max(1,POP/2))]];
      const child = new Array(12);
      for (let k=0;k<12;k++){
        child[k] = rng()<0.5 ? pa[k] : pb[k];
        if (rng()<0.15) child[k] = rng();
      }
      next.push(child);
      if (next.length<POP) next.push(pop[idx[i%POP]].slice());
    }
    pop = next; fit = pop.map(evalW);
  }
  let bi=0; for (let i=1;i<POP;i++) if (fit[i]>fit[bi]) bi=i;
  return weightsToScores(pop[bi], base);
}

// 9) 粒子群 PSO
function algoPSO(hist, W=30, NP=24, IT=18){
  const base = baseZodiacScores(hist, W);
  const evalW = w => evalZodiacWeights(w, hist, base, W);
  const pos = Array.from({length:NP},()=>Array.from({length:12},()=>rng()));
  const vel = pos.map(()=>Array.from({length:12},()=> (rng()-0.5)*0.4));
  const fit = pos.map(evalW);
  const pBest = pos.map(p=>p.slice()), pFit = fit.slice();
  let gBest = pos[0].slice(), gFit = fit[0];
  for (let i=1;i<NP;i++) if (fit[i]>gFit){ gFit=fit[i]; gBest=pos[i].slice(); }
  for (let it=0; it<IT; it++){
    for (let i=0;i<NP;i++){
      for (let d=0; d<12; d++){
        vel[i][d] = 0.7*vel[i][d] + 1.5*rng()*(pBest[i][d]-pos[i][d]) + 1.7*rng()*(gBest[d]-pos[i][d]);
        pos[i][d] = clamp(pos[i][d] + vel[i][d], 0, 1);
      }
      const f = evalW(pos[i]);
      if (f>pFit[i]){ pFit[i]=f; pBest[i]=pos[i].slice(); }
      if (f>gFit){ gFit=f; gBest=pos[i].slice(); }
    }
  }
  return weightsToScores(gBest, base);
}

// ================= 模型注册表（v7：精简至9个） =================
// 保留理由：TCN/N-HiTS/LightGBM/CatBoost 权重>10%（实测有效且互补）；卡尔曼/PSO/GA 权重5-9%（思路独特）；
//          LSTM/CNN 权重低但保留（深度学习代表，且有"起死回生"机制）。
// 已删除：DLinear(被N-HiTS取代) / Transformer(停权) / SSA(停权) / 贝叶斯优化(停权) / 模拟退火+蚁群(与PSO/GA同质)
// 删除理由：停权模型白烧算力不投票；同质模型抄多遍稀释共识——模型越多共识越难达成(12个58.3%→15个46.7%)，影响实际下注。
const MODEL_KEYS = ['nhits','lstm','cnn','kalman','lgbm','catboost','ga','pso'];
// 慢模型：预训练一次 + 回测期只前向
const SLOW_KEYS = ['lstm','cnn','lgbm','catboost'];
const SLOW_TRAIN = { lstm: lstmTrain, cnn: cnnTrain, lgbm: lgbmTrain, catboost: catTrain };
const SLOW_FWD   = { lstm: lstmFwd, cnn: cnnFwd, lgbm: lgbmFwd, catboost: catFwd };
// 轻量/优化类：每期直接算（解析计算 + 优化迭代）
const FAST_FNS = { nhits: algoNHiTS, kalman: algoKalman, ga: algoGA, pso: algoPSO };

// N-HiTS 无需训练（纯解析计算），单独处理
function modelScore(key, hist, model){
  let sc;
  if (key==='nhits') sc = algoNHiTS(hist);
  else if (SLOW_FWD[key]) sc = SLOW_FWD[key](model, hist);
  else sc = FAST_FNS[key](hist);
  // 防御：NaN/Inf 隔离（曾因 TCN 层间维度错配导致整条融合链 NaN，23码退化成1-23）
  for (let i=0;i<sc.length;i++) if (!isFinite(sc[i])) sc[i] = 0.5;
  return sc;
}
function modelTrain(key, hist){
  if (key==='nhits') return null;
  return SLOW_TRAIN[key](hist);
}

// ================= Walk-forward 回测优化权重 =================
const WF_BASELINE = 0.4786;
function walkForwardWeights(hist, w=60){
  const hist26 = hist.filter(d => d.period >= '2026001');
  const bt = hist26.slice(-w);
  const hits={}; MODEL_KEYS.forEach(n=>hits[n]=[0,0]);
  if (bt.length >= 10) {
    const cutoff = bt[0].period;
    const models={};
    for(const n of SLOW_KEYS){
      const sub=hist.filter(d => d.period < cutoff);
      if(sub.length>=60) models[n]=modelTrain(n, sub);
    }
    const histIdx=new Map(hist.map((d,i)=>[d.period,i]));
    for(const d of bt){
      const h=hist.slice(0, histIdx.get(d.period));
      const seven=balls(d);
      for(const n of MODEL_KEYS){
        const sc = modelScore(n, h, models[n]);
        const tz=top1Z(sc);
        const hit=seven.some(x=>zOf(x)===tz)?1:0;
        hits[n][1]++; hits[n][0]+=hit;
      }
    }
  }
  const W={}; let anyPos=false;
  for(const n of MODEL_KEYS){
    const rate=hits[n][1] ? hits[n][0]/hits[n][1] : WF_BASELINE;
    const wt=Math.max(0, rate-WF_BASELINE);
    W[n]=wt; if(wt>0) anyPos=true;
  }
  if(!anyPos) MODEL_KEYS.forEach(n=>W[n]=1);
  return W;
}

// ================= 共识投票层 =================
// 12模型主推生肖投票：一致度≥阈值 → 正常下注；分歧大 → 跳过不下注（保护本金）
const CONSENSUS_THRESHOLD = 0.6;
function consensusOf(details){
  const votes={};
  const keys=Object.keys(details);
  for(const k of keys){ const z=details[k].zodiac; votes[z]=(votes[z]||0)+1; }
  const total=keys.length;
  const sorted=Object.entries(votes).sort((a,b)=>b[1]-a[1]);
  const top = sorted[0] || [null,0];
  const agree = total ? top[1]/total : 0;
  const action = agree >= CONSENSUS_THRESHOLD ? 'full' : 'skip';
  return { agree, votes, top: top[0], count: top[1], total, action, threshold: CONSENSUS_THRESHOLD };
}

// ================= Ensemble 主入口 =================
function predictAll(hist){
  rngReset(20261019);
  // 1) 各模型出分
  const models={};
  for(const n of SLOW_KEYS) models[n]=modelTrain(n, hist);
  const raw={};
  for(const n of MODEL_KEYS) raw[n]=modelScore(n, hist, models[n]);
  // 2) Walk-forward 权重
  const W=walkForwardWeights(hist, 200);   // v8：窗口60→200期，修短视（60期让TCN/LCG这类短期运气模型混上高权重）
  // 3) Ensemble 融合
  const fused=new Float64Array(49);
  let wsum=0;
  for(const k of MODEL_KEYS){
    const s=normalize(raw[k]);
    for(let n=0;n<49;n++) fused[n]+=W[k]*s[n];
    wsum+=W[k];
  }
  for(let n=0;n<49;n++) fused[n]/=(wsum||1);
  // 4) 明细 + 共识投票
  const details={};
  for(const k of MODEL_KEYS) details[k]=derive(normalize(raw[k]));
  const consensus = consensusOf(details);
  // 4.5) 融合后的12生肖分数全景（网页展示：让"为什么是这个生肖"看得见）
  const zodiacScores = [];
  for(let z=0;z<12;z++) zodiacScores.push(0);
  const cnt = new Array(12).fill(0);
  for(let n=1;n<=49;n++){ const z=zOf(n); zodiacScores[z]+=fused[n-1]; cnt[z]++; }
  for(let z=0;z<12;z++) zodiacScores[z]/=(cnt[z]||1);
  return { fused: derive(fused), details, weights: W, consensus, zodiacScores };
}

// ================= 多属性命中判定 =================
function judgeDraw(pred, draw){
  const codeHit = pred.codes ? ((pred.codes.includes(draw.special))?1:0) : null;
  const zodiacHit = ([...draw.numbers, draw.special].some(n=>zOf(n)===ZI[pred.zodiac]))?1:0;
  const sixHit = pred.six ? ((pred.six.some(z=>z===ZODIACS[zOf(draw.special)]))?1:0) : null;
  return { codeHit, zodiacHit, sixHit };
}

module.exports = { predictAll, derive, judgeDraw, zOf, ZODIACS, MODEL_KEYS, consensusOf, CONSENSUS_THRESHOLD };
