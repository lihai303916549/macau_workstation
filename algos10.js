'use strict';
/**
 * 六子 —— 预测引擎 v6（2026-10-21 用户定版：12 模型 + 共识投票层）
 * ==================== 模型清单（12个，全部基于历史数据学习）====================
 * 【深度学习 / 现代时序 4 个】
 *   1) DLinear          —— 趋势分解(移动平均) + 残差AR(1)线性回归，线性时序、抗噪
 *   2) LSTM             —— 标准i/f/o/g门，隐层16，滑窗20步BPTT
 *   3) Transformer      —— 单头自注意力 d=16 window=12 + FFN32，正弦位置编码
 *   4) CNN              —— T=20×49时序图，2个kernel=3一维卷积 + FC49
 * 【统计 / 机器学习 3 个】
 *   5) 卡尔曼滤波/状态空间 —— 局部水平模型(随机游走)，逐号贝叶斯平滑+一步外推
 *   6) SSA 奇异谱分析   —— 轨迹矩阵+power iteration主成分+线性外推（对特码序列）
 *   7) LightGBM/GBDT    —— 深度3的提升树×20，每号独立模型（4维特征）
 * 【优化 / 进化 5 个】（都优化"生肖权重向量"，适应度=近30期平特肖命中）
 *   8)  遗传算法 GA     —— 锦标赛选择+交叉+变异
 *   9)  粒子群 PSO      —— 速度/位置更新
 *  10) 模拟退火 SA      —— 邻域扰动+退火
 *  11) 蚁群 ACO         —— 信息素轮盘+挥发
 *  12) 贝叶斯优化 BO    —— 高斯过程代理 + 期望改进(EI) + 随机投影降维(12D→4D)
 * 【框架层】
 *   1) Ensemble 动态融合 —— 12模型归一化后按 Walk-forward 超额权重加权融合
 *   2) Walk-forward 优化 —— 近60期影子回测，命中率超基线(0.4786)=当天话语权
 *   3) 多属性命中判定    —— judgeDraw：23码/平特肖/6肖三层判定
 *   4) 共识投票层(新)    —— 12模型主推生肖投票；一致度≥60% 正常下注，分歧大则【跳过不下注】
 *
 * ⚠️ 成本控制：慢模型(rnn→dlinear/lstm/transformer/cnn/lgbm)预训练到回测起点后只前向；
 *   优化类(ga/pso/sa/aco/bo)每期都要重跑（种群30/代15/回测30期），单次≈50ms。
 * 训练范围：全量历史（最早一期到最新一期）。固定种子20261019 → 同数据必得同预测。
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

// ================= 1) DLinear（趋势分解 + 残差AR(1)） =================
function algoDLinear(hist, W=60, K=25){
  const win = hist.slice(-W);
  const sc = new Array(49);
  for (let n=1;n<=49;n++){
    const y = win.map(d => balls(d).includes(n) ? 1 : 0);
    // 趋势：移动平均核 K
    const trend = y.map((_, t) => {
      let s=0,c=0; for(let k=Math.max(0,t-K+1);k<=t;k++){ s+=y[k]; c++; } return s/c;
    });
    // 残差 AR(1)：r_t = a*r_{t-1} + b（最小二乘）
    const r = y.map((v,t)=>v-trend[t]);
    let Sxy=0,Sxx=0,Sx=0,Sy=0,n2=0;
    for(let t=1;t<r.length;t++){ Sxy+=r[t]*r[t-1]; Sxx+=r[t-1]*r[t-1]; Sx+=r[t-1]; Sy+=r[t]; n2++; }
    const den = n2*Sxx - Sx*Sx;
    const a = Math.abs(den)>1e-9 ? (n2*Sxy - Sx*Sy)/den : 0;
    const b = n2>0 ? (Sy - a*Sx)/n2 : 0;
    // 趋势线性外推一步
    const tLast=trend[trend.length-1], tPrev=trend[trend.length-2]||tLast;
    const trendNext = tLast + (tLast - tPrev);
    const rNext = a*r[r.length-1] + b;
    sc[n-1] = trendNext + rNext;
  }
  return sc;
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

// ================= 3) Transformer =================
const TR_D = 16, TR_WIN = 12, TR_EPOCHS = 3;
function trTrain(hist){
  const data=hist.map(feats49);
  const Emb=Array.from({length:49},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const Wq=Array.from({length:TR_D},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const Wk=Array.from({length:TR_D},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const Wv=Array.from({length:TR_D},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const Wo=Array.from({length:TR_D},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const Wf1=Array.from({length:32},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const Wf2=Array.from({length:TR_D},()=>Array.from({length:32},()=>randn()*0.2));
  const Wy=Array.from({length:49},()=>Array.from({length:TR_D},()=>randn()*0.2));
  const by=new Array(49).fill(0);
  const lr=0.05;
  const posEnc=(pos)=>{ const v=new Array(TR_D); for(let i=0;i<TR_D;i++) v[i]=(i%2===0?Math.sin(pos/Math.pow(10000,i/TR_D)):Math.cos(pos/Math.pow(10000,(i-1)/TR_D))); return v; };
  const samples=Math.max(1, data.length-TR_WIN);
  for(let ep=0;ep<TR_EPOCHS;ep++){
    for(let s=0;s<samples;s+=4){
      const win=data.slice(s,s+TR_WIN);
      const y=data[s+TR_WIN] || data[s+TR_WIN-1];
      const E=win.map((x,pos)=>{ const e=new Array(TR_D); const p=posEnc(pos);
        for(let d=0;d<TR_D;d++){ let s2=p[d]; for(let k=0;k<49;k++) s2+=Emb[k][d]*x[k]; e[d]=s2; } return e; });
      const sc=new Array(TR_WIN);
      for(let i=0;i<TR_WIN;i++){
        const q=new Array(TR_D), kk=new Array(TR_D);
        for(let d=0;d<TR_D;d++){ let aq=0,ak=0; for(let e=0;e<TR_D;e++){ aq+=Wq[d][e]*E[i][e]; ak+=Wk[d][e]*E[i][e]; } q[d]=aq; kk[d]=ak; }
        for(let j=0;j<TR_WIN;j++){ let s2=0; for(let d=0;d<TR_D;d++) s2+=q[d]*kk[d]; sc[j]=s2/Math.sqrt(TR_D); }
        const mx=Math.max(...sc); const ex=sc.map(v=>Math.exp(v-mx)); const sum=ex.reduce((a,b)=>a+b,0)||1;
        const O=new Array(TR_D).fill(0);
        for(let j=0;j<TR_WIN;j++){ const a=ex[j]/sum;
          for(let d=0;d<TR_D;d++){ let vv=0; for(let e=0;e<TR_D;e++) vv+=Wv[d][e]*E[j][e]; O[d]+=a*vv; } }
        for(let d=0;d<TR_D;d++){ let o=0; for(let e=0;e<TR_D;e++) o+=Wo[d][e]*O[d]; E[i][d]=E[i][d]+o; }
      }
      const F=E.map(o=>{ const h=new Array(32); for(let j=0;j<32;j++){ let s2=0; for(let d=0;d<TR_D;d++) s2+=Wf1[j][d]*o[d]; h[j]=Math.max(0,s2); }
        const f=new Array(TR_D); for(let d=0;d<TR_D;d++){ let s2=0; for(let j=0;j<32;j++) s2+=Wf2[d][j]*h[j]; f[d]=s2; } return f; });
      const last=E[TR_WIN-1].map((v,d)=>v+F[TR_WIN-1][d]);
      const out=new Array(49);
      for(let k=0;k<49;k++){ let s2=by[k]; for(let d=0;d<TR_D;d++) s2+=Wy[k][d]*last[d]; out[k]=sigmoid(s2); }
      for(let k=0;k<49;k++){ const g=out[k]-y[k];
        for(let d=0;d<TR_D;d++) Wy[k][d]-=lr*clip(g);
        by[k]-=lr*clip(g); }
    }
  }
  return {Emb,Wq,Wk,Wv,Wo,Wf1,Wf2,Wy,by,posEnc};
}
function trFwd(m, hist){
  const data=hist.slice(-TR_WIN).map(feats49);
  const win=data.length<TR_WIN ? [...new Array(TR_WIN-data.length).fill(null).map(()=>new Float64Array(49)), ...data] : data;
  const E=win.map((x,pos)=>{ const e=new Array(TR_D); const p=m.posEnc(pos);
    for(let d=0;d<TR_D;d++){ let s=p[d]; for(let k=0;k<49;k++) s+=m.Emb[k][d]*x[k]; e[d]=s; } return e; });
  for(let i=0;i<TR_WIN;i++){
    const q=new Array(TR_D), kk=new Array(TR_D);
    for(let d=0;d<TR_D;d++){ let aq=0,ak=0; for(let e=0;e<TR_D;e++){ aq+=m.Wq[d][e]*E[i][e]; ak+=m.Wk[d][e]*E[i][e]; } q[d]=aq; kk[d]=ak; }
    const sc=new Array(TR_WIN);
    for(let j=0;j<TR_WIN;j++){ let s=0; for(let d=0;d<TR_D;d++) s+=q[d]*kk[d]; sc[j]=s/Math.sqrt(TR_D); }
    const mx=Math.max(...sc); const ex=sc.map(v=>Math.exp(v-mx)); const sum=ex.reduce((a,b)=>a+b,0)||1;
    for(let j=0;j<TR_WIN;j++){ const a=ex[j]/sum;
      for(let d=0;d<TR_D;d++){ let vv=0; for(let e=0;e<TR_D;e++) vv+=m.Wv[d][e]*E[j][e]; E[i][d]+=a*vv; } }
  }
  const F=E.map(o=>{ const h=new Array(32); for(let j=0;j<32;j++){ let s=0; for(let d=0;d<TR_D;d++) s+=m.Wf1[j][d]*o[d]; h[j]=Math.max(0,s); }
    const f=new Array(TR_D); for(let d=0;d<TR_D;d++){ let s=0; for(let j=0;j<32;j++) s+=m.Wf2[d][j]*h[j]; f[d]=s; } return f; });
  const last=E[TR_WIN-1].map((v,d)=>v+F[TR_WIN-1][d]);
  const sc=new Array(49);
  for(let k=0;k<49;k++){ let s=m.by[k]; for(let d=0;d<TR_D;d++) s+=m.Wy[k][d]*last[d]; sc[k]=sigmoid(s); }
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

// ================= 6) SSA 奇异谱分析 =================
// 轨迹矩阵 → power iteration 求主成分 → 线性外推 → 高斯核（对特码序列）
function algoSSA(hist, W=120, L=30, K=3, sigma=7){
  const y = hist.slice(-W).map(d => d.special);
  const N = y.length;
  const sc = new Array(49).fill(1);
  if (N < L + 5) return sc;
  // 轨迹矩阵的行向量：x_t = (y_t, y_{t+1}, ..., y_{t+L-1})，M = Σ x x^T（L×L）
  const M = Array.from({length:L},()=>new Array(L).fill(0));
  for (let t=0;t+L<=N;t++){
    for (let i=0;i<L;i++) for (let j=0;j<L;j++) M[i][j] += y[t+i]*y[t+j];
  }
  // power iteration 求前 K 个主成分（正交化）
  const comps = [];
  for (let c=0;c<K;c++){
    let v = new Array(L).fill(0);
    for (let i=0;i<L;i++) v[i] = randn();
    let norm = Math.sqrt(v.reduce((a,b)=>a+b*b,0)) || 1;
    v = v.map(x=>x/norm);
    for (let it=0; it<40; it++){
      const w = new Array(L).fill(0);
      for (let i=0;i<L;i++){ let s=0; for(let j=0;j<L;j++) s+=M[i][j]*v[j]; w[i]=s; }
      // 与已得成分正交化
      for (const u of comps){
        let d=0; for(let i=0;i<L;i++) d+=w[i]*u[i];
        for(let i=0;i<L;i++) w[i]-=d*u[i];
      }
      const nw = Math.sqrt(w.reduce((a,b)=>a+b*b,0)) || 1;
      v = w.map(x=>x/nw);
    }
    comps.push(v);
  }
  // 每个主成分：投影序列末段线性外推
  let pred = 0;
  for (const v of comps){
    const z = [];
    for (let t=0;t+L<=N;t++){ let s=0; for(let i=0;i<L;i++) s+=v[i]*y[t+i]; z.push(s); }
    const seg = z.slice(-8);
    // 最小二乘直线外推
    const m = seg.length;
    let Sx=0,Sy=0,Sxy=0,Sxx=0;
    for (let i=0;i<m;i++){ Sx+=i; Sy+=seg[i]; Sxy+=i*seg[i]; Sxx+=i*i; }
    const den = m*Sxx - Sx*Sx;
    const a = Math.abs(den)>1e-9 ? (m*Sxy - Sx*Sy)/den : 0;
    const b = (Sy - a*Sx)/m;
    pred += a*m + b;
  }
  pred = clamp(pred, 1, 49);
  for (let n=1;n<=49;n++) sc[n-1] = Math.exp(-((n-pred)*(n-pred))/(2*sigma*sigma));
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

// 10) 模拟退火 SA
function algoSA(hist, W=30, STEPS=500, T0=0.5){
  const base = baseZodiacScores(hist, W);
  const evalW = w => evalZodiacWeights(w, hist, base, W);
  let cur = Array.from({length:12},()=>rng()), curF = evalW(cur);
  let best = cur.slice(), bestF = curF;
  for (let s=0;s<STEPS;s++){
    const T = T0 * (1 - s/STEPS) + 1e-3;
    const cand = cur.slice();
    const k = 1 + Math.floor(rng()*3);
    for (let i=0;i<k;i++) cand[Math.floor(rng()*12)] = clamp(cand[Math.floor(rng()*12)] + (rng()-0.5)*0.5, 0, 1);
    const f = evalW(cand);
    if (f > curF || rng() < Math.exp((f-curF)/T)){ cur = cand; curF = f; }
    if (f > bestF){ bestF = f; best = cand.slice(); }
  }
  return weightsToScores(best, base);
}

// 11) 蚁群 ACO（信息素轮盘 + 挥发强化）
function algoACO(hist, W=30, ANTS=16, ROUNDS=18){
  const base = baseZodiacScores(hist, W);
  const evalW = w => evalZodiacWeights(w, hist, base, W);
  const tau = new Array(12).fill(1);
  let best=null, bestF=-1;
  for (let r=0;r<ROUNDS;r++){
    let roundBest = new Array(12).fill(0); let rf=-1;
    for (let a=0;a<ANTS;a++){
      // 按信息素轮盘选 3 个重点生肖（权重集中在这些生肖上）
      const chosen = new Set();
      for (let c=0;c<3;c++){
        const tot = tau.reduce((x,y)=>x+y,0)||1;
        let pick = rng()*tot, acc=0, idx=0;
        for (let z=0;z<12;z++){ acc+=tau[z]; if(pick<=acc){ idx=z; break; } }
        chosen.add(idx);
      }
      const w = new Array(12).fill(0.1);
      for (const z of chosen) w[z] = 1;
      const f = evalW(w);
      for (const z of chosen) tau[z] += 0.15 * f;
      if (f>rf){ rf=f; roundBest=w; }
    }
    for (let z=0;z<12;z++) tau[z] = tau[z]*0.85 + 0.05;
    if (rf>bestF){ bestF=rf; best=roundBest; }
  }
  return weightsToScores(best || new Array(12).fill(1), base);
}

// 12) 贝叶斯优化 BO（高斯过程代理 + 期望改进EI + 随机投影 12D→4D）
function gpFit(X, y, ls=1.0, sn=0.1){
  const n = X.length;
  const K = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>
    ls<=0?0:Math.exp(-sqDist(X[i],X[j])/(2*ls*ls)) + (i===j?sn:0)));
  // 高斯消元求 K^{-1}
  const A = K.map((row,i)=>row.slice().concat([y[i]]));
  for (let i=0;i<n;i++){
    let piv=i; for(let r=i+1;r<n;r++) if(Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
    [A[i],A[piv]]=[A[piv],A[i]];
    if(Math.abs(A[i][i])<1e-10) continue;
    for(let r=0;r<n;r++){ if(r===i)continue; const f=A[r][i]/A[i][i]; for(let c=i;c<=n;c++) A[r][c]-=f*A[i][c]; }
  }
  const alpha=new Array(n);
  for(let i=0;i<n;i++) alpha[i]=A[i][n]/(A[i][i]||1e-10);
  const ymean = y.reduce((a,b)=>a+b,0)/(n||1);
  return { X, ymean, alpha, ls, sn };
}
function sqDist(a,b){ let s=0; for(let i=0;i<a.length;i++){ const d=a[i]-b[i]; s+=d*d; } return s; }
function gpPredict(gp, x){
  const n=gp.X.length;
  let mu=gp.ymean, var1=0;
  for(let i=0;i<n;i++){
    const k = gp.ls<=0?0:Math.exp(-sqDist(gp.X[i],x)/(2*gp.ls*gp.ls));
    mu += gp.alpha[i]*k;
    var1 += gp.alpha[i]*k;
  }
  const kxx = gp.ls<=0?0:1 + gp.sn;
  const v = Math.max(1e-8, kxx - var1);
  return { mu, sd: Math.sqrt(v) };
}
function algoBO(hist, W=30, DIM=4, EVALS=20){
  const base = baseZodiacScores(hist, W);
  const evalW = w => evalZodiacWeights(w, hist, base, W);
  // 随机投影：12维 → DIM 维
  const P = Array.from({length:12},()=>Array.from({length:DIM},()=>randn()));
  const proj = w => { const v=new Array(DIM).fill(0);
    for(let i=0;i<12;i++) for(let d=0;d<DIM;d++) v[d]+=w[i]*P[i][d];
    return v; };
  const unproj = z => { const w=new Array(12).fill(0);
    for(let i=0;i<12;i++) for(let d=0;d<DIM;d++) w[i]+=z[d]*P[i][d];
    return w.map(v=>clamp(v,0,1)); };
  const X=[], Y=[];
  for (let i=0;i<6;i++){
    const w = Array.from({length:12},()=>rng());
    X.push(proj(w)); Y.push(evalW(w));
  }
  let gp = gpFit(X, Y, 1.0, 0.05);
  for (let e=0;e<EVALS-6;e++){
    let bestX=null, bestEI=-Infinity;
    for (let t=0;t<200;t++){
      const z = Array.from({length:DIM},()=>rng());
      const { mu, sd } = gpPredict(gp, z);
      const ei = mu + 0.5*sd;   // 期望改进（简化）
      if (ei>bestEI){ bestEI=ei; bestX=z; }
    }
    const w = unproj(bestX);
    X.push(bestX); Y.push(evalW(w));
    gp = gpFit(X, Y, 1.0, 0.05);
  }
  let bi=0; for(let i=1;i<Y.length;i++) if (Y[i]>Y[bi]) bi=i;
  return weightsToScores(unproj(X[bi]), base);
}

// ================= 模型注册表 =================
const MODEL_KEYS = ['dlinear','lstm','transformer','cnn','kalman','ssa','lgbm','ga','pso','sa','aco','bo'];
// 慢模型：预训练一次 + 回测期只前向
const SLOW_KEYS = ['dlinear','lstm','transformer','cnn','lgbm'];
const SLOW_TRAIN = { dlinear: null, lstm: lstmTrain, transformer: trTrain, cnn: cnnTrain, lgbm: lgbmTrain };
const SLOW_FWD   = { dlinear: null, lstm: lstmFwd, transformer: trFwd, cnn: cnnFwd, lgbm: lgbmFwd };
// 轻量/优化类：每期直接算
const FAST_FNS = { kalman: algoKalman, ssa: algoSSA, ga: algoGA, pso: algoPSO, sa: algoSA, aco: algoACO, bo: algoBO };

// DLinear 无需训练（纯解析计算），单独处理
function modelScore(key, hist, model){
  if (key==='dlinear') return algoDLinear(hist);
  if (SLOW_FWD[key]) return SLOW_FWD[key](model, hist);
  return FAST_FNS[key](hist);
}
function modelTrain(key, hist){
  if (key==='dlinear') return null;
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
  const W=walkForwardWeights(hist, 60);
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
  return { fused: derive(fused), details, weights: W, consensus };
}

// ================= 多属性命中判定 =================
function judgeDraw(pred, draw){
  const codeHit = pred.codes ? ((pred.codes.includes(draw.special))?1:0) : null;
  const zodiacHit = ([...draw.numbers, draw.special].some(n=>zOf(n)===ZI[pred.zodiac]))?1:0;
  const sixHit = pred.six ? ((pred.six.some(z=>z===ZODIACS[zOf(draw.special)]))?1:0) : null;
  return { codeHit, zodiacHit, sixHit };
}

module.exports = { predictAll, derive, judgeDraw, zOf, ZODIACS, MODEL_KEYS, consensusOf, CONSENSUS_THRESHOLD };
