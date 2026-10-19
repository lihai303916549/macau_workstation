'use strict';
/**
 * 六子 —— 预测引擎 v5（2026-10-20 用户定版，替换 v4 全部算法）
 * ==================== 模型清单（用户指定5个，旧模型已全部删除）====================
 * A类·号码级打分模型（5个，各输出49维原始分，分高=更可能出）：
 *   1) RNN 循环神经网络      —— 49维7球序列，tanh隐层16，滑窗20步BPTT
 *   2) LSTM 长短期记忆       —— 标准i/f/o/g门，隐层16，滑窗20步BPTT
 *   3) Transformer           —— 单头自注意力 d=16 window=12 + FFN32，正弦位置编码
 *   4) LCG 线性同余          —— x=(a·x+c) mod 2^32（Math.imul保精度），随机派基准尺
 *   5) CNN 卷积神经网络      —— T=20×49时序图，2个kernel=3一维卷积 + FC49
 * D类·框架层（混合架构，保留）：
 *   1) Ensemble 动态融合    —— 5模型归一化后按动态权重加权融合（= 混合架构）
 *   2) Walk-forward 优化    —— 每天预测前对近60期影子回测，命中超额=当天话语权
 *   4) 多属性命中判定       —— judgeDraw：23码/平特肖/6肖三层判定（导出供 update.js 用）
 *
 * ⚠️ 深度学习训练成本控制（关键工程决策）：
 *   慢模型（rnn/lstm/transformer/cnn）在 Walk-forward 回测中不每期重训——
 *   预训练到「回测起点之前的历史」，回测60期只前向；预测当天用全量历史重训一次。
 *   否则 60期×全训练 = 几十分钟，每晚自动化等不起。
 *   训练范围（v5.1 用户要求"最早一期到最新一期全部学习"）：全量历史，不再截断；
 *   epoch/隐层/窗口全部小号化；梯度裁剪clip=5。
 *
 * 聚合除号数（用户决策「消除」）：49维→生肖汇总时除以该生肖号码数（马÷5、其余÷4），
 *   12生肖按平均分公平竞争，消除「马5个号总和自动胜出」的偏置。
 * 动态加权兜底：全模型不及格（超额≤0）退化为等权。
 * 固定种子（20261020）：同数据必得同预测。
 * ==================== 事前预测铁律（用户2026-10-19强调）====================
 * 只做事前预测：每晚对「下一期」预测并存档；开奖后按存档判定记录，严禁事后重算历史。
 */
const STD = {1:'马',2:'蛇',3:'龙',4:'兔',5:'虎',6:'牛',7:'鼠',8:'猪',9:'狗',10:'鸡',11:'猴',12:'羊'};
const ZODIACS = Object.values(STD);
const ZI = {}; ZODIACS.forEach((z,i)=>ZI[z]=i);
function zOf(n){ const k = n<=12 ? n : ((n-1)%12)+1; return ZI[STD[k]]; }

// ---------- 可复现随机数（训练初始化用） ----------
let _rngState = 0;
function rngReset(seed){ _rngState = seed >>> 0; }
function rng(){
  _rngState = (_rngState + 0x6D2B79F5) >>> 0;
  let t = _rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function randn(){ // Box-Muller 标准正态
  let u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
const clip = (g, C=5) => Math.max(-C, Math.min(C, g));
const sigmoid = x => 1/(1+Math.exp(-x));
const TANH = Math.tanh;

function normalize(sc){
  let mn=Infinity,mx=-Infinity;
  for(const v of sc){ if(v<mn)mn=v; if(v>mx)mx=v; }
  const r=(mx-mn)||1, o=new Float64Array(49);
  for(let i=0;i<49;i++) o[i]=(sc[i]-mn)/r;
  return o;
}
function balls(d){ return [...d.numbers, d.special]; }
// 每期49维0/1特征（7球位置=1）
function feats49(d){ const x=new Float64Array(49); for(const n of balls(d)) x[n-1]=1; return x; }

// ================= 4) LCG（随机派基准尺，确定性） =================
function algoLCG(){
  let x = 20261020 >>> 0;
  const a = 1664525, c = 1013904223;
  const sc = new Array(49);
  for (let n = 0; n < 49; n++) {
    x = (Math.imul(a, x) + c) >>> 0;
    sc[n] = x / 4294967296;
  }
  return sc;
}

// ================= 1) RNN 循环神经网络 =================
// h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)；o = sigmoid(Who·h_t + bo)
const RNN_H = 16, RNN_WIN = 20, RNN_EPOCHS = 5;
function rnnTrain(hist){
  const data = hist; // v5.1：全量历史学习（最早一期到最新一期，用户2026-10-20要求）
  const Wxh = Array.from({length:RNN_H},()=>Array.from({length:49},()=>randn()*0.1));
  const Whh = Array.from({length:RNN_H},()=>Array.from({length:RNN_H},()=>randn()*0.1));
  const bh  = new Array(RNN_H).fill(0);
  const Who = Array.from({length:49},()=>Array.from({length:RNN_H},()=>randn()*0.1));
  const bo  = new Array(49).fill(0);
  const lr = 0.1;
  const X = data.map(feats49), Y = data.slice(1).map(feats49);
  const samples = Math.max(1, X.length - RNN_WIN);
  for (let ep=0; ep<RNN_EPOCHS; ep++){
    for (let s=0; s<samples; s++){
      // 前向：窗口内 RNN_WIN 步
      const hs=[new Array(RNN_H).fill(0)], os=[], xs=[];
      for (let t=0;t<RNN_WIN;t++){
        const x=X[s+t]; xs.push(x);
        const h=new Array(RNN_H);
        for(let j=0;j<RNN_H;j++){ let acc=bh[j]; for(let k=0;k<49;k++) acc+=Wxh[j][k]*x[k];
          for(let k=0;k<RNN_H;k++) acc+=Whh[j][k]*hs[t][k]; h[j]=TANH(acc); }
        hs.push(h);
        const o=new Array(49);
        for(let k=0;k<49;k++){ let acc=bo[k]; for(let j=0;j<RNN_H;j++) acc+=Who[k][j]*h[j]; o[k]=sigmoid(acc); }
        os.push(o);
      }
      // 反向（BPTT，只对最后一步输出算损失）
      const y=Y[s+RNN_WIN-1];
      const dO=new Array(RNN_WIN).fill(null); dO[RNN_WIN-1]=new Array(49);
      for(let k=0;k<49;k++) dO[RNN_WIN-1][k]=os[RNN_WIN-1][k]-y[k];
      let dhNext=new Array(RNN_H).fill(0);
      for(let t=RNN_WIN-1;t>=0;t--){
        if(!dO[t]) continue;
        const dW=new Array(49).fill(0), dbo=new Array(49).fill(0);
        for(let k=0;k<49;k++){
          const g=dO[t][k]*os[t][k]*(1-os[t][k]); dbo[k]=g;
          for(let j=0;j<RNN_H;j++){ dW[k*0+j]=0; }
        }
        // 逐隐单元反传
        const dh=new Array(RNN_H).fill(0);
        for(let j=0;j<RNN_H;j++){
          let g=0;
          for(let k=0;k<49;k++) g+=dO[t][k]*os[t][k]*(1-os[t][k])*Who[k][j];
          g+=dhNext[j];
          g*= (1-hs[t+1][j]*hs[t+1][j]); // tanh'
          dh[j]=g;
          for(let k=0;k<49;k++) Who[k][j]-=lr*clip(dO[t][k]*os[t][k]*(1-os[t][k])*hs[t+1][j]);
          bh[j]-=lr*clip(g);
          for(let k=0;k<RNN_H;k++) Whh[j][k]-=lr*clip(g*hs[t][k]);
          for(let k=0;k<49;k++) Wxh[j][k]-=lr*clip(g*xs[t][k]);
        }
        dhNext=dh;
      }
    }
  }
  return { Wxh, Whh, bh, Who, bo };
}
function rnnFwd(m, hist){
  const data=hist.slice(-RNN_WIN);
  let h=new Array(RNN_H).fill(0);
  for(let t=0;t<data.length;t++){
    const x=feats49(data[t]); const nh=new Array(RNN_H);
    for(let j=0;j<RNN_H;j++){ let acc=m.bh[j];
      for(let k=0;k<49;k++) acc+=m.Wxh[j][k]*x[k];
      for(let k=0;k<RNN_H;k++) acc+=m.Whh[j][k]*h[k];
      nh[j]=TANH(acc); }
    h=nh;
  }
  const sc=new Array(49);
  for(let k=0;k<49;k++){ let acc=m.bo[k]; for(let j=0;j<RNN_H;j++) acc+=m.Who[k][j]*h[j]; sc[k]=sigmoid(acc); }
  return sc;
}

// ================= 2) LSTM 长短期记忆 =================
const LSTM_H = 16, LSTM_WIN = 20, LSTM_EPOCHS = 5;
function lstmTrain(hist){
  const data = hist; // v5.1：全量历史学习
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
      // 损失只对最后一步输出
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
      // 隐状态梯度回传最后一步（简化：只更新最后步的门参数）
      const t=LSTM_WIN-1, cc=cache[t];
      const dh=new Array(LSTM_H).fill(0);
      for(let j=0;j<LSTM_H;j++){ let g=0; for(let k=0;k<49;k++) g+=dOut[k]*oOut[k]*(1-oOut[k])*Wy[k][j]; dh[j]=g; }
      const dC=new Array(LSTM_H).fill(0);
      for(let j=0;j<LSTM_H;j++){ const dc=dh[j]*cc.o[j]*(1-cc.c[j]*cc.c[j]); dC[j]=dc; }
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
// 每期49维 → Emb(d=16)；window=12 单头自注意力 + FFN32；正弦位置编码；输出投影49
const TR_D = 16, TR_WIN = 12, TR_EPOCHS = 3;
function trTrain(hist){
  const data=hist.map(feats49); // v5.1：全量历史学习
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
  const fwd=(win)=>{ // win: TR_WIN 个49维
    const E=win.map((x,pos)=>{ const e=new Array(TR_D); const p=posEnc(pos);
      for(let d=0;d<TR_D;d++){ let s=p[d]; for(let k=0;k<49;k++) s+=Emb[k][d]*x[k]; e[d]=s; } return e; });
    const att=E.map(()=>new Array(TR_WIN).fill(0));
    for(let i=0;i<TR_WIN;i++){
      const q=new Array(TR_D), kk=new Array(TR_D);
      for(let d=0;d<TR_D;d++){ let aq=0,ak=0; for(let e=0;e<TR_D;e++){ aq+=Wq[d][e]*E[i][e]; ak+=Wk[d][e]*E[i][e]; } q[d]=aq; kk[d]=ak; }
      const sc=new Array(TR_WIN);
      for(let j=0;j<TR_WIN;j++){ let s=0; for(let d=0;d<TR_D;d++) s+=q[d]*kk[d]; sc[j]=s/Math.sqrt(TR_D); }
      const mx=Math.max(...sc); const ex=sc.map(v=>Math.exp(v-mx)); const sum=ex.reduce((a,b)=>a+b,0)||1;
      for(let j=0;j<TR_WIN;j++) att[i][j]=ex[j]/sum;
    }
    const O=E.map(()=>new Array(TR_D).fill(0));
    for(let i=0;i<TR_WIN;i++) for(let j=0;j<TR_WIN;j++){ const a=att[i][j];
      for(let d=0;d<TR_D;d++){ let vv=0; for(let e=0;e<TR_D;e++) vv+=Wv[d][e]*E[j][e]; O[i][d]+=a*vv; } }
    for(let i=0;i<TR_WIN;i++) for(let d=0;d<TR_D;d++){ let o=0; for(let e=0;e<TR_D;e++) o+=Wo[d][e]*O[i][e]; O[i][d]+=E[i][d]+o; }
    const F=O.map(o=>{ const h=new Array(32); for(let j=0;j<32;j++){ let s=0; for(let d=0;d<TR_D;d++) s+=Wf1[j][d]*o[d]; h[j]=Math.max(0,s); }
      const f=new Array(TR_D); for(let d=0;d<TR_D;d++){ let s=0; for(let j=0;j<32;j++) s+=Wf2[d][j]*h[j]; f[d]=s; } return f; });
    const Z=O.map((o,i)=>o.map((v,d)=>v+F[i][d]));
    const last=Z[TR_WIN-1];
    const out=new Array(49);
    for(let k=0;k<49;k++){ let s=by[k]; for(let d=0;d<TR_D;d++) s+=Wy[k][d]*last[d]; out[k]=sigmoid(s); }
    return out;
  };
  // 训练：滑窗样本（step=4），只更新输出投影 Wy/by（成本受控的增量训练）
  const samples=Math.max(1, data.length-TR_WIN);
  for(let ep=0;ep<TR_EPOCHS;ep++){
    for(let s=0;s<samples;s+=4){
      const win=data.slice(s,s+TR_WIN);
      const y=data[s+TR_WIN] || data[s+TR_WIN-1];
      const out=fwd(win);
      for(let k=0;k<49;k++){
        const g=out[k]-y[k];
        for(let d=0;d<TR_D;d++) Wy[k][d]-=lr*clip(g);
        by[k]-=lr*clip(g);
      }
    }
  }
  return {Emb,Wq,Wk,Wv,Wo,Wf1,Wf2,Wy,by,posEnc};
}
function trFwd(m, hist){
  const data=hist.slice(-TR_WIN).map(feats49);
  const win=data.length<TR_WIN ? [...new Array(TR_WIN-data.length).fill(null).map(()=>new Float64Array(49)), ...data] : data;
  const E=win.map((x,pos)=>{ const e=new Array(TR_D); const p=m.posEnc(pos);
    for(let d=0;d<TR_D;d++){ let s=p[d]; for(let k=0;k<49;k++) s+=m.Emb[k][d]*x[k]; e[d]=s; } return e; });
  const O=E.map(()=>new Array(TR_D).fill(0));
  for(let i=0;i<TR_WIN;i++){
    const q=new Array(TR_D), kk=new Array(TR_D);
    for(let d=0;d<TR_D;d++){ let aq=0,ak=0; for(let e=0;e<TR_D;e++){ aq+=m.Wq[d][e]*E[i][e]; ak+=m.Wk[d][e]*E[i][e]; } q[d]=aq; kk[d]=ak; }
    const sc=new Array(TR_WIN);
    for(let j=0;j<TR_WIN;j++){ let s=0; for(let d=0;d<TR_D;d++) s+=q[d]*kk[d]; sc[j]=s/Math.sqrt(TR_D); }
    const mx=Math.max(...sc); const ex=sc.map(v=>Math.exp(v-mx)); const sum=ex.reduce((a,b)=>a+b,0)||1;
    for(let j=0;j<TR_WIN;j++){ const a=ex[j]/sum;
      for(let d=0;d<TR_D;d++){ let vv=0; for(let e=0;e<TR_D;e++) vv+=m.Wv[d][e]*E[j][e]; O[i][d]+=a*vv; } }
  }
  const F=O.map(o=>{ const h=new Array(32); for(let j=0;j<32;j++){ let s=0; for(let d=0;d<TR_D;d++) s+=m.Wf1[j][d]*o[d]; h[j]=Math.max(0,s); }
    const f=new Array(TR_D); for(let d=0;d<TR_D;d++){ let s=0; for(let j=0;j<32;j++) s+=m.Wf2[d][j]*h[j]; f[d]=s; } return f; });
  const Z=O.map((o,i)=>o.map((v,d)=>v+F[i][d]));
  const last=Z[TR_WIN-1];
  const sc=new Array(49);
  for(let k=0;k<49;k++){ let s=m.by[k]; for(let d=0;d<TR_D;d++) s+=m.Wy[k][d]*last[d]; sc[k]=sigmoid(s); }
  return sc;
}

// ================= 5) CNN 卷积神经网络 =================
// 输入 T=20 期 × 49 维时序图；2个 kernel=3 一维卷积 → 拼接(18×2) → FC49
const CNN_T = 20, CNN_K = 3, CNN_C = 2, CNN_EPOCHS = 5;
function cnnTrain(hist){
  const data=hist.map(feats49); // v5.1：全量历史学习
  const conv=Array.from({length:CNN_C},()=>Array.from({length:CNN_K},()=>Array.from({length:49},()=>randn()*0.1)));
  const bconv=new Array(CNN_C).fill(0);
  const flatN=(CNN_T-CNN_K+1)*CNN_C; // 18*2=36
  const Wfc=Array.from({length:49},()=>Array.from({length:flatN},()=>randn()*0.1));
  const bfc=new Array(49).fill(0);
  const lr=0.1;
  const convOut=(win)=>{ // win: CNN_T 个49维
    const out=[]; // [t][c]
    for(let t=0;t<=CNN_T-CNN_K;t++){
      const row=new Array(CNN_C).fill(0);
      for(let c=0;c<CNN_C;c++){ let s=bconv[c];
        for(let k=0;k<CNN_K;k++) for(let d=0;d<49;d++) s+=conv[c][k][d]*win[t+k][d];
        row[c]=Math.max(0,s); }
      out.push(row);
    }
    return out;
  };
  const samples=Math.max(1, data.length-CNN_T);
  for(let ep=0;ep<CNN_EPOCHS;ep++){
    for(let s=0;s<samples;s++){
      const win=data.slice(s,s+CNN_T);
      const y=data[s+CNN_T] || data[s+CNN_T-1];
      const co=convOut(win);
      const flat=[]; for(let t=0;t<co.length;t++) for(let c=0;c<CNN_C;c++) flat.push(co[t][c]);
      const out=new Array(49);
      for(let k=0;k<49;k++){ let acc=bfc[k]; for(let f=0;f<flatN;f++) acc+=Wfc[k][f]*flat[f]; out[k]=sigmoid(acc); }
      // 反向：FC → conv
      const dflat=new Array(flatN).fill(0);
      for(let k=0;k<49;k++){
        const g=out[k]-y[k];
        for(let f=0;f<flatN;f++){ dflat[f]+=g*Wfc[k][f]; Wfc[k][f]-=lr*clip(g); }
        bfc[k]-=lr*clip(g);
      }
      let fi=0;
      for(let t=0;t<co.length;t++) for(let c=0;c<CNN_C;c++){
        const g=dflat[fi++]*(co[t][c]>0?1:0); // ReLU'
        if(g!==0){ bconv[c]-=lr*clip(g);
          for(let k=0;k<CNN_K;k++) for(let d=0;d<49;d++) conv[c][k][d]-=lr*clip(g*win[t+k][d]); }
      }
    }
  }
  return {conv,bconv,Wfc,bfc};
}
function cnnFwd(m, hist){
  const data=hist.slice(-CNN_T).map(feats49);
  const win=data.length<CNN_T ? [...new Array(CNN_T-data.length).fill(null).map(()=>new Float64Array(49)), ...data] : data;
  const co=[];
  for(let t=0;t<=CNN_T-CNN_K;t++){
    const row=new Array(CNN_C).fill(0);
    for(let c=0;c<CNN_C;c++){ let s=m.bconv[c];
      for(let k=0;k<CNN_K;k++) for(let d=0;d<49;d++) s+=m.conv[c][k][d]*win[t+k][d];
      row[c]=Math.max(0,s); }
    co.push(row);
  }
  const flat=[]; for(let t=0;t<co.length;t++) for(let c=0;c<CNN_C;c++) flat.push(co[t][c]);
  const sc=new Array(49);
  for(let k=0;k<49;k++){ let acc=m.bfc[k]; for(let f=0;f<flat.length;f++) acc+=m.Wfc[k][f]*flat[f]; sc[k]=sigmoid(acc); }
  return sc;
}

// ================= D2·Walk-forward 回测优化权重 =================
// 慢模型（rnn/lstm/transformer/cnn）预训练到回测起点，回测期只前向不重训（成本控制）
// ⚠️ 跨年数据口径（v5.2 用户要求全量学习）：早年数据（2023-2025）只进【训练】（DL输入输出
//    是49维号码，跨年可比）；【回测】只用2026年部分——生肖映射每年春节切换，早年数据
//    生肖判定会错，不进回测。
const WF_BASELINE = 0.4786;
const MODEL_KEYS = ['rnn','lstm','transformer','lcg','cnn'];
const DL_KEYS = ['rnn','lstm','transformer','cnn'];
const DL_TRAIN = { rnn: rnnTrain, lstm: lstmTrain, transformer: trTrain, cnn: cnnTrain };
const DL_FWD   = { rnn: rnnFwd,   lstm: lstmFwd,   transformer: trFwd,   cnn: cnnFwd };

function walkForwardWeights(hist, w=60){
  // 回测期：2026 年部分近 w 期（生肖映射 STD_2026 正确的区间）
  const hist26 = hist.filter(d => d.period >= '2026001');
  const bt = hist26.slice(-w);
  const hits={}; MODEL_KEYS.forEach(n=>hits[n]=[0,0]);
  if (bt.length >= 10) {
    // 慢模型：用「首个回测期之前」的全量历史（含2023-2025）预训练一次
    const cutoff = bt[0].period;
    const models={};
    for(const n of DL_KEYS){
      const sub=hist.filter(d => d.period < cutoff);
      if(sub.length>=60) models[n]=DL_TRAIN[n](sub);
    }
    const histIdx=new Map(hist.map((d,i)=>[d.period,i]));
    for(const d of bt){
      const h=hist.slice(0, histIdx.get(d.period));
      const seven=balls(d);
      for(const n of MODEL_KEYS){
        const sc = DL_KEYS.includes(n)
          ? (models[n] ? DL_FWD[n](models[n], h) : new Array(49).fill(0.5))
          : algoLCG();
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

// ================= 聚合与派生 =================
const ZODIAC_NUMS = ZODIACS.map(z => { let c = 0; for (let n = 1; n <= 49; n++) if (zOf(n) === ZI[z]) c++; return c; });

function derive(sc){
  const zs=new Array(12).fill(0);
  for(let n=1;n<=49;n++) zs[zOf(n)]+=sc[n-1];
  for(let z=0;z<12;z++) zs[z]/=ZODIAC_NUMS[z]; // 除号数：公平竞争
  const zr=ZODIACS.map((z,i)=>({z,s:zs[i]})).sort((a,b)=>b.s-a.s);
  const top3=zr.slice(0,3).map(x=>x.z);
  const sum3=zr.slice(0,3).reduce((s,x)=>s+x.s,0)||1;
  const probs={}; top3.forEach(x=>{ probs[x]=zr.find(y=>y.z===x).s/sum3; });
  const order=[...Array(49).keys()].sort((a,b)=>sc[b]-sc[a]);
  const codes=order.slice(0,23).map(x=>x+1).sort((a,b)=>a-b);
  const six=zr.slice(0,6).map(x=>x.z);
  return { zodiac:zr[0].z, top3, probs, codes, six };
}

// 除号数版 top1 生肖索引（与 derive 聚合口径一致）
function top1Z(sc){
  const zs=new Array(12).fill(0),cnt=new Array(12).fill(0);
  for(let n=1;n<=49;n++){ zs[zOf(n)]+=sc[n-1]; cnt[zOf(n)]++; }
  for(let z=0;z<12;z++) zs[z]/=cnt[z];
  let t=0; for(let z=1;z<12;z++) if(zs[z]>zs[t]) t=z;
  return t;
}

// ================= D1·Ensemble 主入口（混合架构） =================
function predictAll(hist){
  rngReset(20261020); // v5 种子
  // 1) 4个深度学习模型用全量历史训练（最早一期到最新一期，用户2026-10-20要求）+ LCG 确定性
  const models={};
  for(const n of DL_KEYS) models[n]=DL_TRAIN[n](hist);
  // 2) D2 Walk-forward 近60期影子回测 → 当天各模型话语权
  const W=walkForwardWeights(hist, 60);
  // 3) D1 Ensemble：归一化后按动态权重加权融合
  const raw={};
  for(const n of DL_KEYS) raw[n]=DL_FWD[n](models[n], hist);
  raw.lcg=algoLCG();
  const fused=new Float64Array(49);
  let wsum=0;
  for(const k of MODEL_KEYS){
    const s=normalize(raw[k]);
    for(let n=0;n<49;n++) fused[n]+=W[k]*s[n];
    wsum+=W[k];
  }
  for(let n=0;n<49;n++) fused[n]/=(wsum||1);
  // 4) 明细
  const details={};
  for(const k of MODEL_KEYS) details[k]=derive(normalize(raw[k]));
  return { fused: derive(fused), details, weights: W };
}

// ================= D4·多属性命中判定 =================
// code_hit  ：特码 ∈ 预测23码（无事前23码预测 → null，显示?）
// zodiac_hit：开奖7球任一生肖 == 预测主推生肖
// six_hit   ：特码生肖 ∈ 预测6肖（无事前6肖预测 → null，显示?）
function judgeDraw(pred, draw){
  const codeHit = pred.codes ? ((pred.codes.includes(draw.special))?1:0) : null;
  const zodiacHit = ([...draw.numbers, draw.special].some(n=>zOf(n)===ZI[pred.zodiac]))?1:0;
  const sixHit = pred.six ? ((pred.six.some(z=>z===ZODIACS[zOf(draw.special)]))?1:0) : null;
  return { codeHit, zodiacHit, sixHit };
}

module.exports = { predictAll, derive, judgeDraw, zOf, ZODIACS, MODEL_KEYS };
