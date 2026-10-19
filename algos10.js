'use strict';
/**
 * 六子 —— 预测引擎 v4（2026-10-20 用户定版，v3 基础上删除 HMM/谱分析/自相关）
 * ==================== 模型清单（用户指定，旧模型已全部删除）====================
 * A类·号码级打分模型（5个，各输出49维原始分，分高=更可能出）：
 *   【随机派 4 个】——RNG 不依赖历史，纯产随机分；在 Ensemble 中充当「内部基准尺」：
 *     1) 线性同余 LCG        —— x=(a·x+c) mod 2^32（Math.imul 保精度），49维均匀分
 *     2) LFSR                —— 16位斐波那契LFSR（taps 16/14/13/11），位流映射49维
 *     3) 梅森旋转 MT19937    —— 标准624状态梅森旋转，49维均匀分
 *     4) 截断输出 PRNG       —— MT 输出仅取高11位（tempering 后截断），49维分
 *   【统计派 1 个】：
 *     5) AR/ARIMA            —— 特码序列 AR(2)（水平+一阶差分融合）→ 高斯核打分
 * D类·框架层（4个，保留不变）：
 *   1) Ensemble 动态融合    —— 5模型归一化后按动态权重加权融合
 *   2) Walk-forward 优化    —— 每天预测前对近60期影子回测，命中超额=当天话语权
 *   4) 多属性命中判定       —— judgeDraw：23码/平特肖/6肖三层判定（导出供 update.js 用）
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

// ---------- 通用随机数（可复现，供 LFSR 外的一切确定性流程） ----------
let _rngState = 0;
function rngReset(seed){ _rngState = seed >>> 0; }
function rng(){
  _rngState = (_rngState + 0x6D2B79F5) >>> 0;
  let t = _rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function normalize(sc){
  let mn=Infinity,mx=-Infinity;
  for(const v of sc){ if(v<mn)mn=v; if(v>mx)mx=v; }
  const r=(mx-mn)||1, o=new Float64Array(49);
  for(let i=0;i<49;i++) o[i]=(sc[i]-mn)/r;
  return o;
}
function balls(d){ return [...d.numbers, d.special]; }

// ================= 随机派 4 个（RNG 基准尺） =================

// 1) 线性同余生成器 LCG：x' = (1664525·x + 1013904223) mod 2^32
function algoLCG(){
  let x = 20261020 >>> 0;
  const a = 1664525, c = 1013904223;
  const sc = new Array(49);
  for (let n = 0; n < 49; n++) {
    x = (Math.imul(a, x) + c) >>> 0;   // imul 保 32 位低精度，等效 mod 2^32
    sc[n] = x / 4294967296;
  }
  return sc;
}

// 2) 线性反馈移位寄存器 LFSR：16位 Fibonacci 型，taps = 16,14,13,11
function algoLFSR(){
  let s = 0xACE1 >>> 0;                // 非零种子
  const sc = new Array(49);
  for (let n = 0; n < 49; n++) {
    let v = 0;
    for (let b = 0; b < 16; b++) {
      const bit = ((s >>> 15) ^ ((s >>> 13) & 1) ^ ((s >>> 11) & 1) ^ ((s >>> 9) & 1)) & 1;
      s = ((s << 1) | bit) & 0xFFFF;
      v = (v << 1) | bit;
    }
    sc[n] = v / 65536;
  }
  return sc;
}

// 3) 梅森旋转 MT19937（标准实现）
const MT = new Array(624); let mti = 625;
function mtInit(seed){
  MT[0] = seed >>> 0;
  for (let i = 1; i < 624; i++)
    MT[i] = (Math.imul(1812433253, MT[i - 1] ^ (MT[i - 1] >>> 30)) + i) >>> 0;
  mti = 624;
}
function mtGen(){
  if (mti >= 624) {
    for (let i = 0; i < 624; i++) {
      const y = (MT[i] & 0x80000000) + (MT[(i + 1) % 624] & 0x7fffffff);
      MT[i] = MT[(i + 397) % 624] ^ (y >>> 1);
      if (y & 1) MT[i] ^= 0x9908b0df;
    }
    mti = 0;
  }
  let y = MT[mti++];
  y ^= y >>> 11;
  y ^= (y << 7) & 0x9d2c5680;
  y ^= (y << 15) & 0xefc60000;
  y ^= y >>> 18;
  return y >>> 0;
}
function algoMT(){
  mtInit(20261020);
  const sc = new Array(49);
  for (let n = 0; n < 49; n++) sc[n] = mtGen() / 4294967296;
  return sc;
}

// 4) 截断输出的 PRNG：MT 输出仅保留高 11 位（低位比特截断丢弃）
function algoTruncPRNG(){
  mtInit(20261021);
  const sc = new Array(49);
  for (let n = 0; n < 49; n++) sc[n] = (mtGen() >>> 21) / 2048;
  return sc;
}

// ================= 统计派 1 个 =================

// AR(p) 最小二乘拟合（高斯消元 3x3，含截距）
function fitAR(y){
  const P = 2;
  let S11=0,S12=0,S13=0,S22=0,S23=0,S33=0,T1=0,T2=0,T3=0;
  for (let t = P; t < y.length; t++) {
    const x1 = y[t-1], x2 = y[t-2];
    S11+=x1*x1; S12+=x1*x2; S13+=x1; S22+=x2*x2; S23+=x2; S33+=1;
    T1+=x1*y[t]; T2+=x2*y[t]; T3+=y[t];
  }
  const A=[[S11,S12,S13],[S12,S22,S23],[S13,S23,S33]], b=[T1,T2,T3];
  for (let i=0;i<3;i++){
    let piv=i; for(let r=i+1;r<3;r++) if(Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
    [A[i],A[piv]]=[A[piv],A[i]]; [b[i],b[piv]]=[b[piv],b[i]];
    if(Math.abs(A[i][i])<1e-9) continue;
    for(let r=0;r<3;r++){ if(r===i)continue; const f=A[r][i]/A[i][i]; for(let c=0;c<3;c++)A[r][c]-=f*A[i][c]; b[r]-=f*b[i]; }
  }
  return { a1: b[0]/(A[0][0]||1), a2: b[1]/(A[1][1]||1), c: b[2]/(A[2][2]||0) };
}

// 5) AR/ARIMA：特码序列 AR(2)（水平）与一阶差分 AR(2) 融合预测 → 高斯核打分
function algoARIMA(hist, W=150, sigma=7){
  const y = hist.slice(-W).map(d => d.special);
  if (y.length < 20) return new Array(49).fill(1);
  const pL = fitAR(y);
  const predLevel = pL.a1*y[y.length-1] + pL.a2*y[y.length-2] + pL.c;
  const dy = []; for (let i=1;i<y.length;i++) dy.push(y[i]-y[i-1]);
  const pD = fitAR(dy);
  const predDiff = pD.a1*dy[dy.length-1] + pD.a2*dy[dy.length-2] + pD.c;
  const pred = Math.min(49, Math.max(1, (predLevel + (y[y.length-1] + predDiff)) / 2));
  const sc = new Array(49);
  for (let n=1;n<=49;n++) sc[n-1] = Math.exp(-((n-pred)*(n-pred))/(2*sigma*sigma));
  return sc;
}

// ================= D2·Walk-forward 回测优化权重 =================
const WF_BASELINE = 0.4786; // 平特肖7球口径经验基线
const MODEL_KEYS = ['lcg','lfsr','mt','truncprng','arima'];
const MODEL_FNS = {
  lcg: algoLCG, lfsr: algoLFSR, mt: algoMT, truncprng: algoTruncPRNG,
  arima: algoARIMA,
};

function walkForwardWeights(hist, w=60){
  const start=Math.max(60, hist.length-w);
  const hits={}; MODEL_KEYS.forEach(n=>hits[n]=[0,0]);
  for(let i=start;i<hist.length;i++){
    const h=hist.slice(0,i);
    const seven=balls(hist[i]);
    for(const n of MODEL_KEYS){
      const tz=top1Z(MODEL_FNS[n](h));
      const hit=seven.some(x=>zOf(x)===tz)?1:0;
      hits[n][1]++; hits[n][0]+=hit;
    }
  }
  const W={}; let anyPos=false;
  for(const n of MODEL_KEYS){
    const rate=hits[n][1] ? hits[n][0]/hits[n][1] : WF_BASELINE;
    const wt=Math.max(0, rate-WF_BASELINE);
    W[n]=wt; if(wt>0) anyPos=true;
  }
  if(!anyPos) MODEL_KEYS.forEach(n=>W[n]=1); // 全员不及格：等权保底
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

// ================= D1·Ensemble 主入口 =================
function predictAll(hist){
  rngReset(20261020); // v3 种子
  // 1) 5个A类模型各算49维原始分
  const raw={};
  for(const k of MODEL_KEYS) raw[k]=MODEL_FNS[k](hist);
  // 2) D2 Walk-forward 近60期影子回测 → 当天各模型话语权
  const W=walkForwardWeights(hist, 60);
  // 3) D1 Ensemble：归一化后按动态权重加权融合
  const fused=new Float64Array(49);
  let wsum=0;
  for(const k of MODEL_KEYS){
    const s=normalize(raw[k]);
    for(let n=0;n<49;n++) fused[n]+=W[k]*s[n];
    wsum+=W[k];
  }
  for(let n=0;n<49;n++) fused[n]/=(wsum||1);
  // 4) 明细（每模型单独 derive，供页面展示）
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
