'use strict';
/**
 * 六子 —— 十模型预测引擎 v2（2026-10-19 用户定版，替换 v1 八算法动态加权）
 * ==================== 模型清单（用户指定10种，旧模型已全部删除）====================
 * A类·号码级打分模型（7个，各输出49维原始分，分高=更可能出）：
 *   3) 热冷混合 hotcold     —— 近30期频率(热)×0.6 + 遗漏值(冷,回补)×0.4
 *   5) 马尔可夫 markov      —— 一阶转移：上期7球→下期各号出现频次的条件分布
 *   6) 指数加权 ewma        —— 半衰期衰减加权频率，越近期权重越大
 *   7) 遗漏值回归 omission  —— 遗漏期数线性打分（遗漏越大回补压力越大）
 *   8) 移动平均 ma          —— 分段窗口频率移动平均
 *   9) 蒙特卡洛 montecarlo  —— 近50期频率分布采样2万次
 *   10) 热号 hot            —— 近30期7球频率
 * D类·框架层（4个）：
 *   1) Ensemble 动态融合    —— 7模型归一化后按动态权重加权融合
 *   2) Walk-forward 优化    —— 每天预测前对近60期影子回测，命中超额=当天话语权
 *   4) 多属性命中判定       —— judgeDraw：23码/平特肖/6肖三层判定（导出供 update.js 用）
 *
 * 聚合除号数（用户决策「消除」）：49维→生肖汇总时除以该生肖号码数（马÷5、其余÷4），
 *   12生肖按平均分公平竞争，消除「马5个号总和自动胜出」的偏置。
 * 动态加权兜底：全模型不及格（超额≤0）退化为等权。
 * 固定种子 mulberry32（20261019）：同数据必得同预测，蒙特卡洛可复现。
 * ==================== 事前预测铁律（用户2026-10-19强调）====================
 * 只做事前预测：每晚对「下一期」预测并存档；开奖后按存档判定记录，严禁事后重算历史。
 */
const STD = {1:'马',2:'蛇',3:'龙',4:'兔',5:'虎',6:'牛',7:'鼠',8:'猪',9:'狗',10:'鸡',11:'猴',12:'羊'};
const ZODIACS = Object.values(STD);
const ZI = {}; ZODIACS.forEach((z,i)=>ZI[z]=i);
function zOf(n){ const k = n<=12 ? n : ((n-1)%12)+1; return ZI[STD[k]]; }

// 固定种子随机数（可复现）
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
// 每期7球集合
function balls(d){ return [...d.numbers, d.special]; }
// 遗漏值：距最近一次出现过了多少期（0=上期刚出，未出现=窗口长度）
function omissionOf(win, n){
  for(let k=win.length-1;k>=0;k--) if(win[k].includes(n)) return win.length-1-k;
  return win.length;
}

// ================= A类·7个号码级打分模型 =================

// 10) 热号模型：近30期7球频率
function algoHot(hist, W=30){
  const win=hist.slice(-W), c=new Array(49).fill(0);
  for(const d of win) for(const n of balls(d)) c[n-1]++;
  return c;
}

// 3) 热冷混合 0.6/0.4：热=近30期频率(归一)；冷=遗漏值(归一,回补压力)
function algoHotCold(hist, WH=30, WC=60){
  const winH=hist.slice(-WH);
  const hot=new Array(49).fill(0);
  for(const d of winH) for(const n of balls(d)) hot[n-1]++;
  const winC=hist.slice(-WC).map(balls);
  const cold=new Array(49).fill(0);
  for(let n=1;n<=49;n++) cold[n-1]=omissionOf(winC, n);
  const nh=normalize(hot), nc=normalize(cold);
  const sc=new Array(49);
  for(let n=0;n<49;n++) sc[n]=0.6*nh[n]+0.4*nc[n];
  return sc;
}

// 8) 移动平均：近30期按每5期一段（共6段），各段频率等权平均
function algoMA(hist, W=30, seg=5){
  const win=hist.slice(-W);
  const segs=Math.max(1, Math.floor(win.length/seg));
  const sc=new Array(49).fill(0);
  for(let s=0;s<segs;s++){
    const part=win.slice(s*seg,(s+1)*seg);
    for(const d of part) for(const n of balls(d)) sc[n-1]++;
  }
  for(let n=0;n<49;n++) sc[n]/=segs;
  return sc;
}

// 6) 指数加权 EWMA：alpha=0.15 衰减加权频率（半衰期约4.3期权重减半）
function algoEWMA(hist, W=60){
  const win=hist.slice(-W);
  const alpha=0.15;
  const sc=new Array(49).fill(0);
  for(let t=0;t<win.length;t++){
    const w=Math.pow(1-alpha, win.length-1-t);
    for(const n of balls(win[t])) sc[n-1]+=w;
  }
  return sc;
}

// 7) 遗漏值回归：遗漏期数线性打分（遗漏越大回补压力越大）
function algoOmission(hist, W=60){
  const win=hist.slice(-W).map(balls);
  const sc=new Array(49);
  for(let n=1;n<=49;n++) sc[n-1]=omissionOf(win, n);
  return sc;
}

// 5) 马尔可夫链：一阶转移
//    M[a][b] = 历史中「上期出a、下期出b」的共现次数；以最后一期为当前状态，
//    得分 = 该期7球指向各目标号的归一化转移概率之和
function algoMarkov(hist, W=120){
  const win=hist.slice(-W);
  const M=new Array(49).fill(0).map(()=>new Array(49).fill(0));
  for(let t=0;t<win.length-1;t++){
    const cur=balls(win[t]);
    const nxt=new Set(balls(win[t+1]));
    for(const a of cur) for(let b=1;b<=49;b++) if(nxt.has(b)) M[a-1][b-1]++;
  }
  const sc=new Array(49).fill(0);
  for(const a of balls(win[win.length-1])){
    const row=M[a-1], tot=row.reduce((x,y)=>x+y,0)||1;
    for(let b=0;b<49;b++) sc[b]+=row[b]/tot;
  }
  return sc;
}

// 9) 蒙特卡洛：近50期频率分布采样2万次
function algoMonteCarlo(hist, W=50, T=20000){
  const win=hist.slice(-W), c=new Array(49).fill(0);
  for(const d of win) for(const n of balls(d)) c[n-1]++;
  const tot=c.reduce((a,b)=>a+b,0)||1, hit=new Array(49).fill(0);
  for(let t=0;t<T;t++){
    const r=rng()*tot; let acc=0,pick=48;
    for(let n=0;n<49;n++){ acc+=c[n]; if(r<=acc){pick=n;break;} }
    hit[pick]++;
  }
  return hit;
}

// ================= D2·Walk-forward 回测优化权重 =================
// 对近 w 期做影子 walk-forward：每期只用当时历史算各模型分→判定平特肖命中，
// 命中超额（-基线0.4786，负则0）作为当天融合话语权；全不及格退化为等权。
const WF_BASELINE = 0.4786; // 平特肖7球口径经验基线
const MODEL_KEYS = ['hot','hotcold','ma','ewma','omission','markov','montecarlo'];
const MODEL_FNS = {
  hot: algoHot, hotcold: algoHotCold, ma: algoMA,
  ewma: algoEWMA, omission: algoOmission, markov: algoMarkov, montecarlo: algoMonteCarlo,
};

function walkForwardWeights(hist, w=60){
  const start=Math.max(60, hist.length-w); // 至少留60期历史给模型
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
// 各生肖号码数（2026年：马5个，其余11生肖各4个）——消除聚合号数偏置
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
  rngReset(20261019); // v2 种子
  // 1) 7个A类模型各算49维原始分
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
