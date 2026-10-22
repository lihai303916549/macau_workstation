// 模型回测：防连押对平特肖胜率的影响 + 阈值敏感性 + 边际分析
// 用检查点缓存慢模型，避免每期重训（单次 predictAll ~27s，不可全量跑）
const fs = require('fs');
const vm = require('vm');

const HTML = process.env.HTML_PATH || 'E:/六子/vercel_deploy/index.html';
const ALGO = process.env.ALGO_PATH || 'E:/六子/vercel_deploy/algos10.js';

const WINDOW = 500;   // 回测最近多少期
const CK = 20;        // 慢模型检查点间隔（期）
const TRAIN_CAP = 400;// 慢模型训练历史上限（控制耗时）

function readEmbeddedData(html) {
  const m = html.match(/const EMBEDDED_DATA = \[([\s\S]*?)\n\];/);
  const body = m[1];
  return [...body.matchAll(/\["(\d{7})","([^"]+)","([^"]+)"\]/g)]
    .map(x => { const nums = x[3].split(',').map(n => parseInt(n, 10));
      return { period: x[1], date: x[2], numbers: nums.slice(0,6), special: nums[6] }; })
    .filter(r => r.numbers.length===6 && !isNaN(r.special));
}

const NAMES = ['马','蛇','龙','兔','虎','牛','鼠','猪','狗','鸡','猴','羊'];
function zOf(n){ return NAMES[((n-1)%12)]; }

const html = fs.readFileSync(HTML,'utf8');
const draws = readEmbeddedData(html);
const L = draws.length;

// 开奖侧：每期生肖集合
const S = draws.map(d => { const set=new Set(); for(const n of [...d.numbers,d.special]) set.add(zOf(n)); return set; });

// 加载算法并注入缓存版预测
const src = fs.readFileSync(ALGO,'utf8');
const ctx = { console:{log(){},warn(){},error(){}}, Math, JSON, Object, Array, String, Number, Map, Set, isFinite, parseInt, isNaN, Date, Float64Array, Infinity, NaN, module:{exports:{}} };
ctx.globalThis = ctx; vm.createContext(ctx); vm.runInContext(src, ctx);
vm.runInContext(`
globalThis.__trainCheckpoint = function(hist){
  rngReset(20261019);
  const slow = {};
  for(const n of SLOW_KEYS) slow[n] = modelTrain(n, hist);
  const wf = walkForwardWeights(hist, 200);
  return { slow, wf };
};
globalThis.__predictCached = function(hist, slow, wf){
  rngReset(20261019);
  const raw = {};
  for(const n of MODEL_KEYS) raw[n] = modelScore(n, hist, slow[n]);
  const W = wf;
  const fused = new Float64Array(49); let wsum=0;
  for(const k of MODEL_KEYS){ const s=normalize(raw[k]); for(let n=0;n<49;n++) fused[n]+=W[k]*s[n]; wsum+=W[k]; }
  for(let n=0;n<49;n++) fused[n]/=(wsum||1);
  const zs=new Array(12).fill(0); const cnt=new Array(12).fill(0);
  for(let n=1;n<=49;n++){ const z=zOf(n); zs[z]+=fused[n-1]; cnt[z]++; }
  for(let z=0;z<12;z++) zs[z]/=(cnt[z]||1);
  const zodiac = derive(fused).zodiac;
  return { zodiac, zodiacScores: zs };
};
`, ctx);

const trainCp = ctx.__trainCheckpoint;
const predictCp = ctx.__predictCached;
const antiStreakBet = (ctx.module && ctx.module.exports && ctx.module.exports.antiStreakBet) || ctx.antiStreakBet;

const startIdx = L - WINDOW;
const endIdx = L;

const strategies = [1,2,3,4,5];
const wins = {}; const switches = {};
strategies.forEach(m=>{ wins[m]=0; switches[m]=0; });
let winsPri = 0; // 无约束（=maxConsec=∞），始终押融合主推
const bets = {}; strategies.forEach(m=>bets[m]=[]);

// m=3 边际分析：在"被迫顺延"的期里，主推 vs 顺延期 谁中
let sw3_total=0, sw3_priWin=0, sw3_betWin=0;

// 主推自身连续分布
const runPrimary=[]; let lastPri=null, curRun=0; const priRunLens=[];

let lastCi = -1, ck=null;
const t0 = Date.now();

for (let t=startIdx; t<endIdx; t++){
  const ci = Math.floor((t-startIdx)/CK);
  if (ci !== lastCi){
    const histCap = draws.slice(0,t).slice(-TRAIN_CAP).map(d=>({period:d.period,numbers:d.numbers,special:d.special}));
    ck = trainCp(histCap);
    lastCi = ci;
    const el = ((Date.now()-t0)/1000).toFixed(0);
    console.log(`[检查点 ${ci}] 已训练慢模型 + 权重，耗时 ${el}s，进度 ${t-startIdx}/${WINDOW}`);
  }
  const hist = draws.slice(0,t).map(d=>({period:d.period,numbers:d.numbers,special:d.special}));
  const { zodiac: primary, zodiacScores: zs } = predictCp(hist, ck.slow, ck.wf);

  // 主推连续
  runPrimary.push(primary);
  if (primary===lastPri){ curRun++; } else { if(curRun>0) priRunLens.push(curRun); curRun=1; lastPri=primary; }

  if (S[t].has(primary)) winsPri++;

  for (const m of strategies){
    const bz = antiStreakBet(zs, bets[m], m);
    if (bz!==primary) switches[m]++;
    if (S[t].has(bz)) wins[m]++;
    if (m===3 && bz!==primary){
      sw3_total++;
      if (S[t].has(primary)) sw3_priWin++;
      if (S[t].has(bz)) sw3_betWin++;
    }
    bets[m].push(bz);
  }
}
if (curRun>0) priRunLens.push(curRun);

const total = WINDOW;
console.log('\n================ 回测结果（最近', total, '期，', draws[startIdx].period, '~', draws[endIdx-1].period, '）================');
console.log('参照：无约束（始终押融合主推）胜率 =', (winsPri/total*100).toFixed(2)+'%', `(${winsPri}/${total})`);

console.log('\n--- 防连押阈值敏感性（maxConsec=几期后顺延）---');
console.log('阈值  胜率     顺延次数   vs无约束(百分点)');
for (const m of strategies){
  const wr = wins[m]/total*100;
  const diff = wr - winsPri/total*100;
  console.log(`${m}期  ${wr.toFixed(2)}%  ${String(switches[m]).padStart(6)}    ${diff>=0?'+':''}${diff.toFixed(2)}`);
}

console.log('\n--- 模型主推本身的连续情况（防连押触发频率）---');
const priRunDist={}; priRunLens.forEach(r=>priRunDist[r]=(priRunDist[r]||0)+1);
console.log('主推连续run长度分布:', JSON.stringify(priRunDist));
const runGe3 = priRunLens.filter(r=>r>=3).length;
console.log('主推连续≥3期的run数:', runGe3, '（即防连押(阈值3)会触发的段数）');

console.log('\n--- m=3 边际分析（被迫顺延的那些期）---');
console.log('顺延期数:', sw3_total);
console.log('若坚持主推(不换) 命中:', sw3_priWin, `(${(sw3_priWin/sw3_total*100).toFixed(1)}%)`);
console.log('换成次高(顺延) 命中:', sw3_betWin, `(${(sw3_betWin/sw3_total*100).toFixed(1)}%)`);
console.log('=> 防连押在"该换"的期里', sw3_betWin>sw3_priWin?'帮了忙(换后更准)':(sw3_betWin<sw3_priWin?'反而有害(换走对的)':'基本中性'));

const summary = {
  window: total, start: draws[startIdx].period, end: draws[endIdx-1].period,
  baseline_no_constraint: +(winsPri/total).toFixed(4),
  by_max_consec: strategies.map(m=>({m, rate:+(wins[m]/total).toFixed(4), switches:switches[m]})),
  primary_run_dist: priRunDist, primary_runs_ge3: runGe3,
  m3_switch_total: sw3_total, m3_primary_win: sw3_priWin, m3_bet_win: sw3_betWin,
  elapsed_sec: +((Date.now()-t0)/1000).toFixed(0),
};
const BT_OUT = process.env.BT_OUT || 'E:/六子/_tools/bt_result.json';
fs.writeFileSync(BT_OUT, JSON.stringify(summary,null,2));
console.log('\n结果已写入 bt_result.json，总耗时', summary.elapsed_sec, '秒');
