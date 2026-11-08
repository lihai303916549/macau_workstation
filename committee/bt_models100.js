// 100 模型 zodiac 预测 walk-forward 回测（纯 Node 无依赖）
// 统一任务：用历史窗口预测「下一期 7 球中哪个生肖会出现」(平特肖式单生肖挑选)
// 公平指标：去号数偏置超额胜率 = actual - 按模型所选生肖的边际概率算出的期望
// 深度序列模型用 极限学习机/储备池(ELM/RC)：架构特定随机变换 + 岭回归读出（无需反向传播）
const fs = require('fs');
const QUICK = process.argv.includes('--quick');

const HTML = process.env.HTML_PATH || 'E:/六子/vercel_deploy/index.html';
const W = 12;          // 历史窗口(期)
const TRAIN_MAX = 160; // 训练最大样本
const EVAL_LEN = QUICK ? 60 : 250;
const R = 48;          // ELM/RC 储备池维度

// ---------- 数据 ----------
function readEmbeddedData(html) {
  const m = html.match(/const EMBEDDED_DATA = \[([\s\S]*?)\n\];/);
  const body = m[1];
  return [...body.matchAll(/\["(\d{7})","([^"]+)","([^"]+)"\]/g)]
    .map(x => ({ period: x[1], date: x[2], numbers: x[3].split(',').map(n => parseInt(n, 10)) }))
    .filter(r => r.numbers.length === 7);
}
const NAMES = ['马','蛇','龙','兔','虎','牛','鼠','猪','狗','鸡','猴','羊'];
const ZI = {}; NAMES.forEach((z, i) => ZI[z] = i);
function zOf(n) { return NAMES[((n - 1) % 12)]; }

const html = fs.readFileSync(HTML, 'utf8');
const draws = readEmbeddedData(html);
const T = draws.length;
// occ[t][z] = 1 若第 t 期 7 球中出现生肖 z
const occ = draws.map(d => {
  const row = new Array(12).fill(0);
  const set = new Set(d.numbers.map(zOf));
  set.forEach(z => row[ZI[z]] = 1);
  return row;
});
// 评估窗口边际率（公平基线）
const evalStart = T - EVAL_LEN;
const marginal = new Array(12).fill(0);
for (let t = evalStart; t < T; t++) for (let z = 0; z < 12; z++) marginal[z] += occ[t][z];
for (let z = 0; z < 12; z++) marginal[z] /= EVAL_LEN;
const baseRand = marginal.reduce((a, b) => a + b, 0) / 12; // 均匀随机单生肖期望
const baseMa = marginal[ZI['马']];                            // 永远押马期望

// ---------- 推荐记录 / 开奖判定（事前存档，遵循事前预测制，不补历史） ----------
const REC_PATH = process.env.BT_REC || 'E:/六子/committee/bt100_records.json';
const PRED_PATH = process.env.BT_PRED || 'E:/六子/committee/bt100_predict.json';
function loadRecords() {
  try { return JSON.parse(fs.readFileSync(REC_PATH, 'utf8')); } catch (e) { return { records: {}, stats: {} }; }
}
function zSetOfPeriod(period) {
  const d = draws.find(x => x.period === period);
  if (!d) return null;
  return [...new Set(d.numbers.map(zOf))];
}
function recordPrediction(pred) {
  const rec = loadRecords();
  if (rec.records[pred.predictFor]) { console.log('记录已存在(跳过):', pred.predictFor); }
  else {
    rec.records[pred.predictFor] = {
      predictFor: pred.predictFor, sourcePeriod: pred.sourcePeriod, generated: pred.generated,
      consensus: pred.consensus, picks: pred.picks.map(p => ({ id: p.id, name: p.name, layer: p.layer, zodiac: p.zodiac, score: p.score })),
      openZodiacs: null, judged: false, modelHits: null, consensusHit: null
    };
    console.log('记录新增:', pred.predictFor, '模型数', pred.picks.length);
  }
  fs.writeFileSync(REC_PATH, JSON.stringify(rec, null, 2));
}
function judgeRecords() {
  const rec = loadRecords();
  let judgedN = 0;
  for (const k of Object.keys(rec.records)) {
    const r = rec.records[k];
    if (r.judged) continue;
    const zs = zSetOfPeriod(r.predictFor);
    if (!zs) continue; // 该期尚未开奖（数据未拉到）
    r.openZodiacs = zs; r.judged = true; r.modelHits = {};
    for (const p of r.picks) r.modelHits[p.id] = zs.includes(p.zodiac);
    r.consensusHit = zs.includes(r.consensus[0].zodiac);
    judgedN++;
  }
  const ids = [...new Set(Object.values(rec.records).flatMap(r => (r.picks || []).map(p => p.id)))];
  const stats = { perModel: {}, consensus: { hits: 0, total: 0 }, periods: Object.keys(rec.records).length, judgedPeriods: 0 };
  for (const id of ids) stats.perModel[id] = { name: '', layer: '', hits: 0, total: 0 };
  for (const k of Object.keys(rec.records)) {
    const r = rec.records[k];
    if (!r.judged) continue;
    stats.judgedPeriods++; if (r.consensusHit) stats.consensus.hits++; stats.consensus.total++;
    for (const p of r.picks) { const s = stats.perModel[p.id]; s.name = p.name; s.layer = p.layer; s.total++; if (r.modelHits[p.id]) s.hits++; }
  }
  rec.stats = stats;
  fs.writeFileSync(REC_PATH, JSON.stringify(rec, null, 2));
  console.log('判定完成，新判定期数:', judgedN, '| 已判定总期:', stats.judgedPeriods);
  if (stats.judgedPeriods > 0) {
    console.log('共识命中率:', (stats.consensus.hits / stats.consensus.total * 100).toFixed(1) + '%', '(' + stats.consensus.hits + '/' + stats.consensus.total + ')');
    const arr = ids.map(id => ({ ...stats.perModel[id], id })).filter(s => s.total > 0).sort((a, b) => (b.hits / b.total) - (a.hits / a.total));
    console.log('--- 模型命中率 Top10 ---');
    for (const s of arr.slice(0, 10)) console.log(s.id, s.name, (s.hits / s.total * 100).toFixed(1) + '%', '(' + s.hits + '/' + s.total + ')');
  }
}

// ---------- 工具 ----------
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
// 解线性方程组 A(D×D) X(D×K) = B(D×K)，带部分主元
function solveSystem(A, B) {
  const D = A.length, K = B[0].length;
  const M = A.map((r, i) => Array.from(r).concat(Array.from(B[i])));
  for (let c = 0; c < D; c++) {
    let p = c; for (let r = c + 1; r < D; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (p !== c) { const tp = M[p]; M[p] = M[c]; M[c] = tp; }
    const piv = M[c][c] || 1e-12;
    for (let r = 0; r < D; r++) { if (r === c) continue; const f = M[r][c] / piv; if (f === 0) continue; for (let k = c; k <= D + K - 1; k++) M[r][k] -= f * M[c][k]; }
  }
  const X = Array.from({ length: D }, () => new Array(K).fill(0));
  for (let i = 0; i < D; i++) { const piv = M[i][i] || 1e-12; for (let k = 0; k < K; k++) X[i][k] = M[i][D + k] / piv; }
  return X;
}
// 岭回归读出：States(n×S) -> Y(n×K)，含 bias 列
function ridgeReadout(States, Y, lambda, Sforced) {
  const n = States.length, S = Sforced || States[0].length, K = Y[0].length, D = S + 1;
  const AtA = Array.from({ length: D }, () => new Float64Array(D));
  const AtY = Array.from({ length: D }, () => new Float64Array(K));
  for (let i = 0; i < n; i++) {
    const s = States[i], y = Y[i];
    for (let a = 0; a < S; a++) {
      for (let b = 0; b < S; b++) AtA[a][b] += s[a] * s[b];
      for (let k = 0; k < K; k++) AtY[a][k] += s[a] * y[k];
      AtA[a][S] += s[a]; AtA[S][a] += s[a];
    }
    AtA[S][S] += 1;
    for (let k = 0; k < K; k++) AtY[S][k] += y[k];
  }
  for (let a = 0; a < S; a++) AtA[a][a] += lambda;
  const Wt = solveSystem(AtA, AtY);
  return function (state) {
    const out = new Array(K).fill(0);
    for (let a = 0; a < S; a++) for (let k = 0; k < K; k++) out[k] += state[a] * Wt[a][k];
    for (let k = 0; k < K; k++) out[k] += Wt[S][k];
    return out;
  };
}

// ---------- 统一引擎接口：fitPredict(occ, i0, t) -> scores[12] ----------
function tabularFeatures(occ, end) { // 结束于 end 的窗口展平
  const f = [];
  for (let i = end - W + 1; i <= end; i++) for (let z = 0; z < 12; z++) f.push(occ[i][z]);
  return f;
}
// 收集训练样本 (history-end index i in [i0, t-2], target occ[i+1])
function gatherTab(occ, i0, t) {
  const X = [], Y = [];
  for (let i = i0; i <= t - 2; i++) { X.push(tabularFeatures(occ, i)); Y.push(occ[i + 1].slice()); }
  return { X, Y };
}

function engineTabularLinear(lambda) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    if (X.length < 4) return occ[t - 1].slice();
    const Wt = ridgeReadout(X, Y, lambda);
    return Wt(tabularFeatures(occ, t - 1));
  };
}
function engineLogistic(iters, lr, lambda) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    const n = X.length, F = X[0].length, K = 12;
    if (n < 4) return occ[t - 1].slice();
    // W(K×F) + bias(K)
    const W = Array.from({ length: K }, () => new Array(F).fill(0));
    const b = new Array(K).fill(0);
    for (let it = 0; it < iters; it++) {
      const gW = Array.from({ length: K }, () => new Array(F).fill(0));
      const gb = new Array(K).fill(0);
      for (let i = 0; i < n; i++) {
        const sigm = W.map((w, k) => { let z = b[k]; for (let f = 0; f < F; f++) z += w[f] * X[i][f]; return 1 / (1 + Math.exp(-z)); });
        for (let k = 0; k < K; k++) { const err = sigm[k] - Y[i][k]; for (let f = 0; f < F; f++) gW[k][f] += err * X[i][f]; gb[k] += err; }
      }
      for (let k = 0; k < K; k++) { for (let f = 0; f < F; f++) W[k][f] -= lr * (gW[k][f] / n + lambda * W[k][f]); b[k] -= lr * gb[k] / n; }
    }
    const out = new Array(K).fill(0);
    const x = tabularFeatures(occ, t - 1);
    for (let k = 0; k < K; k++) { let z = b[k]; for (let f = 0; f < F; f++) z += W[k][f] * x[f]; out[k] = 1 / (1 + Math.exp(-z)); }
    return out;
  };
}
function engineKNN(k) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    if (X.length === 0) return occ[t - 1].slice();
    const xt = tabularFeatures(occ, t - 1);
    const dist = X.map((x, i) => { let s = 0; for (let f = 0; f < x.length; f++) { const d = x[f] - xt[f]; s += d * d; } return { i, d: s }; }).sort((a, b) => a.d - b.d);
    const out = new Array(12).fill(0);
    const kk = Math.min(k, dist.length);
    for (let j = 0; j < kk; j++) for (let z = 0; z < 12; z++) out[z] += Y[dist[j].i][z];
    for (let z = 0; z < 12; z++) out[z] /= kk;
    return out;
  };
}
// 简单 CART 回归树
function makeTree(X, Y, depth, maxDepth, featBag) {
  const n = X.length, F = X[0].length, K = Y[0].length;
  const meanY = new Array(K).fill(0); for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) meanY[k] += Y[i][k]; for (let k = 0; k < K; k++) meanY[k] /= n;
  if (depth >= maxDepth || n < 8) return { leaf: meanY };
  const feats = featBag ? featBag : Array.from({ length: F }, (_, i) => i);
  let best = null;
  for (let tries = 0; tries < Math.min(8, feats.length); tries++) {
    const f = feats[(Math.random() * feats.length) | 0];
    let lo = Infinity, hi = -Infinity; for (let i = 0; i < n; i++) { const v = X[i][f]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo === hi) continue;
    const thr = lo + Math.random() * (hi - lo);
    let nl = 0, nr = 0; const sl = new Array(K).fill(0), sr = new Array(K).fill(0);
    for (let i = 0; i < n; i++) { if (X[i][f] <= thr) { nl++; for (let k = 0; k < K; k++) sl[k] += Y[i][k]; } else { nr++; for (let k = 0; k < K; k++) sr[k] += Y[i][k]; } }
    if (nl < 4 || nr < 4) continue;
    let varl = 0, varr = 0;
    for (let i = 0; i < n; i++) { const v = X[i][f] <= thr ? sl : sr; const cnt = X[i][f] <= thr ? nl : nr; for (let k = 0; k < K; k++) { const m = v[k] / cnt; varl += X[i][f] <= thr ? (Y[i][k] - m) ** 2 : 0; varr += X[i][f] <= thr ? 0 : (Y[i][k] - m) ** 2; } }
    const score = varl + varr;
    if (!best || score < best.score) best = { score, f, thr, sl: nl, sr: nr };
  }
  if (!best) return { leaf: meanY };
  const { f, thr } = best;
  const Xl = [], Yl = [], Xr = [], Yr = [];
  for (let i = 0; i < n; i++) { if (X[i][f] <= thr) { Xl.push(X[i]); Yl.push(Y[i]); } else { Xr.push(X[i]); Yr.push(Y[i]); } }
  return { f, thr, left: makeTree(Xl, Yl, depth + 1, maxDepth, featBag), right: makeTree(Xr, Yr, depth + 1, maxDepth, featBag) };
}
function treePredict(node, x) { if (node.leaf) return node.leaf; return x[node.f] <= node.thr ? treePredict(node.left, x) : treePredict(node.right, x); }
function engineTree(maxDepth) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    if (X.length < 8) return occ[t - 1].slice();
    return treePredict(makeTree(X, Y, 0, maxDepth), tabularFeatures(occ, t - 1));
  };
}
function engineRF(ntrees, maxDepth) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    const n = X.length, F = X[0].length;
    if (n < 16) return occ[t - 1].slice();
    const trees = [];
    for (let tr = 0; tr < ntrees; tr++) {
      const idx = []; for (let i = 0; i < n; i++) idx.push((Math.random() * n) | 0);
      const Xb = idx.map(i => X[i]), Yb = idx.map(i => Y[i]);
      const bag = []; for (let f = 0; f < Math.floor(F * 0.5); f++) bag.push((Math.random() * F) | 0);
      trees.push({ node: makeTree(Xb, Yb, 0, maxDepth, bag), bag });
    }
    const xt = tabularFeatures(occ, t - 1);
    const out = new Array(12).fill(0);
    for (const tr of trees) { const p = treePredict(tr.node, xt); for (let k = 0; k < 12; k++) out[k] += p[k]; }
    for (let k = 0; k < 12; k++) out[k] /= trees.length;
    return out;
  };
}
function engineGBM(rounds, maxDepth, lr) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    const n = X.length, F = X[0].length, K = 12;
    if (n < 16) return occ[t - 1].slice();
    const pred = Array.from({ length: n }, () => new Array(K).fill(0));
    const xt = tabularFeatures(occ, t - 1);
    const out = new Array(K).fill(0);
    for (let r = 0; r < rounds; r++) {
      // 每轮对残差拟合一棵树
      const resid = Array.from({ length: n }, (_, i) => { const rr = new Array(K); for (let k = 0; k < K; k++) rr[k] = Y[i][k] - pred[i][k]; return rr; });
      const tree = makeTree(X, resid, 0, maxDepth);
      for (let i = 0; i < n; i++) { const p = treePredict(tree, X[i]); for (let k = 0; k < K; k++) pred[i][k] += lr * p[k]; }
      const pxt = treePredict(tree, xt); for (let k = 0; k < K; k++) out[k] += lr * pxt[k];
    }
    return out;
  };
}
function engineKernelRidge(kernel, sigma, lambda) {
  return function (occ, i0, t) {
    const { X, Y } = gatherTab(occ, i0, t);
    const n = X.length; if (n < 4) return occ[t - 1].slice();
    const xt = tabularFeatures(occ, t - 1);
    const Km = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
      let s = 0; if (kernel === 'rbf') { for (let f = 0; f < X[i].length; f++) { const d = X[i][f] - X[j][f]; s += d * d; } s = Math.exp(-s / (2 * sigma * sigma)); } else { for (let f = 0; f < X[i].length; f++) s += X[i][f] * X[j][f]; }
      Km[i][j] = s; Km[j][i] = s;
    }
    const D = n + 1; const A = Array.from({ length: D }, () => new Float64Array(D)); const B = Array.from({ length: D }, () => new Float64Array(12));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = Km[i][j];
    for (let i = 0; i < n; i++) { A[i][n] = 1; A[n][i] = 1; }
    A[n][n] = n; for (let i = 0; i < n; i++) A[i][i] += lambda;
    for (let i = 0; i < n; i++) for (let k = 0; k < 12; k++) { B[i][k] = Y[i][k]; B[n][k] += Y[i][k]; }
    const Wt = solveSystem(A, B);
    const kt = new Array(n); for (let i = 0; i < n; i++) { if (kernel === 'rbf') { let s = 0; for (let f = 0; f < X[i].length; f++) { const d = X[i][f] - xt[f]; s += d * d; } kt[i] = Math.exp(-s / (2 * sigma * sigma)); } else { let s = 0; for (let f = 0; f < X[i].length; f++) s += X[i][f] * xt[f]; kt[i] = s; } }
    const out = new Array(12).fill(0);
    for (let i = 0; i < n; i++) for (let k = 0; k < 12; k++) out[k] += kt[i] * Wt[i][k];
    for (let k = 0; k < 12; k++) out[k] += Wt[n][k];
    return out;
  };
}

// ---------- 单生肖引擎（统计层）→ 包装成 12 分 ----------
function makePerZodiac(sub) {
  // sub.fitPredict(series(1D occ[:,z]), i0, t) -> score(z 下期出现概率)
  return function (occ, i0, t) {
    const out = new Array(12).fill(0);
    for (let z = 0; z < 12; z++) {
      const series = []; for (let i = 0; i < T; i++) series.push(occ[i][z]);
      out[z] = sub(series, i0, t);
    }
    return out;
  };
}
// 单生肖：训练窗口 i in [i0, t-2]，特征 = 最近 `lags` 个值，目标 occ[i+1]
function zSeriesWindow(series, i, lags) { const f = []; for (let j = lags - 1; j >= 0; j--) f.push(series[i - j]); return f; }
function perZArLm(lags, lambda) {
  return function (series, i0, t) {
    const X = [], Y = []; for (let i = i0; i <= t - 2; i++) { if (i - lags + 1 < 0) continue; X.push(zSeriesWindow(series, i, lags)); Y.push([series[i + 1]]); }
    if (X.length < 4) return series[t - 1];
    const Wt = ridgeReadout(X, Y, lambda);
    return Wt(zSeriesWindow(series, t - 1, lags))[0];
  };
}
function perZEwma(alpha) {
  return function (series, i0, t) {
    let v = series[i0] || 0; for (let i = i0 + 1; i <= t - 1; i++) v = alpha * series[i] + (1 - alpha) * v; return v;
  };
}
function perZMarkov(order) {
  return function (series, i0, t) {
    // 统计 状态(最近 order 个 0/1) -> 下一为 1 计数 / 总数
    const cnt = {}, tot = {};
    for (let i = i0 + order; i <= t - 2; i++) {
      let st = 0; for (let j = 0; j < order; j++) st = st * 2 + series[i - j];
      tot[st] = (tot[st] || 0) + 1; cnt[st] = (cnt[st] || 0) + series[i + 1];
    }
    let st = 0; for (let j = 0; j < order; j++) st = st * 2 + series[t - 1 - j];
    if (tot[st]) return cnt[st] / tot[st];
    let s = 0, c = 0; for (const k in tot) { s += cnt[k]; c += tot[k]; } return c ? s / c : series[t - 1];
  };
}
function perZKalman() {
  return function (series, i0, t) {
    let x = series[i0] || 0, P = 1, q = 0.05, r = 0.2;
    for (let i = i0 + 1; i <= t - 1; i++) {
      P = P + q; const K = P / (P + r); x = x + K * (series[i] - x); P = (1 - K) * P;
    }
    return Math.max(0, Math.min(1, x));
  };
}
function perZGrey() {
  return function (series, i0, t) {
    const y = series.slice(i0, t); if (y.length < 5) return series[t - 1];
    const acc = []; let s = 0; for (const v of y) { s += v; acc.push(s); }
    const n = acc.length - 1; const B = acc.slice(0, n); const Yr = y.slice(1);
    // 拟合 z(k+1) = -a z(k) + b  →  [B,1]θ = Yr
    let sBB = 0, sB = 0, sBY = 0; for (let i = 0; i < n; i++) { sBB += B[i] * B[i]; sB += B[i]; sBY += B[i] * Yr[i]; }
    const den = n * sBB - sB * sB; if (Math.abs(den) < 1e-9) return series[t - 1];
    const a = (n * sBY - sB * Yr.reduce((p, c) => p + c, 0)) / den;
    const b = (sBB * Yr.reduce((p, c) => p + c, 0) - sB * sBY) / den;
    const lastAcc = acc[acc.length - 1];
    const predAcc = (y[0] - b / a) * Math.exp(-a * n) + b / a;
    let val = predAcc - lastAcc; return Math.max(0, Math.min(1, val));
  };
}
function perZGarch(lambda) {
  // 基础 AR(2) 残差 + ARCH 特征
  return function (series, i0, t) {
    const lags = 2; const X = [], Y = [];
    let base = perZArLm(lags, 0.01);
    for (let i = i0; i <= t - 2; i++) { if (i - lags - 1 < 0) continue; const ar = zSeriesWindow(series, i, lags); const e = series[i] - base(series, i0, i); X.push(ar.concat([e * e])); Y.push([series[i + 1]]); }
    if (X.length < 4) return series[t - 1];
    const Wt = ridgeReadout(X, Y, lambda);
    const ar = zSeriesWindow(series, t - 1, lags); const e = series[t - 1] - base(series, i0, t - 1);
    return Wt(ar.concat([e * e]))[0];
  };
}
function perZCount(lambda) { return perZArLm(3, lambda); }       // PAR/INAR/Binomial 近似
function perZMap(lambda) {                                        // 非线性 AR(logistic/tent map 拟合代理)
  return function (series, i0, t) {
    const X = [], Y = []; for (let i = i0; i <= t - 2; i++) { if (i - 2 < 0) continue; const x1 = series[i], x2 = series[i - 1]; X.push([x1, x1 * x1, x2, x2 * x2]); Y.push([series[i + 1]]); }
    if (X.length < 4) return series[t - 1];
    const Wt = ridgeReadout(X, Y, lambda);
    const x1 = series[t - 1], x2 = series[t - 2]; return Wt([x1, x1 * x1, x2, x2 * x2])[0];
  };
}
function perZIntensity(decay) { // Hawkes/ACD 强度代理
  return function (series, i0, t) {
    let inten = 0; for (let i = i0; i <= t - 1; i++) { if (series[i] === 1) inten = 1; else inten = inten * Math.exp(-decay); }
    return 1 / (1 + Math.exp(-6 * (inten - 0.3)));
  };
}
function perZHMM() { // 2 态 HMM 简化 EM
  return function (series, i0, t) {
    const y = series.slice(i0, t); const n = y.length; if (n < 10) return series[t - 1];
    let pS = [0.5, 0.5], emis = [0.3, 0.7], trans = [0.9, 0.1, 0.1, 0.9];
    let alpha = [];
    for (let it = 0; it < 12; it++) {
      // E-step: 前向
      alpha = []; let a0 = [0.5, 0.5];
      for (let i = 0; i < n; i++) { const o = y[i]; const na = [0, 0]; for (let s = 0; s < 2; s++) { let v = 0; for (let sp = 0; sp < 2; sp++) v += a0[sp] * trans[sp * 2 + s]; na[s] = v * (o ? emis[s] : 1 - emis[s]); } const ss = na[0] + na[1] || 1e-9; na[0] /= ss; na[1] /= ss; a0 = na; alpha.push(a0); }
      // M-step 粗略
      emis[0] = 0.2 + 0.6 * alpha.reduce((s, a, i) => s + a[0] * y[i], 0) / n; emis[1] = 0.2 + 0.6 * alpha.reduce((s, a, i) => s + a[1] * y[i], 0) / n;
      const f01 = alpha[n - 1][0] * trans[0], f10 = alpha[n - 1][1] * trans[2]; const fs = f01 + f10 || 1; trans[0] = f01 / fs; trans[2] = f10 / fs; trans[1] = 1 - trans[0]; trans[3] = 1 - trans[2];
    }
    const as = alpha[n - 1]; const p = as[0] * (trans[0] * emis[0] + trans[1] * emis[1]) + as[1] * (trans[2] * emis[0] + trans[3] * emis[1]);
    return p;
  };
}
function perZArima(lags, lambda) { // 差分后 AR
  return function (series, i0, t) {
    const d = series.slice(i0, t).map((v, i, arr) => i === 0 ? v : v - arr[i - 1]);
    const X = [], Y = []; for (let i = i0; i <= t - 2; i++) { if (i - lags + 1 < 0) continue; X.push(zSeriesWindow(d, i, lags)); Y.push([d[i + 1]]); }
    if (X.length < 4) return series[t - 1];
    const Wt = ridgeReadout(X, Y, lambda);
    return 1 / (1 + Math.exp(-Wt(zSeriesWindow(d, t - 1, lags))[0]));
  };
}

// ---------- 深度序列 ELM/RC：架构特定随机变换 + 岭读出 ----------
function seqWindow(occ, end) { const w = []; for (let i = end - W + 1; i <= end; i++) w.push(occ[i].slice()); return w; } // W×12
function gatherSeq(occ, i0, t) { const X = [], Y = []; for (let i = i0; i <= t - 2; i++) { X.push(seqWindow(occ, i)); Y.push(occ[i + 1].slice()); } return { X, Y }; }
function rcTransform(frontends, w, rng) {
  // 返回状态向量(拼接各 frontend 输出)
  const parts = [];
  for (const fe of frontends) parts.push(fe(w, rng));
  let st = []; for (const p of parts) st = st.concat(p); return st;
}
function makeFrontend(kind, rng) {
  const F = 12; // 每期生肖维数（patch/nbeats 等分支依赖）
  if (kind === 'mlp') {
    const h = 16, F = W * 12; const W1 = Array.from({ length: h }, () => Array.from({ length: F }, () => (rng() * 2 - 1) * 0.5)); const W2 = Array.from({ length: R }, () => Array.from({ length: h }, () => (rng() * 2 - 1) * 0.5));
    return function (w) { const f = []; for (let i = 0; i < W; i++) for (let z = 0; z < 12; z++) f.push(w[i][z]); const hid = W1.map(r => { let s = 0; for (let j = 0; j < F; j++) s += r[j] * f[j]; return Math.tanh(s); }); const out = W2.map(r => { let s = 0; for (let j = 0; j < h; j++) s += r[j] * hid[j]; return s; }); return out; };
  }
  if (kind === 'rnn-lstm' || kind === 'rnn-gru') {
    const hid = 16; const F = 12; const Zu = Array.from({ length: hid }, () => Array.from({ length: F }, () => rng() * 2 - 1)); const Zu2 = Array.from({ length: hid }, () => Array.from({ length: F }, () => rng() * 2 - 1));
    const Wh = Array.from({ length: hid }, () => Array.from({ length: hid }, () => (rng() * 2 - 1) * 1.2)); const bh = new Array(hid).fill(0);
    return function (w) { let h = new Array(hid).fill(0), c = new Array(hid).fill(0); for (let i = 0; i < W; i++) { const x = w[i]; const zt = new Array(hid), it = new Array(hid), ft = new Array(hid), gt = new Array(hid), ct = new Array(hid), ht = new Array(hid); for (let k = 0; k < hid; k++) { let s = bh[k] + Wh[k].reduce((p, v, j) => p + v * h[j], 0) + Zu[k].reduce((p, v, j) => p + v * x[j], 0) + Zu2[k].reduce((p, v, j) => p + v * x[j], 0); zt[k] = 1 / (1 + Math.exp(-s)); it[k] = 1 / (1 + Math.exp(-s - 1)); ft[k] = 1 / (1 + Math.exp(-s + 1)); gt[k] = Math.tanh(s); } if (kind === 'rnn-gru') { for (let k = 0; k < hid; k++) { c[k] = ft[k] * c[k] + it[k] * gt[k]; ht[k] = c[k]; } } else { for (let k = 0; k < hid; k++) { c[k] = ft[k] * c[k] + it[k] * gt[k]; ht[k] = zt[k] * Math.tanh(c[k]); } } h = ht; } const out = Array.from({ length: R }, (_, k) => h[k % hid] * (rng() * 2 - 1)); return out; };
  }
  if (kind === 'cnn' || kind === 'tcn') {
    const chOut = 8; const F = 12; const ker = 3; const Kw = Array.from({ length: chOut }, () => Array.from({ length: F * ker }, () => (rng() * 2 - 1) * 0.5));
    const dil = kind === 'tcn' ? 2 : 1;
    return function (w) { const feats = []; for (let co = 0; co < chOut; co++) { let mx = -1e9; for (let i = 0; i + ker * dil <= W; i += 1) { let s = 0; for (let kk = 0; kk < ker; kk++) { const idx = i + kk * dil; for (let cc = 0; cc < F; cc++) s += Kw[co][kk * F + cc] * w[idx][cc]; } s = Math.tanh(s); if (s > mx) mx = s; } feats.push(mx); } return Array.from({ length: R }, (_, k) => feats[k % chOut] * (rng() * 2 - 1)); };
  }
  if (kind === 'attn') {
    const d = 12; const F = 12; const Q = Array.from({ length: d }, () => Array.from({ length: F }, () => rng() * 2 - 1)); const Kk = Array.from({ length: d }, () => Array.from({ length: F }, () => rng() * 2 - 1)); const Vv = Array.from({ length: d }, () => Array.from({ length: F }, () => rng() * 2 - 1));
    return function (w) { const q = Q.map(r => r.reduce((p, v, j) => p + v * w[W - 1][j], 0)); const keys = w.map(x => Kk.map(r => r.reduce((p, v, j) => p + v * x[j], 0))); const vals = w.map(x => Vv.map(r => r.reduce((p, v, j) => p + v * x[j], 0))); let sum = 0; const e = keys.map(ks => { const dprod = q.reduce((p, v, i) => p + v * ks[i], 0) / Math.sqrt(d); const ev = Math.exp(dprod); sum += ev; return ev; }); const ctx = new Array(d).fill(0); e.forEach((ev, i) => { for (let j = 0; j < d; j++) ctx[j] += (ev / (sum || 1)) * vals[i][j]; }); return Array.from({ length: R }, (_, k) => ctx[k % d] * (rng() * 2 - 1)); };
  }
  if (kind === 'patch') {
    const P = 4; const chOut = 12; const Kw = Array.from({ length: chOut }, () => Array.from({ length: F * P }, () => (rng() * 2 - 1) * 0.5));
    return function (w) { const np = Math.floor(W / P); const feats = []; for (let co = 0; co < chOut; co++) { let mx = -1e9; for (let p = 0; p < np; p++) { let s = 0; for (let kk = 0; kk < P; kk++) for (let cc = 0; cc < F; cc++) s += Kw[co][kk * F + cc] * w[p * P + kk][cc]; s = Math.tanh(s); if (s > mx) mx = s; } feats.push(mx); } return Array.from({ length: R }, (_, k) => feats[k % chOut] * (rng() * 2 - 1)); };
  }
  if (kind === 'nbeats') {
    const basis = 24; const proj = Array.from({ length: R }, () => Array.from({ length: basis }, () => rng() * 2 - 1));
    return function (w) { const f = []; for (let i = 0; i < W; i++) for (let z = 0; z < 12; z++) f.push(w[i][z]); const b = []; for (let k = 0; k < basis; k++) { let s = 0; for (let j = 0; j < f.length; j++) s += Math.sin((k + 1) * j / f.length) * f[j]; b.push(s); } return proj.map(r => r.reduce((p, v, j) => p + v * b[j], 0)); };
  }
  // 默认 mlp
  return makeFrontend('mlp', rng);
}
function engineRC(frontendKinds, seed) {
  return function (occ, i0, t) {
    const { X, Y } = gatherSeq(occ, i0, t);
    if (X.length < 4) return occ[t - 1].slice();
    const rng = mulberry32(seed);
    const fes = frontendKinds.map(k => makeFrontend(k, rng));
    const States = X.map(w => rcTransform(fes, w, rng));
    const readout = ridgeReadout(States, Y, 0.1, R * frontendKinds.length);
    return readout(rcTransform(fes, seqWindow(occ, t - 1), rng));
  };
}
function engineSeasonalNaive() { // Chronos/TimeGPT 零样本代理（无法真预训练）→ 直接用上一期出现模式
  return function (occ, i0, t) { return occ[t - 1].slice(); };
}

// ---------- 模型目录（100+ 命名模型 → 实现映射）----------
function perZ(fn) { return makePerZodiac(fn); }
const CATALOG = [];
function add(id, name, layer, family, engine, note) { CATALOG.push({ id, name, layer, family, engine, note }); }

// ---- 第一层：深度序列（ELM/RC 随机特征）----
add('L1-01','Transformer(LCG未见模数)','L1','深度序列·注意力', engineRC(['attn'], 101), '随机注意力储备池');
add('L1-02','Transformer(PCG)','L1','深度序列·注意力', engineRC(['attn'], 102), '随机注意力储备池');
add('L1-03','Transformer(LCG m=2^32)','L1','深度序列·注意力', engineRC(['attn'], 103), '随机注意力储备池');
add('L1-04','Transformer(深度临界d=3)','L1','深度序列·注意力', engineRC(['attn'], 104), '随机注意力储备池');
add('L1-05','LSTM+BLSTM(LFSR/LCG)','L1','深度序列·RNN', engineRC(['rnn-lstm'], 105), '随机LSTM储备池');
add('L1-06','FFT-ATT-LSTM','L1','深度序列·RNN+注意力', engineRC(['rnn-lstm','attn'], 106), '随机LSTM+注意力储备池');
add('L1-07','LSTM(纯序列下一比特)','L1','深度序列·RNN', engineRC(['rnn-lstm'], 107), '随机LSTM储备池');
add('L1-08','FNN(前馈下一比特)','L1','深度序列·MLP', engineRC(['mlp'], 108), '随机MLP储备池');
add('L1-09','TCN(下一比特)','L1','深度序列·TCN', engineRC(['tcn'], 109), '随机TCN储备池');
add('L1-10','Transformer(短窗口基线)','L1','深度序列·注意力', engineRC(['attn'], 110), '随机注意力储备池');
add('L1-11','LSTM(短/中程>99%)','L1','深度序列·RNN', engineRC(['rnn-lstm'], 111), '随机LSTM储备池');
add('L1-12','Transformer(长程Rule90/62)','L1','深度序列·注意力', engineRC(['attn'], 112), '随机注意力储备池');
add('L1-13','BiLSTM-Transformer-GAN预测器','L1','深度序列·混合', engineRC(['rnn-lstm','attn'], 113), 'GAN无法真训练→随机LSTM+注意力储备池代理');
add('L1-14','CNN-LSTM','L1','深度序列·CNN+RNN', engineRC(['cnn','rnn-lstm'], 114), '随机CNN+LSTM储备池');
add('L1-15','LSTM-CNN','L1','深度序列·RNN+CNN', engineRC(['rnn-lstm','cnn'], 115), '随机LSTM+CNN储备池');
add('L1-16','CNN+LSTM并联','L1','深度序列·并联', engineRC(['cnn','rnn-lstm'], 116), '随机CNN+LSTM并联储备池');
add('L1-17','Parallel CNN-BiLSTM+Transformer','L1','深度序列·混合', engineRC(['cnn','rnn-lstm','attn'], 117), '随机CNN+BiLSTM+注意力储备池');
add('L1-18','TCN-LSTM','L1','深度序列·TCN+RNN', engineRC(['tcn','rnn-lstm'], 118), '随机TCN+LSTM储备池');
add('L1-19','CNN-LSTM-attention','L1','深度序列·混合', engineRC(['cnn','rnn-lstm','attn'], 119), '随机CNN+LSTM+注意力储备池');
add('L1-20','TCN-LSTM-attention','L1','深度序列·混合', engineRC(['tcn','rnn-lstm','attn'], 120), '随机TCN+LSTM+注意力储备池');
add('L1-21','GRU','L1','深度序列·RNN', engineRC(['rnn-gru'], 121), '随机GRU储备池');
add('L1-22','BiGRU','L1','深度序列·RNN', engineRC(['rnn-gru'], 122), '随机双向GRU储备池');
add('L1-23','BiLSTM','L1','深度序列·RNN', engineRC(['rnn-lstm'], 123), '随机双向LSTM储备池');
add('L1-24','深度前馈(ANN LCG回归)','L1','深度序列·MLP', engineRC(['mlp'], 124), '随机MLP储备池');
add('L1-25','ANN(RANDU回归)','L1','深度序列·MLP', engineRC(['mlp'], 125), '随机MLP储备池');
add('L1-26','ANN(ZX81 LCG回归)','L1','深度序列·MLP', engineRC(['mlp'], 126), '随机MLP储备池');
add('L1-27','Transformer-BiLSTM-cGAN','L1','深度序列·混合', engineRC(['attn','rnn-lstm'], 127), 'GAN无法真训练→随机注意力+LSTM储备池代理');
add('L1-28','Mamba/SSM','L1','深度序列·注意力', engineRC(['attn'], 128), 'SSM无库→随机线性注意力储备池代理');
add('L1-29','Perceiver/Perceiver IO','L1','深度序列·注意力', engineRC(['attn'], 129), '交叉注意力→随机注意力储备池代理');
add('L1-30','N-BEATS','L1','深度序列·基函数', engineRC(['nbeats'], 130), '随机基函数储备池');
add('L1-31','N-HiTS','L1','深度序列·基函数', engineRC(['nbeats'], 131), '随机多尺度基函数储备池');
add('L1-32','TimesNet','L1','深度序列·分块', engineRC(['patch'], 132), '随机分块(2D周期)储备池代理');
add('L1-33','PatchTST','L1','深度序列·分块', engineRC(['patch'], 133), '随机分块Transformer储备池');
add('L1-34','Crossformer','L1','深度序列·分块', engineRC(['patch'], 134), '随机跨维分块储备池');
add('L1-35','iTransformer','L1','深度序列·分块', engineRC(['patch'], 135), '随机倒置维度分块储备池');
add('L1-36','DLinear','L1','线性', engineTabularLinear(0.1), '线性映射（去偏基线之一）');
add('L1-37','TiDE','L1','深度序列·MLP', engineRC(['mlp'], 137), '随机密集编码器储备池');
add('L1-38','TFT','L1','深度序列·注意力', engineRC(['attn'], 138), '随机时序融合注意力储备池');
add('L1-39','Chronos(预训练时序LM)','L1','基础模型', engineSeasonalNaive(), '无法真预训练→零样本 seasonal-naive 代理（诚实标注）');
add('L1-40','TimeGPT/时序基础模型','L1','基础模型', engineSeasonalNaive(), '无法真预训练→零样本 seasonal-naive 代理（诚实标注）');

// ---- 第二层：统计时序 / 状态空间 ----
add('L2-41','AR(p)','L2','统计·自回归', perZ(perZArLm(3, 0.05)), '');
add('L2-42','MA(q)','L2','统计·滑动平均', perZ(perZArLm(3, 0.05)), 'MA归并到滞后回归');
add('L2-43','ARMA(p,q)','L2','统计·自回归滑动', perZ(perZArLm(3, 0.05)), 'ARMA→滞后回归');
add('L2-44','ARIMA(p,d,q)','L2','统计·差分AR', perZ(perZArima(3, 0.05)), '差分后AR');
add('L2-45','SARIMA','L2','统计·季节ARIMA', perZ(perZArima(3, 0.05)), '季节项→差分AR代理');
add('L2-46','SARIMAX','L2','统计·外生ARIMA', perZ(perZArima(3, 0.05)), '外生→差分AR代理');
add('L2-47','ARFIMA','L2','统计·长记忆', perZ(perZArLm(5, 0.05)), '分数阶→高阶AR代理');
add('L2-48','指数平滑','L2','统计·指数平滑', perZ(perZEwma(0.3)), 'EWMA');
add('L2-49','Holt线性趋势','L2','统计·指数平滑', perZ(perZEwma(0.3)), 'EWMA代理');
add('L2-50','Holt-Winters','L2','统计·季节平滑', perZ(perZEwma(0.3)), 'EWMA代理');
add('L2-51','ETS状态空间','L2','统计·状态空间', perZ(perZEwma(0.3)), 'EWMA代理');
add('L2-52','Theta方法','L2','统计·Theta', perZ(perZEwma(0.3)), 'EWMA代理');
add('L2-53','TBATS','L2','统计·Box-Cox+季节', perZ(perZEwma(0.3)), 'EWMA代理');
add('L2-54','Kalman局部水平/趋势','L2','状态空间·Kalman', perZ(perZKalman()), '');
add('L2-55','Kalman-ARIMA混合','L2','状态空间·混合', perZ(perZKalman()), 'Kalman代理');
add('L2-56','GARCH(p,q)','L2','统计·条件异方差', perZ(perZGarch(0.05)), 'ARCH特征回归');
add('L2-57','EGARCH','L2','统计·非对称GARCH', perZ(perZGarch(0.05)), 'ARCH特征回归');
add('L2-58','GJR-GARCH','L2','统计·门限GARCH', perZ(perZGarch(0.05)), 'ARCH特征回归');
add('L2-59','GARCH-M','L2','统计·均值GARCH', perZ(perZGarch(0.05)), 'ARCH特征回归');
add('L2-60','ARCH(q)','L2','统计·ARCH', perZ(perZGarch(0.05)), 'ARCH特征回归');
add('L2-61','马尔可夫链(一阶)','L2','统计·马尔可夫', perZ(perZMarkov(1)), '');
add('L2-62','高阶马尔可夫','L2','统计·马尔可夫', perZ(perZMarkov(2)), '');
add('L2-63','隐马尔可夫HMM','L2','统计·HMM', perZ(perZHMM()), '2态简化EM');
add('L2-64','马尔可夫-switching AR','L2','统计·区制', perZ(perZArLm(3, 0.05)), '区制→滞后回归代理');
add('L2-65','泊松自回归PAR','L2','统计·计数', perZ(perZCount(0.05)), '计数→滞后回归');
add('L2-66','INAR','L2','统计·整数AR', perZ(perZCount(0.05)), '→滞后回归');
add('L2-67','二项AR(1)','L2','统计·二项', perZ(perZArLm(1, 0.05)), '→一阶滞后');
add('L2-68','二项MA(q)','L2','统计·二项', perZ(perZArLm(2, 0.05)), '→滞后回归');
add('L2-69','二项ARMA','L2','统计·二项', perZ(perZArLm(3, 0.05)), '→滞后回归');
add('L2-70','随机系数AR','L2','统计·随机系数', perZ(perZArLm(3, 0.05)), '→滞后回归');
add('L2-71','分数高斯噪声fGn','L2','统计·长记忆', perZ(perZArLm(5, 0.05)), '→高阶AR');
add('L2-72','灰色GM(1,1)','L2','灰色·GM', perZ(perZGrey()), '');
add('L2-73','灰色Verhulst','L2','灰色·Verhulst', perZ(perZGrey()), 'S型→GM代理');
add('L2-74','ACD','L2','统计·持续时间', perZ(perZIntensity(0.3)), '强度代理');
add('L2-75','Hawkes过程','L2','统计·自激励', perZ(perZIntensity(0.3)), '强度代理');

// ---- 第三层：经典 ML / 核方法 ----
add('L3-76','线性回归','L3','ML·线性', engineTabularLinear(0.0), '');
add('L3-77','岭回归Ridge','L3','ML·线性', engineTabularLinear(0.1), '');
add('L3-78','LASSO','L3','ML·线性', engineTabularLinear(0.3), 'L1→强岭(近似)');
add('L3-79','Elastic Net','L3','ML·线性', engineTabularLinear(0.2), '弹性网→岭(近似)');
add('L3-80','决策树回归','L3','ML·树', engineTree(4), '');
add('L3-81','决策树分类','L3','ML·树', engineTree(4), '分类→回归树(12输出)');
add('L3-82','随机森林回归','L3','ML·集成', engineRF(20, 4), '');
add('L3-83','随机森林分类','L3','ML·集成', engineRF(20, 4), '同RF');
add('L3-84','梯度提升GBM','L3','ML·集成', engineGBM(30, 2, 0.1), '');
add('L3-85','XGBoost','L3','ML·集成', engineGBM(30, 2, 0.1), 'XGB库不可装→浅层GBM代理');
add('L3-86','LightGBM','L3','ML·集成', engineGBM(30, 2, 0.1), 'LightGBM库不可装→浅层GBM代理');
add('L3-87','CatBoost','L3','ML·集成', engineGBM(30, 2, 0.1), 'CatBoost库不可装→浅层GBM代理');
add('L3-88','SVR(RBF核)','L3','ML·核', engineKernelRidge('rbf', 1.0, 0.1), '');
add('L3-89','SVM分类','L3','ML·核', engineKernelRidge('rbf', 1.0, 0.1), '同SVR');
add('L3-90','高斯过程回归GPR','L3','ML·核', engineKernelRidge('rbf', 1.0, 0.1), 'GPR→RBF核岭');
add('L3-91','GPR+SVR混合','L3','ML·核', engineKernelRidge('rbf', 1.0, 0.1), '同核岭');
add('L3-92','KNN回归','L3','ML·近邻', engineKNN(8), '');
add('L3-93','KNN分类','L3','ML·近邻', engineKNN(8), '同KNN');
add('L3-94','朴素贝叶斯NB','L3','ML·贝叶斯', engineLogistic(40, 0.5, 0.0), 'NB→多输出Logistic(条件独立近似)');
add('L3-95','逻辑回归','L3','ML·线性', engineLogistic(40, 0.5, 0.0), '');
add('L3-96','线性判别LDA','L3','ML·线性', engineTabularLinear(0.05), 'LDA→共享协方差岭(近似)');
add('L3-97','偏最小二乘PLS','L3','ML·线性', engineTabularLinear(0.2), 'PLS→岭(近似)');

// ---- 第四层：混沌/动力学/储备池 ----
add('L4-98','ESN储备池','L4','储备池', engineRC(['mlp'], 198), '随机储备池+岭读出（标准RC）');
add('L4-99','小波-ESN(WESN)','L4','储备池', engineRC(['mlp'], 199), 'ESN代理(无显式小波)');
add('L4-100','RFCM储备池','L4','储备池', engineRC(['mlp'], 200), 'ESN代理');
add('L4-101','Logistic映射外推','L4','混沌·映射', perZ(perZMap(0.05)), '混沌映射→非线性AR代理(诚实标注)');
add('L4-102','Tent映射外推','L4','混沌·映射', perZ(perZMap(0.05)), '混沌映射→非线性AR代理(诚实标注)');

// ---------- 回测主循环 ----------
function runModel(m) {
  const i0 = Math.max(W, T - EVAL_LEN - TRAIN_MAX); // 训练窗口起点
  let wins = 0, expSum = 0, maPicks = 0, n = 0;
  for (let t = T - EVAL_LEN; t < T; t++) {
    let scores;
    try { scores = m.engine(occ, i0, t); } catch (e) { return { error: String(e && e.message || e).slice(0, 120) }; }
    if (!scores || scores.length !== 12) return { error: 'bad output' };
    let best = 0; for (let z = 1; z < 12; z++) if (scores[z] > scores[best]) best = z;
    const hit = occ[t][best]; if (hit) wins++;
    expSum += marginal[best]; if (NAMES[best] === '马') maPicks++;
    n++;
  }
  const winRate = wins / n;
  const expected = expSum / n;
  return { winRate, expected, excess: winRate - expected, maPicks: maPicks / n, n };
}

const models = QUICK ? CATALOG.filter(m => ['L1-08','L1-05','L2-41','L2-54','L2-61','L3-76','L3-82','L4-98','L1-39'].includes(m.id)) : CATALOG;

// ---------- 仅判定模式（重新跑开奖判定 + 统计，不影响预测） ----------
if (process.argv.includes('--judge')) { judgeRecords(); process.exit(0); }

// ---------- 前向预测模式：每个模型对「下一期」单独出 1 个生肖 ----------
if (process.argv.includes('--predict')) {
  const picks = [], tally = new Array(12).fill(0);
  for (const m of models) {
    let scores;
    try { scores = m.engine(occ, W, T); } catch (e) { picks.push({ id: m.id, name: m.name, layer: m.layer, family: m.family, error: String(e && e.message || e).slice(0, 120) }); continue; }
    if (!scores || scores.length !== 12) { picks.push({ id: m.id, name: m.name, layer: m.layer, family: m.family, error: 'bad output' }); continue; }
    let best = 0; for (let z = 1; z < 12; z++) if (scores[z] > scores[best]) best = z;
    tally[best]++;
    picks.push({ id: m.id, name: m.name, layer: m.layer, family: m.family, zodiac: NAMES[best], score: +scores[best].toFixed(4), scores: NAMES.map((z, i) => [z, +scores[i].toFixed(4)]) });
  }
  const consensus = NAMES.map((z, i) => ({ zodiac: z, votes: tally[i] })).sort((a, b) => b.votes - a.votes);
  const lastPeriod = draws[T - 1].period;
  const pred = { generated: new Date().toISOString(), sourcePeriod: lastPeriod, predictFor: String(Number(lastPeriod) + 1), nModels: models.length, consensus, picks };
  // 守卫：预测对象未推进（同一期）则不重写 predict 文件，避免 nightly 每5分钟空转提交（时间戳变化）
  let skipPredWrite = false;
  try { const ex = JSON.parse(fs.readFileSync(PRED_PATH, 'utf8')); if (ex && ex.predictFor === pred.predictFor) skipPredWrite = true; } catch (e) {}
  if (!skipPredWrite) fs.writeFileSync(PRED_PATH, JSON.stringify(pred, null, 2));
  else console.log('predict 文件已是最新(' + pred.predictFor + ')，跳过重写');
  recordPrediction(pred);
  judgeRecords();
  console.log('=== 100模型 对下一期(' + pred.predictFor + ') 单独预测生肖 ===');
  console.log('共识(得票数降序): ' + consensus.map(c => c.zodiac + ':' + c.votes).join('  '));
  console.log('结果已写: E:/六子/_tools/bt100_predict.json');
  process.exit(0);
}

const results = [];
const t0 = Date.now();
for (const m of models) {
  const tm = Date.now();
  const r = runModel(m);
  results.push({ id: m.id, name: m.name, layer: m.layer, family: m.family, note: m.note, ...r });
  if (!QUICK) console.error('[' + results.length + '/' + models.length + ']', m.id, (Date.now() - tm) + 'ms', r.error ? ('ERR ' + r.error) : ('win=' + (r.winRate * 100).toFixed(1) + '% ex=' + (r.excess * 100).toFixed(1) + 'pp ma%=' + (r.maPicks * 100).toFixed(0)));
}
if (QUICK) { console.log('QUICK OK'); process.exit(0); }

// 排序：去偏超额降序
results.sort((a, b) => (b.excess || -9) - (a.excess || -9));
const out = {
  generated: new Date().toISOString(),
  data: 'macaujc2 EMBEDDED_DATA', periods: T, evalLen: EVAL_LEN, window: W,
  baseline: { randomSinglePick: baseRand, alwaysMa: baseMa, marginal },
  results
};
const outPath = process.env.BT_OUT || (QUICK ? 'E:/六子/_tools/bt100_quick.json' : 'E:/六子/_tools/bt100_result.json');
fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log('=== 100模型 zodiac 预测 walk-forward 回测完成 ===');
console.log('基线：随机单生肖=' + (baseRand * 100).toFixed(2) + '% | 永远押马=' + (baseMa * 100).toFixed(2) + '%');
console.log('--- 去偏超额 Top15（actual − 按所选生肖边际期望）---');
for (const r of results.slice(0, 15)) {
  console.log(r.id.padEnd(7), r.name.padEnd(22), 'win=' + (r.winRate * 100).toFixed(1) + '%', 'excess=' + (r.excess * 100).toFixed(1) + 'pp', 'ma%=' + (r.maPicks * 100).toFixed(0), r.error ? ('ERR:' + r.error) : '');
}
console.log('--- 去偏超额 Bottom5 ---');
for (const r of results.slice(-5)) {
  console.log(r.id.padEnd(7), r.name.padEnd(22), 'win=' + (r.winRate * 100).toFixed(1) + '%', 'excess=' + (r.excess * 100).toFixed(1) + 'pp', r.error ? ('ERR:' + r.error) : '');
}
console.log('结果已写:', outPath);
