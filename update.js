/**
 * 六子预测系统 —— 自动更新脚本（本地 / GitHub Actions 通用）
 *
 * 数据源：history.macaumarksix.com 的单期查询接口 /history/macaujc2/expect/<期号>
 *   —— macaujc2 = 「新澳门六合彩」（21:32开奖），用户 2026-10-17 确认要的是这套；
 *      旧的 macaujc（22:32开奖，特码04那期那套）=「澳门六合彩」，已弃用。
 *
 * 流程（v2 事前预测制，用户2026-10-19定版）：
 *   1. 读取现有 EMBEDDED_DATA 作为基线（已是全量历史）
 *   2. 单期接口增量追加缺失的最新期（最多查到未来 40 期，连续 3 期无数据即停）
 *   3. 对末尾 5 期做核对（防历史数据污染复发；仅当 2/2 一致且与本地不同才覆盖）
 *   4. 按实判定：对已开奖但未记录的期，从 predictions.json「事前预测存档」取当时预测，
 *      用 algos10.judgeDraw 判定命中——★严禁事后重算★（存档没有该期预测则跳过不补）
 *   5. 事前预测下一期：algos10 v2 十模型 Ensemble（7号码级模型 + Walk-forward 动态权重），
 *      写 prediction_result.json，并追加进 predictions.json 存档（明晚开奖后按它判定）
 *   6. 写回 index.html / record.json / prediction_result.json / predictions.json
 *
 * 用法：
 *   node update.js            # 只更新本地文件
 *   node update.js --upload   # 更新后上传到 GitHub（本地用；Actions 自带 git push，不传此参）
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = __dirname;
// 本地：脚本在 E:\六子，文件在 vercel_deploy 子目录
// GitHub Actions：脚本与文件同在仓库根目录
const DEPLOY = fs.existsSync(path.join(ROOT, 'vercel_deploy'))
  ? path.join(ROOT, 'vercel_deploy')
  : ROOT;
const HTML_PATH = path.join(DEPLOY, 'index.html');
const RECORD_PATH = path.join(DEPLOY, 'record.json');
const PRED_PATH = path.join(DEPLOY, 'prediction_result.json');
const PRED_ARCHIVE_PATH = path.join(DEPLOY, 'predictions.json'); // 事前预测存档（按期号累积）

// 单期查询接口（macaujc2 = 新澳门六合彩，21:32开奖；2026-10-17 全量切换）
const EXP = 'https://history.macaumarksix.com/history/macaujc2/expect/';

const UPLOAD = process.argv.includes('--upload');

function log(...a) { console.log(...a); }

// ---------- 加载页面算法 ----------
function loadPageAlgorithms() {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const code = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
  const store = {};
  const mk = id => store[id] || (store[id] = {
    textContent: '', innerHTML: '', value: '', style: {}, disabled: false,
    addEventListener() {}, removeEventListener() {}, focus() {}, click() {},
  });
  const ctx = {
    console: { log() {}, warn() {}, error() {} },
    document: { getElementById: mk, querySelector: () => mk('q'), addEventListener: () => {}, hidden: false },
    navigator: { serviceWorker: { register: () => Promise.reject() }, clipboard: { writeText: () => {} } },
    window: { addEventListener: () => {} },
    fetch: () => Promise.reject(new TypeError('offline')),
    setInterval: () => 0, clearInterval: () => 0, setTimeout: () => 0,
    caches: { keys: () => Promise.resolve([]), delete: () => Promise.resolve() },
    atob: (s) => Buffer.from(s, 'base64').toString('latin1'),
    btoa: (s) => Buffer.from(s, 'latin1').toString('base64'),
    alert: () => {}, Date, Math, JSON, Object, Array, String, Number, Map, Set, isNaN, parseInt,
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(code + '\n;globalThis.__api={compositePredict,getNumberZodiac,zodiacPredictSpecial};', ctx);
  return ctx.__api;
}

// ---------- 单期查询（带超时 + 重试） ----------
async function fetchExpect(p) {
  for (let i = 0; i < 4; i++) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 8000);
      const r = await fetch(EXP + p, { cache: 'no-store', signal: ctrl.signal });
      clearTimeout(t);
      const j = await r.json();
      const d = j.data;
      let code = null, time = null;
      if (Array.isArray(d) && d[0]) { code = d[0].openCode; time = d[0].openTime; }
      else if (d && d.expect) { code = d.openCode; time = d.openTime; }
      if (code && /^\d{1,2}(,\d{1,2}){6}$/.test(code)) return { code, time };
    } catch (e) { /* retry */ }
    if (i < 3) await new Promise(r => setTimeout(r, 200));
  }
  return null;
}

// 双次查询取众数，降低源站节点不同步导致的误判
async function fetchExpectVote(p) {
  const seen = {};
  for (let i = 0; i < 2; i++) {
    const r = await fetchExpect(p);
    if (r) {
      const k = r.code + '|' + (r.time || '');
      seen[k] = (seen[k] || 0) + 1;
    }
    if (i < 1) await new Promise(r => setTimeout(r, 150));
  }
  const best = Object.entries(seen).sort((a, b) => b[1] - a[1])[0];
  return best ? { code: best[0].split('|')[0], time: best[0].split('|')[1] } : null;
}

// ---------- 读取现有 EMBEDDED_DATA ----------
function readEmbeddedData(html) {
  const m = html.match(/const EMBEDDED_DATA = \[([\s\S]*?)\n\];/);
  if (!m) throw new Error('未找到 EMBEDDED_DATA 块');
  const body = m[1];
  const rows = [...body.matchAll(/\["(\d{7})","([^"]+)","([^"]+)"\]/g)]
    .map(x => {
      const nums = x[3].split(',').map(n => parseInt(n, 10));
      return { period: x[1], date: x[2], numbers: nums.slice(0, 6), special: nums[6] };
    });
  return rows.filter(r => r.numbers.length === 6 && !isNaN(r.special));
}

// ---------- 增量拉取：追加新期 + 末尾核对 ----------
async function syncDraws(baseRows) {
  const known = new Map(baseRows.map(r => [r.period, r]));
  const result = baseRows.map(r => ({ ...r }));
  const maxP = baseRows[baseRows.length - 1].period;
  let added = 0, fixed = 0;

  // 1) 追加新期：从 maxP+1 查起，连续 3 期无数据即停
  let gap = 0;
  for (let n = (+maxP) + 1; n <= (+maxP) + 40; n++) {
    const p = String(n);
    const r = await fetchExpectVote(p);
    if (!r) { if (++gap >= 3) break; else continue; }
    gap = 0;
    if (known.has(p)) continue;
    const nums = r.code.split(',').map(Number);
    if (nums.length !== 7 || nums.some(isNaN)) continue;
    result.push({ period: p, date: (r.time || '').slice(0, 10), numbers: nums.slice(0, 6), special: nums[6] });
    added++;
    log('🆕 新增期', p, '→', r.code);
  }

  // 2) 末尾 5 期核对（防历史数据污染复发；仅当 2/2 完全一致且与本地不同才覆盖）
  const tail = result.slice(-5);
  for (const r of tail) {
    const remote = await fetchExpectVote(r.period);
    if (!remote) continue;
    const remoteNums = remote.code.split(',').map(Number);
    if (remoteNums.length !== 7 || remoteNums.some(isNaN)) continue;
    const same = remoteNums.join(',') === [...r.numbers, r.special].join(',');
    if (!same) {
      r.numbers = remoteNums.slice(0, 6);
      r.special = remoteNums[6];
      r.date = (remote.time || '').slice(0, 10);
      fixed++;
      log('🔧 修正期', r.period, '→', remote.code);
    }
  }

  result.sort((a, b) => a.period.localeCompare(b.period));
  return { result, added, fixed };
}

// ---------- 判定命中（D4 多属性命中判定，统一走 algos10.judgeDraw）----------
// code_hit  ：特码 ∈ 预测23码
// zodiac_hit：开奖7球任一生肖 == 预测主推生肖
// six_hit   ：特码生肖 ∈ 预测6肖
// 注意：只用「事前预测」判定，绝不做事后重算（用户2026-10-19铁律）

// 下一期期号（连续期号，处理跨年边界）
function nextPeriodOf(period) {
  const y = +period.slice(0, 4);
  const seq = +period.slice(4);
  if (seq >= 366) return (y + 1) + '001';
  return y + String(seq + 1).padStart(3, '0');
}

// ---------- 主流程 ----------
(async () => {
  log('===== 六子自动更新 =====');
  const api = loadPageAlgorithms();
  // algos10 v2 十模型引擎（事前预测 + D4多属性命中判定）
  const algos10 = require(path.join(__dirname, 'algos10.js'));

  let html = fs.readFileSync(HTML_PATH, 'utf8');
  const baseRows = readEmbeddedData(html);
  if (!baseRows.length) { log('❌ 未读到任何本地开奖数据，中止'); process.exit(1); }
  log('本地基线 :', baseRows.length, '期，最大', baseRows[baseRows.length - 1].period);

  const { result: draws, added, fixed } = await syncDraws(baseRows);
  if (!added && !fixed) log('（暂无新开奖 / 末端无变化）');

  const latest = draws[draws.length - 1];
  log('最新开奖 :', latest.period, '(' + latest.date + ')',
      [...latest.numbers, latest.special].join(',') + ' + ' + latest.special);

  const record = JSON.parse(fs.readFileSync(RECORD_PATH, 'utf8'));
  const pred = JSON.parse(fs.readFileSync(PRED_PATH, 'utf8'));
  const history = record.history || [];
  const lastRec = history[history.length - 1];
  log('战绩末期 :', lastRec ? lastRec.period : '(空)');
  log('存档预测 :', pred.next_period, '| 生肖', pred.zodiac);

  // 1) 按实判定：对已开奖未记录的期，从事前预测存档（predictions.json）取「当时」的预测判定
  //    ★用户铁律（2026-10-19）：只做事前预测，事后按实记录，严禁用现在历史重算过去★
  const FIRST_PERIOD = '2026293';   // 2026-10-20 03:18 用户再次清空，从293期重新开始（292已判记录清除）
  let predArchive = {};
  try { predArchive = JSON.parse(fs.readFileSync(PRED_ARCHIVE_PATH, 'utf8')); } catch (e) {}

  // 0) 幂等守卫（v5.4 用户要求「21:35起每5分钟、跑到成功为止」的等效实现）：
  //    今晚工作已完成 = 最新开奖期已判定 + 下一期预测已存档 → 秒退不重复劳动。
  //    后续每5分钟的 cron 触发都空转（几秒退出，无提交无副作用）。
  //    workflow_dispatch 手动触发跳过守卫（GITHUB_EVENT_NAME=workflow_dispatch），永远强制执行。
  const isDispatch = process.env.GITHUB_EVENT_NAME === 'workflow_dispatch';
  const latestP = latest.period;
  const nextP = nextPeriodOf(latestP);
  const recDone = history.some(h => h.period === latestP);
  const archDone = !!(predArchive[nextP] && predArchive[nextP].zodiac);
  if (!isDispatch && recDone && archDone) {
    log('✅ 今晚工作已完成（' + latestP + ' 已判定 + ' + nextP + ' 预测已存档），跳过本次运行');
    log('\n===== 完成（幂等秒退）=====');
    process.exit(0);
  }
  log('今晚待办: ' + latestP + (recDone ? ' 已判定' : ' 待判定') + ' / ' + nextP + (archDone ? ' 预测已存档' : ' 待预测存档'));

  const missing = draws
    .filter(d => !history.some(h => h.period === d.period))
    .filter(d => lastRec ? (d.period > lastRec.period) : (d.period >= FIRST_PERIOD))
    .sort((a, b) => a.period.localeCompare(b.period))
    .slice(0, 5);

  let changed = false;
  for (const d of missing) {
    const saved = predArchive[d.period];   // 事前预测存档
    if (!saved || !saved.zodiac) {
      log('⏭ 跳过', d.period, '（事前无预测存档，按实不补录——严禁事后重算）');
      continue;
    }
    const { codeHit, zodiacHit, sixHit } = algos10.judgeDraw(saved, d);
    history.push({
      period: d.period,
      actual_numbers: d.numbers,
      actual_special: d.special,
      pred_zodiac: saved.zodiac,
      pred_top3: saved.top3 || null,
      pred_six: saved.six || null,
      pred_codes: saved.codes || null,
      code_hit: codeHit,
      zodiac_hit: zodiacHit,
      six_hit: sixHit,
    });
    log('✅ 按实判定', d.period, '→ 事前预测', saved.zodiac,
        '| 23码', codeHit ? '√' : '×', '| 平特肖', zodiacHit ? '√' : '×', '| 6肖', sixHit ? '√' : '×');
    changed = true;
  }
  if (!missing.length) log('（战绩无需补录）');

  // 2) 写 record.json（23码 + 平特肖 + 6肖 三指标）
  const total = history.length;
  const codeWin = history.filter(h => h.code_hit === 1).length;
  const zodiacWin = history.filter(h => h.zodiac_hit === 1).length;
  const sixWin = history.filter(h => h.six_hit === 1).length;
  fs.writeFileSync(RECORD_PATH, JSON.stringify({ total, code_win: codeWin, zodiac_win: zodiacWin, six_win: sixWin, history }, null, 2) + '\n', 'utf8');
  log('战绩更新 : 累计' + total + '期  23码 ' + codeWin + '/' + total +
      ' (' + (total ? (codeWin / total * 100).toFixed(1) : 0) + '%)  平特肖 ' + zodiacWin + '/' + total +
      ' (' + (total ? (zodiacWin / total * 100).toFixed(1) : 0) + '%)  6肖 ' + sixWin + '/' + total +
      ' (' + (total ? (sixWin / total * 100).toFixed(1) : 0) + '%)');

  // 3) 事前预测下一期：algos10 v2 十模型 Ensemble（7号码级 + Walk-forward 动态权重）
  const nextPeriod = nextPeriodOf(latest.period);
  const today = new Date(Date.now() + 8 * 3600 * 1000).toISOString().slice(0, 10);
  let fusedOut = null;
  try {
    fusedOut = algos10.predictAll(draws.map(d => ({
      period: d.period, numbers: d.numbers, special: d.special,
    })));
    log('十模型预测:', Object.entries(fusedOut.details).map(([k, v]) => k + '=' + v.zodiac).join(' '));
    log('动态权重  :', Object.entries(fusedOut.weights).map(([k, v]) => k + '=' + (+v).toFixed(3)).join(' '));
  } catch (e) {
    log('⚠️ 十模型融合失败，回退页面单算法:', e.message);
  }
  const result = api.compositePredict(draws.map(d => ({
    period: d.period, numbers: d.numbers, special: d.special,
  })));
  const predObj = fusedOut ? Object.assign({}, fusedOut.fused, {
    fused: fusedOut.fused,
    details: fusedOut.details,
    weights: fusedOut.weights,
    consensus: fusedOut.consensus || null,   // v6：共识投票（前端展示一致度/投票明细）
    zodiacScores: fusedOut.zodiacScores || null,   // v8.1：融合后12生肖分数全景（网页展示）
    generated_at: today,
    next_period: nextPeriod,
    data_until: latest.period,
  }) : {
    zodiac: result.zodiac, top3: result.top3, probs: result.probs,
    codes: result.codes, six: result.six,
    generated_at: today, next_period: nextPeriod, data_until: latest.period,
  };
  fs.writeFileSync(PRED_PATH, JSON.stringify(predObj, null, 2) + '\n', 'utf8');
  log('新预测   :', nextPeriod, '| 融合生肖', predObj.zodiac, '| 6肖', (predObj.six || []).join('/'));

  // 3.5) 把「下一期的事前预测」追加进 predictions.json 存档（该期开奖后按它判定，绝不重算）
  if (fusedOut) {
    predArchive[nextPeriod] = {
      zodiac: fusedOut.fused.zodiac,
      top3: fusedOut.fused.top3,
      six: fusedOut.fused.six,
      codes: fusedOut.fused.codes,
      consensus: fusedOut.consensus || null,     // v6：共识投票存档
      skipped: !!(fusedOut.consensus && fusedOut.consensus.action === 'skip'), // 共识不足→本期不下注
      saved_at: today,
    };
    fs.writeFileSync(PRED_ARCHIVE_PATH, JSON.stringify(predArchive, null, 2) + '\n', 'utf8');
    log('存档预测 :', nextPeriod, '已存入 predictions.json（共', Object.keys(predArchive).length, '期）');
  }

  // 3.6) 云端生成 bet_table.json（v5.3 全云端化：数据源=predictions存档+record判定，不依赖桌面Excel）
  try {
    const { gen } = require(path.join(__dirname, 'gen_bet_table.js'));
    const bt = gen();
    log('投注表   : bet_table.json 云端生成', bt.rows.length, '行 | 现在本金', bt.kpi.current);
  } catch (e) {
    log('⚠️ bet_table 云端生成失败（不影响其他数据）:', e.message);
  }

  // 4) 重建 EMBEDDED_DATA / EMBEDDED_RECORDS 并写回 index.html
  const edLines = draws.map(d =>
    '["' + d.period + '","' + d.date + '","' +
    [...d.numbers, d.special].map(n => String(n).padStart(2, '0')).join(',') + '"]'
  );
  const edBlock = 'const EMBEDDED_DATA = [\n' + edLines.join(',\n') + '\n];';
  html = html.replace(/const EMBEDDED_DATA = \[[\s\S]*?\n\];/, edBlock);

  const erLines = history.map(h => {
    // 事前预测字段（v2：存档预测随战绩一起内嵌，前端直接读，不重算）
    const predPart = h.pred_zodiac
      ? ',"pred_zodiac":"' + h.pred_zodiac + '"' +
        ',"pred_top3":' + (h.pred_top3 ? JSON.stringify(h.pred_top3) : 'null') +
        ',"pred_six":' + (h.pred_six ? JSON.stringify(h.pred_six) : 'null') +
        ',"pred_codes":' + (h.pred_codes ? JSON.stringify(h.pred_codes) : 'null')
      : '';
    return '  {"period":"' + h.period + '","actual_numbers":[' + h.actual_numbers.join(',') + '],' +
      '"actual_special":' + h.actual_special + predPart +
      ',"code_hit":' + h.code_hit + ',"zodiac_hit":' + h.zodiac_hit + ',"six_hit":' + h.six_hit + '}';
  });
  const erBlock = 'const EMBEDDED_RECORDS = [\n' + erLines.join(',\n') + '\n];';
  html = html.replace(/const EMBEDDED_RECORDS = \[[\s\S]*?\n\];/, erBlock);

  fs.writeFileSync(HTML_PATH, html, 'utf8');
  log('index.html 已更新 : 内嵌', draws.length, '期 / 战绩', history.length, '条');

  // 同步源文件（本地才有 macau_workstation.html）
  const SRC = path.join(ROOT, 'macau_workstation.html');
  if (fs.existsSync(SRC)) fs.copyFileSync(HTML_PATH, SRC);

  if (!changed && !added && !fixed) log('\n（数据无变化）');

  // 5) 可选上传
  if (UPLOAD) {
    log('\n===== 上传 GitHub =====');
    const { spawnSync } = require('child_process');
    const script = path.join(ROOT, 'github_upload.py');
    let ok = false;
    for (const py of ['python', 'python3', 'py',
      'C:/Users/30391/.workbuddy/binaries/python/versions/3.13.12/python.exe']) {
      const r = spawnSync(py, [script], { stdio: 'inherit' });
      if (r.status === 0) { ok = true; break; }
    }
    if (!ok) log('❌ 上传失败：未找到可用的 python，或上传过程出错');
  }
  log('\n===== 完成 =====');
})();
