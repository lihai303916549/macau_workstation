/**
 * 六子预测系统 —— 自动更新脚本（本地 / GitHub Actions 通用）
 *
 * 数据源：history.macaumarksix.com 的单期查询接口 /history/macaujc2/expect/<期号>
 *   —— macaujc2 = 「新澳门六合彩」（21:32开奖），用户 2026-10-17 确认要的是这套；
 *      旧的 macaujc（22:32开奖，特码04那期那套）=「澳门六合彩」，已弃用。
 * 做法：用 vm 加载 index.html 里的算法，保证与页面 100% 一致（不另写一套算法）。
 *
 * 流程：
 *   1. 读取现有 EMBEDDED_DATA 作为基线（已是全量历史）
 *   2. 单期接口增量追加缺失的最新期（最多查到未来 40 期，连续 3 期无数据即停）
 *   3. 对末尾 5 期做核对（防历史数据污染复发；仅当 2/2 一致且与本地不同才覆盖）
 *   4. 用上一期存档预测判定平特肖命中
 *   5. 重写 EMBEDDED_DATA / EMBEDDED_RECORDS，重算下一期预测
 *   6. 写回 index.html / record.json / prediction_result.json
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

// ---------- 判定命中 ----------
// 23码命中：特码(特码)球落在 23 码内
// 平特肖命中：开奖 7 个号（6正码+特码）任一生肖 == 预测生肖（网页一直显示的口径，与 zodiacPredictSpecial 一致）
function judge(pred, draw, getNumberZodiac) {
  const codeHit = (pred.codes && pred.codes.includes(draw.special)) ? 1 : 0;
  const zodiacHit = ([...draw.numbers, draw.special].some(n => getNumberZodiac(n) === pred.zodiac)) ? 1 : 0;
  const sixHit = (pred.six && pred.six.includes(getNumberZodiac(draw.special))) ? 1 : 0;
  return { codeHit, zodiacHit, sixHit };
}

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

  // 1) 补录所有缺失的期（用「该期之前的真实历史」重跑算法生成预测，再判定命中）
  const FIRST_PERIOD = '2026289';
  const missing = draws
    .filter(d => !history.some(h => h.period === d.period))
    .filter(d => lastRec ? (d.period > lastRec.period) : (d.period >= FIRST_PERIOD))
    .sort((a, b) => a.period.localeCompare(b.period))
    .slice(0, 5);

  let changed = false;
  for (const d of missing) {
    const before = draws.filter(x => x.period < d.period);
    if (before.length < 30) continue;   // 历史太少，算法不可靠，跳过
    const p = api.compositePredict(before.map(x => ({
      period: x.period, numbers: x.numbers, special: x.special,
    })));
    const { codeHit, zodiacHit, sixHit } = judge({ codes: p.codes, zodiac: p.zodiac, six: p.six }, d, api.getNumberZodiac);
    history.push({
      period: d.period,
      actual_numbers: d.numbers,
      actual_special: d.special,
      code_hit: codeHit,
      zodiac_hit: zodiacHit,
      six_hit: sixHit,
    });
    log('✅ 补录', d.period, '→ 23码', codeHit ? '√' : '×', '| 平特肖', zodiacHit ? '√' : '×', '| 6肖', sixHit ? '√' : '×', '(按当时历史重算)');
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

  // 3) 生成下一期预测：前10种算法加权融合（权重=2026-10-16 全量613期回测超额）
  const nextPeriod = nextPeriodOf(latest.period);
  const today = new Date(Date.now() + 8 * 3600 * 1000).toISOString().slice(0, 10);
  let fusedOut = null;
  try {
    const algos10 = require(path.join(__dirname, 'algos10.js'));
    fusedOut = algos10.predictAll(draws.map(d => ({
      period: d.period, numbers: d.numbers, special: d.special,
    })));
    log('10算法预测:', Object.entries(fusedOut.details).map(([k, v]) => k + '=' + v.zodiac).join(' '));
  } catch (e) {
    log('⚠️ 10算法融合失败，回退页面单算法:', e.message);
  }
  const result = api.compositePredict(draws.map(d => ({
    period: d.period, numbers: d.numbers, special: d.special,
  })));
  const predObj = fusedOut ? Object.assign({}, fusedOut.fused, {
    fused: fusedOut.fused,
    details: fusedOut.details,
    weights: fusedOut.weights,
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

  // 4) 重建 EMBEDDED_DATA / EMBEDDED_RECORDS 并写回 index.html
  const edLines = draws.map(d =>
    '["' + d.period + '","' + d.date + '","' +
    [...d.numbers, d.special].map(n => String(n).padStart(2, '0')).join(',') + '"]'
  );
  const edBlock = 'const EMBEDDED_DATA = [\n' + edLines.join(',\n') + '\n];';
  html = html.replace(/const EMBEDDED_DATA = \[[\s\S]*?\n\];/, edBlock);

  const erLines = history.map(h =>
    '  {"period":"' + h.period + '","actual_numbers":[' + h.actual_numbers.join(',') + '],' +
    '"actual_special":' + h.actual_special + ',"code_hit":' + h.code_hit + ',"zodiac_hit":' + h.zodiac_hit + ',"six_hit":' + h.six_hit + '}'
  );
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
