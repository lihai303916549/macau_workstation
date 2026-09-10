/**
 * 老六预测系统 —— 自动更新脚本（本地 / GitHub Actions 通用）
 *
 * 做法：用 vm 加载 index.html 里的算法，保证与页面 100% 一致（不另写一套 Python 算法）。
 *
 * 流程：
 *   1. 拉取最新开奖（Node 端无 CORS 限制）
 *   2. 用上一期存档的预测判定命中
 *   3. 重新生成 EMBEDDED_DATA / EMBEDDED_RECORDS
 *   4. 用页面算法生成下一期预测
 *   5. 写回 index.html / record.json / prediction_result.json
 *
 * 用法：
 *   node update.js            # 只更新本地文件
 *   node update.js --upload   # 更新后上传到 GitHub
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
const API = 'https://history.macaumarksix.com/history/macaujc2/y/';

const UPLOAD = process.argv.includes('--upload');

function log(...a) { console.log(...a); }

// ---------- 加载页面算法 ----------
function loadPageAlgorithms() {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const code = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
  const store = {};
  const mk = id => store[id] || (store[id] = { textContent: '', innerHTML: '', style: {}, disabled: false });
  const ctx = {
    console: { log() {}, warn() {}, error() {} },
    document: { getElementById: mk, querySelector: () => mk('q'), addEventListener: () => {}, hidden: false },
    navigator: { serviceWorker: { register: () => Promise.reject() }, clipboard: { writeText: () => {} } },
    window: { addEventListener: () => {} },
    fetch: () => Promise.reject(new TypeError('offline')),
    setInterval: () => 0, clearInterval: () => 0, setTimeout: () => 0,
    caches: { keys: () => Promise.resolve([]), delete: () => Promise.resolve() },
    alert: () => {}, Date, Math, JSON, Object, Array, String, Number, Map, Set, isNaN, parseInt,
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(code + '\n;globalThis.__api={compositePredict,getNumberZodiac,pad2,macauYear};', ctx);
  return ctx.__api;
}

// ---------- 拉取开奖数据 ----------
async function fetchAll() {
  const year = new Date(Date.now() + 8 * 3600 * 1000).getUTCFullYear();
  const rows = [];
  for (const y of [year, year - 1]) {
    try {
      const r = await fetch(API + y, { cache: 'no-store' });
      const j = await r.json();
      for (const it of (j.data || [])) {
        const p = (it.openCode || '').split(',').map(s => parseInt(s.trim(), 10));
        if (p.length !== 7 || p.some(n => isNaN(n))) continue;
        rows.push({
          period: (it.openTime || '').slice(0, 4) + (it.expect || '').slice(-3),
          date: (it.openTime || '').slice(0, 10),
          numbers: p.slice(0, 6),
          special: p[6],
          openTime: it.openTime || '',
        });
      }
    } catch (e) { /* 某一年失败不影响另一年 */ }
  }
  const seen = new Set();
  const out = rows.filter(r => r.period.length === 7 && !seen.has(r.period) && seen.add(r.period));
  out.sort((a, b) => a.openTime.localeCompare(b.openTime));
  return out;
}

// ---------- 判定命中 ----------
// 23码：必须包含当期特码；平特肖：7个号中任一号码生肖 == 预测生肖
function judge(pred, draw, getNumberZodiac) {
  const codeHit = pred.codes_23.includes(draw.special) ? 1 : 0;
  const zodiacHit = [...draw.numbers, draw.special]
    .some(n => getNumberZodiac(n) === pred.zodiac) ? 1 : 0;
  return { codeHit, zodiacHit };
}

// ---------- 主流程 ----------
(async () => {
  log('===== 老六自动更新 =====');
  const api = loadPageAlgorithms();

  const draws = await fetchAll();
  if (!draws.length) { log('❌ 未拉到任何开奖数据，中止'); process.exit(1); }
  const latest = draws[draws.length - 1];
  log('最新开奖 :', latest.period, '(' + latest.date + ')',
      latest.numbers.join(',') + ' + ' + latest.special);

  const record = JSON.parse(fs.readFileSync(RECORD_PATH, 'utf8'));
  const pred = JSON.parse(fs.readFileSync(PRED_PATH, 'utf8'));
  const history = record.history || [];
  const lastRec = history[history.length - 1];
  log('战绩末期 :', lastRec ? lastRec.period : '(空)');
  log('存档预测 :', pred.next_period, '| 生肖', pred.zodiac);

  // 1) 判定：把存档预测对应的那一期补进战绩
  let changed = false;
  if (pred.next_period && latest.period >= pred.next_period) {
    const target = draws.find(d => d.period === pred.next_period);
    if (target && !(history.some(h => h.period === target.period))) {
      const { codeHit, zodiacHit } = judge(pred, target, api.getNumberZodiac);
      history.push({
        period: target.period,
        actual_numbers: target.numbers,
        actual_special: target.special,
        code_hit: codeHit,
        zodiac_hit: zodiacHit,
      });
      log('✅ 补录', target.period, '→ 23码', codeHit ? '√' : '×', '| 平特肖', zodiacHit ? '√' : '×');
      changed = true;
    }
  } else {
    log('（暂无新开奖，战绩不用补录）');
  }

  // 2) 写 record.json
  const total = history.length;
  const codeWin = history.filter(h => h.code_hit === 1).length;
  const zodiacWin = history.filter(h => h.zodiac_hit === 1).length;
  fs.writeFileSync(RECORD_PATH, JSON.stringify({ total, code_win: codeWin, zodiac_win: zodiacWin, history }, null, 2) + '\n', 'utf8');
  log('战绩更新 : 累计' + total + '期 23码 ' + codeWin + '/' + total + ' 平特肖 ' + zodiacWin + '/' + total);

  // 3) 生成下一期预测（用页面同一套算法）
  const dataForPredict = draws.map(d => ({
    period: d.period, numbers: d.numbers, special: d.special, openTime: d.openTime,
  }));
  const result = api.compositePredict(dataForPredict);
  const nextNum = parseInt(latest.period.slice(4), 10) + 1;
  let nextYear = parseInt(latest.period.slice(0, 4), 10);
  let nn = nextNum;
  while (nn > 366) { nn -= 366; nextYear += 1; }
  const nextPeriod = nextYear + '' + String(nn).padStart(3, '0');

  const today = new Date(Date.now() + 8 * 3600 * 1000).toISOString().slice(0, 10);
  fs.writeFileSync(PRED_PATH, JSON.stringify({
    codes_23: result.codes,
    zodiac: result.zodiac,
    generated_at: today,
    next_period: nextPeriod,
    data_until: latest.period,
  }, null, 2) + '\n', 'utf8');
  log('新预测   :', nextPeriod, '| 生肖', result.zodiac, '| 23码', result.codes.length, '个');

  // 4) 重建 EMBEDDED_DATA / EMBEDDED_RECORDS 并写回 index.html
  let html = fs.readFileSync(HTML_PATH, 'utf8');
  const edLines = draws.map(d =>
    '["' + d.period + '","' + d.date + '","' +
    [...d.numbers, d.special].map(n => String(n).padStart(2, '0')).join(',') + '"]'
  );
  const edBlock = 'const EMBEDDED_DATA = [\n' + edLines.join(',\n') + '\n];';
  if (!/const EMBEDDED_DATA = \[[\s\S]*?\n\];/.test(html)) { log('❌ 未找到 EMBEDDED_DATA 块'); process.exit(1); }
  html = html.replace(/const EMBEDDED_DATA = \[[\s\S]*?\n\];/, edBlock);

  const erLines = history.map(h =>
    '  {"period":"' + h.period + '","actual_numbers":[' + h.actual_numbers.join(',') + '],' +
    '"actual_special":' + h.actual_special + ',"code_hit":' + h.code_hit + ',"zodiac_hit":' + h.zodiac_hit + '}'
  );
  const erBlock = 'const EMBEDDED_RECORDS = [\n' + erLines.join(',\n') + '\n];';
  html = html.replace(/const EMBEDDED_RECORDS = \[[\s\S]*?\n\];/, erBlock);

  fs.writeFileSync(HTML_PATH, html, 'utf8');
  log('index.html 已更新 : 内嵌', draws.length, '期 / 战绩', history.length, '条');

  // 同步源文件（本地才有 macau_workstation.html）
  const SRC = path.join(ROOT, 'macau_workstation.html');
  if (fs.existsSync(SRC)) fs.copyFileSync(HTML_PATH, SRC);

  if (!changed) log('\n（数据无变化）');

  // 5) 可选上传
  if (UPLOAD) {
    log('\n===== 上传 GitHub =====');
    const { spawnSync } = require('child_process');
    const script = path.join(ROOT, 'github_upload.py');
    let ok = false;
    for (const py of ['python', 'python3', 'py']) {
      const r = spawnSync(py, [script], { stdio: 'inherit' });
      if (r.status === 0) { ok = true; break; }
    }
    if (!ok) log('❌ 上传失败：未找到可用的 python，或上传过程出错');
  }
  log('\n===== 完成 =====');
})();
