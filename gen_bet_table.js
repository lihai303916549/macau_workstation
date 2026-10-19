/**
 * 六子 —— 云端生成 bet_table.json（v5.3 全云端化，替代桌面 export_table.py）
 * 数据源：predictions.json（每期事前押注肖）+ record.json（开奖后判定）
 * 公式：与桌面表 v7 固定比例法完全一致——
 *   单价 = ROUND(上期本金 or C3 ÷ 风险值7 ÷ 押肖数1)
 *   盈亏 = 赢: 单价×(赔率-押肖数+1) / 亏: -单价×押肖数
 *   本金 = 上期本金 + 盈亏
 *   赔率：押马 0.8 / 其他 1.0
 * 起点：2026292（新中国第一期），本金 1000，风险值 7
 * 无预测存档的期 = 桌面表空行：跳过，本金回退起始值（与表内 IF(ISNUMBER(prev)) 逻辑一致）
 */
'use strict';
const fs = require('fs');
const path = require('path');

const DEPLOY = __dirname;
const PRED_ARCHIVE = path.join(DEPLOY, 'predictions.json');
const RECORD = path.join(DEPLOY, 'record.json');
const OUT = path.join(DEPLOY, 'bet_table.json');

const FIRST_PERIOD = '2026293'; // 2026-10-20 03:18 用户再次清空，从293期重新开始
const CAPITAL_START = 1000;
const RISK = 7;

function xround(x){ return Math.round(x); } // 正数 ROUND = half up（与 Excel 一致）

function gen(){
  let preds = {}, record = { history: [] };
  try { preds = JSON.parse(fs.readFileSync(PRED_ARCHIVE, 'utf8')); } catch(e) {}
  try { record = JSON.parse(fs.readFileSync(RECORD, 'utf8')); } catch(e) {}

  // 判定表：period -> {zodiac_hit}
  const judged = {};
  for (const h of (record.history || [])) judged[h.period] = h;

  // 所有有预测的期，按期间排序（>= FIRST_PERIOD）
  const periods = Object.keys(preds).filter(p => p >= FIRST_PERIOD).sort();

  const rows = [];
  let capital = CAPITAL_START;   // 当前滚动本金
  let trades = 0, wins = 0, totalPnl = 0;
  const now = new Date(Date.now() + 8 * 3600 * 1000).toISOString().slice(0, 19).replace('T', ' ');

  for (const p of periods) {
    const bet = (preds[p] && preds[p].zodiac) || '';
    if (!bet) continue;                       // 无押注预测 → 跳过（等同桌面表空行）
    const odds = bet.indexOf('马') >= 0 ? 0.8 : 1.0;
    const unit = xround((capital || CAPITAL_START) / RISK / 1);   // 押肖数恒 1
    const j = judged[p];
    let zodiac = '', result = '开奖中', pnl = '?';
    if (j) {  // 已开奖已判定
      trades++;
      const hit = j.zodiac_hit === 1;
      if (hit) { wins++; zodiac = bet; result = '赢'; pnl = Math.round(unit * (odds - 1 + 1)); }
      else { zodiac = '猫'; result = '亏'; pnl = -unit; }
      totalPnl += pnl;
      capital += pnl;
    }
    rows.push({
      period: +p,
      zodiac: zodiac,
      odds: odds,
      count: 1,
      result: result,
      unit: unit,
      stake: unit,          // 单押1肖：投注额=单价
      pnl: pnl,
      capital: capital,
      bet: bet + ',',
    });
  }

  const data = {
    generated_at: now + '+08:00',
    capital_start: CAPITAL_START,
    risk: RISK,
    kpi: {
      trades: trades,
      win_rate: trades > 0 ? (wins / trades) : null,
      total_pnl: totalPnl,
      current: capital,
    },
    rows: rows,
  };
  fs.writeFileSync(OUT, JSON.stringify(data, null, 2) + '\n', 'utf8');
  return data;
}

if (require.main === module) {
  const d = gen();
  console.log('云端生成 bet_table.json:', d.rows.length, '行 | kpi:', JSON.stringify(d.kpi));
  for (const r of d.rows) console.log(' ', r.period, r.zodiac || '-', r.result, '单价' + r.unit, '盈亏' + r.pnl, '本金' + r.capital, '押' + r.bet);
}
module.exports = { gen };
