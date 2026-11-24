import io
import json
import os
import time
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ccxt
import numpy as np
import pandas as pd
import requests

# ==========================================
# 1. MEXC 爆擊流 v2 核心設定（含今日次數統計 + 自動模擬追蹤戰績）
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").strip().lower() == "true"
FORCE_DAILY_REPORT = os.environ.get("FORCE_DAILY_REPORT", "false").strip().lower() == "true"

TW_TZ = timezone(timedelta(hours=8))       # 台灣時區 (UTC+8)
STATE_FILE = "burst_state.json"

# 觸發與評分門檻
MIN_1H_GAIN_PCT = 10.0                     # 1小時漲幅門檻：>= +10.0%
MIN_1H_TRIGGER_COUNT = 3                   # 1小時內異常脈衝觸發次數：>= 3 次
MIN_SCORE_TO_ALERT = 65                    # 評分 >= 65 分才推播（自動過濾低分陷阱單）
MIN_SCORE_TO_TRACK = 70                    # 評分 >= 70 分（A級/S級）納入自動戰績追蹤
PULSE_3M_GAIN_PCT = 1.2                    # 單根 3m 漲幅 >= +1.2% 視為一次異常脈衝
PULSE_3M_VOL_MULT = 2.8                    # 或單根 3m 量能 >= 2.8 倍均量視為一次異常脈衝

# 訂單追蹤參數
ORDER_FILL_TIMEOUT_MIN = 120               # 掛單超過 120 分鐘未回踩成交則視為「踏空取消」
ORDER_MAX_TRACK_HOURS = 12                 # 成交後最長追蹤 12 小時

# MEXC 小市值妖幣流動性門檻
MIN_24H_TURNOVER_USDT = 300_000

# 倉位參考（小倉位逐倉博高R）
SAMPLE_MARGIN_USDT = 15.0                  # 標準逐倉保證金：15U
SAMPLE_LEVERAGE = 10                       # 建議逐倉槓桿：10x

BLACKLIST_BASE = {
    "USDC", "FDUSD", "TUSD", "USDP", "EUR", "USD1", "DAI", "XAUT", "PAXG"
}


def now_tw() -> datetime:
    return datetime.now(TW_TZ)


def now_tw_str() -> str:
    return now_tw().strftime("%Y/%m/%d %H:%M:%S")


def today_tw_str() -> str:
    return now_tw().strftime("%Y-%m-%d")


def format_price(p: float) -> str:
    if p == 0:
        return "0"
    abs_p = abs(p)
    if abs_p < 0.00001:
        return f"{p:.8f}"
    elif abs_p < 0.0001:
        return f"{p:.7f}"
    elif abs_p < 0.01:
        return f"{p:.6f}"
    elif abs_p < 1:
        return f"{p:.5f}"
    elif abs_p < 100:
        return f"{p:.4f}"
    else:
        return f"{p:.2f}"


# ==========================================
# 2. 狀態讀寫（維護今日出現次數、滾動脈衝與追蹤訂單）
# ==========================================
def load_state() -> dict:
    today = today_tw_str()
    default = {
        "trigger_timestamps": {},
        "alert_history": {},
        "prev_tickers": {},
        "oi_baseline": {},
        "active_orders": {},        # sym -> 追蹤中的模擬掛單/持倉
        "last_daily_report_date": "",
        "stats": {"wins": 0, "losses": 0, "missed": 0, "net_pnl_u": 0.0},
        "daily_stats": {
            "date": today,
            "alerts_sent": 0,
            "filled": 0,
            "missed": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl_u": 0.0,
            "coin_counts": {},      # 記錄今天每檔幣出現幾次 + 首發價格
        },
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            data.setdefault(k, v)
        if data.get("daily_stats", {}).get("date") != today:
            data["daily_stats"] = default["daily_stats"]
        data["daily_stats"].setdefault("coin_counts", {})
        return data
    except Exception:
        return default


def save_state(state: dict):
    now = time.time()
    cleaned_triggers = {}
    for sym, ts_list in state.get("trigger_timestamps", {}).items():
        valid = [t for t in ts_list if now - t <= 3900]
        if valid:
            cleaned_triggers[sym] = valid
    state["trigger_timestamps"] = cleaned_triggers

    state["alert_history"] = {
        k: v for k, v in state.get("alert_history", {}).items()
        if now - v.get("last_alert_ts", 0) <= 86400
    }
    state["oi_baseline"] = {
        k: v for k, v in state.get("oi_baseline", {}).items()
        if now - v.get("ts", 0) <= 7200
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_footer_summary(state: dict) -> str:
    st = state["stats"]
    dst = state["daily_stats"]
    tot_closed = st["wins"] + st["losses"]
    tot_wr = f"{st['wins']/tot_closed*100:.0f}%" if tot_closed > 0 else "尚無平倉"
    day_closed = dst["wins"] + dst["losses"]
    day_wr = f"{dst['wins']/day_closed*100:.0f}%" if day_closed > 0 else "尚無平倉"
    return (
        f"今日戰績: {dst['wins']}勝{dst['losses']}敗 ({day_wr} | {dst['net_pnl_u']:+.2f}U) ｜ "
        f"總戰績: {tot_wr} ({st['net_pnl_u']:+.2f}U)"
    )


# ==========================================
# 3. 1m 合成 3m K 線 ＋ 爆擊流全套指標計算
# ==========================================
def resample_1m_to_3m(ohlcv_1m: list) -> pd.DataFrame:
    df_1m = pd.DataFrame(ohlcv_1m, columns=["ts", "open", "high", "low", "close", "volume"])
    df_1m["dt"] = pd.to_datetime(df_1m["ts"], unit="ms")
    df_1m = df_1m.set_index("dt")

    df_3m = df_1m.resample("3min", closed="left", label="left").agg({
        "ts": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index(drop=True)
    return df_3m


def compute_burst_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    # 1. 布林帶 BOLL (20, 2) 與 EMA7（追強快線）
    df["boll_mid"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["boll_upper"] = df["boll_mid"] + 2.0 * std20
    df["boll_lower"] = df["boll_mid"] - 2.0 * std20
    df["ema7"] = close.ewm(span=7, adjust=False).mean()

    # 2. 成交量均線與倍數
    df["vol_ma20"] = vol.rolling(20).mean().shift(1)
    df["vol_ratio"] = vol / df["vol_ma20"].replace(0, np.nan)

    # 3. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2

    # 4. KDJ (9, 3, 3)
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = ((close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100.0).fillna(50.0)
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3.0 * df["kdj_k"] - 2.0 * df["kdj_d"]

    # 5. StochRSI (14, 14, 3, 3)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0, np.nan)))
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100.0
    df["stoch_k"] = stoch_rsi.rolling(3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    return df


# ==========================================
# 4. 分析單一 MEXC 合約（含今日第幾次出現 + 高檔疲勞修正 + 動態結構SL）
# ==========================================
def analyze_burst_coin(
    symbol: str,
    mexc_pair: str,
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    trigger_count_1h: int,
    oi_chg_pct: float | None,
    funding_pct: float | None,
    today_occ_num: int,
    first_alert_price: float | None,
) -> dict:
    bar_1h = df_1h.iloc[-1]
    prev_1h = df_1h.iloc[-2]
    open_1h = float(bar_1h["open"])
    cur_price = float(df_3m.iloc[-1]["close"])
    gain_1h_pct = (cur_price - open_1h) / open_1h * 100.0

    rolling_gain_1h = (cur_price - float(prev_1h["open"])) / float(prev_1h["open"]) * 100.0
    if rolling_gain_1h > gain_1h_pct and gain_1h_pct < MIN_1H_GAIN_PCT:
        open_1h = float(prev_1h["open"])
        gain_1h_pct = rolling_gain_1h

    range_1h = float(bar_1h["high"]) - float(bar_1h["low"])
    upper_wick_1h = float(bar_1h["high"]) - max(float(bar_1h["open"]), cur_price)
    wick_ratio_1h = (upper_wick_1h / range_1h) if range_1h > 0 else 0.0

    # --- 評分維度 1：1H 實體真漲度 (滿分 30) ---
    if cur_price > float(bar_1h["open"]) and wick_ratio_1h <= 0.20:
        s_h1 = 30
        h1_verdict = "✅ 1H收飽滿大陽線（極強真漲，無出貨影線）"
    elif cur_price > float(bar_1h["open"]) and wick_ratio_1h <= 0.32:
        s_h1 = 22
        h1_verdict = "✅ 1H收實體強勢真漲（上影線在安全範圍）"
    elif wick_ratio_1h <= 0.45:
        s_h1 = 10
        h1_verdict = "⚡ 1H多頭推進中（留意下小時回踩影線）"
    else:
        s_h1 = 0
        h1_verdict = "⚠️ 1H留長上影線，慎防拉高出貨，嚴禁追高"

    # --- 評分維度 2：1H 脈衝密度與漲幅爆發力 (滿分 25) ---
    if trigger_count_1h >= 10 and gain_1h_pct >= 12.0:
        s_pulse = 25
    elif trigger_count_1h >= 5 and gain_1h_pct >= 10.0:
        s_pulse = 20
    elif trigger_count_1h >= 3:
        s_pulse = 14
    else:
        s_pulse = 6

    # --- 評分維度 3：合約籌碼（OI 持倉激增 + 負費率軋空燃料）(滿分 25) ---
    s_oi = 0
    if oi_chg_pct is not None:
        if oi_chg_pct >= 10.0:
            s_oi = 15
        elif oi_chg_pct >= 3.0:
            s_oi = 10
        elif oi_chg_pct >= 0.0:
            s_oi = 5
        else:
            s_oi = -5
    else:
        s_oi = 5

    s_fr = 0
    if funding_pct is not None:
        if funding_pct <= -0.05:
            s_fr = 10
        elif funding_pct <= -0.01:
            s_fr = 7
        elif funding_pct <= 0.015:
            s_fr = 4
        elif funding_pct >= 0.03:
            s_fr = -3  # 費率過高代表散戶多單擁擠，倒扣防回洗
    else:
        s_fr = 3
    s_chip = max(0, min(25, s_oi + s_fr))

    # --- 3m 布林帶、EMA7 追強位與動態結構 SL ---
    live_3m = df_3m.iloc[-1]
    prev_3m = df_3m.iloc[-2]
    boll_up = float(live_3m["boll_upper"])
    boll_mid = float(live_3m["boll_mid"])
    boll_low = float(live_3m["boll_lower"])
    ema7_3m = float(live_3m["ema7"])

    body_3m = abs(float(live_3m["close"]) - float(live_3m["open"]))
    lower_wick_3m = min(float(live_3m["close"]), float(live_3m["open"])) - float(live_3m["low"])
    is_pinbar_mid = (
        float(live_3m["low"]) <= boll_mid * 1.004
        and cur_price >= boll_mid
        and lower_wick_3m >= max(body_3m * 1.5, cur_price * 0.002)
    )

    dist_to_mid_pct = (cur_price - boll_mid) / boll_mid * 100.0
    if is_pinbar_mid:
        entry_action = "🔥 現價回踩 3m 布林中軌並出現 Pinbar 反轉，可嘗試直接市價進場！"
        suggested_entry = cur_price
        s_entry = 8
    elif abs(dist_to_mid_pct) <= 0.8:
        entry_action = "🎯 現價正處於 3m 布林中軌附近，可直接市價或貼近中軌進場！"
        suggested_entry = boll_mid
        s_entry = 7
    elif cur_price >= boll_up * 0.995:
        entry_action = "🧗 正沿 3m 布林上軌攀爬（穩健掛「紅色線：布林中軌」，怕踏空可掛一半在「EMA7 追強位」）"
        suggested_entry = boll_mid
        s_entry = 5
    else:
        entry_action = "⏳ 位於中上軌之間，建議掛單在「紅色線：3m布林中軌」等待回踩接針"
        suggested_entry = boll_mid
        s_entry = 4

    live_15m = df_15m.iloc[-1]
    if cur_price >= float(live_15m["boll_mid"]):
        s_m15 = 8
        m15_status = "✅ 15m 站穩布林中軌之上（多頭結構極強）"
    elif cur_price > float(live_15m["boll_lower"]):
        s_m15 = 4
        m15_status = "✅ 15m 位於布林下軌支撐之上（結構有效）"
    else:
        s_m15 = -15
        m15_status = "❌ 已跌破 15m 布林下軌（建議放棄不做）"

    macd_expanding = live_3m["macd_hist"] > prev_3m["macd_hist"] and live_3m["macd_hist"] > 0
    s_macd = 4 if macd_expanding else 1
    s_struct = max(0, min(20, s_m15 + s_entry + s_macd))

    # --- 高檔疲勞修正（依據今日第幾次出現 ＋ 距離首發累計漲幅） ---
    cum_from_first_pct = 0.0
    if first_alert_price and first_alert_price > 0:
        cum_from_first_pct = (cur_price - first_alert_price) / first_alert_price * 100.0

    fatigue_penalty = 0
    if today_occ_num == 1:
        occ_tag = "🥇 今日第 1 次出現（首發起爆點・最佳上車時機）"
    elif today_occ_num == 2:
        occ_tag = f"🥈 今日第 2 次出現（二波主升段接力 ｜ 較首發 {cum_from_first_pct:+.1f}%）"
        if cum_from_first_pct >= 20.0:
            fatigue_penalty = 6
    else:
        occ_tag = f"⚠️ 今日第 {today_occ_num} 次出現（高檔魚尾接力 ｜ 較首發已漲 {cum_from_first_pct:+.1f}%）"
        fatigue_penalty = 12 if cum_from_first_pct >= 25.0 else 8

    total_score = int(np.clip(s_h1 + s_pulse + s_chip + s_struct - fatigue_penalty, 0, 100))

    if total_score >= 85:
        grade_badge = "🔥 S級【強烈建議做・主力真拉盤】"
        decision_advice = f"首發成色極佳！建議逐倉 {SAMPLE_LEVERAGE}x 下 {SAMPLE_MARGIN_USDT:.0f}U，貼近紅色中軌進場並嚴格帶好 SL。"
        embed_color = 0x2ECC71
    elif total_score >= 70:
        grade_badge = "⚡ A級【可以做・掛 3m 中軌等回踩】"
        decision_advice = f"結構合格！嚴禁市價追高，在紅色線（3m布林中軌）掛限價單接針，建議倉位 {SAMPLE_MARGIN_USDT:.0f}U。"
        embed_color = 0xF1C40F
    elif total_score >= 55:
        grade_badge = "👀 B級【偏弱/高檔接力・最多半倉試單】"
        decision_advice = f"非首發起爆或部分指標未共振，若要參與建議減半倉位（約 {SAMPLE_MARGIN_USDT/2:.0f}U）掛中軌試單即可。"
        embed_color = 0x95A5A6
    else:
        grade_badge = "🚫 C級【陷阱警告・建議放棄不做】"
        decision_advice = "1H 上影線過長或已屬高檔末端，假漲出貨機率高，建議直接跳過。"
        embed_color = 0xE74C3C

    penalty_str = f" ｜ 高檔疲勞 `-{fatigue_penalty}`" if fatigue_penalty > 0 else ""
    score_breakdown = (
        f"1H實體 `{s_h1}/30` ｜ 脈衝 `{s_pulse}/25` ｜ "
        f"籌碼 `{s_chip}/25` ｜ 支撐 `{s_struct}/20`{penalty_str}"
    )

    # --- 修復止損演算法：使用「3m 近 10 根結構前低」而非遙遠布林下軌，解除 -5.5% 鎖死 ---
    swing_low_10 = float(df_3m["low"].iloc[-10:].min())
    sl_price = swing_low_10 * 0.996
    # 確保止損必定在進場價下方，且寬度落在真實合理的 1.8% ~ 5.2% 區間
    sl_pct = (suggested_entry - sl_price) / suggested_entry
    if sl_pct < 0.018:
        sl_price = suggested_entry * (1 - 0.018)
        sl_pct = 0.018
    elif sl_pct > 0.052:
        sl_price = suggested_entry * (1 - 0.052)
        sl_pct = 0.052

    kdj_str = f"K:{live_3m['kdj_k']:.0f} D:{live_3m['kdj_d']:.0f} J:{live_3m['kdj_j']:.0f}"
    macd_str = "多頭紅柱擴大" if macd_expanding else "動能整理"
    srsi_str = f"{live_3m['stoch_k']:.0f}/{live_3m['stoch_d']:.0f}"

    # 高 R 止盈計算（2.5R 保本位 / 5R 大肉 / 8R 暴賺）
    risk_unit = suggested_entry - sl_price
    tp1_price = suggested_entry + 2.5 * risk_unit
    tp2_price = suggested_entry + 5.0 * risk_unit
    tp3_price = suggested_entry + 8.0 * risk_unit

    notional = SAMPLE_MARGIN_USDT * SAMPLE_LEVERAGE
    fee_u = notional * 0.001
    loss_u = min(SAMPLE_MARGIN_USDT, notional * sl_pct + fee_u)
    win_tp1_u = notional * (sl_pct * 2.5) - fee_u
    win_tp2_u = notional * (sl_pct * 5.0) - fee_u
    win_tp3_u = notional * (sl_pct * 8.0) - fee_u

    stars = "⭐⭐" if trigger_count_1h >= 10 else ("⭐" if trigger_count_1h >= 3 else "")

    return {
        "symbol": symbol,
        "mexc_pair": mexc_pair,
        "score": total_score,
        "grade_badge": grade_badge,
        "decision_advice": decision_advice,
        "score_breakdown": score_breakdown,
        "embed_color": embed_color,
        "today_occ_num": today_occ_num,
        "occ_tag": occ_tag,
        "gain_1h_pct": gain_1h_pct,
        "trigger_count_1h": trigger_count_1h,
        "stars": stars,
        "open_1h": open_1h,
        "trigger_price": cur_price,
        "trigger_time_tw": now_tw_str(),
        "h1_verdict": h1_verdict,
        "m15_status": m15_status,
        "entry_action": entry_action,
        "suggested_entry": suggested_entry,
        "aggressive_entry": ema7_3m,
        "sl_price": sl_price,
        "sl_pct": sl_pct * 100.0,
        "boll_upper": boll_up,
        "boll_mid": boll_mid,
        "boll_lower": boll_low,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "tp3_price": tp3_price,
        "loss_u": loss_u,
        "win_tp1_u": win_tp1_u,
        "win_tp2_u": win_tp2_u,
        "win_tp3_u": win_tp3_u,
        "kdj_str": kdj_str,
        "macd_str": macd_str,
        "srsi_str": srsi_str,
        "oi_chg_pct": oi_chg_pct,
        "funding_pct": funding_pct,
    }


# ==========================================
# 5. 生成雙視角看盤圖表
# ==========================================
def generate_dual_chart(df_1h: pd.DataFrame, df_3m: pd.DataFrame, info: dict) -> io.BytesIO:
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor="#1e1f22")

    def plot_candles_with_boll(ax, df_sub: pd.DataFrame, title: str, draw_levels: bool = False):
        sub = df_sub.iloc[-45:].copy().reset_index(drop=True)
        ax.set_facecolor("#1e1f22")
        ax.grid(True, color="#2b2d31", linestyle="--", alpha=0.7)

        x = np.arange(len(sub))
        up = sub["close"] >= sub["open"]
        down = ~up

        ax.vlines(x[up], sub["low"][up], sub["high"][up], color="#26a69a", linewidth=1.0)
        ax.vlines(x[down], sub["low"][down], sub["high"][down], color="#ef5350", linewidth=1.0)

        heights = np.maximum(np.abs(sub["close"] - sub["open"]), (sub["high"] - sub["low"]) * 0.02)
        bottoms = np.minimum(sub["open"], sub["close"])
        ax.bar(x[up], heights[up], bottom=bottoms[up], width=0.6, color="#26a69a", edgecolor="#26a69a")
        ax.bar(x[down], heights[down], bottom=bottoms[down], width=0.6, color="#ef5350", edgecolor="#ef5350")

        ax.plot(x, sub["boll_upper"], color="#f1c40f", linewidth=1.1, label="BOLL Upper")
        ax.plot(x, sub["boll_mid"], color="#e056fd", linewidth=1.2, label="BOLL Mid (中軌)")
        ax.plot(x, sub["boll_lower"], color="#00d2d3", linewidth=1.1, label="BOLL Lower")
        ax.fill_between(x, sub["boll_lower"], sub["boll_upper"], color="#9b59b6", alpha=0.06)

        if draw_levels:
            entry_p = info["suggested_entry"]
            sl_p = info["sl_price"]
            ax.axhline(entry_p, color="#ff3b30", linestyle="-", linewidth=1.6, label=f"中軌掛單 {format_price(entry_p)}")
            ax.axhline(sl_p, color="#00e676", linestyle="-", linewidth=1.6, label=f"SL 止損 {format_price(sl_p)}")

            ax.text(
                len(sub) - 0.5, entry_p, f" 掛單 {format_price(entry_p)} ",
                color="white", fontsize=8, fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ff3b30", edgecolor="none"),
            )
            ax.text(
                len(sub) - 0.5, sl_p, f" SL {format_price(sl_p)} ",
                color="black", fontsize=8, fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#00e676", edgecolor="none"),
            )

        tick_idx = np.linspace(0, len(sub) - 1, 5, dtype=int)
        tick_labels = [
            (pd.to_datetime(sub["ts"].iloc[i], unit="ms") + pd.Timedelta(hours=8)).strftime("%H:%M")
            for i in tick_idx
        ]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, color="#b5bac1", fontsize=8)
        ax.tick_params(axis="y", colors="#b5bac1", labelsize=8)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.legend(loc="upper left", facecolor="#2b2d31", edgecolor="#3f4147", fontsize=7.5)

    plot_candles_with_boll(ax1, df_1h, "1H K線（辨別實體真漲 vs 拉高出貨）", draw_levels=False)
    plot_candles_with_boll(ax2, df_3m, f"3m K線（紅線=布林中軌掛單 / 綠線=SL -{info['sl_pct']:.2f}%）", draw_levels=True)

    clean_pair = info["mexc_pair"].replace("_", "")
    plt.suptitle(
        f"MEXC {clean_pair} [今日第{info['today_occ_num']}次] | 評分: {info['score']}/100 | 1H漲幅: {info['gain_1h_pct']:+.2f}% | TW {info['trigger_time_tw']}",
        color="#f1c40f",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#1e1f22")
        buf.seek(0)
    finally:
        plt.close(fig)
    return buf


# ==========================================
# 6. Discord 推播（訊號卡片 + 到價戰報 + 每日總結）
# ==========================================
def send_burst_discord_alert(info: dict, df_1h: pd.DataFrame, df_3m: pd.DataFrame, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    clean_pair = info["mexc_pair"].replace("_", "")
    mexc_url = f"https://futures.mexc.com/exchange/{info['mexc_pair']}"

    extra_tags = []
    if info["oi_chg_pct"] is not None and abs(info["oi_chg_pct"]) >= 1.0:
        extra_tags.append(f"🐋 持倉變化: `{info['oi_chg_pct']:+.1f}%`")
    if info["funding_pct"] is not None:
        fr_note = "🔥軋空燃料" if info["funding_pct"] <= -0.01 else "費率"
        extra_tags.append(f"💸 {fr_note}: `{info['funding_pct']:+.4f}%`")
    meta_line = (" ｜ ".join(extra_tags) + "\n") if extra_tags else ""

    core_box = (
        f"```\n"
        f"1小時漲幅：{info['gain_1h_pct']:+.2f} %\n"
        f"1小時觸發次數： {info['trigger_count_1h']} 次 {info['stars']}\n"
        f"開盤價：{format_price(info['open_1h'])}\n"
        f"觸發價格：{format_price(info['trigger_price'])}\n"
        f"觸發：{info['trigger_time_tw']}\n"
        f"```"
    )

    embed = {
        "title": f"📈 異常上漲 | {clean_pair} | 評分 {info['score']}/100 分",
        "description": (
            f"### {info['grade_badge']}\n"
            f"📅 **今日出現頻率**：**{info['occ_tag']}**\n"
            f"🧭 **做單決策**：**{info['decision_advice']}**\n"
            f"📊 **評分拆解**：{info['score_breakdown']}\n"
            f"{core_box}"
            f"{meta_line}"
            f"🔍 **1H 體檢**：{info['h1_verdict']}\n"
            f"🛡️ **15m 支撐**：{info['m15_status']}\n"
            f"💡 **進場手法**：{info['entry_action']}\n"
            f"🚀 **[點此直接開啟 MEXC {clean_pair} 合約下單]({mexc_url})**"
        ),
        "color": info["embed_color"],
        "fields": [
            {
                "name": "🔴 紅色線｜穩健掛單 (3m中軌)",
                "value": f"**`{format_price(info['suggested_entry'])}`**\n激進追強(EMA7): `{format_price(info['aggressive_entry'])}`",
                "inline": True,
            },
            {
                "name": f"🟢 綠色線｜結構止損 (-{info['sl_pct']:.2f}%)",
                "value": f"**`{format_price(info['sl_price'])}`**\n打損最多虧: **`-{info['loss_u']:.2f} U`**",
                "inline": True,
            },
            {
                "name": f"🎰 小倉位逐倉 ({SAMPLE_MARGIN_USDT:.0f}U × {SAMPLE_LEVERAGE}x)",
                "value": f"TP1(2.5R): **`+{info['win_tp1_u']:.1f} U`**\nTP2(5.0R): **`+{info['win_tp2_u']:.1f} U`**",
                "inline": True,
            },
            {
                "name": "🚀 高 R 階梯止盈目標（TP1平半倉推保本，尾倉衝大魚）",
                "value": (
                    f"• **TP1 (2.5R 保本位)**：`{format_price(info['tp1_price'])}` (**`+{info['win_tp1_u']:.1f} U`**)\n"
                    f"• **TP2 (5.0R 主升浪)**：`{format_price(info['tp2_price'])}` (**`+{info['win_tp2_u']:.1f} U`**)\n"
                    f"• **TP3 (8.0R 妖幣頂)**：`{format_price(info['tp3_price'])}` (**`+{info['win_tp3_u']:.1f} U`**)"
                ),
                "inline": False,
            },
            {
                "name": "📐 3m 輔助指標快照（KDJ / MACD / StochRSI）",
                "value": f"`KDJ: {info['kdj_str']}` ｜ `MACD: {info['macd_str']}` ｜ `StochRSI: {info['srsi_str']}`",
                "inline": False,
            },
        ],
        "image": {"url": "attachment://burst_chart.png"},
        "footer": {"text": get_footer_summary(state)},
    }

    payload = {"username": "MEXC 爆擊流異常波動雷達 v2", "embeds": [embed]}
    try:
        chart_buf = generate_dual_chart(df_1h, df_3m, info)
        files = {"file": ("burst_chart.png", chart_buf, "image/png")}
        resp = requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"🚨 已發送 MEXC 爆擊通知: {clean_pair} [今日第{info['today_occ_num']}次] | 評分 {info['score']}/100")
    except Exception as e:
        print(f"❌ 發送異常: {e}")


def send_burst_outcome_alert(order: dict, event_type: str, pnl_u: float, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    clean_pair = order["mexc_pair"].replace("_", "")
    elapsed_min = int((time.time() - order["created_at"]) / 60)

    if event_type == "MISSED":
        title = f"💨 掛單未成交（踏空取消）| {clean_pair}"
        desc = (
            f"🕒 **時間**：`{now_tw_str()}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格起爆後未回踩至掛單價 **`{format_price(order['entry'])}`** 即直接拉飛，已自動取消追蹤（**不計入盈虧**）。"
        )
        color = 0x95A5A6
    elif event_type == "TP1":
        title = f"🎯 TP1 (2.5R) 達標・鎖利推保本 | {clean_pair}"
        desc = (
            f"🕒 **到價時間**：`{now_tw_str()}`（發單後 {elapsed_min} 分鐘）\n"
            f"掛單 `{format_price(order['entry'])}` 成交後已衝破 **`{format_price(order['tp1'])}`**！\n"
            f"💰 先平 **50% 倉位**鎖定 **`+{pnl_u:.2f} U`**，剩餘 50% 止損已上移至進場價保本，繼續追蹤 **TP2 (5R)**！"
        )
        color = 0x1ABC9C
    elif event_type == "TP2":
        title = f"🚀 TP2 (5.0R) 暴賺全數達標！| {clean_pair}"
        desc = (
            f"🕒 **到價時間**：`{now_tw_str()}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格強勢攻克 5R 目標 **`{format_price(order['tp2'])}`**！本單累計淨賺 **`+{pnl_u:.2f} U`**！"
        )
        color = 0x2ECC71
    elif event_type == "BREAKEVEN":
        title = f"🛡️ TP1鎖利後保本離場 | {clean_pair}"
        desc = (
            f"🕒 **到價時間**：`{now_tw_str()}`（發單後 {elapsed_min} 分鐘）\n"
            f"已於 TP1 平半倉鎖利，尾倉回落至保本價離場，整單實賺 **`+{pnl_u:.2f} U`**！"
        )
        color = 0x3498DB
    else:  # SL
        title = f"🛑 觸及結構止損出場 (SL) | {clean_pair}"
        desc = (
            f"🕒 **到價時間**：`{now_tw_str()}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格跌破止損線 **`{format_price(order['sl'])}`**，嚴格控損 **`{pnl_u:+.2f} U`**。"
        )
        color = 0xE74C3C

    embed = {
        "title": title,
        "description": desc,
        "color": color,
        "footer": {"text": get_footer_summary(state)},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "爆擊流戰績追蹤器", "embeds": [embed]}, timeout=10)
    except Exception:
        pass


def send_burst_daily_report(state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    st = state["stats"]
    dst = state["daily_stats"]
    active = state["active_orders"]

    day_closed = dst["wins"] + dst["losses"]
    day_wr = f"{dst['wins']/day_closed*100:.1f}% ({dst['wins']}勝 {dst['losses']}敗)" if day_closed > 0 else "0.0% (今日無平倉單)"

    tot_closed = st["wins"] + st["losses"]
    tot_wr = f"{st['wins']/tot_closed*100:.1f}% ({st['wins']}勝 {st['losses']}敗)" if tot_closed > 0 else "0.0% (尚無平倉單)"

    # 整理今日出現最多次的幣種排行
    coin_counts = dst.get("coin_counts", {})
    top_coins_lines = []
    for pair, rec in sorted(coin_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:6]:
        top_coins_lines.append(f"• **{pair}**：今日共觸發 **`{rec['count']}` 次**（首發價 `{format_price(rec['first_price'])}`）")
    coins_summary = "\n".join(top_coins_lines) if top_coins_lines else "今日尚無達標暴漲幣種。"

    embed = {
        "title": f"📅 MEXC 爆擊流每日戰報總結 | {today_tw_str()} (台灣時間)",
        "description": (
            f"🕒 **結算時間**：`{now_tw_str()}`\n"
            f"今日共發出 **`{dst['alerts_sent']}`** 次警報 ｜ 掛單成交 **`{dst['filled']}`** 單 ｜ 踏空未成交 **`{dst['missed']}`** 單 ｜ 追蹤中 **`{len(active)}`** 單"
        ),
        "color": 0x2ECC71 if dst["net_pnl_u"] >= 0 else 0xE67E22,
        "fields": [
            {"name": "💰 今日爆擊流淨盈虧", "value": f"**`{dst['net_pnl_u']:+.2f} U`**", "inline": True},
            {"name": "🎯 今日實戰勝率", "value": f"**`{day_wr}`**", "inline": True},
            {"name": "🏆 系統累計總盈虧 / 勝率", "value": f"**`{st['net_pnl_u']:+.2f} U`** (`{tot_wr}`)", "inline": True},
            {"name": "🔥 今日暴漲幣出現次數統計", "value": coins_summary, "inline": False},
        ],
        "footer": {"text": "MEXC 爆擊流異常波動雷達 v2 • 每日 23:50 自動結算戰績"},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "爆擊流每日戰報", "embeds": [embed]}, timeout=10)
        print("📅 已發送爆擊流每日戰報總結")
    except Exception as e:
        print(f"[每日戰報發送失敗] {e}")


# ==========================================
# 7. 持續追蹤今日喊單的成交與盈虧狀態
# ==========================================
def check_active_burst_orders(exchange, state: dict):
    active_orders = state["active_orders"]
    if not active_orders:
        return

    now = time.time()
    st = state["stats"]
    dst = state["daily_stats"]

    for sym in list(active_orders.keys()):
        order = active_orders[sym]
        elapsed_min = (now - order["created_at"]) / 60.0

        # 超過最長追蹤時間直接清理
        if elapsed_min > ORDER_MAX_TRACK_HOURS * 60:
            del active_orders[sym]
            continue

        try:
            since_ms = int(order.get("last_checked_ms", order["created_at"] * 1000))
            candles_1m = exchange.fetch_ohlcv(sym, "1m", since=since_ms, limit=300)
            if not candles_1m:
                continue

            done = False
            tp1_half_u = round(order["win_tp1_u"] * 0.50, 2)
            tp2_half_u = round(order["win_tp2_u"] * 0.50, 2)

            for c in candles_1m:
                c_ts, _, c_high, c_low, _, _ = c

                # 階段一：尚未成交前，檢查價格是否回踩觸及掛單價 (entry)
                if not order.get("filled", False):
                    if c_low <= order["entry"]:
                        order["filled"] = True
                        dst["filled"] += 1
                    elif (c_ts / 1000.0 - order["created_at"]) / 60.0 >= ORDER_FILL_TIMEOUT_MIN:
                        st["missed"] += 1
                        dst["missed"] += 1
                        send_burst_outcome_alert(order, "MISSED", 0.0, state)
                        del active_orders[sym]
                        done = True
                        break
                    else:
                        continue

                # 階段二：已成交後，檢查 SL / TP1 (2.5R) / TP2 (5.0R)
                if c_low <= order["sl"]:
                    if order.get("tp1_hit"):
                        st["wins"] += 1
                        dst["wins"] += 1
                        st["net_pnl_u"] = round(st["net_pnl_u"] + tp1_half_u, 2)
                        dst["net_pnl_u"] = round(dst["net_pnl_u"] + tp1_half_u, 2)
                        send_burst_outcome_alert(order, "BREAKEVEN", tp1_half_u, state)
                    else:
                        loss_u = order["loss_u"]
                        st["losses"] += 1
                        dst["losses"] += 1
                        st["net_pnl_u"] = round(st["net_pnl_u"] - loss_u, 2)
                        dst["net_pnl_u"] = round(dst["net_pnl_u"] - loss_u, 2)
                        send_burst_outcome_alert(order, "SL", -loss_u, state)
                    del active_orders[sym]
                    done = True
                    break

                if not order.get("tp1_hit") and c_high >= order["tp1"]:
                    order["tp1_hit"] = True
                    order["sl"] = order["entry"] * 1.002  # 保本止損上移 +0.2% 覆蓋手續費
                    send_burst_outcome_alert(order, "TP1", tp1_half_u, state)

                if order.get("tp1_hit") and c_high >= order["tp2"]:
                    total_win = round(tp1_half_u + tp2_half_u, 2)
                    st["wins"] += 1
                    dst["wins"] += 1
                    st["net_pnl_u"] = round(st["net_pnl_u"] + total_win, 2)
                    dst["net_pnl_u"] = round(dst["net_pnl_u"] + total_win, 2)
                    send_burst_outcome_alert(order, "TP2", total_win, state)
                    del active_orders[sym]
                    done = True
                    break

            if not done and sym in active_orders:
                active_orders[sym]["last_checked_ms"] = candles_1m[-1][0]

        except Exception as e:
            print(f"[訂單追蹤異常] {sym}: {e}")


# ==========================================
# 8. MEXC 全市場永續合約掃描主流程
# ==========================================
def run_burst_scan():
    exchange = ccxt.mexc({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    state = load_state()
    trigger_ts_map = state["trigger_timestamps"]
    alert_hist = state["alert_history"]
    prev_tickers = state["prev_tickers"]
    oi_baseline = state["oi_baseline"]
    coin_counts = state["daily_stats"]["coin_counts"]
    now = time.time()

    # 1. 優先檢查先前喊出的單是否成交、止盈或止損
    check_active_burst_orders(exchange, state)

    # 2. 抓取全市場 MEXC USDT 永續合約即時報價
    swap_tickers = exchange.fetch_tickers()
    shortlist = []

    for sym, t in swap_tickers.items():
        if not sym.endswith("/USDT:USDT"):
            continue
        base = sym.split("/")[0]
        if base in BLACKLIST_BASE:
            continue

        last_p = float(t.get("last") or 0)
        if last_p <= 0:
            continue

        info_raw = t.get("info", {})
        mexc_pair = info_raw.get("symbol", f"{base}_USDT")
        turnover_24h = float(info_raw.get("amount24") or t.get("quoteVolume") or 0)
        if turnover_24h == 0:
            turnover_24h = float(t.get("baseVolume") or 0) * last_p

        if turnover_24h < MIN_24H_TURNOVER_USDT:
            continue

        pct_24h = float(t.get("percentage") or 0)
        cur_oi = float(info_raw.get("holdVol") or 0)
        fr_raw = info_raw.get("fundingRate")
        funding_pct = float(fr_raw) * 100.0 if fr_raw is not None else None

        prev_rec = prev_tickers.get(sym)
        short_jump_pct = 0.0
        if prev_rec and (now - prev_rec["ts"]) <= 900 and prev_rec["price"] > 0:
            short_jump_pct = (last_p - prev_rec["price"]) / prev_rec["price"] * 100.0
        prev_tickers[sym] = {"price": last_p, "oi": cur_oi, "ts": now}

        base_rec = oi_baseline.get(sym)
        oi_chg_pct = None
        if cur_oi > 0:
            if not base_rec or (now - base_rec["ts"]) >= 2400:
                oi_baseline[sym] = {"oi": cur_oi, "ts": now}
            elif base_rec["oi"] > 0:
                oi_chg_pct = (cur_oi - base_rec["oi"]) / base_rec["oi"] * 100.0

        if pct_24h >= 4.0 or short_jump_pct >= 1.0:
            shortlist.append((sym, mexc_pair, max(pct_24h, short_jump_pct * 5), oi_chg_pct, funding_pct))

    shortlist.sort(key=lambda x: x[2], reverse=True)
    top_active = shortlist[:40]
    all_analyzed = []

    for sym, mexc_pair, _, oi_chg_pct, funding_pct in top_active:
        try:
            ohlcv_1m = exchange.fetch_ohlcv(sym, "1m", limit=150)
            if len(ohlcv_1m) < 90:
                continue
            df_3m = compute_burst_indicators(resample_1m_to_3m(ohlcv_1m))
            if len(df_3m) < 25:
                continue

            c_now = df_3m.iloc[-1]
            c_prev = df_3m.iloc[-2]
            gain_3m_now = (float(c_now["close"]) - float(c_now["open"])) / float(c_now["open"]) * 100.0
            gain_3m_prev = (float(c_prev["close"]) - float(c_prev["open"])) / float(c_prev["open"]) * 100.0
            vol_r_now = float(c_now["vol_ratio"]) if not np.isnan(c_now["vol_ratio"]) else 0.0
            vol_r_prev = float(c_prev["vol_ratio"]) if not np.isnan(c_prev["vol_ratio"]) else 0.0

            is_pulse = (
                gain_3m_now >= PULSE_3M_GAIN_PCT
                or gain_3m_prev >= PULSE_3M_GAIN_PCT
                or (gain_3m_now > 0.4 and vol_r_now >= PULSE_3M_VOL_MULT)
                or (gain_3m_prev > 0.4 and vol_r_prev >= PULSE_3M_VOL_MULT)
            )

            ts_list = [t for t in trigger_ts_map.get(sym, []) if now - t <= 3600]
            if is_pulse:
                if not ts_list or (now - ts_list[-1]) >= 150:
                    ts_list.append(now)
            trigger_ts_map[sym] = ts_list
            count_1h = len(ts_list)

            ohlcv_1h = exchange.fetch_ohlcv(sym, "1h", limit=50)
            ohlcv_15m = exchange.fetch_ohlcv(sym, "15m", limit=40)
            if len(ohlcv_1h) < 25 or len(ohlcv_15m) < 25:
                continue

            df_1h = compute_burst_indicators(pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"]))
            df_15m = compute_burst_indicators(pd.DataFrame(ohlcv_15m, columns=["ts", "open", "high", "low", "close", "volume"]))

            # 查詢今日該幣種已出現幾次與首發價格
            clean_pair = mexc_pair.replace("_", "")
            coin_rec = coin_counts.get(clean_pair, {"count": 0, "first_price": None})
            next_occ_num = coin_rec["count"] + 1

            info = analyze_burst_coin(
                sym, mexc_pair, df_3m, df_15m, df_1h, count_1h,
                oi_chg_pct, funding_pct, next_occ_num, coin_rec["first_price"]
            )
            all_analyzed.append((info, df_1h, df_3m))

            if (
                info["gain_1h_pct"] >= MIN_1H_GAIN_PCT
                and count_1h >= MIN_1H_TRIGGER_COUNT
                and info["score"] >= MIN_SCORE_TO_ALERT
            ):
                hist = alert_hist.get(sym, {"last_alert_ts": 0, "last_alert_count": 0})
                mins_since_last = (now - hist["last_alert_ts"]) / 60.0
                # 第1次立即推播；第2次起需間隔 >= 45分鐘（防同波段重複轟炸）
                cooldown_req = 0 if next_occ_num == 1 else 45
                if mins_since_last >= cooldown_req:
                    # 更新今日出現次數與首發價
                    if coin_rec["count"] == 0:
                        coin_counts[clean_pair] = {"count": 1, "first_price": info["trigger_price"]}
                    else:
                        coin_counts[clean_pair]["count"] += 1

                    state["daily_stats"]["alerts_sent"] += 1
                    send_burst_discord_alert(info, df_1h, df_3m, state)
                    alert_hist[sym] = {"last_alert_ts": now, "last_alert_count": count_1h}

                    # 若評分 >= 70 分且目前未追蹤此幣，自動納入模擬掛單追蹤
                    if info["score"] >= MIN_SCORE_TO_TRACK and sym not in state["active_orders"]:
                        state["active_orders"][sym] = {
                            "symbol": sym,
                            "mexc_pair": mexc_pair,
                            "entry": info["suggested_entry"],
                            "sl": info["sl_price"],
                            "tp1": info["tp1_price"],
                            "tp2": info["tp2_price"],
                            "loss_u": info["loss_u"],
                            "win_tp1_u": info["win_tp1_u"],
                            "win_tp2_u": info["win_tp2_u"],
                            "filled": False,
                            "tp1_hit": False,
                            "created_at": now,
                            "last_checked_ms": int(now * 1000),
                        }

        except Exception as e:
            print(f"[跳過] {sym}: {e}")

    all_analyzed.sort(key=lambda x: (x[0]["score"], x[0]["trigger_count_1h"]), reverse=True)
    print(f"⚡ [{now_tw_str()}] MEXC 爆擊流 v2 掃描完成（追蹤中: {len(state['active_orders'])} 單），當前評分 Top 5：")
    for inf, _, _ in all_analyzed[:5]:
        print(
            f"   • {inf['mexc_pair']:<14} | 評分: {inf['score']:2d}/100 | "
            f"1H漲幅: {inf['gain_1h_pct']:+6.2f}% | 觸發: {inf['trigger_count_1h']:2d}次"
        )

    if FORCE_TEST and all_analyzed:
        top_info, top_1h, top_3m = all_analyzed[0]
        send_burst_discord_alert(top_info, top_1h, top_3m, state)

    # 每日 23:50 自動發送爆擊流每日戰報總結
    tw_now = now_tw()
    today_s = today_tw_str()
    if (tw_now.hour == 23 and tw_now.minute >= 48 and state.get("last_daily_report_date") != today_s) or FORCE_DAILY_REPORT:
        send_burst_daily_report(state)
        state["last_daily_report_date"] = today_s

    save_state(state)


if __name__ == "__main__":
    run_burst_scan()
