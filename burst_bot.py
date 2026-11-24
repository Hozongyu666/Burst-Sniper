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
# 1. MEXC 爆噴預言機 v3 核心設定（完全對齊原作者功能）
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").strip().lower() == "true"
FORCE_DAILY_REPORT = os.environ.get("FORCE_DAILY_REPORT", "false").strip().lower() == "true"

TW_TZ = timezone(timedelta(hours=8))       # 台灣時區 (UTC+8)
STATE_FILE = "burst_state.json"

# 觸發與信心指數門檻
MIN_1H_GAIN_PCT = 10.0                     # 1小時漲幅門檻：>= +10.0%
MIN_1H_TRIGGER_COUNT = 3                   # 1小時內異常脈衝觸發次數：>= 3 次
MIN_CONFIDENCE_TO_ALERT = 62               # 多重因素信心指數 >= 62 分才推播
MIN_CONFIDENCE_TO_TRACK = 70               # 信心指數 >= 70 分納入自動模擬追蹤
REPEAT_ALERT_COOLDOWN_MIN = 15             # 同幣種再次觸發最小間隔 15 分鐘（對齊 22:00 -> 22:15 動態重定價）

PULSE_3M_GAIN_PCT = 1.2                    # 單根 3m 漲幅 >= +1.2% 視為一次異常脈衝
PULSE_3M_VOL_MULT = 2.6                    # 或單根 3m 量能 >= 2.6 倍均量視為一次異常脈衝

# 訂單追蹤參數
ORDER_FILL_TIMEOUT_MIN = 120               # 掛單超過 120 分鐘未回踩成交則視為「踏空取消」
ORDER_MAX_TRACK_HOURS = 12                 # 成交後最長追蹤 12 小時

# 選幣池設定：專抓 MEXC 中小市值莊家妖幣（排除穩定幣、黃金與盤子過大的前20主流幣）
MIN_24H_TURNOVER_USDT = 150_000            # 最低流動性門檻 15 萬 U
MAX_24H_TURNOVER_USDT = 120_000_000        # 排除破億美金大盤幣，鎖定高彈性妖幣
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "EUR", "USD1", "DAI", "XAUT", "PAXG",
    "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX", "AVAX", "LINK",
    "DOT", "LTC", "BCH", "TON", "SHIB", "SUI", "APT", "NEAR", "UNI", "ATOM",
}

# 小倉位逐倉參考（損 3~4 單，吃一次 TP 起大飛）
SAMPLE_MARGIN_USDT = 15.0                  # 單筆逐倉保證金參考：15U
SAMPLE_LEVERAGE = 10                       # 建議逐倉槓桿：10x


def now_tw() -> datetime:
    return datetime.now(TW_TZ)


def now_tw_str() -> str:
    return now_tw().strftime("%Y/%m/%d %H:%M:%S")


def short_tw_time() -> str:
    return now_tw().strftime("%H:%M")


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
# 2. 狀態讀寫（維護今日出現次數計數表、首發進場價與追蹤訂單）
# ==========================================
def load_state() -> dict:
    today = today_tw_str()
    default = {
        "trigger_timestamps": {},
        "alert_history": {},
        "prev_tickers": {},
        "oi_baseline": {},
        "active_orders": {},
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
            "coin_board": {},       # 記錄今日每檔妖幣的出現次數、首發時間、首發進場價、最新現價
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
        data["daily_stats"].setdefault("coin_board", {})
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
        f"累計戰績: {tot_wr} ({st['net_pnl_u']:+.2f}U) • 信心指數非做多勝率"
    )


def build_today_counter_table(coin_board: dict) -> str:
    """生成今日該訊號出現次數計數表"""
    if not coin_board:
        return "```\n今日尚無達標暴噴幣種紀錄\n```"

    sorted_items = sorted(
        coin_board.items(),
        key=lambda x: (x[1].get("count", 0), x[1].get("last_change_pct", 0)),
        reverse=True,
    )[:8]

    lines = [
        "幣種        | 今日次數 | 首發價    | 目前現價  | 進場後漲跌",
        "------------------------------------------------------",
    ]
    for pair, rec in sorted_items:
        cnt = rec.get("count", 1)
        fire = "🔥" if cnt >= 3 else ("⚡" if cnt == 2 else "  ")
        f_p = format_price(rec.get("first_entry", 0))
        c_p = format_price(rec.get("latest_price", 0))
        chg = rec.get("last_change_pct", 0.0)
        lines.append(f"{pair:<11} | {cnt:2d} 次 {fire} | {f_p:<9} | {c_p:<9} | {chg:+7.2f}%")

    return "```\n" + "\n".join(lines) + "\n```"


# ==========================================
# 3. 1m 合成 3m K 線 ＋ 爆噴預言機全套指標計算
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

    # 1. 布林帶 BOLL (20, 2) 與 EMA7
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
# 4. 爆噴預言機分析引擎（動態重定價 + 出現次數越強加權 + 進場後漲跌追蹤）
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
    board_rec: dict | None,
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

    # --- 維度 1：1H 實體真漲度 (滿分 30) ---
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

    # --- 維度 2：1H 脈衝密度 ＋ 今日出現次數正向加權 (滿分 25) ---
    # 對齊原作者核心邏輯：「今日出現次數越多，該幣種爆噴可能性越強大」
    next_today_count = (board_rec["count"] + 1) if board_rec else 1
    occ_bonus = 10 if next_today_count >= 4 else (7 if next_today_count == 3 else (4 if next_today_count == 2 else 0))
    base_pulse = 15 if trigger_count_1h >= 8 else (12 if trigger_count_1h >= 5 else 8)
    s_pulse = min(25, base_pulse + occ_bonus)

    # --- 維度 3：合約持倉激增 (OI) 與負費率軋空燃料 (滿分 25) ---
    s_oi = 0
    if oi_chg_pct is not None:
        if oi_chg_pct >= 10.0:
            s_oi = 15
        elif oi_chg_pct >= 3.0:
            s_oi = 10
        elif oi_chg_pct >= 0.0:
            s_oi = 5
        else:
            s_oi = -4
    else:
        s_oi = 6

    s_fr = 0
    if funding_pct is not None:
        if funding_pct <= -0.05:
            s_fr = 10
        elif funding_pct <= -0.01:
            s_fr = 7
        elif funding_pct <= 0.02:
            s_fr = 4
    else:
        s_fr = 3
    s_chip = max(0, min(25, s_oi + s_fr))

    # --- 維度 4：最新 3m 布林帶動態重新定價 ＋ 15m 支撐健康檢查 (滿分 20) ---
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
        entry_action = "🔥 現價回踩最新 3m 布林中軌並出現 Pinbar 反轉，可嘗試直接市價進場！"
        suggested_entry = cur_price
        s_entry = 8
    elif abs(dist_to_mid_pct) <= 0.8:
        entry_action = "🎯 現價正處於最新 3m 布林中軌附近，可直接市價或貼近中軌進場！"
        suggested_entry = boll_mid
        s_entry = 7
    elif cur_price >= boll_up * 0.995:
        entry_action = "🧗 正沿最新 3m 布林上軌攀爬（請在下方「最新重新定價紅線：3m中軌」掛限價單等回踩）"
        suggested_entry = boll_mid
        s_entry = 5
    else:
        entry_action = "⏳ 位於中上軌之間，建議掛單在「最新重新定價紅線：3m中軌」等待回踩接針"
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

    # --- 計算多重因素信心指數 (Confidence Index) ---
    confidence_score = int(np.clip(s_h1 + s_pulse + s_chip + s_struct, 0, 100))

    # --- 計算「訊號放出進場後的現價」與「距離進場後目前漲跌 %」 ---
    if board_rec and board_rec.get("first_entry", 0) > 0:
        first_entry = float(board_rec["first_entry"])
        first_time = board_rec.get("first_time", short_tw_time())
        prev_entry = float(board_rec.get("last_entry", first_entry))
        chg_from_first_pct = (cur_price - first_entry) / first_entry * 100.0
        chg_from_prev_pct = (cur_price - prev_entry) / prev_entry * 100.0
        tracking_status = (
            f"📍 **首發進場價 (`{first_time}`)**：`{format_price(first_entry)}` → **目前最新現價**：`{format_price(cur_price)}`\n"
            f"📈 **距離首發進場後漲跌**：**`{chg_from_first_pct:+.2f}%`**（較前次訊號 `{chg_from_prev_pct:+.2f}%`）｜ "
            f"**今日第 `{next_today_count}` 次出現**（出現次數越多爆噴機率越強 🔥）"
        )
    else:
        first_entry = suggested_entry
        chg_from_first_pct = (cur_price - suggested_entry) / suggested_entry * 100.0
        tracking_status = (
            f"🆕 **今日第 `1` 次首發出現** ｜ 基準進場價：`{format_price(suggested_entry)}` → 目前現價：`{format_price(cur_price)}`（`{chg_from_first_pct:+.2f}%`）"
        )

    if confidence_score >= 85:
        grade_badge = "🔥 S級信心指數【極強爆噴結構・主力真拉盤】"
        embed_color = 0x2ECC71
    elif confidence_score >= 70:
        grade_badge = "⚡ A級信心指數【結構合格・掛最新3m中軌接針】"
        embed_color = 0xF1C40F
    else:
        grade_badge = "👀 B級信心指數【觀察名單・建議輕倉試單】"
        embed_color = 0x95A5A6

    score_breakdown = (
        f"1H實體 `{s_h1}/30` ｜ 脈衝與次數熱度 `{s_pulse}/25` ｜ "
        f"籌碼/費率 `{s_chip}/25` ｜ 支撐結構 `{s_struct}/20`"
    )

    # --- 動態調整止損 SL（基於最新 3m 近 10 根結構前低，每次重發都會動態更新） ---
    swing_low_10 = float(df_3m["low"].iloc[-10:].min())
    sl_price = swing_low_10 * 0.996
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
        "score": confidence_score,
        "grade_badge": grade_badge,
        "score_breakdown": score_breakdown,
        "embed_color": embed_color,
        "next_today_count": next_today_count,
        "tracking_status": tracking_status,
        "chg_from_first_pct": chg_from_first_pct,
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
# 5. 重新繪製最新雙時區走勢圖（每次重發皆動態更新紅綠線，不沿用舊圖）
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
            ax.axhline(entry_p, color="#ff3b30", linestyle="-", linewidth=1.6, label=f"最新中軌掛單 {format_price(entry_p)}")
            ax.axhline(sl_p, color="#00e676", linestyle="-", linewidth=1.6, label=f"最新SL止損 {format_price(sl_p)}")

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

    plot_candles_with_boll(ax1, df_1h, "1H 最新走勢（辨別實體真漲 vs 拉高出貨）", draw_levels=False)
    plot_candles_with_boll(ax2, df_3m, f"3m 動態重定價圖（紅線=最新中軌 / 綠線=SL -{info['sl_pct']:.2f}%）", draw_levels=True)

    clean_pair = info["mexc_pair"].replace("_", "")
    plt.suptitle(
        f"MEXC {clean_pair} [今日第{info['next_today_count']}次] | 信心指數: {info['score']}/100 | 1H漲幅: {info['gain_1h_pct']:+.2f}% | TW {info['trigger_time_tw']}",
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
# 6. Discord 推播（含進場後漲跌追蹤 + 今日出現次數計數表）
# ==========================================
def send_burst_discord_alert(info: dict, df_1h: pd.DataFrame, df_3m: pd.DataFrame, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    clean_pair = info["mexc_pair"].replace("_", "")
    mexc_url = f"https://futures.mexc.com/exchange/{info['mexc_pair']}"
    counter_table = build_today_counter_table(state["daily_stats"].get("coin_board", {}))

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
        "title": f"📈 異常上漲 | {clean_pair} | 信心指數 {info['score']}/100 分",
        "description": (
            f"### {info['grade_badge']}\n"
            f"{info['tracking_status']}\n"
            f"📊 **多重因素驗證**：{info['score_breakdown']} *(信心指數非做多勝率)*\n"
            f"{core_box}"
            f"{meta_line}"
            f"🔍 **1H 體檢**：{info['h1_verdict']}\n"
            f"🛡️ **15m 支撐**：{info['m15_status']}\n"
            f"💡 **操作指引**：{info['entry_action']}\n"
            f"🚀 **[點此直接開啟 MEXC {clean_pair} 合約下單]({mexc_url})**"
        ),
        "color": info["embed_color"],
        "fields": [
            {
                "name": "🔴 最新紅色線｜3m布林中軌掛單",
                "value": f"**`{format_price(info['suggested_entry'])}`**\n追強快線(EMA7): `{format_price(info['aggressive_entry'])}`",
                "inline": True,
            },
            {
                "name": f"🟢 最新綠色線｜動態SL (-{info['sl_pct']:.2f}%)",
                "value": f"**`{format_price(info['sl_price'])}`**\n打損最多虧: **`-{info['loss_u']:.2f} U`**",
                "inline": True,
            },
            {
                "name": f"🎰 小倉位逐倉 ({SAMPLE_MARGIN_USDT:.0f}U × {SAMPLE_LEVERAGE}x)",
                "value": f"TP1(2.5R): **`+{info['win_tp1_u']:.1f} U`**\nTP2(5.0R): **`+{info['win_tp2_u']:.1f} U`**",
                "inline": True,
            },
            {
                "name": "🚀 動態調整高 R 止盈目標（TP1平半倉推保本）",
                "value": (
                    f"• **TP1 (2.5R 保本位)**：`{format_price(info['tp1_price'])}` (**`+{info['win_tp1_u']:.1f} U`**)\n"
                    f"• **TP2 (5.0R 主升浪)**：`{format_price(info['tp2_price'])}` (**`+{info['win_tp2_u']:.1f} U`**)\n"
                    f"• **TP3 (8.0R 妖幣頂)**：`{format_price(info['tp3_price'])}` (**`+{info['win_tp3_u']:.1f} U`**)"
                ),
                "inline": False,
            },
            {
                "name": "📋 今日該訊號出現次數計數表（出現次數越多爆噴機率越強）",
                "value": counter_table,
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

    payload = {"username": "MEXC 爆噴預言機 v3 (動態重定價版)", "embeds": [embed]}
    try:
        chart_buf = generate_dual_chart(df_1h, df_3m, info)
        files = {"file": ("burst_chart.png", chart_buf, "image/png")}
        resp = requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"🚨 已發送爆噴訊號: {clean_pair} [今日第{info['next_today_count']}次] | 信心指數 {info['score']}/100")
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
            f"價格起爆後未回踩至掛單價 **`{format_price(order['entry'])}`** 即直接噴出，已自動取消追蹤（**不計入盈虧**）。"
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
    else:
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
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "爆噴預言機戰績追蹤", "embeds": [embed]}, timeout=10)
    except Exception:
        pass


def send_burst_daily_report(state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    st = state["stats"]
    dst = state["daily_stats"]
    active = state["active_orders"]
    counter_table = build_today_counter_table(dst.get("coin_board", {}))

    day_closed = dst["wins"] + dst["losses"]
    day_wr = f"{dst['wins']/day_closed*100:.1f}% ({dst['wins']}勝 {dst['losses']}敗)" if day_closed > 0 else "0.0% (今日無平倉單)"

    tot_closed = st["wins"] + st["losses"]
    tot_wr = f"{st['wins']/tot_closed*100:.1f}% ({st['wins']}勝 {st['losses']}敗)" if tot_closed > 0 else "0.0% (尚無平倉單)"

    embed = {
        "title": f"📅 MEXC 爆噴預言機每日戰報總結 | {today_tw_str()} (台灣時間)",
        "description": (
            f"🕒 **結算時間**：`{now_tw_str()}`\n"
            f"今日共發出 **`{dst['alerts_sent']}`** 次警報 ｜ 掛單成交 **`{dst['filled']}`** 單 ｜ 踏空未成交 **`{dst['missed']}`** 單 ｜ 追蹤中 **`{len(active)}`** 單"
        ),
        "color": 0x2ECC71 if dst["net_pnl_u"] >= 0 else 0xE67E22,
        "fields": [
            {"name": "💰 今日爆噴系統淨盈虧", "value": f"**`{dst['net_pnl_u']:+.2f} U`**", "inline": True},
            {"name": "🎯 今日實戰勝率", "value": f"**`{day_wr}`**", "inline": True},
            {"name": "🏆 系統累計總盈虧 / 勝率", "value": f"**`{st['net_pnl_u']:+.2f} U`** (`{tot_wr}`)", "inline": True},
            {"name": "📋 今日爆噴幣種出現次數總計數表", "value": counter_table, "inline": False},
        ],
        "footer": {"text": "MEXC 爆噴預言機 v3 • 每日 23:50 自動結算戰績"},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "爆噴預言機每日戰報", "embeds": [embed]}, timeout=10)
        print("📅 已發送爆噴預言機每日戰報總結")
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
                    order["sl"] = order["entry"] * 1.002
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
# 8. MEXC 全市場中小型妖幣掃描主流程
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
    coin_board = state["daily_stats"]["coin_board"]
    now = time.time()

    check_active_burst_orders(exchange, state)

    swap_tickers = exchange.fetch_tickers()
    shortlist = []

    for sym, t in swap_tickers.items():
        if not sym.endswith("/USDT:USDT"):
            continue
        base = sym.split("/")[0]
        if base in EXCLUDED_BASES:
            continue

        last_p = float(t.get("last") or 0)
        if last_p <= 0:
            continue

        info_raw = t.get("info", {})
        mexc_pair = info_raw.get("symbol", f"{base}_USDT")
        clean_pair = mexc_pair.replace("_", "")

        # 若該幣種已在今日計數表上，即時同步更新它的最新現價與距離首發進場漲跌%
        if clean_pair in coin_board and coin_board[clean_pair].get("first_entry", 0) > 0:
            f_entry = float(coin_board[clean_pair]["first_entry"])
            coin_board[clean_pair]["latest_price"] = last_p
            coin_board[clean_pair]["last_change_pct"] = round((last_p - f_entry) / f_entry * 100.0, 2)

        turnover_24h = float(info_raw.get("amount24") or t.get("quoteVolume") or 0)
        if turnover_24h == 0:
            turnover_24h = float(t.get("baseVolume") or 0) * last_p

        # 鎖定中小市值高彈性妖幣區間（15萬U ~ 1.2億U）
        if not (MIN_24H_TURNOVER_USDT <= turnover_24h <= MAX_24H_TURNOVER_USDT):
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

        # 擴大初篩捕捉網：24H漲幅 >= 3.5% 或剛出現短線急拉 >= 0.8% 或已在今日計數表上的強勢幣
        if pct_24h >= 3.5 or short_jump_pct >= 0.8 or clean_pair in coin_board:
            shortlist.append((sym, mexc_pair, max(pct_24h, short_jump_pct * 6), oi_chg_pct, funding_pct))

    shortlist.sort(key=lambda x: x[2], reverse=True)
    top_active = shortlist[:45]
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

            clean_pair = mexc_pair.replace("_", "")
            existing_rec = coin_board.get(clean_pair)

            info = analyze_burst_coin(
                sym, mexc_pair, df_3m, df_15m, df_1h, count_1h,
                oi_chg_pct, funding_pct, existing_rec
            )
            all_analyzed.append((info, df_1h, df_3m))

            if (
                info["gain_1h_pct"] >= MIN_1H_GAIN_PCT
                and count_1h >= MIN_1H_TRIGGER_COUNT
                and info["score"] >= MIN_CONFIDENCE_TO_ALERT
            ):
                hist = alert_hist.get(sym, {"last_alert_ts": 0, "last_alert_count": 0})
                mins_since_last = (now - hist["last_alert_ts"]) / 60.0

                # 首次觸發立即播報；之後每隔 >= 15 分鐘若仍符合條件，動態重定價＋重畫走勢圖播報
                if mins_since_last >= REPEAT_ALERT_COOLDOWN_MIN:
                    if not existing_rec:
                        coin_board[clean_pair] = {
                            "count": 1,
                            "first_time": short_tw_time(),
                            "first_entry": info["suggested_entry"],
                            "last_entry": info["suggested_entry"],
                            "latest_price": info["trigger_price"],
                            "last_change_pct": round((info["trigger_price"] - info["suggested_entry"]) / info["suggested_entry"] * 100.0, 2),
                        }
                    else:
                        coin_board[clean_pair]["count"] += 1
                        coin_board[clean_pair]["last_entry"] = info["suggested_entry"]
                        coin_board[clean_pair]["latest_price"] = info["trigger_price"]
                        f_e = float(coin_board[clean_pair]["first_entry"])
                        coin_board[clean_pair]["last_change_pct"] = round((info["trigger_price"] - f_e) / f_e * 100.0, 2)

                    state["daily_stats"]["alerts_sent"] += 1
                    send_burst_discord_alert(info, df_1h, df_3m, state)
                    alert_hist[sym] = {"last_alert_ts": now, "last_alert_count": count_1h}

                    # 動態更新模擬追蹤單：若尚未成交則同步更新最新進場/止損/止盈；若未建單則新建
                    if info["score"] >= MIN_CONFIDENCE_TO_TRACK:
                        if sym not in state["active_orders"]:
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
                        elif not state["active_orders"][sym].get("filled", False):
                            # 尚未成交的掛單同步隨最新圖表動態重新定價調整 TP/SL
                            state["active_orders"][sym].update({
                                "entry": info["suggested_entry"],
                                "sl": info["sl_price"],
                                "tp1": info["tp1_price"],
                                "tp2": info["tp2_price"],
                                "loss_u": info["loss_u"],
                                "win_tp1_u": info["win_tp1_u"],
                                "win_tp2_u": info["win_tp2_u"],
                            })

        except Exception as e:
            print(f"[跳過] {sym}: {e}")

    all_analyzed.sort(key=lambda x: (x[0]["score"], x[0]["trigger_count_1h"]), reverse=True)
    print(f"⚡ [{now_tw_str()}] MEXC 爆噴預言機 v3 掃描完成（追蹤中: {len(state['active_orders'])} 單），當前信心指數 Top 5：")
    for inf, _, _ in all_analyzed[:5]:
        print(
            f"   • {inf['mexc_pair']:<14} | 信心指數: {inf['score']:2d}/100 | "
            f"1H漲幅: {inf['gain_1h_pct']:+6.2f}% | 觸發: {inf['trigger_count_1h']:2d}次"
        )

    if FORCE_TEST and all_analyzed:
        top_info, top_1h, top_3m = all_analyzed[0]
        send_burst_discord_alert(top_info, top_1h, top_3m, state)

    tw_now = now_tw()
    today_s = today_tw_str()
    if (tw_now.hour == 23 and tw_now.minute >= 48 and state.get("last_daily_report_date") != today_s) or FORCE_DAILY_REPORT:
        send_burst_daily_report(state)
        state["last_daily_report_date"] = today_s

    save_state(state)


if __name__ == "__main__":
    run_burst_scan()
