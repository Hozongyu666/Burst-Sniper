import io
import json
import os
import time
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

import ccxt
import numpy as np
import pandas as pd
import requests

# ==========================================
# 1. 爆擊流核心設定區（獨立系統，不影響主趨勢雷達）
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").strip().lower() == "true"

TW_TZ = timezone(timedelta(hours=8))       # 台灣時區 (UTC+8)
STATE_FILE = "burst_state.json"            # 獨立狀態檔（記錄1小時觸發次數與冷卻）

# 觸發門檻（對照 0xDrag 10/12 補充規則：1小時漲幅 10% UP + 觸發 3~5 次星星關注）
MIN_1H_GAIN_PCT = 10.0                     # 1小時漲幅門檻：>= +10.0%
MIN_1H_TRIGGER_COUNT = 3                   # 1小時內異常脈衝觸發次數：>= 3 次（出星星）
PULSE_3M_GAIN_PCT = 1.2                    # 單根 3m 漲幅 >= +1.2% 視為一次異常脈衝
PULSE_3M_VOL_MULT = 2.8                    # 或單根 3m 量能 >= 2.8 倍均量視為一次異常脈衝

# 倉位參考（小倉位逐倉博高R：損 3~4 單，吃一次 TP 起大飛）
SAMPLE_MARGIN_USDT = 15.0                  # 單筆逐倉保證金參考：15U
SAMPLE_LEVERAGE = 10                       # 建議逐倉槓桿：10x

BLACKLIST_BASE = {
    "USDC", "FDUSD", "TUSD", "USDP", "EUR", "USD1", "DAI", "XAUT", "PAXG"
}
VALID_X_CRYPTO = {"XRP", "XLM", "XTZ", "XMR", "XEC", "XDC", "XAI", "XCH", "XRD", "XVS"}


def is_excluded_base(base: str) -> bool:
    if base in BLACKLIST_BASE:
        return True
    if base.startswith("X") and base not in VALID_X_CRYPTO:
        return True
    return False


def now_tw() -> datetime:
    return datetime.now(TW_TZ)


def now_tw_str() -> str:
    return now_tw().strftime("%Y/%m/%d %H:%M:%S")


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
# 2. 狀態讀寫（維護滾動 1 小時觸發次數計數器）
# ==========================================
def load_state() -> dict:
    default = {
        "trigger_timestamps": {},   # sym -> [ts1, ts2, ...] 記錄過去1小時內脈衝時間戳
        "alert_history": {},        # sym -> {"last_alert_ts": float, "last_alert_count": int}
        "prev_tickers": {},         # sym -> {"price": float, "vol": float, "ts": float}
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return default


def save_state(state: dict):
    now = time.time()
    # 只保留過去 65 分鐘內的觸發時間戳
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
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ==========================================
# 3. 爆擊流全套指標計算（BOLL + KDJ + MACD + StochRSI）
# ==========================================
def compute_burst_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # 1. 布林帶 BOLL (20, 2) —— 爆擊流最核心指標
    df["boll_mid"] = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["boll_upper"] = df["boll_mid"] + 2.0 * std20
    df["boll_lower"] = df["boll_mid"] - 2.0 * std20

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
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100.0
    rsv = rsv.fillna(50.0)
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
# 4. 查詢 OKX 永續合約持倉量 (OI) 激增幅度
# ==========================================
def fetch_oi_surge_pct(inst_id: str) -> float | None:
    try:
        resp = requests.get(
            "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history",
            params={"instId": inst_id, "period": "5m", "limit": "6"},
            timeout=5,
        ).json()
        rows = resp.get("data", [])
        if len(rows) >= 4:
            cur_oi = float(rows[0][2])
            old_oi = float(rows[3][2])  # 對比 15 分鐘前持倉量
            if old_oi > 0:
                return (cur_oi - old_oi) / old_oi * 100.0
    except Exception:
        pass
    return None


# ==========================================
# 5. 分析單一幣種是否符合「爆擊流」進場與看盤條件
# ==========================================
def analyze_burst_coin(
    symbol: str,
    inst_id: str,
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    trigger_count_1h: int,
) -> dict:
    # --- A. 1小時大格局檢驗（計算 1H 開盤價、漲幅、是否實體真漲） ---
    bar_1h = df_1h.iloc[-1]
    prev_1h = df_1h.iloc[-2]
    # 取最近 60 分鐘起始參考價（若當根 1H 剛開盤不到 20 分鐘，則與前一根 1H 開盤比較）
    open_1h = float(bar_1h["open"])
    cur_price = float(df_3m.iloc[-1]["close"])
    gain_1h_pct = (cur_price - open_1h) / open_1h * 100.0

    # 若當根 1H 漲幅不足，檢查過去完整 1 小時（對比 prev_1h open）的滾動漲幅
    rolling_gain_1h = (cur_price - float(prev_1h["open"])) / float(prev_1h["open"]) * 100.0
    if rolling_gain_1h > gain_1h_pct and gain_1h_pct < MIN_1H_GAIN_PCT:
        open_1h = float(prev_1h["open"])
        gain_1h_pct = rolling_gain_1h

    # 1H 真漲 vs 拉高出貨檢驗（看實體與上影線比例）
    range_1h = float(bar_1h["high"]) - float(bar_1h["low"])
    upper_wick_1h = float(bar_1h["high"]) - max(float(bar_1h["open"]), cur_price)
    wick_ratio_1h = (upper_wick_1h / range_1h) if range_1h > 0 else 0.0

    if cur_price > float(bar_1h["open"]) and wick_ratio_1h <= 0.28:
        h1_verdict = "✅ 1H收實體強勢真漲（無長上影線）"
    elif wick_ratio_1h > 0.45:
        h1_verdict = "⚠️ 1H留長上影線，慎防拉高出貨，嚴禁追高"
    else:
        h1_verdict = "⚡ 1H多頭推進中（留意下小時回踩影線）"

    # --- B. 3m 布林帶位置與「掛中軌進場 / 中軌下前低 SL」精算 ---
    live_3m = df_3m.iloc[-1]
    prev_3m = df_3m.iloc[-2]
    boll_up = float(live_3m["boll_upper"])
    boll_mid = float(live_3m["boll_mid"])
    boll_low = float(live_3m["boll_lower"])

    # 偵測 3m 是否出現下影線 Pinbar 反轉訊號
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
    elif abs(dist_to_mid_pct) <= 0.8:
        entry_action = "🎯 現價正處於 3m 布林中軌附近，可直接市價或貼近中軌進場！"
        suggested_entry = boll_mid
    elif cur_price >= boll_up * 0.995:
        entry_action = "🧗 正沿 3m 布林上軌攀爬（勿直接追高，請在下方「紅色線：3m布林中軌」掛限價單等回踩）"
        suggested_entry = boll_mid
    else:
        entry_action = "⏳ 位於中上軌之間，建議掛單在「紅色線：3m布林中軌」等待回踩接針"
        suggested_entry = boll_mid

    # 計算止損位 SL：中軌下方的近 8 根 3m 前低 K 棒，並參考布林下軌
    recent_swing_low = float(df_3m["low"].iloc[-8:].min())
    sl_price = min(recent_swing_low * 0.997, boll_low * 0.998)
    sl_pct = (suggested_entry - sl_price) / suggested_entry
    # 保護止損寬度在 1.5% ~ 5.5% 之間，符合 10x 逐倉不易被插針掃掉又能高 R 爆發
    if sl_pct < 0.015:
        sl_price = suggested_entry * (1 - 0.015)
        sl_pct = 0.015
    elif sl_pct > 0.055:
        sl_price = suggested_entry * (1 - 0.055)
        sl_pct = 0.055

    # 15m 支撐健康檢查（0xDrag：如果跌破中軌或下軌，看 5分 15分，都跌破就不用做）
    live_15m = df_15m.iloc[-1]
    m15_ok = cur_price > float(live_15m["boll_lower"])
    m15_status = "✅ 15m 位於布林帶支撐之上（結構有效）" if m15_ok else "❌ 已跌破 15m 布林下軌（建議放棄不做）"

    # 輔助指標狀態（KDJ / MACD / StochRSI）
    kdj_str = f"K:{live_3m['kdj_k']:.0f} D:{live_3m['kdj_d']:.0f} J:{live_3m['kdj_j']:.0f}"
    macd_str = "多頭紅柱擴大" if live_3m["macd_hist"] > prev_3m["macd_hist"] and live_3m["macd_hist"] > 0 else "動能整理"
    srsi_str = f"{live_3m['stoch_k']:.0f}/{live_3m['stoch_d']:.0f}"

    # 查詢 15 分鐘持倉量 (OI) 激增幅度
    oi_15m_chg = fetch_oi_surge_pct(inst_id)

    # 高 R 止盈計算（3R / 5R / 8R —— 吃一次 TP 起大飛）
    risk_unit = suggested_entry - sl_price
    tp_3r = suggested_entry + 3.0 * risk_unit
    tp_5r = suggested_entry + 5.0 * risk_unit
    tp_8r = suggested_entry + 8.0 * risk_unit

    notional = SAMPLE_MARGIN_USDT * SAMPLE_LEVERAGE
    loss_u = min(SAMPLE_MARGIN_USDT, notional * sl_pct + notional * 0.001)
    win_3r_u = notional * (sl_pct * 3.0) - notional * 0.001
    win_5r_u = notional * (sl_pct * 5.0) - notional * 0.001
    win_8r_u = notional * (sl_pct * 8.0) - notional * 0.001

    # 星級評等（對照截圖：3~9次一顆星 ⭐，10次以上兩顆星 ⭐⭐）
    if trigger_count_1h >= 10:
        stars = "⭐⭐"
    elif trigger_count_1h >= 3:
        stars = "⭐"
    else:
        stars = ""

    return {
        "symbol": symbol,
        "inst_id": inst_id,
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
        "sl_price": sl_price,
        "sl_pct": sl_pct * 100.0,
        "boll_upper": boll_up,
        "boll_mid": boll_mid,
        "boll_lower": boll_low,
        "tp_3r": tp_3r,
        "tp_5r": tp_5r,
        "tp_8r": tp_8r,
        "loss_u": loss_u,
        "win_3r_u": win_3r_u,
        "win_5r_u": win_5r_u,
        "win_8r_u": win_8r_u,
        "kdj_str": kdj_str,
        "macd_str": macd_str,
        "srsi_str": srsi_str,
        "oi_15m_chg": oi_15m_chg,
    }


# ==========================================
# 6. 生成雙視角看盤圖表（左 1H 辨真假 ＋ 右 3m 布林中軌紅線/SL綠線）
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

        # 畫影線
        ax.vlines(x[up], sub["low"][up], sub["high"][up], color="#26a69a", linewidth=1.0)
        ax.vlines(x[down], sub["low"][down], sub["high"][down], color="#ef5350", linewidth=1.0)

        # 畫 K 棒實體
        heights = np.maximum(np.abs(sub["close"] - sub["open"]), (sub["high"] - sub["low"]) * 0.02)
        bottoms = np.minimum(sub["open"], sub["close"])
        ax.bar(x[up], heights[up], bottom=bottoms[up], width=0.6, color="#26a69a", edgecolor="#26a69a")
        ax.bar(x[down], heights[down], bottom=bottoms[down], width=0.6, color="#ef5350", edgecolor="#ef5350")

        # 畫布林帶 BOLL (上軌/中軌/下軌)
        ax.plot(x, sub["boll_upper"], color="#f1c40f", linewidth=1.1, label="BOLL Upper")
        ax.plot(x, sub["boll_mid"], color="#e056fd", linewidth=1.2, label="BOLL Mid (中軌)")
        ax.plot(x, sub["boll_lower"], color="#00d2d3", linewidth=1.1, label="BOLL Lower")
        ax.fill_between(x, sub["boll_lower"], sub["boll_upper"], color="#9b59b6", alpha=0.06)

        if draw_levels:
            # 復刻 0xDrag 截圖：紅色線 = 可進場掛單位置（3m布林中軌），綠色線 = SL 止損位置
            entry_p = info["suggested_entry"]
            sl_p = info["sl_price"]
            ax.axhline(entry_p, color="#ff3b30", linestyle="-", linewidth=1.6, label=f"進場掛單 {format_price(entry_p)}")
            ax.axhline(sl_p, color="#00e676", linestyle="-", linewidth=1.6, label=f"SL 止損 {format_price(sl_p)}")

            ax.text(
                len(sub) - 0.5, entry_p, f" 進場 {format_price(entry_p)} ",
                color="white", fontsize=8, fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ff3b30", edgecolor="none"),
            )
            ax.text(
                len(sub) - 0.5, sl_p, f" SL {format_price(sl_p)} ",
                color="black", fontsize=8, fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#00e676", edgecolor="none"),
            )

        # 設定 X 軸台灣時間刻度
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

    plot_candles_with_boll(ax1, df_1h, f"1H K線（辨別實體真漲 vs 拉高出貨）", draw_levels=False)
    plot_candles_with_boll(ax2, df_3m, f"3m K線（紅線=布林中軌掛單 / 綠線=SL止損）", draw_levels=True)

    plt.suptitle(
        f"{info['inst_id']} | 1H漲幅: {info['gain_1h_pct']:+.2f}% | 1H觸發: {info['trigger_count_1h']}次 {info['stars']} | TW {info['trigger_time_tw']}",
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
# 7. Discord 推播（完全復刻「異常上漲」通知卡片格式）
# ==========================================
def send_burst_discord_alert(info: dict, df_1h: pd.DataFrame, df_3m: pd.DataFrame):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 未設定 DISCORD_WEBHOOK_URL")
        return

    clean_pair = info["inst_id"].replace("-USDT-SWAP", "USDT")
    oi_line = f" ｜ 📈 15m持倉激增：`{info['oi_15m_chg']:+.1f}%`" if info["oi_15m_chg"] is not None else ""

    # 完全復刻截圖頂部的「異常上漲」核心數據區塊
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
        "title": f"📈 異常上漲 | {clean_pair} | 合約",
        "description": (
            f"{core_box}"
            f"🔍 **1H 體檢**：{info['h1_verdict']}{oi_line}\n"
            f"🧭 **15m 支撐**：{info['m15_status']}\n"
            f"💡 **操作指引**：{info['entry_action']}\n"
            f"📊 [立即開啟 TradingView 切 1m/3m 盯盤](https://www.tradingview.com/chart/?symbol=OKX:{clean_pair}.P)"
        ),
        "color": 0xFF3B30 if info["trigger_count_1h"] >= 10 else 0xF1C40F,
        "fields": [
            {
                "name": "🔴 紅色線｜3m布林中軌掛單",
                "value": f"**`{format_price(info['suggested_entry'])}`**\n(中軌: `{format_price(info['boll_mid'])}`)",
                "inline": True,
            },
            {
                "name": f"🟢 綠色線｜SL 止損 (-{info['sl_pct']:.2f}%)",
                "value": f"**`{format_price(info['sl_price'])}`**\n(下軌: `{format_price(info['boll_lower'])}`)",
                "inline": True,
            },
            {
                "name": f"🎰 小倉位逐倉 ({SAMPLE_MARGIN_USDT:.0f}U × {SAMPLE_LEVERAGE}x)",
                "value": f"打損最多虧: **`-{info['loss_u']:.2f} U`**\n吃滿大波段可賺 **`+{info['win_5r_u']:.0f}~{info['win_8r_u']:.0f} U`**",
                "inline": True,
            },
            {
                "name": "🚀 高 R 止盈參考目標（損3單吃1單起大飛）",
                "value": (
                    f"• **3R 目標**：`{format_price(info['tp_3r'])}` (**`+{info['win_3r_u']:.1f} U`**)\n"
                    f"• **5R 目標**：`{format_price(info['tp_5r'])}` (**`+{info['win_5r_u']:.1f} U`**)\n"
                    f"• **8R 暴賺**：`{format_price(info['tp_8r'])}` (**`+{info['win_8r_u']:.1f} U`**)"
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
        "footer": {"text": "爆擊流異常波動雷達 • 嚴禁無腦滿倉，務必等回踩3m中軌＋嚴格設SL"},
    }

    payload = {"username": "異常漲跌波動雷達 (爆擊流)", "embeds": [embed]}
    try:
        chart_buf = generate_dual_chart(df_1h, df_3m, info)
        files = {"file": ("burst_chart.png", chart_buf, "image/png")}
        resp = requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"🚨 已發送爆擊流通知: {clean_pair} | 1H漲幅 {info['gain_1h_pct']:+.2f}% | 觸發 {info['trigger_count_1h']} 次 {info['stars']}")
        else:
            print(f"❌ Discord 發送失敗 ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ 發送異常: {e}")


# ==========================================
# 8. 全市場合約 3 分鐘異常脈衝掃描主流程
# ==========================================
def run_burst_scan():
    exchange = ccxt.okx({"enableRateLimit": True})
    state = load_state()
    trigger_ts_map = state["trigger_timestamps"]
    alert_hist = state["alert_history"]
    prev_tickers = state["prev_tickers"]
    now = time.time()

    # 1. 一次取得全市場 OKX USDT 永續合約即時報價
    swap_tickers = exchange.fetch_tickers(params={"instType": "SWAP"})
    shortlist = []

    for sym, t in swap_tickers.items():
        if not sym.endswith("/USDT:USDT"):
            continue
        base = sym.split("/")[0]
        if is_excluded_base(base):
            continue

        last_p = float(t.get("last") or 0)
        if last_p <= 0:
            continue

        info_raw = t.get("info", {})
        inst_id = info_raw.get("instId", f"{base}-USDT-SWAP")
        vol_24h_u = float(info_raw.get("volCcy24h") or 0) * last_p
        pct_24h = float(t.get("percentage") or 0)

        # 過濾掉完全沒流動性的死幣（24H 成交額至少 80 萬 U）
        if vol_24h_u < 800_000:
            continue

        # 與上一輪（約3~5分鐘前）價格/成交額比對，抓出瞬間異動
        prev_rec = prev_tickers.get(sym)
        short_jump_pct = 0.0
        if prev_rec and (now - prev_rec["ts"]) <= 900 and prev_rec["price"] > 0:
            short_jump_pct = (last_p - prev_rec["price"]) / prev_rec["price"] * 100.0

        prev_tickers[sym] = {"price": last_p, "vol": vol_24h_u, "ts": now}

        # 初篩進入精細 K 線掃描池：24H 漲幅 > 4% 或 剛出現短線急拉 > 1.0%
        if pct_24h >= 4.0 or short_jump_pct >= 1.0:
            shortlist.append((sym, inst_id, max(pct_24h, short_jump_pct * 5)))

    shortlist.sort(key=lambda x: x[2], reverse=True)
    # 掃描最活躍的前 35 檔合約的 3m / 15m / 1h K 線
    top_active = shortlist[:35]
    all_analyzed = []

    for sym, inst_id, _ in top_active:
        try:
            ohlcv_3m = exchange.fetch_ohlcv(sym, "3m", limit=55)
            if len(ohlcv_3m) < 30:
                continue
            df_3m = compute_burst_indicators(pd.DataFrame(ohlcv_3m, columns=["ts", "open", "high", "low", "close", "volume"]))

            # 檢查最新兩根已走/在走 3m K 線是否出現「短線異常脈衝」
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
                # 避免同一根 3m K 線在 150 秒內重複計數
                if not ts_list or (now - ts_list[-1]) >= 150:
                    ts_list.append(now)
            trigger_ts_map[sym] = ts_list
            count_1h = len(ts_list)

            # 抓 1H 與 15m K 線進行結構體檢
            ohlcv_1h = exchange.fetch_ohlcv(sym, "1h", limit=50)
            ohlcv_15m = exchange.fetch_ohlcv(sym, "15m", limit=40)
            if len(ohlcv_1h) < 25 or len(ohlcv_15m) < 25:
                continue

            df_1h = compute_burst_indicators(pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"]))
            df_15m = compute_burst_indicators(pd.DataFrame(ohlcv_15m, columns=["ts", "open", "high", "low", "close", "volume"]))

            info = analyze_burst_coin(sym, inst_id, df_3m, df_15m, df_1h, count_1h)
            all_analyzed.append((info, df_1h, df_3m))

            # 判斷是否達到通知門檻（1H漲幅 >= 10% 且 1H觸發次數 >= 3次出星星）
            if info["gain_1h_pct"] >= MIN_1H_GAIN_PCT and count_1h >= MIN_1H_TRIGGER_COUNT:
                hist = alert_hist.get(sym, {"last_alert_ts": 0, "last_alert_count": 0})
                mins_since_last = (now - hist["last_alert_ts"]) / 60.0
                # 首次達 3 次出星、或觸發次數再增加 >= 3 次、或距離上次播報滿 20 分鐘才再次推播（防洗版）
                if mins_since_last >= 20 or (count_1h - hist["last_alert_count"]) >= 3:
                    send_burst_discord_alert(info, df_1h, df_3m)
                    alert_hist[sym] = {"last_alert_ts": now, "last_alert_count": count_1h}

        except Exception as e:
            print(f"[跳過] {sym}: {e}")

    all_analyzed.sort(key=lambda x: (x[0]["trigger_count_1h"], x[0]["gain_1h_pct"]), reverse=True)
    print(f"⚡ [{now_tw_str()}] 爆擊流掃描完成（精細掃描 {len(top_active)} 檔異動合約），當前異動排行 Top 5：")
    for inf, _, _ in all_analyzed[:5]:
        print(
            f"   • {inf['inst_id']:<18} | 1H漲幅: {inf['gain_1h_pct']:+6.2f}% | "
            f"1H觸發: {inf['trigger_count_1h']:2d}次 {inf['stars']:<2} | 現價: {format_price(inf['trigger_price'])}"
        )

    if FORCE_TEST and all_analyzed:
        top_info, top_1h, top_3m = all_analyzed[0]
        print("🧪 手動測試觸發：強制發送當前異動最強合約預覽卡片...")
        send_burst_discord_alert(top_info, top_1h, top_3m)

    save_state(state)


if __name__ == "__main__":
    run_burst_scan()
