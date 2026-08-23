import os
import json
import asyncio
import warnings
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, BayesianRidge, Ridge, Lasso, ElasticNet, HuberRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

STATE_FILE = "state_v2.json"
WATCHLIST = ["3466.HK", "NVDA", "VOO", "RKLB"]

# Explicit decision thresholds (easy to tune)
SCORE_STRONG_BULL = 70
SCORE_NEUTRAL_FLOOR = 45
EARNINGS_BLOCK_DAYS = 7
VOL_SHOCK_PCT = 8.0
VOL_ROC_PCT = 6.0
# Positive options result cached longer; negative/uncertain results expire fast
OPTIONS_CACHE_TTL_POSITIVE = 6 * 3600   # 6 hours
OPTIONS_CACHE_TTL_NEGATIVE = 120        # 2 minutes (rate-limit recovery)
OPTIONS_CACHE_TTL_UNKNOWN = 300         # 5 minutes

YF_SESSION = requests.Session()
YF_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

app = FastAPI(title="AI Quant Trading Platform")

# OPTIONS_CACHE: symbol -> {"has_options": bool, "ts": float, "ttl": int, "source": str}
OPTIONS_CACHE: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------------
# 狀態載入與儲存
# ----------------------------------------------------------------------
def load_state() -> Dict[str, Any]:
    default_state = {
        "watchlist": WATCHLIST,
        "stock_states": {},
        "pred_audit": {},
        "model_maes": {},
        "alert_history": [],
        "earnings_cache": {},  # symbol -> {date, ts, source} — sync Mac/Render under rate-limit
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key in default_state:
                    if key not in data:
                        data[key] = default_state[key]
                return data
        except Exception:
            return default_state
    return default_state

def save_state(state: Dict[str, Any]):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存 {STATE_FILE} 失敗: {e}")

GLOBAL_STATE = load_state()

# ----------------------------------------------------------------------
# 未來 30 天重大事件與季度業績爬取模組
# ----------------------------------------------------------------------
def get_macro_events(start_date: datetime.date, end_date: datetime.date) -> List[Dict[str, Any]]:
    macro_list = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() == 4 and 1 <= curr.day <= 7:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🇺🇸 美國非農就業報告 (NFP) & 失業率",
                "tag": "⚡ 總經大日",
                "tag_color": "bg-amber-950 text-amber-300 border-amber-800",
                "impact": "影響聯準會降息預期與大盤整體波動"
            })
        if curr.day == 12 and curr.weekday() < 5:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🇺🇸 美國 CPI 通膨物價指數公佈",
                "tag": "🔥 重大通膨",
                "tag_color": "bg-rose-950 text-rose-300 border-rose-800",
                "impact": "關鍵通膨指標，常引發美股大盤單日劇烈震盪"
            })
        if curr.day == 26 and curr.weekday() < 5:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🇺🇸 美國核心 PCE 個人消費支出物價指數",
                "tag": "🔥 FED指標",
                "tag_color": "bg-purple-950 text-purple-300 border-purple-800",
                "impact": "聯準會最看重的通膨指標"
            })
        if curr.month == 11 and curr.weekday() == 1 and 2 <= curr.day <= 8:
            if curr.year % 4 == 0:
                macro_list.append({
                    "date": curr.strftime("%Y-%m-%d"),
                    "days_left": (curr - start_date).days,
                    "title": "🏛️ 美國總統大選 (Presidential Election Day)",
                    "tag": "🗳️ 總統大選",
                    "tag_color": "bg-red-950 text-red-300 border-red-800 font-bold animate-pulse",
                    "impact": "決定未來四年產業與稅務政策，選前極致觀望、選後慶祝行情高爆發"
                })
            elif curr.year % 4 == 2:
                macro_list.append({
                    "date": curr.strftime("%Y-%m-%d"),
                    "days_left": (curr - start_date).days,
                    "title": "🏛️ 美國中期選舉 (Midterm Election Day)",
                    "tag": "🗳️ 中期選舉",
                    "tag_color": "bg-amber-950 text-amber-300 border-amber-800 font-bold animate-pulse",
                    "impact": "決定國會兩黨席次，政策不確定性消除後歷史上股市上漲機率極高"
                })
        if curr.month in [3, 6, 9, 12] and curr.weekday() == 4 and 15 <= curr.day <= 21:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🧙‍♀️ 四巫日期權/期貨大結算 (Quadruple Witching)",
                "tag": "🔮 期權結算",
                "tag_color": "bg-indigo-950 text-indigo-300 border-indigo-800 font-bold",
                "impact": "指數/股票期權期貨集中到期，尾盤易出現極致暴量洗盤與高波動"
            })
        if curr.weekday() == 2 and 18 <= curr.day <= 24 and curr.month in [1, 3, 5, 6, 7, 9, 11, 12]:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🏦 FOMC 聯準會利率決議 & 鮑爾記者會",
                "tag": "🎯 FED決議",
                "tag_color": "bg-cyan-950 text-cyan-300 border-cyan-800 font-bold animate-pulse",
                "impact": "決定升降息與點陣圖走向，全球資本市場最核心的資金流向風向球"
            })
        curr += timedelta(days=1)
    return macro_list


def _parse_earnings_date(raw) -> Optional[datetime.date]:
    """Robustly turn Yahoo calendar / earnings_dates value into a date."""
    if raw is None:
        return None
    try:
        if isinstance(raw, datetime):
            return raw.date()
        # Already a date
        if type(raw).__name__ == "date" and hasattr(raw, "year"):
            return raw  # type: ignore
        # pandas Timestamp / datetime-like
        if hasattr(raw, "to_pydatetime"):
            return raw.to_pydatetime().date()
        if hasattr(raw, "date") and callable(raw.date):
            return raw.date()
        s = str(raw)[:10]
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


# Persist last-known upcoming earnings so rate-limited hosts (e.g. Render)
# stay consistent with environments that successfully fetched the date.
# symbol -> {"date": "YYYY-MM-DD", "ts": float, "source": str}
EARNINGS_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_persisted_earnings(symbol: str) -> Optional[datetime.date]:
    try:
        store = GLOBAL_STATE.setdefault("earnings_cache", {})
        item = store.get(symbol) or EARNINGS_CACHE.get(symbol)
        if not item:
            return None
        return _parse_earnings_date(item.get("date"))
    except Exception:
        return None


def _save_persisted_earnings(symbol: str, ed: datetime.date, source: str):
    payload = {
        "date": ed.strftime("%Y-%m-%d"),
        "ts": datetime.now().timestamp(),
        "source": source,
    }
    EARNINGS_CACHE[symbol] = payload
    try:
        store = GLOBAL_STATE.setdefault("earnings_cache", {})
        store[symbol] = payload
        save_state(GLOBAL_STATE)
    except Exception:
        pass


def _fetch_next_earnings_date(
    symbol: str, now_tz: datetime.date, end_date: datetime.date
) -> Tuple[Optional[datetime.date], str]:
    """
    Multi-source earnings lookup.

    Root cause of Mac HARD_BLOCK vs Render SELL_PUT:
    Yahoo calendar is often rate-limited on cloud (Render) → no Earnings event
    → has_near_earnings=false → incorrectly allows options.

    Strategy:
    1. calendar
    2. get_earnings_dates / earnings_dates
    3. persisted cache in state_v2.json (survives rate-limit & restarts)
    """
    candidates: List[Tuple[datetime.date, str]] = []

    # --- Source A: calendar ---
    try:
        stock = yf.Ticker(symbol)  # default session; custom session worsens rate-limit
        cal = None
        try:
            cal = stock.calendar
        except Exception:
            cal = None
        if cal is not None:
            raw_list: List[Any] = []
            if isinstance(cal, dict) and "Earnings Date" in cal:
                v = cal["Earnings Date"]
                raw_list = list(v) if isinstance(v, (list, tuple)) else [v]
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    raw_list = cal.loc["Earnings Date"].dropna().tolist()
                elif "Earnings Date" in cal.columns:
                    raw_list = cal["Earnings Date"].dropna().tolist()
            for raw in raw_list:
                ed = _parse_earnings_date(raw)
                if ed is not None:
                    candidates.append((ed, "calendar"))
    except Exception:
        pass

    # --- Source B: get_earnings_dates / earnings_dates ---
    try:
        stock = yf.Ticker(symbol)
        ed_df = None
        try:
            if hasattr(stock, "get_earnings_dates"):
                ed_df = stock.get_earnings_dates(limit=12)
        except Exception:
            ed_df = None
        if ed_df is None:
            try:
                ed_df = getattr(stock, "earnings_dates", None)
            except Exception:
                ed_df = None
        if isinstance(ed_df, pd.DataFrame) and not ed_df.empty:
            for idx in ed_df.index:
                ed = _parse_earnings_date(idx)
                if ed is not None:
                    candidates.append((ed, "earnings_dates"))
    except Exception:
        pass

    upcoming = sorted(
        [(d, src) for d, src in candidates if now_tz <= d <= end_date],
        key=lambda x: x[0],
    )
    if upcoming:
        best_d, best_src = upcoming[0]
        _save_persisted_earnings(symbol, best_d, best_src)
        return best_d, best_src

    # --- Source C: persisted (Mac may have saved it; Render reuses) ---
    cached = _load_persisted_earnings(symbol)
    if cached is not None and now_tz <= cached <= end_date:
        return cached, "persisted_cache"

    # Remember any future date even outside 30d window for later
    future_any = sorted(
        [(d, src) for d, src in candidates if d >= now_tz],
        key=lambda x: x[0],
    )
    if future_any:
        _save_persisted_earnings(symbol, future_any[0][0], future_any[0][1])

    return None, "none"


def fetch_stock_events(symbol: str) -> List[Dict[str, Any]]:
    symbol = symbol.strip().upper()
    tz_str = "Asia/Hong_Kong" if symbol.endswith(".HK") else "America/New_York"
    now_tz = datetime.now(ZoneInfo(tz_str)).date()
    end_date = now_tz + timedelta(days=30)

    events: List[Dict[str, Any]] = []
    ed, earn_src = _fetch_next_earnings_date(symbol, now_tz, end_date)
    if ed is not None:
        days_left = (ed - now_tz).days
        events.append({
            "date": ed.strftime("%Y-%m-%d"),
            "days_left": days_left,
            "title": f"📊 {symbol} 季度業績報告發佈 (Earnings)",
            "tag": "🚨 業績日",
            "tag_color": "bg-rose-950 text-rose-300 border-rose-800 font-black animate-pulse",
            "impact": f"極高波動風險！股價單日可能有 > ±8% 暴升暴跌 (來源: {earn_src})",
        })

    macro_events = get_macro_events(now_tz, end_date)
    events.extend(macro_events)
    events.sort(key=lambda x: x["days_left"])
    return events[:5]


# ----------------------------------------------------------------------
# 高階期權風險量化引擎 (Black-Scholes-Merton + Greeks + Risk Metrics)
# ----------------------------------------------------------------------
def bs_greeks_and_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0, "pop": 0.0, "d1": 0.0, "d2": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        pop = norm.cdf(d2)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100.0
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        pop = norm.cdf(-d2)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        rho = (- K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100.0

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = (S * pdf_d1 * np.sqrt(T)) / 100.0

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
        "pop": float(pop),
        "d1": float(d1),
        "d2": float(d2)
    }


def _is_likely_us_optionable(symbol: str) -> bool:
    """
    Heuristic: most liquid US equities/ETFs have listed options on Yahoo.
    HK / CN / other suffixes often do not (or chains are sparse / blocked).
    Used only as optimistic fallback when Yahoo rate-limits the options endpoint.
    """
    s = symbol.strip().upper()
    if not s:
        return False
    # Explicit non-US suffixes that rarely have Yahoo option chains
    non_us_suffixes = (
        ".HK", ".SS", ".SZ", ".TW", ".T", ".KS", ".KQ", ".AX", ".L", ".TO",
        ".V", ".SA", ".MX", ".NS", ".BO", ".SI", ".JK", ".KL", ".BK"
    )
    if any(s.endswith(suf) for suf in non_us_suffixes):
        return False
    # Pure ticker like NVDA / VOO / RKLB / AAPL → treat as US-optionable fallback
    return True


def _check_has_options(symbol: str) -> bool:
    """
    Robust options availability check.

    Problems this solves:
    - Yahoo frequently rate-limits `Ticker.options` → empty / exception.
    - Caching a False result for 1 hour made all US stocks show「不能購買」.
    - Custom requests.Session sometimes interferes with options endpoint.

    Strategy:
    1. Honour cache if still within its own TTL.
    2. Try yfinance WITHOUT custom session (more reliable for options).
    3. On hard empty chain → cache False (short TTL for non-US, longer if confirmed).
    4. On rate-limit / any exception:
       - US-like symbols → optimistic True (short TTL so we re-check soon)
       - Non-US → False (short TTL)
    """
    symbol = symbol.strip().upper()
    now_ts = datetime.now().timestamp()
    cached = OPTIONS_CACHE.get(symbol)
    if cached:
        ttl = cached.get("ttl", OPTIONS_CACHE_TTL_NEGATIVE)
        if (now_ts - cached.get("ts", 0)) < ttl:
            return bool(cached.get("has_options", False))

    has = False
    source = "unknown"
    ttl = OPTIONS_CACHE_TTL_UNKNOWN

    try:
        # Prefer default yfinance session for options — custom Session is a common
        # cause of empty chains / rate-limit amplification.
        stock = yf.Ticker(symbol)
        opts = getattr(stock, "options", None)
        if opts is not None and len(opts) > 0:
            has = True
            source = "yahoo_chain"
            ttl = OPTIONS_CACHE_TTL_POSITIVE
        else:
            # Explicitly empty chain (not an exception)
            has = False
            source = "yahoo_empty"
            # For US names still allow optimistic override if we never saw a chain
            if _is_likely_us_optionable(symbol):
                has = True
                source = "us_optimistic_empty"
                ttl = OPTIONS_CACHE_TTL_UNKNOWN
            else:
                ttl = OPTIONS_CACHE_TTL_NEGATIVE
    except Exception as e:
        err_name = type(e).__name__
        # Rate limit / network / parse errors → do not permanently blacklist
        if _is_likely_us_optionable(symbol):
            has = True
            source = f"us_optimistic_on_error:{err_name}"
            ttl = OPTIONS_CACHE_TTL_UNKNOWN
        else:
            has = False
            source = f"error:{err_name}"
            ttl = OPTIONS_CACHE_TTL_NEGATIVE

    OPTIONS_CACHE[symbol] = {
        "has_options": has,
        "ts": now_ts,
        "ttl": ttl,
        "source": source,
    }
    return has


def _select_real_option_contract(
    symbol: str,
    close_price: float,
    target_strike: float,
    option_type: str,
    now_date: datetime.date,
    prefer_dte: int = 35,
) -> dict:
    """
    Pick a real Yahoo option contract near target DTE & strike.
    Returns dict with strike, exp_date, dte, iv, bid, ask, last, mid, contractSymbol.
    Falls back to empty fields if chain unavailable (caller uses theoretical).
    """
    result = {
        "strike": target_strike,
        "exp_date": None,
        "dte": None,
        "iv": None,
        "bid": None,
        "ask": None,
        "last": None,
        "mid": None,
        "contract": None,
        "source": "theoretical",
    }
    try:
        stock = yf.Ticker(symbol)
        exps = list(getattr(stock, "options", None) or [])
        if not exps:
            return result

        # Choose expiration closest to prefer_dte (and at least 7 DTE)
        best_exp = None
        best_diff = 10**9
        for e in exps:
            try:
                ed = datetime.strptime(e, "%Y-%m-%d").date()
            except Exception:
                continue
            dte = (ed - now_date).days
            if dte < 7:
                continue
            diff = abs(dte - prefer_dte)
            if diff < best_diff:
                best_diff = diff
                best_exp = e
        if best_exp is None:
            # take farthest available if all short-dated
            best_exp = exps[-1]

        exp_dt = datetime.strptime(best_exp, "%Y-%m-%d").date()
        dte = max(1, (exp_dt - now_date).days)
        chain = stock.option_chain(best_exp)
        table = chain.calls if option_type == "call" else chain.puts
        if table is None or table.empty:
            result["exp_date"] = best_exp
            result["dte"] = dte
            return result

        # Nearest strike to target
        table = table.copy()
        table["strike_diff"] = (table["strike"] - target_strike).abs()
        row = table.loc[table["strike_diff"].idxmin()]
        strike = float(row["strike"])
        bid = float(row["bid"]) if pd.notna(row.get("bid")) else None
        ask = float(row["ask"]) if pd.notna(row.get("ask")) else None
        last = float(row["lastPrice"]) if pd.notna(row.get("lastPrice")) else None
        iv = float(row["impliedVolatility"]) if pd.notna(row.get("impliedVolatility")) else None
        if iv is not None and (iv <= 0 or iv > 5):
            iv = None  # junk
        mid = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = round((bid + ask) / 2, 2)
        elif last is not None and last > 0:
            mid = round(last, 2)

        result.update({
            "strike": round(strike, 2),
            "exp_date": best_exp,
            "dte": dte,
            "iv": iv,
            "bid": bid,
            "ask": ask,
            "last": last,
            "mid": mid,
            "contract": str(row.get("contractSymbol", "")) or None,
            "source": "yahoo_chain",
        })
    except Exception:
        pass
    return result


def calculate_options_recommendation(
    symbol: str,
    df: pd.DataFrame,
    score: int,
    shock_data: dict,
    events: list
) -> dict:
    """
    Quant options engine:
    - Decision gates unchanged
    - Strike/expiry prefer REAL chain contracts
    - Sigma prefers market IV when available, else 30D HV
    - Short-put PoP = N(d2) = P(S_T > K) under RN measure
    """
    debug = {
        "score": score,
        "score_strong_bull": SCORE_STRONG_BULL,
        "score_neutral_floor": SCORE_NEUTRAL_FLOOR,
        "has_near_earnings": False,
        "earnings_days_left": None,
        "is_high_volatile": bool(shock_data.get("is_high_volatile", False)),
        "shock_72h_pct": shock_data.get("shock_72h_pct"),
        "roc_72h_pct": shock_data.get("roc_72h_pct"),
        "has_options": False,
        "options_source": None,
        "df_len": len(df),
        "gate": None,
        "reason_detail": ""
    }

    # Gate 1: options market existence
    has_opts = _check_has_options(symbol)
    debug["has_options"] = has_opts
    cache_meta = OPTIONS_CACHE.get(symbol, {})
    debug["options_source"] = cache_meta.get("source")
    if not has_opts:
        debug["gate"] = "NO_OPTIONS_MARKET"
        debug["reason_detail"] = (
            f"{symbol} 無可用期權鏈 "
            f"(source={cache_meta.get('source', 'n/a')})"
        )
        return {
            "strategy": "🚫 不能購買 (無期權市場)",
            "action": "NONE",
            "reason": (
                f"系統檢測到 {symbol} 目前不支援期權交易或無可用的期權合約。"
                f" (偵測來源: {cache_meta.get('source', 'n/a')})"
            ),
            "strike_price": "N/A",
            "exp_date": "N/A",
            "take_profit_target": "N/A",
            "greeks": {},
            "risk_metrics": {},
            "stress_test": [],
            "debug": debug
        }

    # Gate 2: enough history for HV
    if len(df) < 30:
        debug["gate"] = "INSUFFICIENT_HISTORY"
        debug["reason_detail"] = f"歷史 K 線僅 {len(df)} 根 (<30)"
        return {
            "strategy": "🚫 數據不足",
            "action": "NONE",
            "reason": "歷史數據少於 30 日，無法精確計算 30D 年化波動率與 Greeks。",
            "strike_price": "N/A",
            "exp_date": "N/A",
            "take_profit_target": "N/A",
            "greeks": {},
            "risk_metrics": {},
            "stress_test": [],
            "debug": debug
        }

    latest = df.iloc[-1]
    close_price = float(latest['Close'])
    atr = float(latest['ATR'])
    bb_lower = float(latest['BB_Lower'])
    bb_upper = float(latest['BB_Upper'])

    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    hv_30d = float(log_returns.tail(30).std() * np.sqrt(252))
    if pd.isna(hv_30d) or hv_30d <= 0:
        hv_30d = 0.30

    tz_str = "Asia/Hong_Kong" if symbol.endswith(".HK") else "America/New_York"
    now_date = datetime.now(ZoneInfo(tz_str)).date()
    target_dt = now_date + timedelta(days=35)
    days_to_friday = (4 - target_dt.weekday()) % 7
    exp_dt = target_dt + timedelta(days=days_to_friday)
    exp_date = exp_dt.strftime("%Y-%m-%d")

    days_to_exp = max(1, (exp_dt - now_date).days)
    T = days_to_exp / 365.0
    r = 0.045

    # Earnings proximity (timezone-aware)
    near_earn = False
    earn_days = None
    for e in events:
        title = e.get("title", "")
        days = e.get("days_left", 99)
        if "業績" in title or "Earnings" in title:
            earn_days = days
            if days <= EARNINGS_BLOCK_DAYS:
                near_earn = True
                break
    debug["has_near_earnings"] = near_earn
    debug["earnings_days_left"] = earn_days
    is_high_volatile = bool(shock_data.get("is_high_volatile", False))

    # Gate 3: hard block
    if near_earn or is_high_volatile or score < SCORE_NEUTRAL_FLOOR:
        reasons = []
        if near_earn:
            reasons.append(f"近 {EARNINGS_BLOCK_DAYS} 天有業績日 (剩餘 {earn_days} 天)")
        if is_high_volatile:
            reasons.append(
                f"高波動 (72h振幅 {shock_data.get('shock_72h_pct')}% / "
                f"淨漲跌 {shock_data.get('roc_72h_pct')}%)"
            )
        if score < SCORE_NEUTRAL_FLOOR:
            reasons.append(f"技術分 {score} < {SCORE_NEUTRAL_FLOOR}")
        debug["gate"] = "HARD_BLOCK"
        debug["reason_detail"] = " | ".join(reasons)
        return {
            "strategy": "🚫 暫不建議期權操作",
            "action": "NONE",
            "reason": (
                "近 7 天有重大業績發佈、股價極致震盪或技術打分偏弱，"
                "波動率溢價 (IV Crush Risk) 過高或方向極不明確。"
                f"（{debug['reason_detail']}）"
            ),
            "strike_price": "N/A",
            "exp_date": "N/A",
            "take_profit_target": "N/A",
            "greeks": {
                "hv_30d": f"{hv_30d*100:.1f}%",
                "dte": f"{days_to_exp} 天"
            },
            "risk_metrics": {},
            "stress_test": [],
            "debug": debug
        }

    # Gate 4: strong bull → Buy Call
    if score >= SCORE_STRONG_BULL:
        ideal_strike = round(close_price + (1.0 * atr), 2)
        contract = _select_real_option_contract(
            symbol, close_price, ideal_strike, "call", now_date, prefer_dte=35
        )
        strike = float(contract["strike"])
        if contract.get("exp_date"):
            exp_date = contract["exp_date"]
            days_to_exp = int(contract["dte"] or days_to_exp)
            T = days_to_exp / 365.0
        pct_otm = round((strike / close_price - 1) * 100, 1)

        sigma = float(contract["iv"]) if contract.get("iv") else hv_30d
        sigma_label = "IV" if contract.get("iv") else "HV"
        bs_res = bs_greeks_and_price(close_price, strike, T, r, sigma, "call")
        theo = max(bs_res["price"], 0.01)
        # Prefer market mid for premium estimate when buying
        call_price = float(contract["mid"]) if contract.get("mid") else theo
        call_price = max(call_price, 0.01)

        var_95_stock = 1.645 * sigma * np.sqrt(T) * close_price
        pop_pct = round(bs_res["pop"] * 100, 1)  # call: P(S_T > K)

        stress_test = []
        for change_pct in [-10, -5, 0, 5, 10]:
            sim_s = close_price * (1 + change_pct / 100.0)
            sim_res = bs_greeks_and_price(sim_s, strike, T, r, sigma, "call")
            sim_pnl = ((sim_res["price"] - call_price) / call_price) * 100
            stress_test.append({
                "price_change": f"{change_pct:+d}%",
                "target_price": round(sim_s, 2),
                "opt_price": round(sim_res["price"], 2),
                "pnl_pct": f"{sim_pnl:+.1f}%"
            })

        mkt_note = ""
        if contract.get("bid") is not None and contract.get("ask") is not None:
            mkt_note = f"｜市價 Bid/Ask ${contract['bid']:.2f}/${contract['ask']:.2f}"
        src = contract.get("source", "theoretical")

        debug["gate"] = "BUY_CALL"
        debug["reason_detail"] = f"score {score} >= {SCORE_STRONG_BULL} | chain={src}"
        return {
            "strategy": "🐂 建議 Buy Call (買看漲期權)",
            "action": "BUY_CALL",
            "reason": (
                f"技術面得分為強勢多頭區間 ({score}分)，採用 Buy Call 槓桿參與上漲行情。"
                f" 合約來源: {src}{mkt_note}"
            ),
            "strike_price": f"${strike} (OTM +{pct_otm}%)",
            "exp_date": f"{exp_date} ({days_to_exp} 天 DTE)",
            "take_profit_target": "權利金獲利 +50% ~ +100% 或正股觸及布林上軌時提前平倉",
            "greeks": {
                "hv_30d": f"{hv_30d*100:.1f}%",
                "sigma": f"{sigma*100:.1f}% ({sigma_label})",
                "est_premium": f"${call_price:.2f}",
                "delta": f"{bs_res['delta']:.3f}",
                "gamma": f"{bs_res['gamma']:.4f}",
                "theta": f"${bs_res['theta']:.3f}/日",
                "vega": f"${bs_res['vega']:.3f}/1% IV"
            },
            "risk_metrics": {
                "pop": f"{pop_pct}% (到期價內勝率)",
                "max_loss": f"${call_price:.2f} / 股 (${call_price*100:.0f} / 張)",
                "var_95": f"${var_95_stock:.2f} (正股 95% 波動風險值)",
                "risk_reward": f"1 : {(atr*2)/call_price:.1f} (理論盈虧比)",
                "contract": contract.get("contract") or "N/A",
            },
            "stress_test": stress_test,
            "debug": debug
        }

    # Gate 5: neutral → Sell Put
    ideal_strike = round(min(close_price - (1.5 * atr), bb_lower), 2)
    contract = _select_real_option_contract(
        symbol, close_price, ideal_strike, "put", now_date, prefer_dte=35
    )
    strike = float(contract["strike"])
    if contract.get("exp_date"):
        exp_date = contract["exp_date"]
        days_to_exp = int(contract["dte"] or days_to_exp)
        T = days_to_exp / 365.0
    pct_otm = round((1 - strike / close_price) * 100, 1)

    sigma = float(contract["iv"]) if contract.get("iv") else hv_30d
    sigma_label = "IV" if contract.get("iv") else "HV"
    bs_res = bs_greeks_and_price(close_price, strike, T, r, sigma, "put")
    theo = max(bs_res["price"], 0.01)
    # Selling: conservative credit ≈ bid if available, else mid/theo
    if contract.get("bid") is not None and contract["bid"] > 0:
        put_price = float(contract["bid"])
    elif contract.get("mid"):
        put_price = float(contract["mid"])
    else:
        put_price = theo
    put_price = max(put_price, 0.01)

    # Short put profit if S_T > K → PoP = N(d2)
    pop_pct = round(norm.cdf(bs_res["d2"]) * 100, 1)
    var_95_stock = 1.645 * sigma * np.sqrt(T) * close_price

    stress_test = []
    for change_pct in [-10, -5, 0, 5, 10]:
        sim_s = close_price * (1 + change_pct / 100.0)
        sim_res = bs_greeks_and_price(sim_s, strike, T, r, sigma, "put")
        sim_pnl_dollar = put_price - sim_res["price"]
        sim_pnl_pct = (sim_pnl_dollar / put_price) * 100
        stress_test.append({
            "price_change": f"{change_pct:+d}%",
            "target_price": round(sim_s, 2),
            "opt_price": round(sim_res["price"], 2),
            "pnl_pct": f"{sim_pnl_pct:+.1f}%"
        })

    mkt_note = ""
    if contract.get("bid") is not None and contract.get("ask") is not None:
        mkt_note = f"｜市價 Bid/Ask ${contract['bid']:.2f}/${contract['ask']:.2f}"
    src = contract.get("source", "theoretical")

    debug["gate"] = "SELL_PUT"
    debug["reason_detail"] = (
        f"{SCORE_NEUTRAL_FLOOR} <= score {score} < {SCORE_STRONG_BULL} | chain={src}"
    )
    return {
        "strategy": "🛡️ 建議 Sell Put (賣看跌期權/收取權利金)",
        "action": "SELL_PUT",
        "reason": (
            f"技術面處於中性盤整區間 ({score}分)，適合透過 Sell Put 賺取時間價值衰減 (Theta decay)。"
            f" 合約來源: {src}{mkt_note}"
        ),
        "strike_price": f"${strike} (OTM -{pct_otm}%)",
        "exp_date": f"{exp_date} ({days_to_exp} 天 DTE)",
        "take_profit_target": "當獲得最大權利金收益之 50% ~ 75% 時提前買回平倉 (Buy to Close)",
        "greeks": {
            "hv_30d": f"{hv_30d*100:.1f}%",
            "sigma": f"{sigma*100:.1f}% ({sigma_label})",
            "est_premium": f"${put_price:.2f}",
            "delta": f"{bs_res['delta']:.3f}",
            "gamma": f"{bs_res['gamma']:.4f}",
            "theta": f"${abs(bs_res['theta']):.3f}/日 (收益)",
            "vega": f"${bs_res['vega']:.3f}/1% IV"
        },
        "risk_metrics": {
            "pop": f"{pop_pct}% (到期不履約/獲利勝率)",
            "max_gain": f"${put_price:.2f} / 股 (${put_price*100:.0f} / 張)",
            "var_95": f"${var_95_stock:.2f} (正股 95% 波動風險值)",
            "breakeven": f"${strike - put_price:.2f} (損益平衡點)",
            "contract": contract.get("contract") or "N/A",
        },
        "stress_test": stress_test,
        "debug": debug
    }


# ----------------------------------------------------------------------
# 市場狀態與即時股價獲取
# ----------------------------------------------------------------------
def get_stock_session(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.endswith(".HK"):
        now_hk = datetime.now(ZoneInfo("Asia/Hong_Kong"))
        if now_hk.weekday() >= 5:
            return "CLOSED"
        time_num = now_hk.hour * 100 + now_hk.minute
        if 900 <= time_num < 930:
            return "PRE"
        elif 930 <= time_num < 1200:
            return "REGULAR"
        elif 1200 <= time_num < 1300:
            return "LUNCH_BREAK"
        elif 1300 <= time_num < 1610:
            return "REGULAR"
        else:
            return "CLOSED"
    else:
        now_us = datetime.now(ZoneInfo("America/New_York"))
        if now_us.weekday() >= 5:
            return "CLOSED"
        time_num = now_us.hour * 100 + now_us.minute
        if 400 <= time_num < 930:
            return "PRE"
        elif 930 <= time_num < 1600:
            return "REGULAR"
        elif 1600 <= time_num < 2000:
            return "POST"
        else:
            return "CLOSED"


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.isdigit():
        if len(symbol) <= 4:
            return f"{symbol.zfill(4)}.HK"
    return symbol


def get_realtime_price(stock) -> Optional[float]:
    try:
        df_1m = stock.history(period="1d", interval="1m", auto_adjust=False, prepost=True)
        if not df_1m.empty:
            price = float(df_1m['Close'].iloc[-1])
            if price > 0:
                return price
    except Exception:
        pass
    try:
        if hasattr(stock, 'fast_info'):
            price = stock.fast_info.get('lastPrice') or getattr(stock.fast_info, 'last_price', None)
            if price and not pd.isna(price) and float(price) > 0:
                return float(price)
    except Exception:
        pass
    return None


def get_realtime_open_price(stock, symbol) -> Optional[float]:
    symbol = symbol.strip().upper()
    tz_str = "Asia/Hong_Kong" if symbol.endswith(".HK") else "America/New_York"
    try:
        df_intra = stock.history(period="5d", interval="1m", auto_adjust=False, prepost=True)
        if not df_intra.empty:
            df_intra.index = df_intra.index.tz_convert(ZoneInfo(tz_str))
            today_date = datetime.now(ZoneInfo(tz_str)).date()
            df_today = df_intra[df_intra.index.date == today_date]
            if not df_today.empty:
                first_open = float(df_today['Open'].iloc[0])
                if first_open > 0:
                    return first_open
    except Exception:
        pass
    try:
        if hasattr(stock, 'fast_info'):
            open_p = stock.fast_info.get('open') or getattr(stock.fast_info, 'open', None)
            if open_p and not pd.isna(open_p) and float(open_p) > 0:
                return float(open_p)
    except Exception:
        pass
    return None


def fetch_stock_data(symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
    symbol = symbol.strip().upper()
    stock = yf.Ticker(symbol, session=YF_SESSION)
    df = stock.history(period="180d", auto_adjust=False)

    if df.empty or len(df) < 15:
        return None, "CLOSED"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=['Close']).copy()
    session = get_stock_session(symbol)

    # Only overwrite last bar when market is potentially live
    if session in ["PRE", "REGULAR", "LUNCH_BREAK", "POST", "CLOSED"]:
        real_open = get_realtime_open_price(stock, symbol)
        if real_open and real_open > 0:
            df.iloc[-1, df.columns.get_loc("Open")] = real_open
        else:
            curr_open = df.iloc[-1]["Open"]
            if pd.isna(curr_open) or curr_open <= 0:
                df.iloc[-1, df.columns.get_loc("Open")] = df.iloc[-1]["Close"]

        live_price = get_realtime_price(stock)
        if live_price and live_price > 0:
            df.iloc[-1, df.columns.get_loc("Close")] = live_price
            df.iloc[-1, df.columns.get_loc("High")] = max(float(df.iloc[-1]["High"]), live_price)
            df.iloc[-1, df.columns.get_loc("Low")] = min(float(df.iloc[-1]["Low"]), live_price)

    return df, session


# ----------------------------------------------------------------------
# 技術指標與異動警訊偵測
# ----------------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Close_MA20_Ratio'] = df['Close'] / df['MA20']

    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
    df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min).replace(0, 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['ATR'] = tr.rolling(window=14).mean()

    plus_dm = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    minus_dm = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    atr14 = df['ATR'].replace(0, 1e-9)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr14)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA5'].replace(0, 1e-9)

    return df.ffill().bfill()


def check_signals(df: pd.DataFrame) -> List[str]:
    if len(df) < 30:
        return []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(latest['Close'])
    vol_ratio = float(latest['Vol_Ratio'])
    bb_u, bb_l = float(latest['BB_Upper']), float(latest['BB_Lower'])
    bb_w = float(latest['BB_Width'])
    rsi = float(latest['RSI'])
    k, d = float(latest['K']), float(latest['D'])
    prev_k, prev_d = float(prev['K']), float(prev['D'])

    recent_bb_w = df['BB_Width'].tail(60)
    is_squeeze = bb_w <= np.percentile(recent_bb_w, 20)
    signals = []

    low_10 = df['Low'].tail(10).values
    rsi_10 = df['RSI'].tail(10).values
    bullish_div = (low_10[-1] < low_10[-5]) and (rsi_10[-1] > rsi_10[-5]) if len(low_10) >= 10 else False
    if (is_squeeze and close > bb_u and vol_ratio >= 2.0) or (bullish_div and (prev_k <= prev_d and k > d)):
        signals.append("🚀 暴升警訊：布林壓縮爆量向上突破 / 底背離共振！")

    high_10 = df['High'].tail(10).values
    bearish_div = (high_10[-1] > high_10[-5]) and (rsi_10[-1] < rsi_10[-5]) if len(high_10) >= 10 else False
    if (is_squeeze and close < bb_l and vol_ratio >= 2.0) or (bearish_div and (prev_k >= prev_d and k < d)):
        signals.append("🩸 暴跌警訊：布林壓縮爆量向下跌破 / 頂背離！")

    if (rsi < 32 and prev_k <= prev_d and k > d) or (close <= bb_l * 1.005 and rsi < 35):
        signals.append("🎯 抄底警訊：超賣區極致超跌 (RSI < 35) + KDJ 觸底黃金交叉！")

    if (rsi > 68 and prev_k >= prev_d and k < d) or (close >= bb_u * 0.995 and rsi > 65):
        signals.append("⚠️ 逃頂警訊：超買區過熱 (RSI > 65) + KDJ 高位死亡交叉！")

    return signals


def calculate_price_shock(df: pd.DataFrame) -> dict:
    """
    Use completed bars only when possible to reduce live-bar noise that
    causes Mac vs cloud divergence.
    """
    if len(df) < 4:
        return {"shock_72h_pct": 0.0, "roc_72h_pct": 0.0, "is_high_volatile": False}

    # Prefer last 3 completed bars; fall back to last 3 including current
    lookback = df.iloc[-4:-1] if len(df) >= 4 else df.tail(3)
    if lookback.empty:
        lookback = df.tail(3)

    latest_close = float(df['Close'].iloc[-1])
    high_72h = float(lookback['High'].max())
    low_72h = float(lookback['Low'].min())
    if low_72h <= 0:
        return {"shock_72h_pct": 0.0, "roc_72h_pct": 0.0, "is_high_volatile": False}

    shock_72h_pct = round(((high_72h - low_72h) / low_72h) * 100, 2)

    close_3d_ago = float(lookback['Close'].iloc[0])
    roc_72h_pct = round(((latest_close - close_3d_ago) / close_3d_ago) * 100, 2) if close_3d_ago > 0 else 0.0

    is_high_volatile = (shock_72h_pct >= VOL_SHOCK_PCT) or (abs(roc_72h_pct) >= VOL_ROC_PCT)

    return {
        "shock_72h_pct": shock_72h_pct,
        "roc_72h_pct": roc_72h_pct,
        "is_high_volatile": is_high_volatile
    }


def calculate_rigorous_score(df: pd.DataFrame):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    close = float(latest['Close'])
    prev_close = float(prev['Close'])
    ma5, ma10, ma20 = float(latest['MA5']), float(latest['MA10']), float(latest['MA20'])
    rsi = float(latest['RSI'])
    k, d = float(latest['K']), float(latest['D'])
    prev_k, prev_d = float(prev['K']), float(prev['D'])
    macd_h, prev_macd_h = float(latest['MACD_Hist']), float(prev['MACD_Hist'])
    vol_ratio = float(latest['Vol_Ratio'])
    adx = float(latest['ADX'])

    score = 50
    reasons = []

    if close > ma5 and ma5 > ma10 and ma10 > ma20:
        score += 20
        reasons.append("• 均線系統：完美多頭排列 (股價 > MA5 > MA10 > MA20) (+20分)")
    elif close < ma5 and ma5 < ma10 and ma10 < ma20:
        score -= 20
        reasons.append("• 均線系統：完全空頭排列 (股價 < MA5 < MA10 < MA20) (-20分)")
    elif close >= ma20:
        score += 10
        reasons.append("• 均線系統：股價守在 MA20 月線之上，具備中期底部支撐 (+10分)")
    else:
        score -= 10
        reasons.append("• 均線系統：股價跌破 MA20 月線，上方存在明顯反壓 (-10分)")

    if macd_h > 0 and macd_h > prev_macd_h:
        score += 15
        reasons.append("• MACD 動能：零軸之上紅柱持續放大，多方攻擊動能增強 (+15分)")
    elif macd_h > 0 and macd_h <= prev_macd_h:
        score += 5
        reasons.append("• MACD 動能：紅柱縮短，多方動能雖在但逐漸放緩 (+5分)")
    elif macd_h < 0 and macd_h < prev_macd_h:
        score -= 15
        reasons.append("• MACD 動能：零軸之下綠柱持續放大，空方賣壓沉重 (-15分)")
    else:
        score -= 5
        reasons.append("• MACD 動能：綠柱縮短，空方下殺力道開始收斂 (-5分)")

    if prev_k <= prev_d and k > d:
        score += 10
        reasons.append(f"• 擺盪轉折：KDJ 觸發黃金交叉 (K:{k:.1f}, D:{d:.1f})，短線轉強 (+10分)")
    elif prev_k >= prev_d and k < d:
        score -= 10
        reasons.append(f"• 擺盪轉折：KDJ 觸發死亡交叉 (K:{k:.1f}, D:{d:.1f})，短線轉弱 (-10分)")
    elif rsi > 70:
        score -= 5
        reasons.append(f"• 擺盪轉折：RSI 進入超買區 ({rsi:.1f})，需留意短線高檔回檔風險 (-5分)")
    elif rsi < 30:
        score += 5
        reasons.append(f"• 擺盪轉折：RSI 進入超賣區 ({rsi:.1f})，隨時醞釀技術性反彈 (+5分)")
    else:
        reasons.append(f"• 擺盪轉折：RSI 數值 {rsi:.1f}，處於中性震盪區間 (+0分)")

    if adx >= 25:
        score += 10
        reasons.append(f"• 趨勢強度：ADX 數值達 {adx:.1f}，顯示目前趨勢明確且強勁 (+10分)")
    else:
        reasons.append(f"• 趨勢強度：ADX 數值僅 {adx:.1f} (<25)，市場偏向盤整或無明顯方向 (+0分)")

    if vol_ratio >= 1.5 and close > prev_close:
        score += 10
        reasons.append(f"• 量價結構：帶量上漲 (成交量達均量 {vol_ratio:.2f} 倍) (+10分)")
    elif vol_ratio >= 1.5 and close < prev_close:
        score -= 10
        reasons.append(f"• 量價結構：帶量下殺 (成交量達均量 {vol_ratio:.2f} 倍) (-10分)")
    elif vol_ratio < 0.8:
        reasons.append(f"• 量價結構：量能萎縮 ({vol_ratio:.2f}x)，觀望氣氛濃厚 (+0分)")
    else:
        reasons.append(f"• 量價結構：量價表現平穩 ({vol_ratio:.2f}x) (+0分)")

    shock_data = calculate_price_shock(df)
    if shock_data["is_high_volatile"]:
        if shock_data["roc_72h_pct"] > 0:
            reasons.append(f"• ⚠️ 風控警示：近 72h 振幅達 {shock_data['shock_72h_pct']}% (防追高機制) (-5分)")
            score -= 5
        else:
            reasons.append(f"• ⚠️ 風控警示：近 72h 震盪下挫 {shock_data['roc_72h_pct']}% (防接刀機制) (-10分)")
            score -= 10

    score = max(0, min(100, int(round(score))))
    advice = "多頭格局可偏多操作" if score >= SCORE_STRONG_BULL else (
        "中性盤整觀望" if score >= SCORE_NEUTRAL_FLOOR else "空頭弱勢建議避險"
    )
    return score, reasons, advice, shock_data


# ----------------------------------------------------------------------
# 13 大 AI 機器學習預估引擎
# ----------------------------------------------------------------------
def _get_model_weights(symbol: str, model_names: List[str]) -> Dict[str, float]:
    """
    Inverse-MAE weights from historical verification feedback loop.
    Models with lower O/H/L mean absolute error get higher weight.
    Cold-start → equal weights.
    """
    maes = GLOBAL_STATE.get("model_maes", {}).get(symbol, {})
    raw = {}
    for name in model_names:
        info = maes.get(name, {})
        mae = float(info.get("mae_ohl", 0) or 0)
        n = int(info.get("n", 0) or 0)
        # Need at least 2 settled samples before trusting MAE
        if n >= 2 and mae > 0:
            raw[name] = 1.0 / (mae + 1e-6)
        else:
            raw[name] = 1.0
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def _update_model_maes(symbol: str, model_preds: List[dict], actual_open: float, actual_high: float, actual_low: float):
    """Exponential-ish running MAE per model after a day is fully settled."""
    if not model_preds or actual_open <= 0:
        return
    store = GLOBAL_STATE.setdefault("model_maes", {})
    sym_maes = store.setdefault(symbol, {})
    alpha = 0.35  # weight on newest error

    for m in model_preds:
        name = m.get("name")
        if not name:
            continue
        err = (
            abs(float(m.get("open", 0)) - actual_open)
            + abs(float(m.get("high", 0)) - actual_high)
            + abs(float(m.get("low", 0)) - actual_low)
        ) / 3.0
        prev = sym_maes.get(name, {"mae_ohl": err, "n": 0})
        n = int(prev.get("n", 0)) + 1
        old_mae = float(prev.get("mae_ohl", err))
        new_mae = err if n == 1 else (1 - alpha) * old_mae + alpha * err
        sym_maes[name] = {"mae_ohl": round(new_mae, 4), "n": n}

    # Ensemble-level stats for confidence display
    ens = sym_maes.get("_ensemble", {"mae_ohl": 0.0, "n": 0, "open_hits": 0})
    # caller also passes ensemble diffs via side channel — updated in audit
    store[symbol] = sym_maes


def _ensemble_accuracy_summary(symbol: str) -> dict:
    """Build human-readable accuracy stats for UI."""
    maes = GLOBAL_STATE.get("model_maes", {}).get(symbol, {})
    ens = maes.get("_ensemble", {})
    n = int(ens.get("n", 0) or 0)
    mae = float(ens.get("mae_ohl", 0) or 0)
    open_hits = int(ens.get("open_hits", 0) or 0)
    open_hit_pct = round(100.0 * open_hits / n, 1) if n > 0 else None

    # Rank models by MAE (best first)
    ranked = []
    for name, info in maes.items():
        if name.startswith("_"):
            continue
        if int(info.get("n", 0) or 0) >= 2:
            ranked.append((name, float(info.get("mae_ohl", 999))))
    ranked.sort(key=lambda x: x[1])

    if n >= 5 and mae > 0:
        if mae < 1.5:
            conf = "高"
        elif mae < 3.0:
            conf = "中"
        else:
            conf = "低"
        conf_text = f"{conf} (近 {n} 日 MAE ${mae:.2f}"
        if open_hit_pct is not None:
            conf_text += f", 開盤方向命中 {open_hit_pct}%"
        conf_text += ")"
    elif n > 0:
        conf_text = f"累積中 (已結算 {n} 日)"
    else:
        conf_text = "冷啟動 (等結算回饋)"

    return {
        "confidence": conf_text,
        "settled_n": n,
        "ensemble_mae": round(mae, 2) if n else None,
        "open_hit_pct": open_hit_pct,
        "top_models": [{"name": a[0], "mae": round(a[1], 2)} for a in ranked[:3]],
    }


def predict_prices_with_13_models(df: pd.DataFrame, symbol: str = ""):
    """
    13-model ensemble with inverse-MAE weighting from verification feedback.
    Does NOT add more models — improves the existing set via learned weights.
    """
    try:
        symbol = (symbol or "").strip().upper()
        data = df.copy()
        data['Target_Open_Ret'] = (data['Open'].shift(-1) - data['Close']) / data['Close']
        data['Target_High_Ret'] = (data['High'].shift(-1) - data['Close']) / data['Close']
        data['Target_Low_Ret'] = (data['Low'].shift(-1) - data['Close']) / data['Close']

        features = ['RSI', 'K', 'D', 'Close_MA20_Ratio', 'BB_Width', 'MACD_Hist', 'Vol_Ratio', 'ADX']
        train_data = data.dropna(subset=['Target_Open_Ret', 'Target_High_Ret', 'Target_Low_Ret'] + features)

        if len(train_data) < 15:
            return None

        X = train_data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        latest_X = scaler.transform(data[features].iloc[[-1]])
        curr_close = float(data['Close'].iloc[-1])

        gpr_kernel = C(1.0) * RBF(1.0) + WhiteKernel()

        models = {
            "BayesianRidge": BayesianRidge(),
            "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
            "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=30, max_depth=3, random_state=42),
            "GBDT": GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42),
            "GaussianProcess": GaussianProcessRegressor(kernel=gpr_kernel, alpha=1e-2, random_state=42),
            "HistGBDT": HistGradientBoostingRegressor(max_iter=30, max_depth=3, random_state=42),
            "Huber": HuberRegressor(max_iter=1000),
            "Lasso": Lasso(alpha=0.001),
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42),
            "Ridge": Ridge(alpha=2.0),
            "SVR": SVR(C=1.0, kernel='rbf')
        }

        open_details, high_details, low_details = {}, {}, {}

        for name, model in models.items():
            try:
                model.fit(X_scaled, train_data['Target_Open_Ret'])
                po = curr_close * (1 + float(model.predict(latest_X)[0]))

                model.fit(X_scaled, train_data['Target_High_Ret'])
                ph = curr_close * (1 + float(model.predict(latest_X)[0]))

                model.fit(X_scaled, train_data['Target_Low_Ret'])
                pl = curr_close * (1 + float(model.predict(latest_X)[0]))

                ph = max(ph, po, curr_close * 1.001)
                pl = min(pl, po, curr_close * 0.999)

                open_details[name] = round(po, 2)
                high_details[name] = round(ph, 2)
                low_details[name] = round(pl, 2)
            except Exception:
                pass

        if not open_details:
            return None

        weights = _get_model_weights(symbol, list(open_details.keys()))

        def wavg(detail: dict) -> float:
            s = sum(weights.get(k, 0) * v for k, v in detail.items())
            return round(float(s), 2)

        avg_open = wavg(open_details)
        avg_high = wavg(high_details)
        avg_low = wavg(low_details)
        # Keep OHLC consistency
        avg_high = max(avg_high, avg_open, round(curr_close * 1.001, 2))
        avg_low = min(avg_low, avg_open, round(curr_close * 0.999, 2))
        open_pct = round(((avg_open - curr_close) / curr_close) * 100, 1)

        acc = _ensemble_accuracy_summary(symbol)

        model_list = []
        for name in sorted(open_details.keys()):
            model_list.append({
                "name": name,
                "open": open_details[name],
                "high": high_details[name],
                "low": low_details[name],
                "weight": round(weights.get(name, 0) * 100, 1),  # %
            })
        # Sort by weight desc for UI
        model_list.sort(key=lambda x: -x["weight"])

        return {
            "pred_open": avg_open,
            "open_pct": f"+{open_pct}%" if open_pct >= 0 else f"{open_pct}%",
            "pred_high": avg_high,
            "pred_low": avg_low,
            "confidence": acc["confidence"],
            "accuracy": acc,
            "model_list": model_list,
            "weighted": True,
        }
    except Exception as e:
        print(f"ML Error: {e}")
        return None


# ----------------------------------------------------------------------
# 審計對比算法
# ----------------------------------------------------------------------
def update_pred_audit(symbol: str, df: pd.DataFrame, session: str, pred_res: dict):
    symbol = symbol.strip().upper()
    raw_audit = GLOBAL_STATE.get("pred_audit", {}).get(symbol, [])
    tz_str = "Asia/Hong_Kong" if symbol.endswith(".HK") else "America/New_York"
    now_tz = datetime.now(ZoneInfo(tz_str))

    audit_dict = {}
    for item in raw_audit:
        b_date = item.get("base_date") or item.get("date")
        if b_date and b_date != "舊版歷史紀錄":
            item["base_date"] = b_date
            audit_dict[b_date] = item

    base_date_str = df.index[-1].strftime("%Y-%m-%d")
    current_time_num = now_tz.hour * 100 + now_tz.minute

    if pred_res is not None:
        model_snapshot = pred_res.get("model_list") or []
        if base_date_str in audit_dict:
            if not audit_dict[base_date_str].get("full_verified"):
                audit_dict[base_date_str]["pred_open"] = pred_res["pred_open"]
                audit_dict[base_date_str]["pred_high"] = pred_res["pred_high"]
                audit_dict[base_date_str]["pred_low"] = pred_res["pred_low"]
                if model_snapshot and not audit_dict[base_date_str].get("model_preds"):
                    audit_dict[base_date_str]["model_preds"] = model_snapshot
        else:
            audit_dict[base_date_str] = {
                "base_date": base_date_str,
                "target_date": "等待下個交易日",
                "pred_open": pred_res["pred_open"],
                "pred_high": pred_res["pred_high"],
                "pred_low": pred_res["pred_low"],
                "model_preds": model_snapshot,
                "open_verified": False,
                "full_verified": False,
                "mae_updated": False,
                "status": "⏳ 待結算"
            }

    sorted_dates = sorted(audit_dict.keys())
    audit_list = [audit_dict[d] for d in sorted_dates]
    df_date_strs = list(df.index.strftime("%Y-%m-%d"))

    for item in audit_list:
        b_date = item["base_date"]

        if item.get("target_date") == "等待下個交易日":
            future_dates = [d for d in df_date_strs if d > b_date]
            if future_dates:
                item["target_date"] = future_dates[0]

        target_date_str = item.get("target_date")
        if target_date_str and target_date_str != "等待下個交易日" and target_date_str in df_date_strs:
            target_row = df[df.index.strftime("%Y-%m-%d") == target_date_str].iloc[0]
            is_today = (target_date_str == base_date_str)

            if is_today:
                if symbol.endswith(".HK"):
                    is_truly_closed = (current_time_num >= 1610) or (session == "CLOSED")
                else:
                    is_truly_closed = (current_time_num >= 1600) or (session == "CLOSED")
            else:
                is_truly_closed = True

            act_open = round(float(target_row['Open']), 2)
            if act_open > 0 and not item.get("open_verified"):
                item["actual_open"] = act_open
                item["diff_open"] = round(abs(item["pred_open"] - act_open), 2)
                item["open_verified"] = True
                item["status"] = "開盤結算"

            if is_truly_closed and not item.get("full_verified"):
                act_high = round(float(target_row['High']), 2)
                act_low = round(float(target_row['Low']), 2)
                if act_high > 0 and act_low > 0:
                    item["actual_high"] = act_high
                    item["actual_low"] = act_low
                    item["diff_high"] = round(abs(item["pred_high"] - act_high), 2)
                    item["diff_low"] = round(abs(item["pred_low"] - act_low), 2)
                    item["full_verified"] = True
                    item["status"] = "完全結算"

            # Feedback loop: update per-model MAE once when first fully settled
            if item.get("full_verified") and not item.get("mae_updated"):
                act_o = float(item.get("actual_open") or 0)
                act_h = float(item.get("actual_high") or 0)
                act_l = float(item.get("actual_low") or 0)
                if act_o > 0 and act_h > 0 and act_l > 0:
                    _update_model_maes(
                        symbol,
                        item.get("model_preds") or [],
                        act_o, act_h, act_l,
                    )
                    # Ensemble MAE + open direction hit
                    ens_err = (
                        abs(float(item["pred_open"]) - act_o)
                        + abs(float(item["pred_high"]) - act_h)
                        + abs(float(item["pred_low"]) - act_l)
                    ) / 3.0
                    store = GLOBAL_STATE.setdefault("model_maes", {})
                    sym = store.setdefault(symbol, {})
                    ens = sym.get("_ensemble", {"mae_ohl": ens_err, "n": 0, "open_hits": 0})
                    n = int(ens.get("n", 0)) + 1
                    old = float(ens.get("mae_ohl", ens_err))
                    alpha = 0.35
                    new_mae = ens_err if n == 1 else (1 - alpha) * old + alpha * ens_err
                    # Open direction: pred vs prior close approximated by sign of pred move vs actual
                    # Hit if |pred_open - actual_open| <= 1% of price (practical tolerance)
                    tol = max(act_o * 0.01, 0.05)
                    hits = int(ens.get("open_hits", 0))
                    if abs(float(item["pred_open"]) - act_o) <= tol:
                        hits += 1
                    sym["_ensemble"] = {
                        "mae_ohl": round(new_mae, 4),
                        "n": n,
                        "open_hits": hits,
                    }
                    item["mae_updated"] = True
                    # Drop bulky model_preds after feedback to keep state small
                    if "model_preds" in item:
                        del item["model_preds"]

    final_list = []
    seen_targets = set()

    for item in reversed(audit_list):
        t_date = item.get("target_date")
        if t_date == "等待下個交易日":
            if "PENDING" in seen_targets:
                continue
            seen_targets.add("PENDING")
            final_list.append(item)
        else:
            if t_date in seen_targets:
                continue
            seen_targets.add(t_date)
            final_list.append(item)

    final_list.reverse()
    GLOBAL_STATE["pred_audit"][symbol] = final_list[-10:]
    save_state(GLOBAL_STATE)
    return final_list[-10:]


def calculate_dynamic_risk(close_price: float, atr: float):
    atr_pct = (atr / close_price) * 100 if close_price > 0 else 1.0
    suggested_pos = max(5, min(25, int(1.5 / (atr_pct / 100 + 1e-9))))
    stop_loss = round(close_price - (1.5 * atr), 2)
    take_profit = round(close_price + (2.5 * atr), 2)
    trailing_stop = round(close_price - (2.0 * atr), 2)
    return f"{suggested_pos}%", stop_loss, take_profit, trailing_stop


# ----------------------------------------------------------------------
# 全自動核心處理與背景定時輪詢任務
# ----------------------------------------------------------------------
def process_single_stock(symbol: str):
    symbol = symbol.strip().upper()
    df, session = fetch_stock_data(symbol)
    if df is None or df.empty:
        return

    df = calculate_indicators(df)
    signals = check_signals(df)

    latest = df.iloc[-1]
    curr_price = round(float(latest['Close']), 2)
    atr = float(latest['ATR'])

    score, reasons, advice, shock_data = calculate_rigorous_score(df)
    pred_res = predict_prices_with_13_models(df, symbol=symbol)
    history_logs = update_pred_audit(symbol, df, session, pred_res)
    pos_str, stop_loss, take_profit, trailing_stop = calculate_dynamic_risk(curr_price, atr)

    upcoming_events = fetch_stock_events(symbol)

    options_rec = calculate_options_recommendation(
        symbol=symbol,
        df=df,
        score=score,
        shock_data=shock_data,
        events=upcoming_events
    )

    session_map = {
        "PRE": "🌅 盤前/24小時盤",
        "REGULAR": "🔔 正常盤中",
        "LUNCH_BREAK": "🍱 中午休市",
        "POST": "🌃 盤後交易",
        "CLOSED": "🔒 已收盤"
    }
    session_text = session_map.get(session, "🔒 已收盤")

    if signals:
        alert_history = GLOBAL_STATE.get("alert_history", [])
        time_str = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%m-%d %H:%M:%S")
        for sig in signals:
            is_dup = False
            if alert_history:
                for past_alert in alert_history:
                    if past_alert.get("symbol") == symbol and past_alert.get("signal") == sig:
                        is_dup = True
                        break
            if not is_dup:
                alert_history.insert(0, {
                    "id": f"{symbol}_{time_str}_{hash(sig)}",
                    "symbol": symbol,
                    "price": curr_price,
                    "signal": sig,
                    "session": session_text,
                    "time": time_str
                })
        GLOBAL_STATE["alert_history"] = alert_history[:10]
        save_state(GLOBAL_STATE)

    GLOBAL_STATE["stock_states"][symbol] = {
        "symbol": symbol,
        "session_text": session_text,
        "current_price": curr_price,
        "score": score,
        "advice": advice,
        "reasons": reasons,
        "signals": signals,
        "upcoming_events": upcoming_events,
        "options_rec": options_rec,
        "last_update": datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%H:%M:%S"),
        "tech": {
            "rsi": round(float(latest['RSI']), 1),
            "k": round(float(latest['K']), 1),
            "d": round(float(latest['D']), 1),
            "macd_hist": round(float(latest['MACD_Hist']), 3),
            "ma5": round(float(latest['MA5']), 2),
            "ma10": round(float(latest['MA10']), 2),
            "ma20": round(float(latest['MA20']), 2),
            "vol_ratio": round(float(latest['Vol_Ratio']), 2),
            "adx": round(float(latest['ADX']), 1),
            "atr": round(atr, 2),
            "bb_width": round(float(latest['BB_Width']) * 100, 1)
        },
        "shock_data": shock_data,
        "pred": pred_res,
        "risk": {
            "position": pos_str,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": trailing_stop
        },
        "history_logs": history_logs
    }


async def background_market_loop():
    while True:
        watchlist = list(GLOBAL_STATE.get("watchlist", WATCHLIST))
        for symbol in watchlist:
            try:
                await asyncio.to_thread(process_single_stock, symbol)
                await asyncio.sleep(1.5)
            except Exception as e:
                print(f"全自動輪詢更新 {symbol} 失敗: {e}")

        await asyncio.sleep(20)


# ----------------------------------------------------------------------
# FastAPI 路由 API
# ----------------------------------------------------------------------
class StockReq(BaseModel):
    symbol: str


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_market_loop())


@app.get("/api/state")
def get_state():
    return {
        "watchlist": GLOBAL_STATE.get("watchlist", WATCHLIST),
        "states": GLOBAL_STATE.get("stock_states", {}),
        "alert_history": GLOBAL_STATE.get("alert_history", [])
    }


@app.post("/api/stocks")
def add_stock(req: StockReq, bg_tasks: BackgroundTasks):
    symbol = normalize_symbol(req.symbol)
    watchlist = GLOBAL_STATE.get("watchlist", WATCHLIST)
    if symbol not in watchlist:
        watchlist.append(symbol)
        GLOBAL_STATE["watchlist"] = watchlist
        save_state(GLOBAL_STATE)
        bg_tasks.add_task(process_single_stock, symbol)
    return {"status": "ok", "watchlist": watchlist}


@app.delete("/api/stocks/{symbol}")
def delete_stock(symbol: str):
    symbol = normalize_symbol(symbol)
    watchlist = GLOBAL_STATE.get("watchlist", WATCHLIST)
    if symbol in watchlist:
        watchlist.remove(symbol)
        GLOBAL_STATE["watchlist"] = watchlist
        if "stock_states" in GLOBAL_STATE and symbol in GLOBAL_STATE["stock_states"]:
            del GLOBAL_STATE["stock_states"][symbol]
        if "pred_audit" in GLOBAL_STATE and symbol in GLOBAL_STATE["pred_audit"]:
            del GLOBAL_STATE["pred_audit"][symbol]
        save_state(GLOBAL_STATE)
    return {"status": "ok", "watchlist": watchlist}


@app.post("/api/test-alert")
def trigger_test_alert():
    alert_history = GLOBAL_STATE.get("alert_history", [])
    time_str = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%m-%d %H:%M:%S")
    test_id = f"TEST_{datetime.now().timestamp()}"

    test_alert = {
        "id": test_id,
        "symbol": "TEST.US",
        "price": 888.88,
        "signal": "🚀 暴升警訊：[測試訊號] 布林帶爆量突破 + KDJ 底背離觸發！",
        "session": "🔔 測試模擬",
        "time": time_str
    }
    alert_history.insert(0, test_alert)
    GLOBAL_STATE["alert_history"] = alert_history[:10]
    save_state(GLOBAL_STATE)
    return {"status": "ok", "alert": test_alert}


# ----------------------------------------------------------------------
# 網頁控制台前端 (已全面支援 BSM Greeks 與壓力測試 UI)
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="zh-Hant" class="dark">
<head>
    <meta charset="UTF-8">
    <title>⏰【全自動即時行情與 AI 預測】主控台</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0b0f19; color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .card-bg { background-color: #111827; border: 1px solid #1f2937; }
        .block-bg { background-color: #1f2937; }
        @keyframes pulse-border {
            0%, 100% { border-color: rgba(245, 158, 11, 0.9); box-shadow: 0 0 15px rgba(245, 158, 11, 0.4); }
            50% { border-color: rgba(239, 68, 68, 0.9); box-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
        }
        .alert-modal-glow { animation: pulse-border 2s infinite; }
    </style>
</head>
<body class="p-4 md:p-6 relative">

    <header class="max-w-7xl mx-auto flex flex-wrap justify-between items-center pb-4 mb-6 border-b border-gray-800 gap-4">
        <div>
            <div class="flex items-center gap-2">
                <h1 class="text-xl md:text-2xl font-black text-cyan-400">⏰【全自動行情與 AI 預測】主控台</h1>
                <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    全自動即時連線中
                </span>
            </div>
            <p class="text-xs text-gray-400 mt-1">13 大 AI 機器學習模型全自動持續更新 + BSM 期權風險量化引擎 (Greeks / PoP / Stress Test)</p>
        </div>
        <div class="flex items-center gap-2">
            <input type="text" id="stockInput" placeholder="輸入代碼 (例: 3466.HK, NVDA)"
                   onkeypress="if(event.key==='Enter') addStock()"
                   class="bg-gray-900 border border-gray-700 px-3 py-1.5 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500 uppercase">
            <button onclick="addStock()" class="bg-cyan-600 hover:bg-cyan-500 text-white font-bold px-4 py-1.5 rounded-lg text-sm transition">新增標的</button>
        </div>
    </header>

    <main class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-6">

        <div class="lg:col-span-1 space-y-4">
            <div class="card-bg rounded-xl p-4 border border-amber-500/30 sticky top-4">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2 mb-3">
                    <h2 class="text-sm font-bold text-amber-400 flex items-center gap-1">
                        <span>🚨 市場即時警訊中心</span>
                    </h2>
                    <button onclick="sendTestAlert()" class="bg-amber-600/30 hover:bg-amber-600 text-amber-300 hover:text-white border border-amber-500/50 text-[11px] px-2 py-0.5 rounded font-bold transition flex items-center gap-1" title="測試警訊彈出視窗功能">
                        🧪 測試警訊
                    </button>
                </div>
                <div id="alertsContainer" class="space-y-3 max-h-[700px] overflow-y-auto pr-1">
                    <div class="text-xs text-gray-500 text-center py-4">全自動掃描市場警訊中...</div>
                </div>
            </div>
        </div>

        <div class="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-6" id="cardGrid">
            <div class="col-span-full text-center text-gray-400 py-12">正在載入全自動 AI 運算數據...</div>
        </div>

    </main>

    <div id="alertModal" class="fixed inset-0 bg-black/80 backdrop-blur-md hidden flex items-center justify-center z-50 p-4 transition-all duration-300">
        <div class="card-bg alert-modal-glow rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4 transform scale-100">
            <div class="flex justify-between items-center border-b border-gray-800 pb-3">
                <div class="flex items-center gap-2">
                    <span class="text-2xl animate-bounce">🚨</span>
                    <h3 class="text-lg font-black text-amber-400">觸發重大市場異動提醒！</h3>
                </div>
                <button onclick="closeAlertModal()" class="text-gray-400 hover:text-white font-bold text-2xl leading-none">&times;</button>
            </div>

            <div id="alertModalBody" class="space-y-3 max-h-80 overflow-y-auto pr-1"></div>

            <div class="pt-2 border-t border-gray-800">
                <button onclick="closeAlertModal()" class="w-full bg-gradient-to-r from-amber-600 to-rose-600 hover:from-amber-500 hover:to-rose-500 text-white font-black py-2.5 rounded-xl shadow-lg transition duration-200 text-sm tracking-wide">
                    我知道了 (關閉提醒)
                </button>
            </div>
        </div>
    </div>

    <script>
        let dismissedAlertIds = new Set(JSON.parse(localStorage.getItem('dismissedAlertIds') || '[]'));

        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                renderDashboard(data.watchlist, data.states, data.alert_history || []);
            } catch(e) { console.error(e); }
        }

        async function addStock() {
            const input = document.getElementById('stockInput');
            const symbol = input.value.trim();
            if(!symbol) return;
            await fetch('/api/stocks', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({symbol})
            });
            input.value = '';
            fetchState();
        }

        async function removeStock(symbol) {
            if(!confirm(`確定要從追蹤清單中移除 ${symbol} 嗎？`)) return;
            try {
                await fetch(`/api/stocks/${encodeURIComponent(symbol)}`, {
                    method: 'DELETE'
                });
                fetchState();
            } catch(e) {
                console.error(e);
            }
        }

        async function sendTestAlert() {
            try {
                await fetch('/api/test-alert', { method: 'POST' });
                fetchState();
            } catch(e) {
                console.error("發送測試警訊失敗:", e);
            }
        }

        function closeAlertModal() {
            document.getElementById('alertModal').classList.add('hidden');
        }

        function checkAndTriggerModal(alertHistory) {
            if(!alertHistory || alertHistory.length === 0) return;

            const newAlerts = alertHistory.filter(a => {
                const alertId = a.id || `${a.symbol}_${a.time}_${a.signal}`;
                return !dismissedAlertIds.has(alertId);
            });

            if(newAlerts.length > 0) {
                const modalBody = document.getElementById('alertModalBody');
                modalBody.innerHTML = newAlerts.map(a => {
                    let alertStyle = "border-amber-500/50 bg-amber-950/40 text-amber-200";
                    if(a.signal.includes("暴升")) alertStyle = "border-emerald-500/50 bg-emerald-950/40 text-emerald-200";
                    if(a.signal.includes("暴跌")) alertStyle = "border-rose-500/50 bg-rose-950/40 text-rose-200";
                    if(a.signal.includes("抄底")) alertStyle = "border-cyan-500/50 bg-cyan-950/40 text-cyan-200";

                    return `
                        <div class="border rounded-xl p-3.5 text-xs space-y-1.5 ${alertStyle}">
                            <div class="flex justify-between items-center font-bold">
                                <span class="text-base font-black text-white">${a.symbol}</span>
                                <span class="font-mono text-sm">$${a.price.toFixed(2)}</span>
                            </div>
                            <div class="font-bold text-sm leading-snug">${a.signal}</div>
                            <div class="flex justify-between items-center text-[11px] opacity-80 pt-1">
                                <span>${a.session}</span>
                                <span class="font-mono">${a.time || ''}</span>
                            </div>
                        </div>
                    `;
                }).join('');

                newAlerts.forEach(a => {
                    const alertId = a.id || `${a.symbol}_${a.time}_${a.signal}`;
                    dismissedAlertIds.add(alertId);
                });

                const idsArray = Array.from(dismissedAlertIds).slice(-50);
                localStorage.setItem('dismissedAlertIds', JSON.stringify(idsArray));

                document.getElementById('alertModal').classList.remove('hidden');
            }
        }

        function renderDashboard(watchlist, states, alertHistory) {
            const alertsContainer = document.getElementById('alertsContainer');
            const cardGrid = document.getElementById('cardGrid');

            checkAndTriggerModal(alertHistory);

            if(!watchlist || watchlist.length === 0) {
                cardGrid.innerHTML = `<div class="col-span-full text-center text-gray-500 py-10">追蹤清單為空</div>`;
                alertsContainer.innerHTML = `<div class="text-xs text-gray-500 text-center py-2">無追蹤標的</div>`;
                return;
            }

            if(!alertHistory || alertHistory.length === 0) {
                alertsContainer.innerHTML = `
                    <div class="block-bg rounded-lg p-3 text-xs text-emerald-400 border border-emerald-900/40 text-center leading-relaxed">
                        🟢 當前追蹤標的均無暴升、暴跌、抄底或逃頂異動
                    </div>
                `;
            } else {
                alertsContainer.innerHTML = alertHistory.map(a => {
                    let alertStyle = "border-amber-500/50 bg-amber-950/30 text-amber-200";
                    if(a.signal.includes("暴升")) alertStyle = "border-emerald-500/50 bg-emerald-950/30 text-emerald-200";
                    if(a.signal.includes("暴跌")) alertStyle = "border-rose-500/50 bg-rose-950/30 text-rose-200";
                    if(a.signal.includes("抄底")) alertStyle = "border-cyan-500/50 bg-cyan-950/30 text-cyan-200";

                    return `
                        <div class="border rounded-lg p-3 text-xs space-y-1.5 ${alertStyle}">
                            <div class="flex justify-between items-center font-bold">
                                <span class="text-sm font-black text-white">${a.symbol}</span>
                                <span class="font-mono">$${a.price.toFixed(2)}</span>
                            </div>
                            <div class="font-bold leading-tight">${a.signal}</div>
                            <div class="flex justify-between items-center text-[10px] opacity-75">
                                <span>${a.session}</span>
                                <span class="font-mono">${a.time || ''}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }

            cardGrid.innerHTML = watchlist.map(symbol => {
                const item = states[symbol];
                if(!item) {
                    return `
                        <div class="card-bg rounded-xl p-5 shadow-lg space-y-3 relative">
                            <div class="flex justify-between items-center">
                                <div class="font-bold text-cyan-400 text-lg">標的： ${symbol}</div>
                                <button onclick="removeStock('${symbol}')" class="text-xs bg-rose-900/50 hover:bg-rose-600 text-rose-200 px-2.5 py-1 rounded border border-rose-800 transition">
                                    🗑️ 移除
                                </button>
                            </div>
                            <div class="text-xs text-cyan-300 flex items-center gap-2">
                                <span class="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                                <span>系統正在全自動初始化並進行 13 大 AI 模型與 BSM 期權風險擬合...</span>
                            </div>
                        </div>
                    `;
                }

                const t = item.tech;
                const p = item.pred;
                const r = item.risk;
                const s = item.shock_data;
                const ev = item.upcoming_events || [];
                const opt = item.options_rec || {};

                const reasonsHtml = (item.reasons || []).map(r => `<div>${r}</div>`).join('');

                let eventsHtml = '<div class="text-gray-500 py-1">近 30 天無重大事件</div>';
                if(ev.length > 0) {
                    eventsHtml = ev.map(e => `
                        <div class="flex items-center justify-between border-b border-gray-800/80 pb-1.5 pt-0.5">
                            <div class="space-y-0.5">
                                <div class="flex items-center gap-2">
                                    <span class="px-1.5 py-0.2 rounded text-[10px] font-bold border ${e.tag_color}">${e.tag}</span>
                                    <span class="font-bold text-gray-200 text-xs">${e.title}</span>
                                </div>
                                <div class="text-[10px] text-gray-400">${e.impact}</div>
                            </div>
                            <div class="text-right pl-2 shrink-0">
                                <div class="text-xs font-bold text-amber-400 font-mono">${e.date}</div>
                                <div class="text-[10px] text-cyan-400 font-bold">倒數 ${e.days_left} 天</div>
                            </div>
                        </div>
                    `).join('');
                }

                let optionsHtml = '';
                if(opt && opt.strategy) {
                    let greeksHtml = '';
                    if(opt.greeks && Object.keys(opt.greeks).length > 0) {
                        greeksHtml = `
                            <div class="mt-2 pt-2 border-t border-purple-900/40 grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px]">
                                <div>• 30D 年化波動率: <span class="text-white font-bold">${opt.greeks.hv_30d || 'N/A'}</span></div>
                                ${opt.greeks.sigma ? `<div>• 定價波動率: <span class="text-white font-bold">${opt.greeks.sigma}</span></div>` : ''}
                                ${opt.greeks.est_premium ? `<div>• 預估權利金: <span class="text-amber-300 font-bold">${opt.greeks.est_premium}</span></div>` : ''}
                                ${opt.greeks.delta ? `<div>• Delta (Δ): <span class="text-cyan-300 font-bold">${opt.greeks.delta}</span></div>` : ''}
                                ${opt.greeks.gamma ? `<div>• Gamma (Γ): <span class="text-cyan-300 font-bold">${opt.greeks.gamma}</span></div>` : ''}
                                ${opt.greeks.theta ? `<div>• Theta (Θ): <span class="text-emerald-300 font-bold">${opt.greeks.theta}</span></div>` : ''}
                                ${opt.greeks.vega ? `<div>• Vega (ν): <span class="text-purple-300 font-bold">${opt.greeks.vega}</span></div>` : ''}
                            </div>
                        `;
                    }

                    let riskMetricsHtml = '';
                    if(opt.risk_metrics && Object.keys(opt.risk_metrics).length > 0) {
                        riskMetricsHtml = `
                            <div class="mt-2 pt-2 border-t border-purple-900/40 grid grid-cols-1 sm:grid-cols-2 gap-1 text-[11px]">
                                ${opt.risk_metrics.pop ? `<div>• 到期勝率 (PoP): <span class="text-emerald-400 font-bold">${opt.risk_metrics.pop}</span></div>` : ''}
                                ${opt.risk_metrics.max_loss ? `<div>• 最大風險損失: <span class="text-rose-400 font-bold">${opt.risk_metrics.max_loss}</span></div>` : ''}
                                ${opt.risk_metrics.max_gain ? `<div>• 最大可能收益: <span class="text-emerald-400 font-bold">${opt.risk_metrics.max_gain}</span></div>` : ''}
                                ${opt.risk_metrics.var_95 ? `<div>• 正股 95% VaR: <span class="text-amber-400 font-bold">${opt.risk_metrics.var_95}</span></div>` : ''}
                                ${opt.risk_metrics.breakeven ? `<div>• 損益平衡點: <span class="text-white font-bold">${opt.risk_metrics.breakeven}</span></div>` : ''}
                                ${opt.risk_metrics.risk_reward ? `<div>• 理論盈虧比: <span class="text-cyan-300 font-bold">${opt.risk_metrics.risk_reward}</span></div>` : ''}
                                ${opt.risk_metrics.contract && opt.risk_metrics.contract !== 'N/A' ? `<div>• 參考合約: <span class="text-white font-bold">${opt.risk_metrics.contract}</span></div>` : ''}
                            </div>
                        `;
                    }

                    let stressTestHtml = '';
                    if(opt.stress_test && opt.stress_test.length > 0) {
                        const rows = opt.stress_test.map(st => `
                            <tr class="border-b border-purple-900/30 text-center text-[10px]">
                                <td class="py-1 px-1 font-bold ${st.price_change.includes('+') ? 'text-emerald-400' : (st.price_change.includes('-') ? 'text-rose-400' : 'text-gray-300')}">${st.price_change}</td>
                                <td class="py-1 px-1 font-mono">$${st.target_price}</td>
                                <td class="py-1 px-1 font-mono">$${st.opt_price}</td>
                                <td class="py-1 px-1 font-bold ${st.pnl_pct.includes('+') ? 'text-emerald-400' : (st.pnl_pct.includes('-') ? 'text-rose-400' : 'text-gray-300')}">${st.pnl_pct}</td>
                            </tr>
                        `).join('');

                        stressTestHtml = `
                            <div class="mt-2 pt-2 border-t border-purple-900/40">
                                <div class="text-[11px] font-bold text-purple-300 mb-1.5 flex items-center gap-1">
                                    <span>⚡ 價格變動壓力測試 (PnL Simulation):</span>
                                </div>
                                <table class="w-full text-left border-collapse">
                                    <thead>
                                        <tr class="border-b border-purple-800/60 text-[10px] text-purple-300 text-center bg-purple-950/60">
                                            <th class="py-1">標的漲跌</th>
                                            <th class="py-1">預估股價</th>
                                            <th class="py-1">期權估價</th>
                                            <th class="py-1">預期 PnL</th>
                                        </tr>
                                    </thead>
                                    <tbody>${rows}</tbody>
                                </table>
                            </div>
                        `;
                    }

                    optionsHtml = `
                        <div class="space-y-1">
                            <div class="text-xs font-bold text-purple-400 flex items-center gap-1">
                                <span>🎯 AI 期權風險量化與交易建議 (BSM Engine)：</span>
                            </div>
                            <div class="bg-purple-950/40 border border-purple-900/60 rounded-lg p-3 text-xs text-purple-200 space-y-1 font-mono">
                                <div class="font-bold text-sm text-purple-300">${opt.strategy}</div>
                                <div>• 建議行使價: <span class="text-white font-bold">${opt.strike_price || '無'}</span></div>
                                <div>• 建議到期日: <span class="text-white font-bold">${opt.exp_date || '無'}</span></div>
                                <div class="text-amber-300">• 提前平倉目標: ${opt.take_profit_target || 'N/A'}</div>
                                <div class="text-[11px] text-gray-400 mt-1">• 量化說明: ${opt.reason || ''}</div>
                                ${greeksHtml}
                                ${riskMetricsHtml}
                                ${stressTestHtml}
                            </div>
                        </div>
                    `;
                }

                let modelsHtml = '<div>全自動運算中...</div>';
                if(p && p.model_list) {
                    modelsHtml = p.model_list.map(m => {
                        const w = (m.weight != null) ? ` <span class="text-cyan-400">[${m.weight}%]</span>` : '';
                        return `<div>• <b>${m.name}</b>${w}: $${Number(m.open).toFixed(2)} / $${Number(m.high).toFixed(2)} / $${Number(m.low).toFixed(2)}</div>`;
                    }).join('');
                }

                let historyHtml = (item.history_logs || []).map(h => {
                    let displayDate = h.target_date || "待下個交易日";
                    if (displayDate === "等待下個交易日") displayDate = "待下個交易日";

                    if (h.full_verified || h.status === '✅ 完全結算' || h.status === '完全結算') {
                        return `
                            <div class="mb-2 border-b border-gray-800/60 pb-1.5">
                                <div class="font-bold text-gray-200">• ${displayDate} [<span class="text-emerald-400">完全結算</span>]</div>
                                <div>開: 預 $${(h.pred_open||0).toFixed(2)} / 實 $${(h.actual_open||0).toFixed(2)} (差 $${(h.diff_open||0).toFixed(2)})</div>
                                <div>高: 預 $${(h.pred_high||0).toFixed(2)} / 實 $${(h.actual_high||0).toFixed(2)} (差 $${(h.diff_high||0).toFixed(2)})</div>
                                <div>低: 預 $${(h.pred_low||0).toFixed(2)} / 實 $${(h.actual_low||0).toFixed(2)} (差 $${(h.diff_low||0).toFixed(2)})</div>
                            </div>
                        `;
                    } else if (h.open_verified || h.status === '開盤結算') {
                        return `
                            <div class="mb-2 border-b border-gray-800/60 pb-1.5">
                                <div class="font-bold text-gray-200">• ${displayDate} [<span class="text-amber-400">開盤結算</span>]</div>
                                <div>開: 預 $${(h.pred_open||0).toFixed(2)} / 實 $${(h.actual_open||0).toFixed(2)} (差 $${(h.diff_open||0).toFixed(2)})</div>
                                <div>高: 預 $${(h.pred_high||0).toFixed(2)} | 低: 預 $${(h.pred_low||0).toFixed(2)}</div>
                            </div>
                        `;
                    } else {
                        return `
                            <div class="mb-2 border-b border-gray-800/60 pb-1.5">
                                <div class="font-bold text-gray-200">• ${displayDate} [<span class="text-cyan-400">⏳ 待結算</span>]</div>
                                <div>開: 預 $${(h.pred_open||0).toFixed(2)} | 高: 預 $${(h.pred_high||0).toFixed(2)} | 低: 預 $${(h.pred_low||0).toFixed(2)}</div>
                            </div>
                        `;
                    }
                }).reverse().join('');

                return `
                    <div class="card-bg rounded-xl p-5 shadow-2xl space-y-4 border border-gray-800">
                        <div class="border-b border-gray-800 pb-3 flex justify-between items-center">
                            <div>
                                <span class="text-xs text-gray-400 block font-bold">⏰【全自動即時行情與預測】</span>
                                <span class="text-lg font-black text-cyan-400">標的： ${symbol}</span>
                                <span class="text-xs text-gray-400 font-normal ml-1">(${item.session_text})</span>
                            </div>
                            <div class="flex items-center gap-3">
                                <div class="text-right">
                                    <span class="text-xs text-gray-400 block">當前價格 (最後更新 ${item.last_update || ''})</span>
                                    <span class="text-xl font-extrabold text-amber-400 font-mono">$${item.current_price.toFixed(2)}</span>
                                </div>
                                <button onclick="removeStock('${symbol}')" title="刪除此標的" class="bg-rose-950/60 hover:bg-rose-600 text-rose-300 hover:text-white p-1.5 rounded-lg border border-rose-800/80 transition text-xs font-bold flex items-center gap-1">
                                    🗑️ <span class="hidden sm:inline">刪除</span>
                                </button>
                            </div>
                        </div>

                        <div class="space-y-1">
                            <div class="text-xs font-bold text-amber-400 flex items-center gap-1">
                                <span>📅【未來 30 天】影響股價重大事件與業績預警：</span>
                            </div>
                            <div class="block-bg rounded-lg p-3 text-xs space-y-2 font-mono border border-amber-500/20">
                                ${eventsHtml}
                            </div>
                        </div>

                        <div class="block-bg rounded-lg p-3 text-sm border-l-4 border-emerald-500">
                            <div class="font-bold text-emerald-400 text-base mb-1">
                                💯 技術打分：${item.score} 分 👉 ${item.advice}
                            </div>
                            <div class="text-xs text-gray-300 font-bold mt-2">【打分理由】</div>
                            <div class="text-xs text-gray-300 space-y-0.5 mt-1 leading-relaxed">${reasonsHtml}</div>
                        </div>

                        ${optionsHtml}

                        <div class="space-y-1">
                            <div class="text-xs font-bold text-gray-400">📊 技術細節：</div>
                            <div class="block-bg rounded-lg p-3 text-xs text-gray-300 space-y-1 font-mono leading-relaxed">
                                <div>• RSI (14): ${t.rsi}</div>
                                <div>• KDJ: K:${t.k} / D:${t.d}</div>
                                <div>• MACD 柱狀體: ${t.macd_hist >= 0 ? '+' : ''}${t.macd_hist}</div>
                                <div>• 均線: MA5($${t.ma5.toFixed(2)}) | MA10($${t.ma10.toFixed(2)}) | MA20($${t.ma20.toFixed(2)})</div>
                                <div>• 量價結構 (成交量倍數): ${t.vol_ratio}x</div>
                                <div class="${s && s.is_high_volatile ? 'text-amber-400 font-bold' : ''}">• ⚡ 近 72h 振幅: ${s ? s.shock_72h_pct : 0}% | 近 72h 淨漲跌: ${s ? s.roc_72h_pct : 0}%</div>
                                <div>• ATR 波動: $${t.atr.toFixed(2)} | 布林帶寬: ${t.bb_width}%</div>
                            </div>
                        </div>

                        ${p ? `
                        <div class="space-y-1">
                            <div class="text-xs font-bold text-cyan-400">🔮 13 大 AI 模型預估「下一個交易日」(獨立 O/H/L 加權)：</div>
                            <div class="block-bg rounded-lg p-3 text-xs text-cyan-200 space-y-1 font-mono">
                                <div>• 預估開盤：<b>$${p.pred_open.toFixed(2)}</b> (${p.open_pct})</div>
                                <div>• 預估最高：<b>$${p.pred_high.toFixed(2)}</b></div>
                                <div>• 預估最低：<b>$${p.pred_low.toFixed(2)}</b></div>
                                <div>• 信心度：${p.confidence}</div>
                                ${p.weighted ? '<div class="text-[10px] text-gray-500">• 集成方式：歷史 MAE 反比加權（結算後自動更新）</div>' : ''}
                            </div>
                        </div>

                        <div class="space-y-1">
                            <div class="text-xs font-bold text-gray-400">🤖 13 大 AI 模型預測明細（依權重排序，共 ${p.model_list ? p.model_list.length : 13} 個）：</div>
                            <div class="block-bg rounded-lg p-3 text-[11px] text-gray-300 grid grid-cols-1 gap-1 max-h-36 overflow-y-auto font-mono">
                                ${modelsHtml}
                            </div>
                        </div>
                        ` : ''}

                        <div class="space-y-1">
                            <div class="text-xs font-bold text-indigo-400">🛡️ 動態風控建議：</div>
                            <div class="bg-indigo-950/40 border border-indigo-900/60 rounded-lg p-3 text-xs text-indigo-200 space-y-1 font-mono">
                                <div>• 建議倉位: ${r.position}</div>
                                <div>• 建議停損: $${r.stop_loss.toFixed(2)}</div>
                                <div>• 建議停利: $${r.take_profit.toFixed(2)}</div>
                                <div>• 移動停損: $${r.trailing_stop.toFixed(2)}</div>
                            </div>
                        </div>

                        <div class="space-y-1">
                            <div class="text-xs font-bold text-gray-400">📜 最近 O/H/L 預測對比明細：</div>
                            <div class="block-bg rounded-lg p-3 text-[11px] text-gray-300 font-mono leading-relaxed max-h-48 overflow-y-auto">
                                ${historyHtml || '<div>暫無歷史結算數據</div>'}
                            </div>
                        </div>

                        <div class="w-full bg-gray-900/80 text-xs py-2 rounded-lg text-emerald-400 font-mono text-center border border-emerald-900/50 flex items-center justify-center gap-2">
                            <span class="relative flex h-2 w-2">
                                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                            </span>
                            <span>🤖 AI 全自動即時計算與預測中</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        fetchState();
        setInterval(fetchState, 3000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
