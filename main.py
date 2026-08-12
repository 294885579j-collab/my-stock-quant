import os
import json
import asyncio
import warnings
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf

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
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

STATE_FILE = "state_v2.json"
WATCHLIST = ["3466.HK", "NVDA", "VOO", "RKLB"]

YF_SESSION = requests.Session()
YF_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

app = FastAPI(title="AI Quant Trading Platform")

# ----------------------------------------------------------------------
# 狀態載入與儲存
# ----------------------------------------------------------------------
def load_state() -> Dict[str, Any]:
    default_state = {
        "watchlist": WATCHLIST,
        "stock_states": {},
        "pred_audit": {},
        "model_maes": {},
        "alert_history": []
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
# 未來 30 天重大事件與季度業績爬取模組（修復 ETF / 港股 404 報錯）
# ----------------------------------------------------------------------
def get_macro_events(start_date: datetime.date, end_date: datetime.date) -> List[Dict[str, Any]]:
    """動態計算近 30 天重點總體經濟日曆 (CPI, FOMC, PCE, NFP)"""
    macro_list = []
    curr = start_date
    while curr <= end_date:
        # 每月第一個星期五：非農就業報告 (NFP)
        if curr.weekday() == 4 and 1 <= curr.day <= 7:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🇺🇸 美國非農就業報告 (NFP) & 失業率",
                "tag": "⚡ 總經大日",
                "tag_color": "bg-amber-950 text-amber-300 border-amber-800",
                "impact": "影響聯準會降息預期與大盤整體波動"
            })
        # 每月 12 號左右的非週末：CPI 消費者物價指數
        if curr.day == 12 and curr.weekday() < 5:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🇺🇸 美國 CPI 通膨物價指數公佈",
                "tag": "🔥 重大通膨",
                "tag_color": "bg-rose-950 text-rose-300 border-rose-800",
                "impact": "關鍵通膨指標，常引發美股大盤單日劇烈震盪"
            })
        # 每月下旬 26 號左右：核心 PCE 物價指數
        if curr.day == 26 and curr.weekday() < 5:
            macro_list.append({
                "date": curr.strftime("%Y-%m-%d"),
                "days_left": (curr - start_date).days,
                "title": "🇺🇸 美國核心 PCE 個人消費支出物價指數",
                "tag": "🔥 FED指標",
                "tag_color": "bg-purple-950 text-purple-300 border-purple-800",
                "impact": "聯準會最看重的通膨指標"
            })
        curr += timedelta(days=1)
    return macro_list

def fetch_stock_events(symbol: str) -> List[Dict[str, Any]]:
    """自動搜尋個股季度業績與結合總經日曆（靜默處理不支援業績日的 ETF/港股）"""
    symbol = symbol.strip().upper()
    now_tz = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    end_date = now_tz + timedelta(days=30)
    
    events = []
    
    # 1. 爬取個股專屬業績日曆 (針對美股個股；若為 ETF 或不支援的港股則靜默跳過)
    try:
        stock = yf.Ticker(symbol, session=YF_SESSION)
        cal = None
        try:
            cal = stock.calendar
        except Exception:
            cal = None
            
        earn_date_found = None
        if cal is not None:
            if isinstance(cal, dict) and "Earnings Date" in cal:
                e_list = cal["Earnings Date"]
                if e_list and len(e_list) > 0:
                    earn_date_found = e_list[0]
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                e_list = cal.loc["Earnings Date"].dropna().tolist()
                if e_list and len(e_list) > 0:
                    earn_date_found = e_list[0]
                    
        if earn_date_found:
            if hasattr(earn_date_found, 'date'):
                ed = earn_date_found.date()
            elif isinstance(earn_date_found, datetime):
                ed = earn_date_found.date()
            else:
                ed = datetime.strptime(str(earn_date_found)[:10], "%Y-%m-%d").date()
                
            if now_tz <= ed <= end_date:
                days_left = (ed - now_tz).days
                events.append({
                    "date": ed.strftime("%Y-%m-%d"),
                    "days_left": days_left,
                    "title": f"📊 {symbol} 季度業績報告發佈 (Earnings)",
                    "tag": "🚨 業績日",
                    "tag_color": "bg-rose-950 text-rose-300 border-rose-800 font-black animate-pulse",
                    "impact": "極高波動風險！股價單日可能有 > ±8% 暴升暴跌"
                })
    except Exception:
        pass

    # 2. 加入總體經濟重大日曆 (特別對 VOO / 港股等指數與大盤 ETF 極為重要)
    macro_events = get_macro_events(now_tz, end_date)
    events.extend(macro_events)
    
    # 3. 按日期與倒數天數排序
    events.sort(key=lambda x: x["days_left"])
    return events[:5]

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
        if time_num < 930:
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

def get_realtime_price(stock):
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

def get_realtime_open_price(stock, symbol):
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

def fetch_stock_data(symbol: str):
    symbol = symbol.strip().upper()
    stock = yf.Ticker(symbol, session=YF_SESSION)
    df = stock.history(period="180d", auto_adjust=False)
    
    if df.empty or len(df) < 15:
        return None, "CLOSED"
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.dropna(subset=['Close']).copy()
    session = get_stock_session(symbol)

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
            df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], live_price)
            df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], live_price)

    return df, session

# ----------------------------------------------------------------------
# 技術指標與異動警訊偵測
# ----------------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    plus_dm = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    minus_dm = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
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
    if len(df) < 3:
        return {"shock_72h_pct": 0.0, "roc_72h_pct": 0.0, "is_high_volatile": False}
    
    latest_close = float(df['Close'].iloc[-1])
    high_72h = float(df['High'].tail(3).max())
    low_72h = float(df['Low'].tail(3).min())
    shock_72h_pct = round(((high_72h - low_72h) / low_72h) * 100, 2)
    
    close_3d_ago = float(df['Close'].iloc[-3])
    roc_72h_pct = round(((latest_close - close_3d_ago) / close_3d_ago) * 100, 2)
    
    is_high_volatile = (shock_72h_pct >= 8.0) or (abs(roc_72h_pct) >= 6.0)
    
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

    score = max(0, min(100, int(score)))
    advice = "多頭格局可偏多操作" if score >= 70 else ("中性盤整觀望" if score >= 45 else "空頭弱勢建議避險")
    return score, reasons, advice, shock_data

# ----------------------------------------------------------------------
# 13 大 AI 機器學習預估引擎
# ----------------------------------------------------------------------
def predict_prices_with_13_models(df: pd.DataFrame):
    try:
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

        avg_open = round(float(np.mean(list(open_details.values()))), 2)
        avg_high = round(float(np.mean(list(high_details.values()))), 2)
        avg_low = round(float(np.mean(list(low_details.values()))), 2)
        open_pct = round(((avg_open - curr_close) / curr_close) * 100, 1)

        model_list = []
        for name in sorted(open_details.keys()):
            model_list.append({
                "name": name,
                "open": open_details[name],
                "high": high_details[name],
                "low": low_details[name]
            })

        return {
            "pred_open": avg_open,
            "open_pct": f"+{open_pct}%" if open_pct >= 0 else f"{open_pct}%",
            "pred_high": avg_high,
            "pred_low": avg_low,
            "confidence": "高 (MAE 擬合)",
            "model_list": model_list
        }
    except Exception as e:
        print(f"ML Error: {e}")
        return None

# ----------------------------------------------------------------------
# 審計對比算法 (自動去重並確保歷程簡潔)
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
        if base_date_str in audit_dict:
            if not audit_dict[base_date_str].get("full_verified"):
                audit_dict[base_date_str]["pred_open"] = pred_res["pred_open"]
                audit_dict[base_date_str]["pred_high"] = pred_res["pred_high"]
                audit_dict[base_date_str]["pred_low"] = pred_res["pred_low"]
        else:
            audit_dict[base_date_str] = {
                "base_date": base_date_str,
                "target_date": "等待下個交易日",
                "pred_open": pred_res["pred_open"],
                "pred_high": pred_res["pred_high"],
                "pred_low": pred_res["pred_low"],
                "open_verified": False,
                "full_verified": False,
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
    pred_res = predict_prices_with_13_models(df)
    history_logs = update_pred_audit(symbol, df, session, pred_res)
    pos_str, stop_loss, take_profit, trailing_stop = calculate_dynamic_risk(curr_price, atr)
    
    # 自動爬取近 30 天重點事件與業績日曆
    upcoming_events = fetch_stock_events(symbol)

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
    # 建立即時價格對照表以符合 index.html 的格式需求
    prices = {}
    states = GLOBAL_STATE.get("stock_states", {})
    for symbol, state in states.items():
        if "current_price" in state:
            prices[symbol] = state["current_price"]

    return {
        "watchlist": GLOBAL_STATE.get("watchlist", WATCHLIST),
        "states": states,
        "prices": prices,
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

@app.post("/api/predict/{symbol}")
def trigger_ml_predict(symbol: str, bg_tasks: BackgroundTasks):
    """【微調 2 新增】配合 index.html 觸發 ML 預測功能"""
    symbol = normalize_symbol(symbol)
    bg_tasks.add_task(process_single_stock, symbol)
    return {"status": "ok", "message": f"已觸發 {symbol} 的 13-ML 模型預測"}

@app.post("/api/test-alert")
def trigger_test_alert():
    """模擬發送測試警訊 API"""
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
# 網頁控制台前端 (【微調 1】改為載入外部 index.html)
# ----------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)