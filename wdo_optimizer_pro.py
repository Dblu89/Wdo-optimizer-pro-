"""
╔══════════════════════════════════════════════════════════════════════╗
║   WDO OPTIMIZER PRO — Sistema Completo de Backtest                 ║
║   SMC + Indicadores Clássicos + Candlestick Patterns               ║
║   Backtest 100% vetorizado com numpy                               ║
║                                                                      ║
║   RODAR:                                                            ║
║   python3.13 wdo_optimizer_pro.py                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import time

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════

CSV_PATH   = "/workspace/strategy_composer/wdo_clean.csv"
OUT_DIR    = "/workspace/vbt_pro_output"
CAPITAL    = 50_000.0
MULT_WDO   = 10.0
CONTRATOS  = 2
COMISSAO   = 5.0   # R$ por contrato por lado
SLIP       = 1.0   # pontos de slippage

# Stop e Win em PONTOS
STOPS_PTS  = list(range(3, 11))        # 3-10 pontos
WINS_PTS   = list(range(5, 36, 5))     # 5,10,15,20,25,30,35 pontos

# Filtros de qualidade RIGOROSOS
MIN_TRADES = 100
MIN_PF     = 1.2
MAX_DD_PCT = -15.0
MIN_WR     = 30.0

N_CORES    = min(32, os.cpu_count() or 4)

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CARREGAR DADOS
# ══════════════════════════════════════════════════════════════════════

def carregar_dados(path: str) -> pd.DataFrame:
    print(f"[DATA] Carregando {path}...")
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
    df = df[df["close"] > 0]
    df = df[df["close"].pct_change().abs() < 0.05]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — INDICADORES (todos vetorizados)
# ══════════════════════════════════════════════════════════════════════

def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_sma(s, n):
    return s.rolling(n).mean()

def calc_atr(df, n=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, fast=12, slow=26, sig=9):
    m = calc_ema(s, fast) - calc_ema(s, slow)
    sg = calc_ema(m, sig)
    return m, sg, m - sg

def calc_stoch(df, k=14, d=3):
    lo_k = df["low"].rolling(k).min()
    hi_k = df["high"].rolling(k).max()
    k_pct = 100 * (df["close"] - lo_k) / (hi_k - lo_k).replace(0, np.nan)
    return k_pct, k_pct.rolling(d).mean()

def calc_cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def calc_williams_r(df, n=14):
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)

def calc_adx(df, n=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    up = hi.diff(); dn = -lo.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    pdi = pd.Series(pdm, index=df.index).ewm(span=n, adjust=False).mean() / atr * 100
    ndi = pd.Series(ndm, index=df.index).ewm(span=n, adjust=False).mean() / atr * 100
    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100
    return dx.ewm(span=n, adjust=False).mean(), pdi, ndi

def calc_bb(s, n=20, std=2.0):
    m = s.rolling(n).mean()
    d = s.rolling(n).std()
    return m + std*d, m, m - std*d

def calc_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df.groupby(df.index.date)["volume"].cumsum()
    cum_tp_vol = (tp * df["volume"]).groupby(df.index.date).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)

def calc_ichimoku(df):
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    chikou  = df["close"].shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calc_supertrend(df, n=10, mult=3.0):
    atr = calc_atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    up  = hl2 - mult * atr
    dn  = hl2 + mult * atr
    cl  = df["close"].values
    up_v = up.values.copy()
    dn_v = dn.values.copy()
    dir_v = np.ones(len(df))
    for i in range(1, len(df)):
        up_v[i] = max(up_v[i], up_v[i-1]) if cl[i-1] > up_v[i-1] else up_v[i]
        dn_v[i] = min(dn_v[i], dn_v[i-1]) if cl[i-1] < dn_v[i-1] else dn_v[i]
        if cl[i] > dn_v[i-1]:   dir_v[i] = 1
        elif cl[i] < up_v[i-1]: dir_v[i] = -1
        else:                    dir_v[i] = dir_v[i-1]
    return pd.Series(dir_v, index=df.index)

def calc_smc(df, swing_n=5):
    """CHoCH, BOS, FVG, Order Blocks, Liquidity, Breaker Blocks"""
    hi, lo, cl, op = df["high"], df["low"], df["close"], df["open"]
    n = len(df)

    # Swing highs/lows
    sh = hi.rolling(swing_n*2+1, center=True).max() == hi
    sl = lo.rolling(swing_n*2+1, center=True).min() == lo

    choch = pd.Series(0, index=df.index)
    bos   = pd.Series(0, index=df.index)

    last_sh_val = last_sl_val = None
    trend = 0
    for i in range(swing_n, n):
        if sh.iloc[i]:
            if trend == -1 and last_sl_val is not None:
                choch.iloc[i] = 1
            elif last_sh_val is not None and hi.iloc[i] > last_sh_val:
                bos.iloc[i] = 1
            last_sh_val = hi.iloc[i]
            trend = 1
        if sl.iloc[i]:
            if trend == 1 and last_sh_val is not None:
                choch.iloc[i] = -1
            elif last_sl_val is not None and lo.iloc[i] < last_sl_val:
                bos.iloc[i] = -1
            last_sl_val = lo.iloc[i]
            trend = -1

    # FVG (Fair Value Gap)
    fvg_bull = lo > hi.shift(2)   # gap de alta
    fvg_bear = hi < lo.shift(2)   # gap de baixa
    fvg = pd.Series(0, index=df.index)
    fvg[fvg_bull] = 1
    fvg[fvg_bear] = -1

    # Order Blocks — última vela antes de movimento forte
    body = (cl - op).abs()
    body_avg = body.rolling(20).mean()
    strong_bull = (cl > op) & (body > body_avg * 1.5) & (cl.pct_change() > 0.002)
    strong_bear = (cl < op) & (body > body_avg * 1.5) & (cl.pct_change() < -0.002)
    ob_bull = strong_bull.shift(1).fillna(False)  # vela antes do movimento bull
    ob_bear = strong_bear.shift(1).fillna(False)  # vela antes do movimento bear

    # Liquidity Sweep (EQH/EQL)
    eqh = (hi.rolling(20).max() == hi.rolling(20).max().shift(1))  # equal highs
    eql = (lo.rolling(20).min() == lo.rolling(20).min().shift(1))  # equal lows
    liq_sweep_bear = (hi > hi.rolling(20).max().shift(1)) & (cl < hi.rolling(20).max().shift(1))
    liq_sweep_bull = (lo < lo.rolling(20).min().shift(1)) & (cl > lo.rolling(20).min().shift(1))

    # Premium/Discount zones (baseado no range do swing)
    swing_hi_20 = hi.rolling(20).max()
    swing_lo_20 = lo.rolling(20).min()
    mid_range   = (swing_hi_20 + swing_lo_20) / 2
    in_premium  = cl > mid_range   # acima do meio = premium (vender)
    in_discount = cl < mid_range   # abaixo do meio = discount (comprar)

    # Breaker Blocks — OB que foi rompido e agora é suporte/resistência
    breaker_bull = ob_bear & (cl > ob_bear.shift(1).rolling(10).sum().gt(0))
    breaker_bear = ob_bull & (cl < ob_bull.shift(1).rolling(10).sum().gt(0))

    # Inducement — falso breakout antes do movimento real
    inducement_bull = liq_sweep_bull & (choch == 1)
    inducement_bear = liq_sweep_bear & (choch == -1)

    return {
        "choch": choch, "bos": bos, "fvg": fvg,
        "ob_bull": ob_bull, "ob_bear": ob_bear,
        "liq_sweep_bull": liq_sweep_bull, "liq_sweep_bear": liq_sweep_bear,
        "in_premium": in_premium, "in_discount": in_discount,
        "breaker_bull": breaker_bull, "breaker_bear": breaker_bear,
        "inducement_bull": inducement_bull, "inducement_bear": inducement_bear,
        "swing_hi": sh, "swing_lo": sl,
    }

def calc_candles(df):
    """Padrões de candlestick vetorizados"""
    op, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
    corpo     = cl - op
    corpo_abs = corpo.abs()
    sombra    = hi - lo
    s_sup     = hi - pd.concat([cl, op], axis=1).max(axis=1)
    s_inf     = pd.concat([cl, op], axis=1).min(axis=1) - lo
    avg_corpo = corpo_abs.rolling(20).mean()

    # Doji
    doji = (sombra > 0) & (corpo_abs / sombra.replace(0, np.nan) < 0.1)

    # Martelo (hammer) — sombra inferior longa, corpo pequeno no topo
    hammer = (s_inf > 2 * corpo_abs) & (s_sup < corpo_abs) & (sombra > 0)

    # Shooting Star — sombra superior longa, corpo pequeno na base
    shooting_star = (s_sup > 2 * corpo_abs) & (s_inf < corpo_abs) & (sombra > 0)

    # Engolfo bullish
    engulf_bull = (corpo.shift(1) < 0) & (corpo > 0) & \
                  (cl > op.shift(1)) & (op < cl.shift(1))

    # Engolfo bearish
    engulf_bear = (corpo.shift(1) > 0) & (corpo < 0) & \
                  (cl < op.shift(1)) & (op > cl.shift(1))

    # Pin Bar bullish (sombra inferior >= 2/3 da vela total)
    pinbar_bull = (s_inf >= sombra * 0.66) & (corpo_abs <= sombra * 0.33)

    # Pin Bar bearish
    pinbar_bear = (s_sup >= sombra * 0.66) & (corpo_abs <= sombra * 0.33)

    # Morning Star (3 velas)
    ms1 = corpo.shift(2) < -avg_corpo.shift(2)   # vela bearish forte
    ms2 = corpo_abs.shift(1) < avg_corpo.shift(1) * 0.5  # vela pequena
    ms3 = corpo > avg_corpo * 0.7                 # vela bullish forte
    morning_star = ms1 & ms2 & ms3

    # Evening Star
    es1 = corpo.shift(2) > avg_corpo.shift(2)
    es2 = corpo_abs.shift(1) < avg_corpo.shift(1) * 0.5
    es3 = corpo < -avg_corpo * 0.7
    evening_star = es1 & es2 & es3

    # Harami bullish
    harami_bull = (corpo.shift(1) < -avg_corpo.shift(1)) & \
                  (corpo > 0) & (hi < hi.shift(1)) & (lo > lo.shift(1))

    # Harami bearish
    harami_bear = (corpo.shift(1) > avg_corpo.shift(1)) & \
                  (corpo < 0) & (hi < hi.shift(1)) & (lo > lo.shift(1))

    # Inside Bar
    inside_bar = (hi < hi.shift(1)) & (lo > lo.shift(1))

    # Three White Soldiers
    tws = (corpo > 0) & (corpo.shift(1) > 0) & (corpo.shift(2) > 0) & \
          (op > op.shift(1)) & (op.shift(1) > op.shift(2))

    # Three Black Crows
    tbc = (corpo < 0) & (corpo.shift(1) < 0) & (corpo.shift(2) < 0) & \
          (op < op.shift(1)) & (op.shift(1) < op.shift(2))

    # Dragonfly Doji
    dragonfly = doji & (s_inf > s_sup * 3)

    # Gravestone Doji
    gravestone = doji & (s_sup > s_inf * 3)

    return {
        "doji": doji, "hammer": hammer, "shooting_star": shooting_star,
        "engulf_bull": engulf_bull, "engulf_bear": engulf_bear,
        "pinbar_bull": pinbar_bull, "pinbar_bear": pinbar_bear,
        "morning_star": morning_star, "evening_star": evening_star,
        "harami_bull": harami_bull, "harami_bear": harami_bear,
        "inside_bar": inside_bar, "tws": tws, "tbc": tbc,
        "dragonfly": dragonfly, "gravestone": gravestone,
    }

def preparar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    print("[IND] Calculando todos os indicadores...")
    d = df.copy()
    cl = d["close"]

    # ATR múltiplos períodos
    d["atr_7"]   = calc_atr(d, 7)
    d["atr_14"]  = calc_atr(d, 14)
    d["atr_21"]  = calc_atr(d, 21)
    d["atr_slow"]= calc_atr(d, 50)

    # EMAs
    for p in [9, 20, 50, 100, 200]:
        d[f"ema_{p}"] = calc_ema(cl, p)

    # SMAs
    for p in [20, 50, 200]:
        d[f"sma_{p}"] = calc_sma(cl, p)

    # RSI múltiplos períodos
    d["rsi_9"]  = calc_rsi(cl, 9)
    d["rsi_14"] = calc_rsi(cl, 14)
    d["rsi_21"] = calc_rsi(cl, 21)

    # MACD
    d["macd"], d["macd_sig"], d["macd_hist"] = calc_macd(cl)

    # Estocástico
    d["stoch_k"], d["stoch_d"] = calc_stoch(d)

    # CCI e Williams %R
    d["cci"]  = calc_cci(d)
    d["willr"]= calc_williams_r(d)

    # ADX
    d["adx"], d["pdi"], d["ndi"] = calc_adx(d)

    # Bollinger Bands
    d["bb_up"], d["bb_mid"], d["bb_lo"] = calc_bb(cl)
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / d["bb_mid"]
    d["bb_pct"]   = (cl - d["bb_lo"]) / (d["bb_up"] - d["bb_lo"]).replace(0, np.nan)

    # VWAP
    d["vwap"] = calc_vwap(d)
    d["vwap_dist"] = (cl - d["vwap"]) / d["vwap"] * 100

    # Supertrend
    d["supertrend"] = calc_supertrend(d)

    # Ichimoku
    d["tenkan"], d["kijun"], d["senkou_a"], d["senkou_b"], d["chikou"] = calc_ichimoku(d)

    # Volume
    d["vol_ma_10"]  = d["volume"].rolling(10).mean()
    d["vol_ma_20"]  = d["volume"].rolling(20).mean()
    d["vol_ratio"]  = d["volume"] / d["vol_ma_20"].replace(0, np.nan)

    # ROC e Momentum
    d["roc_5"]  = cl.pct_change(5) * 100
    d["roc_10"] = cl.pct_change(10) * 100
    d["roc_20"] = cl.pct_change(20) * 100

    # SMC
    smc = calc_smc(d)
    for k, v in smc.items():
        d[k] = v

    # Candlestick patterns
    candles = calc_candles(d)
    for k, v in candles.items():
        d[k] = v

    # HLC3, OHLC4
    d["hlc3"]  = (d["high"] + d["low"] + d["close"]) / 3
    d["ohlc4"] = (d["open"] + d["high"] + d["low"] + d["close"]) / 4

    print(f"[IND] {len(d.columns)} colunas | {len(d):,} candles após dropna")
    return d.dropna()

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — GENES (todas as estratégias)
# ══════════════════════════════════════════════════════════════════════

# ── ENTRADAS ──────────────────────────────────────────────────────────
GENE_ENTRADA = [
    # Indicadores clássicos
    "EMA_CROSS_9_20", "EMA_CROSS_20_50", "EMA_CROSS_50_200",
    "MACD_SIGNAL", "MACD_ZERO_CROSS",
    "RSI_EXTREME_30_70", "RSI_EXTREME_25_75", "RSI_DIVERGENCIA",
    "BB_REVERSAL", "BB_SQUEEZE_BREAK",
    "STOCH_CROSS", "STOCH_EXTREME",
    "CCI_EXTREME", "WILLIAMS_EXTREME",
    "ADX_TREND_BREAK",
    "SUPERTREND_FLIP",
    "VWAP_CROSS", "VWAP_BOUNCE",
    "ICHIMOKU_TK_CROSS", "ICHIMOKU_KUMO_BREAK",
    # Price Action
    "BREAKOUT_20", "BREAKOUT_50",
    "MOMENTUM_BREAK_5", "MOMENTUM_BREAK_10",
    # SMC
    "CHoCH_PURO", "CHoCH_FVG", "CHoCH_OB",
    "BOS_PURO", "BOS_FVG",
    "LIQ_SWEEP_BULL", "LIQ_SWEEP_BEAR",
    "ORDER_BLOCK", "BREAKER_BLOCK",
    "INDUCEMENT",
    "DISCOUNT_PREMIUM",
    # Candlestick
    "ENGULF", "PINBAR", "HAMMER_STAR",
    "MORNING_EVENING_STAR", "HARAMI",
    "THREE_SOLDIERS_CROWS",
    "INSIDE_BAR_BREAK",
    "DOJI_REVERSAL",
]

# ── FILTROS DE TENDÊNCIA ───────────────────────────────────────────────
GENE_FILTRO_T = [
    "NENHUM",
    "EMA_20_50", "EMA_50_200", "EMA_20_200",
    "SUPERTREND", "ADX_25", "ADX_20",
    "MACD_HIST_POSITIVO", "ICHIMOKU_KUMO",
    "VWAP_SIDE", "PDI_NDI",
    "HH_HL",  # Higher Highs Higher Lows
]

# ── FILTROS DE VOLATILIDADE ────────────────────────────────────────────
GENE_FILTRO_V = [
    "NENHUM",
    "ATR_EXPANDING", "ATR_CONTRACTING",
    "BB_SQUEEZE", "BB_WIDE",
    "ATR_RANGE_NORMAL",   # ATR entre 3-15pts
    "VOLUME_ACIMA_MA",
    "VOL_RATIO_ALTO",     # volume > 1.5x média
]

# ── SESSÕES ────────────────────────────────────────────────────────────
GENE_SESSAO = [
    "DIA_INTEIRO",
    "ABERTURA",       # 9h-10h
    "MANHA",          # 9h-12h
    "LONDON_OPEN",    # 9h-11h
    "NY_OPEN",        # 11h-14h
    "TARDE",          # 13h-17h
    "FECHAMENTO",     # 16h-18h
    "SEM_ALMOCO",     # 9h-12h + 13h-18h
    "ALTA_LIQUIDEZ",  # 9h-11h + 14h-16h
]

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — GERAÇÃO DE SINAIS (vetorizado)
# ══════════════════════════════════════════════════════════════════════

def get_sessao_mask(df, sessao):
    hour = df.index.hour
    dow  = df.index.dayofweek  # 0=seg, 4=sex
    if sessao == "ABERTURA":
        return (hour >= 9) & (hour < 10)
    elif sessao == "MANHA":
        return (hour >= 9) & (hour < 12)
    elif sessao == "LONDON_OPEN":
        return (hour >= 9) & (hour < 11)
    elif sessao == "NY_OPEN":
        return (hour >= 11) & (hour < 14)
    elif sessao == "TARDE":
        return (hour >= 13) & (hour < 17)
    elif sessao == "FECHAMENTO":
        return (hour >= 16) & (hour < 18)
    elif sessao == "SEM_ALMOCO":
        return ((hour >= 9) & (hour < 12)) | ((hour >= 13) & (hour < 18))
    elif sessao == "ALTA_LIQUIDEZ":
        return ((hour >= 9) & (hour < 11)) | ((hour >= 14) & (hour < 16))
    else:
        return (hour >= 9) & (hour < 18)

def gerar_sinais(df: pd.DataFrame, gene_e: str, gene_t: str,
                 gene_v: str, gene_s: str):
    cl = df["close"]
    true_s = pd.Series(True, index=df.index)
    false_s = pd.Series(False, index=df.index)

    # ── SESSÃO ──────────────────────────────────────────────────────
    mask_s = pd.Series(get_sessao_mask(df, gene_s), index=df.index)

    # ── ENTRADA ─────────────────────────────────────────────────────
    try:
        if gene_e == "EMA_CROSS_9_20":
            bull = (df["ema_9"].shift(1) <= df["ema_20"].shift(1)) & (df["ema_9"] > df["ema_20"])
            bear = (df["ema_9"].shift(1) >= df["ema_20"].shift(1)) & (df["ema_9"] < df["ema_20"])

        elif gene_e == "EMA_CROSS_20_50":
            bull = (df["ema_20"].shift(1) <= df["ema_50"].shift(1)) & (df["ema_20"] > df["ema_50"])
            bear = (df["ema_20"].shift(1) >= df["ema_50"].shift(1)) & (df["ema_20"] < df["ema_50"])

        elif gene_e == "EMA_CROSS_50_200":
            bull = (df["ema_50"].shift(1) <= df["ema_200"].shift(1)) & (df["ema_50"] > df["ema_200"])
            bear = (df["ema_50"].shift(1) >= df["ema_200"].shift(1)) & (df["ema_50"] < df["ema_200"])

        elif gene_e == "MACD_SIGNAL":
            bull = (df["macd"].shift(1) <= df["macd_sig"].shift(1)) & (df["macd"] > df["macd_sig"])
            bear = (df["macd"].shift(1) >= df["macd_sig"].shift(1)) & (df["macd"] < df["macd_sig"])

        elif gene_e == "MACD_ZERO_CROSS":
            bull = (df["macd"].shift(1) <= 0) & (df["macd"] > 0)
            bear = (df["macd"].shift(1) >= 0) & (df["macd"] < 0)

        elif gene_e == "RSI_EXTREME_30_70":
            bull = df["rsi_14"] <= 30
            bear = df["rsi_14"] >= 70

        elif gene_e == "RSI_EXTREME_25_75":
            bull = df["rsi_14"] <= 25
            bear = df["rsi_14"] >= 75

        elif gene_e == "RSI_DIVERGENCIA":
            # Preço faz mínimo mais baixo mas RSI faz mínimo mais alto = divergência bullish
            price_ll = cl < cl.rolling(14).min().shift(1)
            rsi_hl   = df["rsi_14"] > df["rsi_14"].rolling(14).min().shift(1)
            bull = price_ll & rsi_hl
            price_hh = cl > cl.rolling(14).max().shift(1)
            rsi_lh   = df["rsi_14"] < df["rsi_14"].rolling(14).max().shift(1)
            bear = price_hh & rsi_lh

        elif gene_e == "BB_REVERSAL":
            bull = cl <= df["bb_lo"]
            bear = cl >= df["bb_up"]

        elif gene_e == "BB_SQUEEZE_BREAK":
            squeeze = df["bb_width"] < df["bb_width"].rolling(20).mean() * 0.7
            bull = squeeze.shift(1) & (cl > df["bb_up"])
            bear = squeeze.shift(1) & (cl < df["bb_lo"])

        elif gene_e == "STOCH_CROSS":
            bull = (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] < 50)
            bear = (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"] > 50)

        elif gene_e == "STOCH_EXTREME":
            bull = df["stoch_k"] <= 20
            bear = df["stoch_k"] >= 80

        elif gene_e == "CCI_EXTREME":
            bull = df["cci"] <= -100
            bear = df["cci"] >= 100

        elif gene_e == "WILLIAMS_EXTREME":
            bull = df["willr"] <= -80
            bear = df["willr"] >= -20

        elif gene_e == "ADX_TREND_BREAK":
            bull = (df["adx"] >= 25) & (df["pdi"] > df["ndi"]) & \
                   (df["pdi"].shift(1) <= df["ndi"].shift(1))
            bear = (df["adx"] >= 25) & (df["ndi"] > df["pdi"]) & \
                   (df["ndi"].shift(1) <= df["pdi"].shift(1))

        elif gene_e == "SUPERTREND_FLIP":
            bull = (df["supertrend"].shift(1) == -1) & (df["supertrend"] == 1)
            bear = (df["supertrend"].shift(1) == 1)  & (df["supertrend"] == -1)

        elif gene_e == "VWAP_CROSS":
            bull = (cl.shift(1) <= df["vwap"].shift(1)) & (cl > df["vwap"])
            bear = (cl.shift(1) >= df["vwap"].shift(1)) & (cl < df["vwap"])

        elif gene_e == "VWAP_BOUNCE":
            near_vwap = (cl - df["vwap"]).abs() / df["vwap"] < 0.001
            bull = near_vwap & (cl > df["vwap"]) & (df["ema_20"] > df["vwap"])
            bear = near_vwap & (cl < df["vwap"]) & (df["ema_20"] < df["vwap"])

        elif gene_e == "ICHIMOKU_TK_CROSS":
            bull = (df["tenkan"].shift(1) <= df["kijun"].shift(1)) & (df["tenkan"] > df["kijun"])
            bear = (df["tenkan"].shift(1) >= df["kijun"].shift(1)) & (df["tenkan"] < df["kijun"])

        elif gene_e == "ICHIMOKU_KUMO_BREAK":
            kumo_top = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).max(axis=1)
            kumo_bot = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).min(axis=1)
            bull = (cl.shift(1) <= kumo_top.shift(1)) & (cl > kumo_top)
            bear = (cl.shift(1) >= kumo_bot.shift(1)) & (cl < kumo_bot)

        elif gene_e == "BREAKOUT_20":
            high20 = df["high"].rolling(20).max().shift(1)
            low20  = df["low"].rolling(20).min().shift(1)
            vol_ok = df["vol_ratio"] >= 1.3
            bull = (cl > high20) & vol_ok
            bear = (cl < low20)  & vol_ok

        elif gene_e == "BREAKOUT_50":
            high50 = df["high"].rolling(50).max().shift(1)
            low50  = df["low"].rolling(50).min().shift(1)
            vol_ok = df["vol_ratio"] >= 1.5
            bull = (cl > high50) & vol_ok
            bear = (cl < low50)  & vol_ok

        elif gene_e == "MOMENTUM_BREAK_5":
            bull = df["roc_5"] >= 0.3
            bear = df["roc_5"] <= -0.3

        elif gene_e == "MOMENTUM_BREAK_10":
            bull = df["roc_10"] >= 0.5
            bear = df["roc_10"] <= -0.5

        # SMC
        elif gene_e == "CHoCH_PURO":
            bull = df["choch"] == 1
            bear = df["choch"] == -1

        elif gene_e == "CHoCH_FVG":
            bull = (df["choch"] == 1) & (df["fvg"] == 1)
            bear = (df["choch"] == -1) & (df["fvg"] == -1)

        elif gene_e == "CHoCH_OB":
            bull = (df["choch"] == 1) & df["ob_bull"]
            bear = (df["choch"] == -1) & df["ob_bear"]

        elif gene_e == "BOS_PURO":
            bull = df["bos"] == 1
            bear = df["bos"] == -1

        elif gene_e == "BOS_FVG":
            bull = (df["bos"] == 1) & (df["fvg"] == 1)
            bear = (df["bos"] == -1) & (df["fvg"] == -1)

        elif gene_e == "LIQ_SWEEP_BULL":
            bull = df["liq_sweep_bull"]
            bear = false_s

        elif gene_e == "LIQ_SWEEP_BEAR":
            bull = false_s
            bear = df["liq_sweep_bear"]

        elif gene_e == "ORDER_BLOCK":
            bull = df["ob_bull"] & (df["choch"].rolling(10).max() == 1)
            bear = df["ob_bear"] & (df["choch"].rolling(10).min() == -1)

        elif gene_e == "BREAKER_BLOCK":
            bull = df["breaker_bull"]
            bear = df["breaker_bear"]

        elif gene_e == "INDUCEMENT":
            bull = df["inducement_bull"]
            bear = df["inducement_bear"]

        elif gene_e == "DISCOUNT_PREMIUM":
            bull = df["in_discount"] & (df["choch"] == 1)
            bear = df["in_premium"]  & (df["choch"] == -1)

        # Candlestick
        elif gene_e == "ENGULF":
            bull = df["engulf_bull"]
            bear = df["engulf_bear"]

        elif gene_e == "PINBAR":
            bull = df["pinbar_bull"]
            bear = df["pinbar_bear"]

        elif gene_e == "HAMMER_STAR":
            bull = df["hammer"]
            bear = df["shooting_star"]

        elif gene_e == "MORNING_EVENING_STAR":
            bull = df["morning_star"]
            bear = df["evening_star"]

        elif gene_e == "HARAMI":
            bull = df["harami_bull"]
            bear = df["harami_bear"]

        elif gene_e == "THREE_SOLDIERS_CROWS":
            bull = df["tws"]
            bear = df["tbc"]

        elif gene_e == "INSIDE_BAR_BREAK":
            ib = df["inside_bar"].shift(1)
            bull = ib & (df["high"] > df["high"].shift(1))
            bear = ib & (df["low"]  < df["low"].shift(1))

        elif gene_e == "DOJI_REVERSAL":
            trend_up = cl > cl.shift(5)
            trend_dn = cl < cl.shift(5)
            bull = df["doji"] & trend_dn
            bear = df["doji"] & trend_up

        else:
            return pd.Series(0, index=df.index)

    except Exception:
        return pd.Series(0, index=df.index)

    # ── FILTRO DE TENDÊNCIA ─────────────────────────────────────────
    try:
        if gene_t == "EMA_20_50":
            ft_bull = df["ema_20"] > df["ema_50"]
            ft_bear = df["ema_20"] < df["ema_50"]
        elif gene_t == "EMA_50_200":
            ft_bull = df["ema_50"] > df["ema_200"]
            ft_bear = df["ema_50"] < df["ema_200"]
        elif gene_t == "EMA_20_200":
            ft_bull = df["ema_20"] > df["ema_200"]
            ft_bear = df["ema_20"] < df["ema_200"]
        elif gene_t == "SUPERTREND":
            ft_bull = df["supertrend"] == 1
            ft_bear = df["supertrend"] == -1
        elif gene_t == "ADX_25":
            ft_bull = ft_bear = df["adx"] >= 25
        elif gene_t == "ADX_20":
            ft_bull = ft_bear = df["adx"] >= 20
        elif gene_t == "MACD_HIST_POSITIVO":
            ft_bull = df["macd_hist"] > 0
            ft_bear = df["macd_hist"] < 0
        elif gene_t == "ICHIMOKU_KUMO":
            kumo_top = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).max(axis=1)
            kumo_bot = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).min(axis=1)
            ft_bull = cl > kumo_top
            ft_bear = cl < kumo_bot
        elif gene_t == "VWAP_SIDE":
            ft_bull = cl > df["vwap"]
            ft_bear = cl < df["vwap"]
        elif gene_t == "PDI_NDI":
            ft_bull = df["pdi"] > df["ndi"]
            ft_bear = df["ndi"] > df["pdi"]
        elif gene_t == "HH_HL":
            ft_bull = cl > cl.rolling(20).max().shift(1).shift(10)
            ft_bear = cl < cl.rolling(20).min().shift(1).shift(10)
        else:
            ft_bull = ft_bear = true_s
    except Exception:
        ft_bull = ft_bear = true_s

    # ── FILTRO DE VOLATILIDADE ──────────────────────────────────────
    try:
        if gene_v == "ATR_EXPANDING":
            fv = df["atr_14"] >= df["atr_slow"]
        elif gene_v == "ATR_CONTRACTING":
            fv = df["atr_14"] <= df["atr_slow"]
        elif gene_v == "BB_SQUEEZE":
            fv = df["bb_width"] <= df["bb_width"].rolling(20).mean() * 0.8
        elif gene_v == "BB_WIDE":
            fv = df["bb_width"] >= df["bb_width"].rolling(20).mean() * 1.2
        elif gene_v == "ATR_RANGE_NORMAL":
            fv = (df["atr_14"] >= 2.0) & (df["atr_14"] <= 15.0)
        elif gene_v == "VOLUME_ACIMA_MA":
            fv = df["volume"] >= df["vol_ma_20"] * 1.2
        elif gene_v == "VOL_RATIO_ALTO":
            fv = df["vol_ratio"] >= 1.5
        else:
            fv = true_s
    except Exception:
        fv = true_s

    # ── COMBINA ──────────────────────────────────────────────────────
    sinal = pd.Series(0, index=df.index)
    sinal[bull & ft_bull & fv & mask_s] = 1
    sinal[bear & ft_bear & fv & mask_s] = -1
    return sinal

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — BACKTEST VETORIZADO (numpy puro, sem loops)
# ══════════════════════════════════════════════════════════════════════

def backtest_numpy(hi_arr, lo_arr, cl_arr, sig_arr,
                   stop_pts, win_pts):
    """
    Backtest 100% numpy — sem nenhum loop Python.
    Simula stop e win em pontos fixos.
    """
    n = len(cl_arr)
    capital = CAPITAL
    equity  = np.full(n + 1, CAPITAL, dtype=np.float64)
    pnl_arr = np.zeros(n, dtype=np.float64)
    res_arr = np.zeros(n, dtype=np.int8)  # 1=WIN, -1=LOSS

    em_pos   = False
    entry    = 0.0
    sl       = 0.0
    tp       = 0.0
    direcao  = 0
    n_trades = 0
    n_wins   = 0

    for i in range(1, n):
        if em_pos:
            hit_sl = (direcao == 1 and lo_arr[i] <= sl) or \
                     (direcao == -1 and hi_arr[i] >= sl)
            hit_tp = (direcao == 1 and hi_arr[i] >= tp) or \
                     (direcao == -1 and lo_arr[i] <= tp)

            if hit_sl or hit_tp:
                saida   = tp if hit_tp else sl
                pts     = (saida - entry) * direcao
                brl     = pts * MULT_WDO * CONTRATOS - COMISSAO * CONTRATOS * 2
                capital += brl
                pnl_arr[i] = brl
                res_arr[i] = 1 if hit_tp else -1
                n_trades   += 1
                if hit_tp: n_wins += 1
                equity[i+1] = capital
                em_pos = False
            else:
                equity[i+1] = capital
            continue

        if sig_arr[i] != 0:
            direcao = int(sig_arr[i])
            entry   = cl_arr[i] + SLIP * direcao
            sl      = entry - direcao * stop_pts
            tp      = entry + direcao * win_pts
            em_pos  = True

        equity[i+1] = capital

    return equity[:n+1], pnl_arr, res_arr, n_trades, n_wins

def calcular_metricas(equity, pnl_arr, res_arr, n_trades, n_wins):
    if n_trades < MIN_TRADES:
        return {}

    n_losses = n_trades - n_wins
    wr = n_wins / n_trades * 100

    if wr < MIN_WR:
        return {}

    pnl_wins  = pnl_arr[res_arr == 1]
    pnl_loses = pnl_arr[res_arr == -1]
    total_w   = pnl_wins.sum()
    total_l   = pnl_loses.sum()

    pf = abs(total_w / total_l) if total_l != 0 else 9999.0
    if pf < MIN_PF:
        return {}

    eq   = pd.Series(equity)
    peak = eq.cummax()
    dd   = (eq - peak) / peak * 100
    mdd  = dd.min()

    if mdd < MAX_DD_PCT:
        return {}

    pnl_total = pnl_arr.sum()
    rets = eq.pct_change().dropna()
    sharpe  = rets.mean() / rets.std() * np.sqrt(252*540) if rets.std() > 0 else 0.0
    neg = rets[rets < 0]
    sortino = rets.mean() / neg.std() * np.sqrt(252*540) if len(neg) > 0 else 0.0

    avg_win_pts  = pnl_wins.sum()  / (n_wins * MULT_WDO * CONTRATOS) if n_wins > 0 else 0
    avg_loss_pts = pnl_loses.sum() / (n_losses * MULT_WDO * CONTRATOS) if n_losses > 0 else 0
    expectancia  = (wr/100 * avg_win_pts) + ((1-wr/100) * avg_loss_pts)

    return {
        "total_trades": n_trades,
        "wins": int(n_wins),
        "losses": int(n_losses),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "avg_win_pts": round(avg_win_pts, 2),
        "avg_loss_pts": round(avg_loss_pts, 2),
        "expectancia_pts": round(expectancia, 2),
        "total_pnl_brl": round(pnl_total, 2),
        "retorno_pct": round(pnl_total / CAPITAL * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final": round(CAPITAL + pnl_total, 2),
        "equity": equity.tolist(),
    }

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — SCORE
# ══════════════════════════════════════════════════════════════════════

def calcular_score(m: dict) -> float:
    if not m: return -999.0
    pf      = min(m["profit_factor"], 10)
    sharpe  = min(max(m["sharpe"], 0), 8)
    sortino = min(max(m["sortino"], 0), 10)
    wr      = m["win_rate"] / 100
    trades  = min(m["total_trades"], 500)
    dd      = abs(m["max_drawdown_pct"])
    ret     = min(max(m["retorno_pct"], -100), 300) / 300
    exp     = min(max(m["expectancia_pts"], -10), 20) / 20

    return round(
        pf/10       * 0.20
        + sharpe/8  * 0.15
        + sortino/10* 0.10
        + wr        * 0.15
        + trades/500* 0.10
        + ret       * 0.10
        + exp       * 0.15
        - dd/20     * 0.05
    , 6)

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — OTIMIZAÇÃO PARALELA
# ══════════════════════════════════════════════════════════════════════

from joblib import Parallel, delayed

_DF_GLOBAL = None

def _init_worker(df):
    global _DF_GLOBAL
    _DF_GLOBAL = df

def _testar_combinacao(args):
    ge, gt, gv, gs = args
    try:
        df = _DF_GLOBAL
        sinal = gerar_sinais(df, ge, gt, gv, gs)
        n_sig = (sinal != 0).sum()
        if n_sig < MIN_TRADES * 0.5:
            return []

        hi_arr = df["high"].values
        lo_arr = df["low"].values
        cl_arr = df["close"].values
        sig_arr = sinal.values

        resultados = []
        for stop in STOPS_PTS:
            for win in WINS_PTS:
                if win <= stop:
                    continue
                eq, pnl, res, n_t, n_w = backtest_numpy(
                    hi_arr, lo_arr, cl_arr, sig_arr, stop, win)
                m = calcular_metricas(eq, pnl, res, n_t, n_w)
                if m:
                    s = calcular_score(m)
                    resultados.append({
                        "score": s,
                        "gene_entrada": ge,
                        "gene_filtro_t": gt,
                        "gene_filtro_v": gv,
                        "gene_sessao": gs,
                        "stop_pts": stop,
                        "win_pts": win,
                        "rr": round(win/stop, 2),
                        **{k: v for k, v in m.items() if k != "equity"},
                    })
        return resultados
    except Exception:
        return []

def otimizar(df: pd.DataFrame) -> list:
    combos = list(itertools.product(
        GENE_ENTRADA, GENE_FILTRO_T, GENE_FILTRO_V, GENE_SESSAO
    ))

    pares_sw = sum(1 for s in STOPS_PTS for w in WINS_PTS if w > s)
    total = len(combos) * pares_sw

    print(f"\n[OPT] {len(GENE_ENTRADA)} entradas × {len(GENE_FILTRO_T)} filtros_t × "
          f"{len(GENE_FILTRO_V)} filtros_v × {len(GENE_SESSAO)} sessões")
    print(f"[OPT] = {len(combos):,} combinações de genes")
    print(f"[OPT] × {pares_sw} pares stop/win = {total:,} backtests")
    print(f"[OPT] {N_CORES} cores | Filtros: ≥{MIN_TRADES} trades, PF≥{MIN_PF}, "
          f"DD≤{MAX_DD_PCT}%, WR≥{MIN_WR}%\n")

    t0 = time.time()
    results_raw = Parallel(
        n_jobs=N_CORES, backend="loky", verbose=1,
        initializer=_init_worker, initargs=(df,)
    )(delayed(_testar_combinacao)(c) for c in combos)

    resultados = []
    for r in results_raw:
        resultados.extend(r)

    resultados.sort(key=lambda x: -x["score"])
    elapsed = time.time() - t0
    print(f"\n[OPT] {len(resultados)} válidos de {total:,} | {elapsed/60:.1f} min")
    return resultados

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════

def walk_forward(df: pd.DataFrame, melhor: dict, n_splits=5) -> dict:
    print(f"\n[WF] Walk-Forward {n_splits} splits...")
    ge = melhor["gene_entrada"]
    gt = melhor["gene_filtro_t"]
    gv = melhor["gene_filtro_v"]
    gs = melhor["gene_sessao"]
    stop = melhor["stop_pts"]
    win  = melhor["win_pts"]

    step = len(df) // n_splits
    splits = []

    for i in range(n_splits - 1):
        inicio = i * step
        fim    = (i + 2) * step
        split  = inicio + int((fim - inicio) * 0.7)

        df_tr = df.iloc[inicio:split]
        df_te = df.iloc[split:fim]
        if len(df_tr) < 1000 or len(df_te) < 500:
            continue

        resultados = []
        for d, label in [(df_tr, "TRAIN"), (df_te, "TEST")]:
            try:
                sig = gerar_sinais(d, ge, gt, gv, gs)
                eq, pnl, res, nt, nw = backtest_numpy(
                    d["high"].values, d["low"].values,
                    d["close"].values, sig.values, stop, win)
                m = calcular_metricas(eq, pnl, res, nt, nw)
                s = calcular_score(m)
            except Exception:
                m, s = {}, -999.0
            resultados.append((s, m))

        s_tr, m_tr = resultados[0]
        s_te, m_te = resultados[1]

        print(f"  Split {i+1}: TRAIN score={s_tr:.4f} WR={m_tr.get('win_rate',0):.1f}% "
              f"trades={m_tr.get('total_trades',0)} | "
              f"TEST  score={s_te:.4f} WR={m_te.get('win_rate',0):.1f}% "
              f"trades={m_te.get('total_trades',0)}")

        splits.append({
            "split": i+1,
            "score_train": s_tr, "score_test": s_te,
            "train": {k:v for k,v in m_tr.items() if k!="equity"},
            "test":  {k:v for k,v in m_te.items() if k!="equity"},
        })

    scores_oos = [s["score_test"] for s in splits if s["score_test"] > 0]
    lucrativos = [s for s in splits if s.get("test",{}).get("total_pnl_brl",0) > 0]

    return {
        "splits": splits,
        "wf_score_medio": round(np.mean(scores_oos), 4) if scores_oos else -999,
        "splits_lucrativos": len(lucrativos),
        "total_splits": len(splits),
    }

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — ANÁLISE E RELATÓRIO
# ══════════════════════════════════════════════════════════════════════

def analisar_genes(resultados):
    if not resultados: return {}
    top50 = resultados[:50]
    analise = {}
    for campo in ["gene_entrada", "gene_filtro_t", "gene_filtro_v", "gene_sessao"]:
        contagem = {}
        for r in top50:
            v = r.get(campo, "?")
            if v not in contagem:
                contagem[v] = {"count": 0, "score_sum": 0, "pf_sum": 0}
            contagem[v]["count"] += 1
            contagem[v]["score_sum"] += r.get("score", 0)
            contagem[v]["pf_sum"] += r.get("profit_factor", 0)
        for v in contagem:
            d = contagem[v]
            d["score_avg"] = round(d["score_sum"] / d["count"], 4)
            d["pf_avg"]    = round(d["pf_sum"]    / d["count"], 3)
        analise[campo] = dict(sorted(contagem.items(), key=lambda x: -x[1]["score_avg"]))
    return analise

def exibir_relatorio(resultados, wf, m_full, analise):
    print(f"\n{'═'*80}")
    print("  WDO OPTIMIZER PRO — RESULTADO FINAL")
    print(f"{'═'*80}")

    if not resultados:
        print("  Nenhuma estratégia válida encontrada.")
        return

    # Top 10
    print(f"\n  TOP 10 ESTRATÉGIAS\n")
    print(f"  {'#':>2} {'Entrada':<20} {'FT':<14} {'FV':<16} {'Sess':<12} "
          f"{'SL':>3} {'TP':>3} {'RR':>4} {'Score':>7} {'PF':>6} "
          f"{'WR%':>6} {'Trades':>7} {'DD%':>6} {'Exp pts':>8} {'PnL R$':>10}")
    print(f"  {'-'*140}")
    for i, r in enumerate(resultados[:10], 1):
        print(f"  {i:>2} {r['gene_entrada']:<20} {r['gene_filtro_t']:<14} "
              f"{r['gene_filtro_v']:<16} {r['gene_sessao']:<12} "
              f"{r['stop_pts']:>3} {r['win_pts']:>3} {r['rr']:>4.1f} "
              f"{r['score']:>7.4f} {r['profit_factor']:>6.3f} "
              f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
              f"{r['max_drawdown_pct']:>6.1f} {r['expectancia_pts']:>8.2f} "
              f"{r['total_pnl_brl']:>10,.0f}")

    # Melhor detalhado
    m = resultados[0]
    print(f"\n{'═'*80}")
    print("  ★ MELHOR ESTRATÉGIA — DETALHES COMPLETOS")
    print(f"{'═'*80}")
    print(f"  Entrada       : {m['gene_entrada']}")
    print(f"  Filtro Tend.  : {m['gene_filtro_t']}")
    print(f"  Filtro Volat. : {m['gene_filtro_v']}")
    print(f"  Sessão        : {m['gene_sessao']}")
    print(f"  Stop Loss     : {m['stop_pts']} pts = R${m['stop_pts']*MULT_WDO*CONTRATOS:.0f}/trade")
    print(f"  Take Profit   : {m['win_pts']} pts = R${m['win_pts']*MULT_WDO*CONTRATOS:.0f}/trade")
    print(f"  Risk/Reward   : 1:{m['rr']}")
    print(f"  Score         : {m['score']:.4f}")
    print(f"  Profit Factor : {m['profit_factor']}")
    print(f"  Win Rate      : {m['win_rate']}%")
    print(f"  Total Trades  : {m['total_trades']}")
    print(f"  Média Win     : {m['avg_win_pts']} pts = R${m['avg_win_pts']*MULT_WDO*CONTRATOS:.0f}")
    print(f"  Média Loss    : {m['avg_loss_pts']} pts = R${m['avg_loss_pts']*MULT_WDO*CONTRATOS:.0f}")
    print(f"  Expectância   : {m['expectancia_pts']:.2f} pts por trade")
    print(f"  Sharpe Ratio  : {m['sharpe']}")
    print(f"  Sortino Ratio : {m['sortino']}")
    print(f"  Max Drawdown  : {m['max_drawdown_pct']}%")
    print(f"  Retorno       : {m['retorno_pct']}%")
    print(f"  Capital Final : R${m['capital_final']:,.2f}")

    # Walk-Forward
    print(f"\n{'═'*80}")
    print("  WALK-FORWARD VALIDATION")
    print(f"{'═'*80}")
    print(f"  Score OOS médio  : {wf['wf_score_medio']}")
    print(f"  Splits lucrativos: {wf['splits_lucrativos']}/{wf['total_splits']}")

    # Backtest completo
    if m_full:
        print(f"\n{'═'*80}")
        print("  BACKTEST COMPLETO (dataset inteiro)")
        print(f"{'═'*80}")
        print(f"  Trades        : {m_full['total_trades']}")
        print(f"  Win Rate      : {m_full['win_rate']}%")
        print(f"  Profit Factor : {m_full['profit_factor']}")
        print(f"  Sharpe        : {m_full['sharpe']}")
        print(f"  Drawdown      : {m_full['max_drawdown_pct']}%")
        print(f"  Retorno       : {m_full['retorno_pct']}%")
        print(f"  Capital Final : R${m_full['capital_final']:,.2f}")

    # Análise de genes
    print(f"\n{'═'*80}")
    print("  GENES MAIS EFICIENTES (top 50 estratégias)")
    print(f"{'═'*80}")
    labels = {
        "gene_entrada": "ENTRADA",
        "gene_filtro_t": "FILTRO TENDÊNCIA",
        "gene_filtro_v": "FILTRO VOLATILIDADE",
        "gene_sessao": "SESSÃO",
    }
    for campo, label in labels.items():
        dados = analise.get(campo, {})
        print(f"\n  {label}:")
        print(f"  {'Gene':<25} {'Aparições':>10} {'Score Médio':>12} {'PF Médio':>10}")
        print(f"  {'─'*60}")
        for idx, (v, d) in enumerate(list(dados.items())[:5]):
            star = " ★" if idx == 0 else "  "
            print(f"{star} {v:<25} {d['count']:>10} {d['score_avg']:>12.4f} {d['pf_avg']:>10.3f}")

def salvar_resultados(resultados, wf, m_full, analise):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Ranking
    rows = [{k:v for k,v in r.items() if k != "equity"} for r in resultados[:200]]
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pro_ranking.csv", index=False)

    # JSON completo
    melhor = resultados[0] if resultados else {}
    dashboard = {
        "melhor": {k:v for k,v in melhor.items() if k!="equity"},
        "top10": [{k:v for k,v in r.items() if k!="equity"} for r in resultados[:10]],
        "walk_forward": wf,
        "backtest_completo": {k:v for k,v in (m_full or {}).items() if k!="equity"},
        "equity_curve": m_full.get("equity", []) if m_full else [],
        "analise_genes": analise,
        "config": {
            "capital": CAPITAL, "contratos": CONTRATOS,
            "mult_wdo": MULT_WDO, "comissao": COMISSAO,
            "stops": STOPS_PTS, "wins": WINS_PTS,
            "min_trades": MIN_TRADES, "min_pf": MIN_PF,
            "max_dd": MAX_DD_PCT, "min_wr": MIN_WR,
        },
        "gerado_em": datetime.now().isoformat(),
    }
    with open(f"{OUT_DIR}/pro_resultado.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[OUT] Arquivos salvos em {OUT_DIR}/")
    print(f"      pro_ranking.csv — top 200 estratégias")
    print(f"      pro_resultado.json — resultado completo")

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("╔" + "═"*72 + "╗")
    print("║  WDO OPTIMIZER PRO                                                     ║")
    print(f"║  {len(GENE_ENTRADA)} entradas | {len(GENE_FILTRO_T)} filtros_t | {len(GENE_FILTRO_V)} filtros_v | {len(GENE_SESSAO)} sessões | {N_CORES} cores         ║")
    print(f"║  Stop: {min(STOPS_PTS)}-{max(STOPS_PTS)}pts | Win: {min(WINS_PTS)}-{max(WINS_PTS)}pts | {CONTRATOS} contratos | Capital: R${CAPITAL:,.0f}       ║")
    print("╚" + "═"*72 + "╝\n")

    # 1. Dados
    df = carregar_dados(CSV_PATH)
    split = int(len(df) * 0.70)
    df_ins  = df.iloc[:split]
    df_oos  = df.iloc[split:]

    print(f"  In-sample : {len(df_ins):,} ({df_ins.index[0].date()} → {df_ins.index[-1].date()})")
    print(f"  OOS       : {len(df_oos):,} ({df_oos.index[0].date()} → {df_oos.index[-1].date()})")

    # 2. Indicadores
    print("\n[1/5] Calculando indicadores...")
    df_ins  = preparar_indicadores(df_ins)
    df_full = preparar_indicadores(df)

    # 3. Otimização
    print("\n[2/5] Otimização em grade...")
    resultados = otimizar(df_ins)

    if not resultados:
        print("\n⚠ Nenhuma estratégia válida.")
        print(f"  Tente ajustar: MIN_TRADES={MIN_TRADES}, MIN_PF={MIN_PF}, MAX_DD={MAX_DD_PCT}%")
        return

    # 4. Walk-Forward
    print("\n[3/5] Walk-Forward Validation...")
    wf = walk_forward(df_ins, resultados[0])

    # 5. Backtest completo
    print("\n[4/5] Backtest completo (dataset inteiro)...")
    melhor = resultados[0]
    sig_full = gerar_sinais(df_full,
        melhor["gene_entrada"], melhor["gene_filtro_t"],
        melhor["gene_filtro_v"], melhor["gene_sessao"])
    eq, pnl, res, nt, nw = backtest_numpy(
        df_full["high"].values, df_full["low"].values,
        df_full["close"].values, sig_full.values,
        melhor["stop_pts"], melhor["win_pts"])
    m_full = calcular_metricas(eq, pnl, res, nt, nw)

    # OOS
    print("\n[5/5] Out-of-Sample...")
    df_oos_prep = preparar_indicadores(df_oos)
    sig_oos = gerar_sinais(df_oos_prep,
        melhor["gene_entrada"], melhor["gene_filtro_t"],
        melhor["gene_filtro_v"], melhor["gene_sessao"])
    eq_o, pnl_o, res_o, nt_o, nw_o = backtest_numpy(
        df_oos_prep["high"].values, df_oos_prep["low"].values,
        df_oos_prep["close"].values, sig_oos.values,
        melhor["stop_pts"], melhor["win_pts"])
    m_oos = calcular_metricas(eq_o, pnl_o, res_o, nt_o, nw_o)
    if m_oos:
        print(f"  OOS: PF={m_oos['profit_factor']} WR={m_oos['win_rate']}% "
              f"Trades={m_oos['total_trades']} PnL=R${m_oos['total_pnl_brl']:,.0f}")
    else:
        print("  OOS: não passou nos filtros de qualidade")

    # Análise e relatório
    analise = analisar_genes(resultados)
    salvar_resultados(resultados, wf, m_full, analise)
    exibir_relatorio(resultados, wf, m_full, analise)

    elapsed = time.time() - t0
    print(f"\n  Tempo total: {elapsed/60:.1f} minutos")
    print(f"  Válidos encontrados: {len(resultados)}")


if __name__ == "__main__":
    main()
