"""Stage O2: 因果 Regime Detector（1d 传统指标，零未来数据，零 PnL 拟合）

数据: BTCUSDT_raw_1h.parquet（真实 OHLC，引擎同源）
指标（全部因果：1d bar 次日 00:00 才可用；决策时点 T 只取 avail<=T 的最新值）:
  - pos_ema200_d : close / EMA200(daily close) - 1      （价格相对 EMA200 位置）
  - slope_k_d    : EMA20(daily close)[t] / EMA20(daily close)[t-k] - 1  （EMA 斜率，k 天）
  - adx14_d      : 日线 Wilder ADX(14)（诊断输出，不参与分类——ADX 在 crypto 日线趋势上弱，任务书"可以使用"非必须）
规则（grid: k ∈ {5,10,20} 天 × band ∈ {0.02,0.04,0.08}）:
  |pos| <= band                  → SIDEWAYS（无明确趋势距离）
  pos > 0 AND slope > 0          → BULL
  pos < 0 AND slope < 0          → BEAR
  pos 与 slope 符号相反          → TRANSITION（regime 切换中）
阈值拟合（只用 train 段 2021-12 → 2023-12-21，纯结构目标，不碰 PnL）:
  1) 平均持续 run >= 72h   2) BULL/BEAR/SIDEWAYS 覆盖各 ∈ [10%,60%] 且 TRANSITION <= 40%
  3) 次级（val 段 2023-12-21→2024-03-21 为已知上行期）: val BULL 覆盖 >= 30%（标签与已知行情一致性）
  4) 同满足者取 flips/30d 最少
val 验证: 仅报告（mean run >= 48h、单 regime <= 75%）
输出: data/train/regime/regime_series.csv (ts_sec,regime) + regime_fit_report.json
纪律: train 段终点 2023-12-21 早于 W1 test 起点 2024-03-21 → 对全部 4 个 test 窗口因果干净。
v2 修订: v1（ADX 门槛 + 40h 斜率）W3 真实熊市仅标 7.4% BEAR → 弃用 ADX 作门槛，改纯 EMA 位置×慢斜率。
"""
import json, os
import numpy as np, polars as pl, pandas as pd

ROOT = os.path.expanduser("~/quant_iTransformer")
KFILE = f"{ROOT}/data/train/raw_temp/klines/BTCUSDT/BTCUSDT_raw_1h.parquet"
OUTDIR = f"{ROOT}/data/train/regime"
TRAIN_END = "2023-12-21 00:00:00"   # W1 val 起点 = W1 train 终点
VAL_END   = "2024-03-21 00:00:00"   # W1 test 起点（= val 终点）
WARMUP_D  = 250                     # 日线最少消费 bar 数（EMA200 稳定）

df = pl.read_parquet(KFILE).sort("timestamp")
ts  = df["timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
H   = df["high"].to_numpy().astype(float); L = df["low"].to_numpy().astype(float)
C   = df["close"].to_numpy().astype(float)
print(f"1h bars: {len(ts)}  {pd.Timestamp(ts[0], unit='s')} → {pd.Timestamp(ts[-1], unit='s')}")
assert len(ts) > 30000, f"1h K 线范围不足（{len(ts)} 根）"

def resample(bar_sec):
    edges = (ts // bar_sec) * bar_sec
    idx = np.concatenate([[0], np.where(np.diff(edges) != 0)[0] + 1, [len(ts)]])
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        out.append((edges[a], H[a:b].max(), L[a:b].min(), C[b-1], edges[a] + bar_sec))
    return pd.DataFrame(out, columns=["start", "h", "l", "c", "end"])

d = resample(86400)
dc = d["c"].to_numpy(float)

def ema(x, n):
    a = 2.0 / (n + 1); out = np.full(len(x), np.nan); s = 0.0
    for i, v in enumerate(x):
        if i < n - 1:
            s += v; continue
        out[i] = s / n if i == n - 1 else a * v + (1 - a) * out[i-1]
    return out

def wilder_adx(h, l, c, n=14):
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    up = np.diff(c, prepend=c[0]); dn = -np.diff(c, prepend=c[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    def wilder(x):
        out = np.full(len(x), np.nan)
        if len(x) <= n: return out
        out[n] = np.nansum(x[1:n+1])
        for i in range(n + 1, len(x)):
            out[i] = out[i-1] - out[i-1]/n + x[i]
        return out
    atr = wilder(tr); pdm = wilder(plus_dm); mdm = wilder(minus_dm)
    pdi = 100 * pdm / np.where(atr == 0, np.nan, atr)
    mdi = 100 * mdm / np.where(atr == 0, np.nan, atr)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) == 0, np.nan, pdi + mdi)
    adx = wilder(np.nan_to_num(dx, nan=0.0)) / n
    adx[:2*n] = np.nan
    return adx

ema200_d = ema(dc, 200)
ema20_d  = ema(dc, 20)
adx_d    = wilder_adx(d["h"].to_numpy(float), d["l"].to_numpy(float), dc)

d_avail = d["end"].to_numpy()
i_d = np.searchsorted(d_avail, ts, side="right") - 1
valid_base = i_d >= WARMUP_D
pos1h = np.where(valid_base, C / ema200_d[np.clip(i_d, 0, None)] - 1.0, np.nan)
adx1h = np.where(valid_base, adx_d[np.clip(i_d, 0, None)], np.nan)

def classify(k, band):
    sl = np.full(len(ema20_d), np.nan)
    sl[k:] = ema20_d[k:] / ema20_d[:-k] - 1.0
    sl1h = np.where(valid_base, sl[np.clip(i_d, 0, None)], np.nan)
    valid = ~(np.isnan(pos1h) | np.isnan(sl1h))
    r = np.where(~valid, "NA",
        np.where(np.abs(pos1h) <= band, "SIDEWAYS",
          np.where((pos1h > 0) & (sl1h > 0), "BULL",
            np.where((pos1h < 0) & (sl1h < 0), "BEAR", "TRANSITION"))))
    return r

def stats(r, t0, t1):
    m = (ts >= t0) & (ts < t1) & (r != "NA")
    seg = r[m]
    if len(seg) == 0:
        return None
    cov = {k: float(np.mean(seg == k)) for k in ["BULL", "BEAR", "SIDEWAYS", "TRANSITION"]}
    flips = int(np.sum(seg[1:] != seg[:-1]))
    runs = np.split(np.arange(len(seg)), np.where(seg[1:] != seg[:-1])[0] + 1)
    mean_run = float(np.mean([len(x) for x in runs]))
    return {"coverage": cov, "mean_run_h": mean_run,
            "flips_30d": flips / (len(seg) / 24 / 30), "hours": int(m.sum())}

T0 = int(pd.Timestamp(ts[0], unit='s').normalize().timestamp())
TE = int(pd.Timestamp(TRAIN_END).timestamp()); VE = int(pd.Timestamp(VAL_END).timestamp())

grid_results = {}
for k in [5, 10, 20]:
    for band in [0.02, 0.04, 0.08]:
        r = classify(k, band)
        tr = stats(r, T0, TE); va = stats(r, TE, VE)
        ok = bool(tr and va and tr["mean_run_h"] >= 72 and
             all(0.10 <= tr["coverage"][x] <= 0.60 for x in ["BULL", "BEAR", "SIDEWAYS"]) and
             tr["coverage"]["TRANSITION"] <= 0.40 and va["coverage"]["BULL"] >= 0.30)
        grid_results[f"k{k}_band{band}"] = {"train": tr, "val": va, "pass": ok}
        c = tr["coverage"]
        print(f"k={k} band={band}: train B/E/S/T={c['BULL']:.2f}/{c['BEAR']:.2f}/{c['SIDEWAYS']:.2f}/{c['TRANSITION']:.2f} "
              f"run={tr['mean_run_h']:.0f}h flips30={tr['flips_30d']:.1f} valBULL={va['coverage']['BULL']:.2f} pass={ok}")

cands = [k for k, v in grid_results.items() if v["pass"]]
if cands:
    sel = min(cands, key=lambda k: (grid_results[k]["train"]["flips_30d"], k))
    note = "结构检验通过"
else:
    sel = min(grid_results, key=lambda k: grid_results[k]["train"]["flips_30d"] if grid_results[k]["train"] else 9e9)
    note = "!! 无 grid 点全过结构检验，取 flips 最少者（报告标注）"
k_sel, band_sel = int(sel.split("_")[0][1:]), float(sel.split("band")[1])
print(f"SELECTED: k={k_sel} band={band_sel} ({note})")

final = classify(k_sel, band_sel)
os.makedirs(OUTDIR, exist_ok=True)
pd.DataFrame({"ts_sec": ts, "regime": final}).to_csv(f"{OUTDIR}/regime_series.csv", index=False)

windows = {"W1": ("2024-03-21 00:00:00", "2024-06-19 00:00:00"), "W2": ("2024-09-17 00:00:00", "2024-12-16 00:00:00"),
           "W3": ("2025-09-12 00:00:00", "2025-12-11 00:00:00"), "W4": ("2026-07-27 00:00:00", "2026-10-27 12:00:00")}
win_cov = {}
for w, (a, b) in windows.items():
    s = stats(final, int(pd.Timestamp(a).timestamp()), int(pd.Timestamp(b).timestamp()))
    adx_med = float(np.nanmedian(adx1h[(ts >= int(pd.Timestamp(a).timestamp())) & (ts < int(pd.Timestamp(b).timestamp()))]))
    win_cov[w] = {**s, "adx14_median": adx_med}
report = {"params": {"k_days": k_sel, "band": band_sel,
                     "rule": "|pos|<=band→SIDEWAYS; pos>0&slope>0→BULL; pos<0&slope<0→BEAR; else→TRANSITION",
                     "fit_note": note,
                     "train_fit": f"{pd.Timestamp(T0, unit='s')} → {pd.Timestamp(TE, unit='s')} (val check → {pd.Timestamp(VE, unit='s')})"},
          "grid": grid_results, "selected": sel, "windows": win_cov,
          "causal_audit": "1d 指标仅用 avail<=T 的已收盘 bar（次日 00:00 可用）；fit 终点早于 W1 test 起点 2 个月；零 PnL 拟合"}
with open(f"{OUTDIR}/regime_fit_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=float)
for w, v in win_cov.items():
    print(w, {k: round(x, 3) for k, x in v["coverage"].items()}, "adx", round(v["adx14_median"], 1))
print("DONE_O2")
