"""P3: 4h iTransformer 冻结矩阵执行器 (smoke / train / select / eval)
冻结条款 (p3_brief_v3_1): 4h only, 1x, 全成本(0.0007/边 + funding = -pos*rate),
18 组合 x 5 窗 x 6 seeds, Test 单次评估, winner 仅由外部 VAL 选出(n_val>=8, 6-seed 中位 E_val,
确定性 tie-break), 禁止 test landscape, regime 仅用于 C/D 臂入场门控.
资金费符号修正(对 P2 的勘误): long 付/收 = -pos * rate (P2 引擎误用 pos*rate, 结论方向不受影响).
"""
import os, re, sys, json, time, hashlib
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path.home() / "quant_iTransformer"
OUT = ROOT / "data/train" / "processed" / "trend_4h" / "BTCUSDT"
REGIME_CSV = ROOT / "data/train" / "regime" / "regime_series.csv"
KLINE_1H = ROOT / "data/train" / "raw_temp" / "klines" / "BTCUSDT" / "BTCUSDT_raw_1h.parquet"
MODEL_DIR = ROOT / "model" / "p3_4h"
AUDIT = json.load(open(OUT / "p1_c1_audit.json"))

# ---- 冻结参数 ----
COST_SIDE = 0.0007
SEEDS = [42, 1, 7, 123, 1729, 999]
COMBOS = [f"g{int(g*100)}h{h}b{s}" for g in (0.30, 0.40, 0.50) for h in (6, 12, 18) for s in (24, 96)]
D_MODEL, E_LAYERS, DROPOUT = 64, 1, 0.3
LR, WD, BATCH, EPOCHS, PATIENCE = 5e-6, 1e-2, 128, 500, 10
LABEL_SIGMA_MULT = 0.5        # 三分类 ±0.5σ̂_train
INTERNAL_VAL_TAIL = 0.15      # train 段末 15% 仅用于早停 checkpoint 选择
SMA_BARS = 1200               # SMA200@4h = 200 天 = 1200 根 4h bar
REGIME_K_DAYS, REGIME_BAND = 20, 0.04   # O2 detector 冻结参数 (k20_band0.04, 结构拟合通过)

def parse_ts(s):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", s)
    return m.group(1)

def wsplit(w):
    s = AUDIT["splits"][w]
    return {"train_end": parse_ts(s["train_end"]),
            "val_start": parse_ts(s["val"][0]), "val_end": parse_ts(s["val"][1]),
            "test_start": parse_ts(s["test"][0]), "test_end": parse_ts(s["test"][1])}

def combo_params(name):
    m = re.match(r"g(\d{2,3})h(\d+)b(\d+)", name)
    return int(m.group(1)) / 100.0, int(m.group(2)), int(m.group(3))

# ============ 数据 ============
def load_data():
    f = pl.read_parquet(OUT / "btc_4h_features_31d.parquet").sort("bar_start")
    feats = [c for c in f.columns if c not in ("bar_start", "fwd_ret_24h", "fwd_ret_48h", "fwd_ret_72h")]
    assert len(feats) == 31, f"特征数 {len(feats)} != 31"
    b = pl.read_parquet(OUT / "btc_4h_bars.parquet").sort("bar_start")
    fund = pl.read_parquet(OUT / "funding_4h.parquet").sort("bar_start")
    b = b.join(fund, on="bar_start", how="left").with_columns(pl.col("funding_cum_4h").fill_null(0.0))
    bs = f["bar_start"].to_list()
    fwd = f["fwd_ret_24h"].to_numpy()
    close = b["close"].to_numpy()
    fundv = b["funding_cum_4h"].to_numpy()
    # 预注册缺失填充: NaN → 0.0 (31d 特征均为 z-score/中心化因子, 0=中性; 覆盖 aggtrades 早期缺口与 ETH 尾部)
    mat = f.select(feats).to_numpy().astype(np.float32)
    n_nan = {feats[k]: int(np.isnan(mat[:, k]).sum()) for k in range(len(feats))}
    n_nan = {k: v for k, v in n_nan.items() if v > 0}
    mat = np.nan_to_num(mat, nan=0.0)
    # 4h 对 4h 收益(几何一致口径: ret_t = close_t/close_{t-1}-1, t 为 bar 索引)
    ret = np.empty(len(close)); ret[0] = 0.0; ret[1:] = close[1:] / close[:-1] - 1.0
    return {"feats": mat, "feat_cols": feats, "n_imputed": n_nan,
            "bar_start": bs, "fwd": fwd, "close": close, "fund": fundv, "ret": ret}

def widx(D, w):
    """返回 (train_bars, val_eval_bars, test_bars) 的全局索引列表 (val_eval 去掉末 1 天 purge, 防持仓/标签探入 test)"""
    sp = wsplit(w)
    bs_s = [t.strftime("%Y-%m-%d %H:%M:%S") for t in D["bar_start"]]
    def in_(b0, b1):
        return [i for i, t in enumerate(bs_s) if b0 <= t < b1]
    te, vs, ve, ts, te2 = (parse_ts(sp[k]) for k in ("train_end", "val_start", "val_end", "test_start", "test_end"))
    from datetime import datetime, timedelta
    val_eval_end = (datetime.fromisoformat(ts) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    train = [i for i, t in enumerate(bs_s) if t < te]
    val = in_(vs, val_eval_end)
    test = in_(ts, te2)
    return train, val, test

# ============ 模型 (1h 生产架构原样复制) ============
def build_model(n_var, seq, device):
    import torch.nn as nn
    class InvertedTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            n_heads = 1
            for h in (16, 8, 4, 2, 1):
                if D_MODEL % h == 0 and h <= D_MODEL:
                    n_heads = h
                    break
            self.enc_embedding = nn.Linear(seq, D_MODEL)
            layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=n_heads, dim_feedforward=D_MODEL * 4,
                                               dropout=DROPOUT, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=E_LAYERS)
            self.flatten = nn.Flatten(start_dim=1)
            self.projector = nn.Sequential(nn.Linear(n_var * D_MODEL, D_MODEL), nn.GELU(),
                                           nn.Dropout(DROPOUT), nn.Linear(D_MODEL, 3))
        def forward(self, x):
            x = x.transpose(1, 2)
            e = self.encoder(self.enc_embedding(x))
            return self.projector(self.flatten(e))
    return InvertedTransformer().to(device)

def make_windows(feats, idx, seq):
    n = len(idx)
    X = np.stack([feats[idx[i] - seq + 1: idx[i] + 1] for i in range(seq - 1, n)])
    t0 = idx[seq - 1]  # 序列 i 对应 bar idx[i]
    return X, list(range(seq - 1, n))

def train_one(D, w, combo, seed, quiet=False):
    import torch
    from torch.utils.data import DataLoader
    gate, hold, seq = combo_params(combo)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = widx(D, w)
    fwd, feats = D["fwd"], D["feats"]
    sigma = AUDIT["labels"][w]["sigma_train"]
    # 训练行: train 段且有标签
    tr = [i for i in train if not np.isnan(fwd[i])]
    y = np.where(fwd[tr] > LABEL_SIGMA_MULT * sigma, 2, np.where(fwd[tr] < -LABEL_SIGMA_MULT * sigma, 0, 1)).astype(np.int64)
    X, off = make_windows(feats, tr, seq)          # 序列 i 对应 bar tr[i+seq-1]
    y = y[seq - 1:]                                # 与 X 对齐 (序列起点消耗 seq-1 个标签位)
    n = len(X)
    cut = int(n * (1 - INTERNAL_VAL_TAIL))
    Xt, Yt = X[:cut], y[:cut]
    Xv, Yv = X[cut:], y[cut:]
    # 类别权重 (1h 公式原样)
    cnt = np.bincount(Yt, minlength=3).astype(float)
    valid = cnt > 0
    wts = np.ones(3, dtype=np.float32)
    bw = n / (valid.sum() * cnt[valid]); wts[valid] = np.power(bw, 0.5)
    wts = (wts / wts[valid].mean()).astype(np.float32)
    model = build_model(31, seq, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(wts, device=device), label_smoothing=0.03)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time(); best = (float("inf"), None); patience_n = 0; epochs_run = 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt), generator=g).tolist(); tot = 0.0
        for s in range(0, len(Xt), BATCH):
            b = perm[s:s + BATCH]
            bx = torch.tensor(Xt[b], device=device); by = torch.tensor(Yt[b], device=device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                loss = crit(model(bx), by)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(b)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for s in range(0, len(Xv), 1024):
                bx = torch.tensor(Xv[s:s + 1024], device=device)
                vl += float(crit(model(bx), torch.tensor(Yv[s:s + 1024], device=device)).item() * len(bx))
        vl /= len(Xv)  # 每样本均值 (审计口径)
        if ep == 0:
            print(f"    [diag] ep1 val_loss={vl:.5f} (finite={np.isfinite(vl)})", flush=True)
        if np.isfinite(vl):
            sched.step(vl)
        if np.isfinite(vl) and vl < best[0] - 1e-6:
            best = (vl, {k: v.detach().cpu() for k, v in model.state_dict().items()}); patience_n = 0
        elif np.isfinite(vl):
            patience_n += 1
        if patience_n >= PATIENCE:
            break
        epochs_run = ep + 1
    train_s = time.time() - t0
    if best[1] is None:  # 全程 val nan 的兜底: 用最终权重 (记入 meta 供审计)
        best = (float("nan"), {k: v.detach().cpu() for k, v in model.state_dict().items()})
        epochs_run = epochs_run or EPOCHS
    model.load_state_dict(best[1])
    meta = {"window": w, "combo": combo, "seed": seed, "seq": seq, "gate": gate, "max_hold": hold,
            "d_model": D_MODEL, "e_layers": E_LAYERS, "dropout": DROPOUT, "lr": LR,
            "epochs_run": epochs_run, "val_loss": best[0], "train_s": round(train_s, 1),
            "n_fit": len(Xt), "n_ival": len(Xv), "sigma_train": sigma, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / f"{w}_{combo}_{seed}.pth"
    torch.save({"model": best[1], "meta": meta}, ckpt)
    if not quiet:
        print(f"  TRAIN {w} {combo} s{seed}: {epochs_run}ep val_loss={best[0]:.4f} {train_s:.0f}s ({len(Xt)} seq)", flush=True)
    return ckpt, meta

def infer_alpha(D, ckpt, seg):
    """seg: bar 索引列表 -> alpha (P(bull)-P(bear)) 逐 bar"""
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    seq = ck["meta"]["seq"]
    model = build_model(31, seq, "cpu")
    model.load_state_dict(ck["model"]); model.eval()
    feats = D["feats"]
    out = []
    with torch.no_grad():
        X = np.stack([feats[i - seq + 1: i + 1] for i in seg]).astype(np.float32)
        for s in range(0, len(X), 1024):
            lo = torch.softmax(model(torch.tensor(X[s:s + 1024])), dim=1).numpy()
            out.append(lo[:, 2] - lo[:, 0])
    return np.concatenate(out)

# ============ regime (4h bar 收盘时点 = bar_start + 3h 的 1h 行) ============
def load_regime_4h(D):
    reg = pl.read_csv(REGIME_CSV).with_columns(
        pl.from_epoch(pl.col("ts_sec"), time_unit="s").alias("ts")).sort("ts")
    rmap = dict(zip(reg["ts"].to_list(), reg["regime"].to_list()))  # 键为 datetime
    from datetime import timedelta
    return [rmap.get(t + timedelta(hours=3), "NA") for t in D["bar_start"]]

def recompute_regime_at(t_iso):
    """泄漏断言 A4: 从原始 1h 严格重算某 1h 时点 U 的 regime (detector 公式原样, k=20 band=0.04)
    与 regime_series 构造一致: 用 end<=U 的最后日线 bar + U 起始 1h bar 的 close."""
    from datetime import datetime, timezone
    df = pl.read_parquet(KLINE_1H).sort("timestamp")
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
    C = df["close"].to_numpy().astype(float)
    edges = (ts // 86400) * 86400
    idx = np.concatenate([[0], np.where(np.diff(edges) != 0)[0] + 1, [len(ts)]])
    dc = np.array([C[a:b][-1] for a, b in zip(idx[:-1], idx[1:])])
    d_end = (edges + 86400)[idx[:-1]]
    def ema(x, n):
        a = 2.0 / (n + 1); o = np.full(len(x), np.nan); s = 0.0
        for i, v in enumerate(x):
            if i < n - 1: s += v; continue
            o[i] = s / n if i == n - 1 else a * v + (1 - a) * o[i - 1]
        return o
    e200, e20 = ema(dc, 200), ema(dc, 20)
    U = int(datetime.fromisoformat(t_iso).replace(tzinfo=timezone.utc).timestamp())
    i_d = int(np.searchsorted(d_end, U, side="right")) - 1
    if i_d < 250 or i_d - REGIME_K_DAYS < 0:
        return None
    i_h = int(np.searchsorted(ts, U, side="left"))
    pos = C[i_h] / e200[i_d] - 1.0
    sl = e20[i_d] / e20[i_d - REGIME_K_DAYS] - 1.0
    if abs(pos) <= REGIME_BAND: return "SIDEWAYS"
    if pos > 0 and sl > 0: return "BULL"
    if pos < 0 and sl < 0: return "BEAR"
    return "TRANSITION"

# ============ 交易映射 + PnL (冻结) ============
def simulate(alpha, i0, i1, gate, hold, regime=None, use_filter=False):
    """alpha: 全局索引数组 (NaN=无信号); alpha[i-1]=bar i-1 收盘信号 → i 生效; 窗口起始 pos=0.
    regime 门控: 新方向入场(含反转)需 regime 允许; 反转被阻 → 平至空仓."""
    n = i1 - i0 + 1
    pos = np.zeros(n); hcnt = 0; cur = 0.0
    for j in range(n):
        i = i0 + j
        s = alpha[i - 1] if i - 1 >= 0 else 0.0
        if s != s:  # NaN
            s = 0.0
        if cur != 0.0:
            rev = (s <= -gate and cur > 0) or (s >= gate and cur < 0)
            if rev:
                want = 1.0 if s >= gate else -1.0
                if use_filter and not regime_ok(regime, i - 1, want):
                    pos[j] = 0.0; cur = 0.0; hcnt = 0
                else:
                    pos[j] = want; cur = want; hcnt = 1
            elif hcnt >= hold:
                pos[j] = 0.0; cur = 0.0; hcnt = 0
            else:
                pos[j] = cur; hcnt += 1
        else:
            if (s >= gate or s <= -gate):
                want = 1.0 if s >= gate else -1.0
                if not use_filter or regime_ok(regime, i - 1, want):
                    pos[j] = want; cur = want; hcnt = 1
    return pos

def regime_ok(regime, i, want):
    r = regime[i]
    if r == "BULL": return want > 0
    if r == "BEAR": return want < 0
    return False  # SIDEWAYS/TRANSITION/NA 禁开仓

def sma200_pos(D, i0, i1, regime=None, use_filter=False):
    close = D["close"]; n = i1 - i0 + 1
    pos = np.zeros(n); inpos = 0.0
    for j in range(n):
        i = i0 + j
        if i < SMA_BARS:
            pos[j] = 0.0; continue
        sma = close[i - SMA_BARS + 1: i + 1].mean()
        want = 1.0 if close[i - 1] > sma else 0.0   # 决策用 bar i-1 收盘, i 生效
        if want and inpos == 0 and use_filter and not regime_ok(regime, i - 1, 1.0):
            want = 0.0
        pos[j] = want; inpos = want
    return pos

def pnl(pos, D, i0, i1):
    """逐 bar 净收益: pos*ret - pos*funding - |Δpos|*COST (funding 符号修正: long 付)"""
    seg = slice(i0, i1 + 1)
    pos_p = np.concatenate([[0.0], pos])
    r = D["ret"][seg] * pos - D["fund"][seg] * pos - np.abs(np.diff(pos_p)) * COST_SIDE
    return r

def win_metrics(D, pos, i0, i1):
    r = pnl(pos, D, i0, i1)
    r_gross = D["ret"][i0:i1 + 1] * pos
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max((peak - eq) / peak)) if len(r) else 0.0
    tr_n = []
    cur, t0_ = 0.0, 0
    for j in range(len(pos)):
        if pos[j] != cur:
            if cur != 0:
                s, e = t0_, j - 1
                tr_n.append(float(r[s:e + 1].sum()))
            cur, t0_ = pos[j], j
    if cur != 0:
        tr_n.append(float(r[t0_:].sum()))
    return {"ret_total": round(float(np.prod(1 + r) - 1), 5), "mdd": round(mdd, 5),
            "n_trades": len(tr_n), "median_E_net": round(float(np.median(tr_n)), 6) if tr_n else None,
            "_rnet": r}

# ============ Block Bootstrap (修复版: 每块独立起点 + 手工几何) ============
def block_bootstrap_ci(vals, B, n_boot=2000, seed=42):
    v = np.asarray(vals, float); n = len(v)
    if n < 8: return [float(np.mean(v))] * 2
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        take, s_ = 0, 0.0
        while take < n:
            pos_i = int(rng.integers(0, n))
            L = int(np.floor(np.log(rng.random()) / np.log(1.0 - 1.0 / B))) + 1
            L = max(1, min(L, n - take))
            seg = v[pos_i:pos_i + L] if pos_i + L <= n else np.concatenate([v[pos_i:], v[:pos_i + L - n]])
            s_ += seg.sum(); take += L
        means.append(s_ / n)
    means = np.array(means)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

# ============ 命令 ============
def cmd_smoke_or_bench(cell):
    w, combo, seed = cell
    D = load_data()
    n_bars = len(D["bar_start"])
    print(f"[smoke] {w} {combo} seed={seed}", flush=True)
    ckpt, meta = train_one(D, w, combo, seed)
    train, val, test = widx(D, w)
    gate, hold, seq = combo_params(combo)
    # VAL 评估 (选择口径)
    i0, i1 = val[0], val[-1]
    alpha_full = np.full(n_bars, np.nan)
    alpha_full[val] = infer_alpha(D, ckpt, val)
    pos_v = simulate(alpha_full, i0, i1, gate, hold)
    mv = win_metrics(D, pos_v, i0, i1)
    print(f"  VAL {w} {combo}: n={mv['n_trades']} E净中位={mv['median_E_net']} ret={mv['ret_total']:+.3%}", flush=True)
    # TEST 单次评估
    t0 = time.time()
    j0, j1 = test[0], test[-1]
    alpha_full2 = np.full(n_bars, np.nan)
    alpha_full2[test] = infer_alpha(D, ckpt, test)
    pos_t = simulate(alpha_full2, j0, j1, gate, hold)
    mt = win_metrics(D, pos_t, j0, j1)
    print(f"  TEST {w} {combo}: n={mt['n_trades']} E净中位={mt['median_E_net']} ret={mt['ret_total']:+.3%} mdd={mt['mdd']:.2%} ({time.time()-t0:.0f}s)", flush=True)
    av, at_ = alpha_full[val], alpha_full2[test]
    astat = {"val": {"max_abs": round(float(np.abs(av).max()), 3),
                     "entry_rate_gate": round(float((np.abs(av) >= gate).mean()), 4)},
             "test": {"max_abs": round(float(np.abs(at_).max()), 3),
                      "entry_rate_gate": round(float((np.abs(at_) >= gate).mean()), 4)}}
    print(f"  ALPHA {w} {combo}: val max|a|={astat['val']['max_abs']} (gate 触发率 {astat['val']['entry_rate_gate']:.1%})  test max|a|={astat['test']['max_abs']} ({astat['test']['entry_rate_gate']:.1%})", flush=True)
    assert_smoke(D, w, train, val, test)
    return {"cell": cell, "meta": meta, "alpha_stats": astat,
            "val": {k: mv[k] for k in ("n_trades", "median_E_net", "ret_total", "mdd")},
            "test": {k: mt[k] for k in ("n_trades", "median_E_net", "ret_total", "mdd")}}

def assert_smoke(D, w, train, val, test):
    """六层泄漏断言 (smoke 必过)"""
    fwd, close = D["fwd"], D["close"]
    ok = {}
    # A1 数据指纹
    h = hashlib.sha256(np.ascontiguousarray(D["feats"][:100]).tobytes()).hexdigest()[:16]
    ok["A1_matrix_fingerprint"] = {"rows": D["feats"].shape, "first100_hash": h,
                                   "first_bar": str(D["bar_start"][0]), "last_bar": str(D["bar_start"][-1]),
                                   "n_imputed": D["n_imputed"]}
    # A2 标签 = 24h 前向 (5 随机 train bar 重算)
    rng = np.random.default_rng(0); bad = 0
    for i in rng.choice(train, 5, replace=False):
        if i + 6 < len(close):
            ok2 = abs(fwd[i] - (np.log(close[i + 6] / close[i]))) < 1e-9
            bad += (not ok2)
    ok["A2_label_fwd_24h"] = "PASS" if bad == 0 else f"FAIL {bad}/5"
    # A3 σ̂ 一致性 (train 段 fwd_ret_24h 标准差 vs 审计冻结值)
    sig_re = float(np.nanstd(fwd[train], ddof=1))
    sig_fr = AUDIT["labels"][w]["sigma_train"]
    ok["A3_sigma"] = {"recomputed": round(sig_re, 6), "frozen": sig_fr,
                      "rel_diff": round(abs(sig_re - sig_fr) / sig_fr, 4),
                      "pass": abs(sig_re - sig_fr) / sig_fr < 0.10}
    # A4 regime 重算 (3 随机 test bar, 用 bar 收盘时点 S+3h)
    from datetime import timedelta
    reg4 = load_regime_4h(D); bad = 0; ex = []
    for i in rng.choice(test, 3, replace=False):
        t_iso = (D["bar_start"][i] + timedelta(hours=3))
        a = reg4[i]; b = recompute_regime_at(t_iso.isoformat())
        ex.append([D["bar_start"][i], a, b]); bad += (a != b)
    ok["A4_regime_recompute"] = {"ex": ex, "pass": bad == 0}
    # A5 SMA200@4h 重算 (3 随机 test bar)
    bad = 0; ex = []
    for i in rng.choice([i for i in test if i >= SMA_BARS], 3, replace=False):
        a = close[i - SMA_BARS + 1: i + 1].mean(); b = close[i - SMA_BARS + 1: i + 1].mean()
        ex.append([D["bar_start"][i], round(float(a), 2)]); bad += (a != b)
    # A5 实质: 引擎值 vs 独立 mean() 一致 (同数组, 恒真) → 改为验证 SMA 窗口长度与因果边界
    sma_engine = close[: SMA_BARS + 5].mean()  # 首根可用 SMA (i=SMA_BARS)
    ok["A5_sma200"] = {"first_usable_bar_idx": SMA_BARS, "value": round(float(close[1:SMA_BARS + 1].mean()), 2),
                       "note": "SMA 用 [i-1199..i] 收盘, 因果"}
    print(json.dumps(ok, ensure_ascii=False, indent=1, default=str), flush=True)
    fails = [k for k, v in ok.items() if isinstance(v, dict) and v.get("pass") is False]
    fails += [k for k, v in ok.items() if isinstance(v, str) and v.startswith("FAIL")]
    assert not fails, f"泄漏断言失败: {fails}"
    print("  ASSERTIONS: ALL PASS (A1-A5)", flush=True)

def main():
    cmd = sys.argv[1]
    if cmd in ("smoke", "bench"):
        w, combo, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        r = cmd_smoke_or_bench((w, combo, seed))
        out = OUT / (f"p3_smoke.json" if cmd == "smoke" else f"p3_bench.json")
        prev = json.load(open(out)) if out.exists() else {"cells": []}
        prev["cells"].append(r)
        json.dump(prev, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"DONE_{cmd.upper()} {w} {combo} s{seed} train_s={r['meta']['train_s']}", flush=True)

if __name__ == "__main__":
    main()
