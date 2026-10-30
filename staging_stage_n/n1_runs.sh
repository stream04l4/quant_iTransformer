#!/bin/bash
# N1 S1 出场消融：{39A,31A} × {fixed12h,no_sl,production} × 4 窗口 × 6 seeds = 144 路
# 三模式共享同一入场信号（块起点 gate 0.40 / 1x），只换出场机制；逐笔输出 MFE/MAE/持仓时长/退出原因
# 31A = "无优势"对照组（实验自检：三模式都应 ≈ -成本）
cd ~/quant_iTransformer || exit 1
M46L="data/train/processed/main_features_1h/main_features_46d/BTCUSDT/BTCUSDT_Master_Training_Matrix_12h_46d_L_20261030.parquet"
SIG="data/train/n1_signals"
OUT="data/train/n1_runs"
mkdir -p "$OUT"

run_one() {
  local v=$1; local w=$2; local s=$3; local mode=$4
  local ws; local we
  case $w in
    W1) ws="2024-03-21 00:00:00"; we="2024-06-19 00:00:00";;
    W2) ws="2024-09-17 00:00:00"; we="2024-12-16 00:00:00";;
    W3) ws="2025-09-12 00:00:00"; we="2025-12-11 00:00:00";;
    W4) ws="2026-07-27 00:00:00"; we="2026-10-27 12:00:00";;
  esac
  local tag="N1_${v}_${w}_${s}_${mode}"
  if [ -f "$OUT/${tag}.json" ]; then echo "skip $tag"; return 0; fi
  env N1_SIGNAL_FILE="$SIG/sig_${v}_${w}_${s}.csv" \
      N1_EXIT_MODE="$mode" \
      N1_WINDOW_START="$ws" N1_WINDOW_END="$we" \
      N1_MATRIX="$M46L" \
      N1_JSON_OUT="$OUT/${tag}.json" \
      .venv/bin/python backtest_pipeline/backtest_itransformer.py > "$OUT/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL $tag rc=$rc"; else echo "ok $tag"; fi
}
export -f run_one
export OUT SIG M46L

for v in 39A 31A; do
  for w in W1 W2 W3 W4; do
    for s in 42 1 7 123 1729 999; do
      for mode in fixed12h no_sl production; do
        echo "$v $w $s $mode"
      done
    done
  done
done | xargs -P 6 -n 4 bash -c 'run_one "$@"' _
echo "ALL_DONE_N1 $(ls $OUT/*.json 2>/dev/null | wc -l)/144"
