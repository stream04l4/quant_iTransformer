#!/bin/bash
# Stage O 重跑: O1(4) + O3(16)，含 start_equity + 逐小时净值曲线
cd ~/quant_iTransformer || exit 1
MODELS="EXP_718D47C0:0.4,EXP_220A157A:0.4,EXP_066DC380:0.2"
REGIME="$HOME/quant_iTransformer/data/train/regime/regime_series.csv"
o1() {  # $1=tag $2=w $3=st $4=en $5=suf
  local envs=(BT_TEST_START_TS="$3" BT_TEST_END_TS="$4" BT_PLOT_TAG="$1" BT_TRADES_OUT="/tmp/O1_${2}_trades.json")
  if [ -n "$5" ]; then envs+=(BT_THRESH_SUFFIX="$5"); fi
  env "${envs[@]}" timeout 500 .venv/bin/python backtest_pipeline/backtest_itransformer.py --models "$MODELS" > /tmp/O1_${2}.log 2>&1
  echo "[$(date +%H:%M:%S)] $1 done rc=$?"
}
o1 O1a W3 "2025-09-12 00:00:00" "2025-12-11 00:00:00" ""
o1 O1b W3s "2025-09-12 00:00:00" "2025-12-11 00:00:00" "_P0S_W3"
o1 O1c W4 "2026-07-27 00:00:00" "2026-10-27 12:00:00" ""
o1 O1d W4s "2026-07-27 00:00:00" "2026-10-27 12:00:00" "_P0S"
o3() {  # $1=strat $2=w $3=st $4=en
  local envs=(BT_TEST_START_TS="$3" BT_TEST_END_TS="$4" BT_PLOT_TAG="O3_${1}_${2}" BT_TRADES_OUT="/tmp/O3_${1}_${2}.json" BT_LEV_T1=1 BT_LEV_T2=1 BT_LEV_T3=1)
  case $1 in
    long)   envs+=(BT_LONG_ONLY=1);;
    short)  envs+=(BT_SHORT_ONLY=1);;
    regime) envs+=(BT_REGIME_FILE="$REGIME" BT_REGIME_FLAT=1);;
  esac
  env "${envs[@]}" timeout 500 .venv/bin/python backtest_pipeline/backtest_itransformer.py --models "$MODELS" > /tmp/O3_${1}_${2}.log 2>&1
  echo "[$(date +%H:%M:%S)] O3_${1}_${2} done rc=$?"
}
for item in "W1|2024-03-21 00:00:00|2024-06-19 00:00:00" "W2|2024-09-17 00:00:00|2024-12-16 00:00:00" "W3|2025-09-12 00:00:00|2025-12-11 00:00:00" "W4|2026-07-27 00:00:00|2026-10-27 12:00:00"; do
  IFS='|' read -r w st en <<< "$item"
  for strat in prod long short regime; do o3 "$strat" "$w" "$st" "$en"; done
done
echo "ALL_DONE_O_RERUN"
