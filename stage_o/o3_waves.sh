#!/bin/bash
# Stage O3: 条件化策略对比（全 1x，原生产阈值，逐笔 JSON）
# 策略: prod=原生产(1x全档) / long=纯多 / short=纯空 / regime=regime 条件化(BULL 才多, BEAR 才空, 其余空仓+强平)
cd ~/quant_iTransformer || exit 1
MODELS="EXP_718D47C0:0.4,EXP_220A157A:0.4,EXP_066DC380:0.2"
REGIME="$HOME/quant_iTransformer/data/train/regime/regime_series.csv"
run() {  # $1=strat $2=window $3=start $4=end
  local strat=$1 w=$2 st=$3 en=$4
  local envs=(BT_TEST_START_TS="$st" BT_TEST_END_TS="$en" BT_PLOT_TAG="O3_${strat}_${w}" BT_TRADES_OUT="/tmp/O3_${strat}_${w}.json" BT_LEV_T1=1 BT_LEV_T2=1 BT_LEV_T3=1)
  case $strat in
    long)   envs+=(BT_LONG_ONLY=1);;
    short)  envs+=(BT_SHORT_ONLY=1);;
    regime) envs+=(BT_REGIME_FILE="$REGIME" BT_REGIME_FLAT=1);;
  esac
  env "${envs[@]}" timeout 500 .venv/bin/python backtest_pipeline/backtest_itransformer.py --models "$MODELS" > /tmp/O3_${strat}_${w}.log 2>&1
  echo "[$(date +%H:%M:%S)] $strat $w done rc=$?"
}
ITEMS=(
  "W1|2024-03-21 00:00:00|2024-06-19 00:00:00"
  "W2|2024-09-17 00:00:00|2024-12-16 00:00:00"
  "W3|2025-09-12 00:00:00|2025-12-11 00:00:00"
  "W4|2026-07-27 00:00:00|2026-10-27 12:00:00"
)
for item in "${ITEMS[@]}"; do
  IFS='|' read -r w st en <<< "$item"
  for strat in prod long short regime; do
    run "$strat" "$w" "$st" "$en"
  done
done
echo "ALL_DONE_O3"
