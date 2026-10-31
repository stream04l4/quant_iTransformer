#!/bin/bash
# Stage O1: W3/W4 生产策略逐笔归因（原杠杆档位 2/5/8x，BT_TRADES_OUT 逐笔 JSON）
cd ~/quant_iTransformer || exit 1
MODELS="EXP_718D47C0:0.4,EXP_220A157A:0.4,EXP_066DC380:0.2"
run() {  # $1=tag $2=window $3=start $4=end $5=thresh_suffix
  local tag=$1 w=$2 st=$3 en=$4 suf=$5
  local envs=(BT_TEST_START_TS="$st" BT_TEST_END_TS="$en" BT_PLOT_TAG="$tag" BT_TRADES_OUT="/tmp/O1_${w}_trades.json")
  if [ -n "$suf" ]; then envs+=(BT_THRESH_SUFFIX="$suf"); fi
  echo "[$(date +%H:%M:%S)] $tag 启动"
  env "${envs[@]}" timeout 500 .venv/bin/python backtest_pipeline/backtest_itransformer.py --models "$MODELS" \
    > /tmp/O1_${w}.log 2>&1
  echo "[$(date +%H:%M:%S)] $tag 结束 rc=$?"
}
run O1a W3 "2025-09-12 00:00:00" "2025-12-11 00:00:00" ""
run O1b W3s "2025-09-12 00:00:00" "2025-12-11 00:00:00" "_P0S_W3"
run O1c W4 "2026-07-27 00:00:00" "2026-10-27 12:00:00" ""
run O1d W4s "2026-07-27 00:00:00" "2026-10-27 12:00:00" "_P0S"
echo "ALL_DONE_O1"
