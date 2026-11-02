#!/bin/bash
# O4 Phase B: 每窗口 top2 组合 × 3 新 seeds(1/7/123) = 24 路 (seed42 复用 Phase A 权重)
# 前置: /tmp/O4_selected.txt (每行一个 O4A tag) + /tmp/O4_combos.csv
cd ~/quant_iTransformer || exit 1
[ -f /tmp/O4_selected.txt ] || { echo "!! 缺 /tmp/O4_selected.txt" >> /tmp/o4_phaseB.log; exit 1; }

C39="is_synthetic,log_return,max_ret_24h,ma_bias,upper_shadow,lower_shadow,bb_squeeze,ppo_hist,rsi_14,rsi_48,taker_ratio_delta,amihud_illiquidity_zscore,realized_vol_1h_zscore,vol_skew_24h,oi_delta_pct,price_pos_24h,price_pos_72h,price_pos_168h,rv_term_structure,funding_rate_momentum,premium_delta,cross_asset_return,hour_sin,hour_cos,week_sin,week_cos,cvd_zscore,whale_net_flow_zscore,whale_ls_ratio_zscore,retail_ratio,cvd_oi_divergence,taker_imbalance,volume_zscore,trend_efficiency_ratio,downside_volatility,funding_rate_zscore,oi_price_divergence,volatility_of_volatility,return_24h"
WEND_W1="2024-03-20T12:00:00"; WSPL_W1="2023-12-21T00:00:00"
WEND_W2="2024-09-16T12:00:00"; WSPL_W2="2024-06-19T00:00:00"
WEND_W3="2025-09-11T12:00:00"; WSPL_W3="2025-06-14T00:00:00"
WEND_W4="2026-07-26T12:00:00"; WSPL_W4="2026-04-28T00:00:00"

ITEMS=()
: > /tmp/O4_phaseB_tags.txt
while read -r tag; do
  [ -z "$tag" ] && continue
  w=${tag:4:2}; idx=$((10#${tag:8:2}))
  for seed in 1 7 123; do
    ITEMS+=("$w $idx $seed")
    echo "O4B_${w}_c$(printf '%02d' "$idx")_${seed}" >> /tmp/O4_phaseB_tags.txt
  done
done < /tmp/O4_selected.txt
echo "[$(date +%H:%M:%S)] O4 Phase B 启动: ${#ITEMS[@]} 路" >> /tmp/o4_phaseB.log

launch() {
  local w=$1 idx=$2 seed=$3
  local el dm lr
  read -r el dm lr <<< "$(awk -F, -v i="$idx" '$1==i {print $2, $3, $4}' /tmp/O4_combos.csv)"
  local wend wspl
  case $w in W1) wend=$WEND_W1; wspl=$WSPL_W1;; W2) wend=$WEND_W2; wspl=$WSPL_W2;; W3) wend=$WEND_W3; wspl=$WSPL_W3;; W4) wend=$WEND_W4; wspl=$WSPL_W4;; esac
  local tag="O4B_${w}_c$(printf '%02d' "$idx")_${seed}"
  env OPT_SEED=$seed OPT_EPOCHS=150 OPT_CLASS_THRESHOLD=0.018 OPT_D_MODEL=$dm OPT_E_LAYERS=$el \
      OPT_DROPOUT=0.3 OPT_BATCH_SIZE=128 OPT_SEQ_LEN=672 OPT_LEARNING_RATE=$lr \
      OPT_FEATURE_SET=46 OPT_FEATURE_COLS="$C39" OPT_LABEL_COL=label_A \
      OPT_DATA_END_TS="$wend" OPT_SPLIT_TS="$wspl" \
      OPT_EXP_ID=$tag PYTHONUNBUFFERED=1 \
      setsid .venv/bin/python historical_pipeline/train_itransformer.py > /tmp/${tag}.log 2>&1 < /dev/null &
  PIDS+=($!)
}

while true; do
  n=$(ps aux | grep "[t]rain_itransformer" | wc -l)
  [ "$n" -le 1 ] && break
  sleep 120
done

i=0
while [ $i -lt ${#ITEMS[@]} ]; do
  PIDS=()
  for j in 0 1 2 3 4 5 6 7; do
    idx=$((i+j))
    [ $idx -lt ${#ITEMS[@]} ] || break
    set -- ${ITEMS[$idx]}
    launch "$1" "$2" "$3"
  done
  echo "[$(date +%H:%M:%S)] wave $((i/8+1)) 启动 (${#PIDS[@]} 路)" >> /tmp/o4_phaseB.log
  for p in "${PIDS[@]}"; do wait "$p"; done
  echo "[$(date +%H:%M:%S)] wave $((i/8+1)) 结束" >> /tmp/o4_phaseB.log
  i=$((i+8))
done
echo "[$(date +%H:%M:%S)] O4 Phase B 全部 ${#ITEMS[@]} 路训练完成" >> /tmp/o4_phaseB.log
echo "ALL_DONE_O4_PHASEB" >> /tmp/o4_phaseB.log
