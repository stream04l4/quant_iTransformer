#!/bin/bash
# O4 Phase A: 18 超参组合 × 4 窗口 × seed42 = 72 路严格 OOS 训练，8 路/波
# 搜索空间: E_LAYERS{2,3} × D_MODEL{64,128,256} × LR{1e-6,5e-6,1e-5}
# 协议与 P0 完全一致: 物理截断 + 12h purge, val=窗口前90天(early stop 用), test=窗口本体(只评一次)
# 泄漏纪律: 本阶段只训练, 不做任何 test 段评估; 选参用 val 段 (o4_eval.py --mode val)
cd ~/quant_iTransformer || exit 1

C39="is_synthetic,log_return,max_ret_24h,ma_bias,upper_shadow,lower_shadow,bb_squeeze,ppo_hist,rsi_14,rsi_48,taker_ratio_delta,amihud_illiquidity_zscore,realized_vol_1h_zscore,vol_skew_24h,oi_delta_pct,price_pos_24h,price_pos_72h,price_pos_168h,rv_term_structure,funding_rate_momentum,premium_delta,cross_asset_return,hour_sin,hour_cos,week_sin,week_cos,cvd_zscore,whale_net_flow_zscore,whale_ls_ratio_zscore,retail_ratio,cvd_oi_divergence,taker_imbalance,volume_zscore,trend_efficiency_ratio,downside_volatility,funding_rate_zscore,oi_price_divergence,volatility_of_volatility,return_24h"
WEND_W1="2024-03-20T12:00:00"; WSPL_W1="2023-12-21T00:00:00"
WEND_W2="2024-09-16T12:00:00"; WSPL_W2="2024-06-19T00:00:00"
WEND_W3="2025-09-11T12:00:00"; WSPL_W3="2025-06-14T00:00:00"
WEND_W4="2026-07-26T12:00:00"; WSPL_W4="2026-04-28T00:00:00"

echo "idx,el,dm,lr" > /tmp/O4_combos.csv
COMBOS=()
i=0
for el in 2 3; do for dm in 64 128 256; do for lr in 1e-6 5e-6 1e-5; do
  i=$((i+1))
  COMBOS+=("$i $el $dm $lr")
  echo "$i,$el,$dm,$lr" >> /tmp/O4_combos.csv
done; done; done

ITEMS=()
for w in W1 W2 W3 W4; do
  for c in "${COMBOS[@]}"; do ITEMS+=("$w $c"); done
done
echo "[$(date +%H:%M:%S)] O4 Phase A 启动: ${#ITEMS[@]} 路" >> /tmp/o4_phaseA.log

launch() {
  local w=$1 idx=$2 el=$3 dm=$4 lr=$5
  local wend wspl
  case $w in W1) wend=$WEND_W1; wspl=$WSPL_W1;; W2) wend=$WEND_W2; wspl=$WSPL_W2;; W3) wend=$WEND_W3; wspl=$WSPL_W3;; W4) wend=$WEND_W4; wspl=$WSPL_W4;; esac
  local tag="O4A_${w}_c$(printf '%02d' "$idx")"
  env OPT_SEED=42 OPT_EPOCHS=150 OPT_CLASS_THRESHOLD=0.018 OPT_D_MODEL=$dm OPT_E_LAYERS=$el \
      OPT_DROPOUT=0.3 OPT_BATCH_SIZE=128 OPT_SEQ_LEN=672 OPT_LEARNING_RATE=$lr \
      OPT_FEATURE_SET=46 OPT_FEATURE_COLS="$C39" OPT_LABEL_COL=label_A \
      OPT_DATA_END_TS="$wend" OPT_SPLIT_TS="$wspl" \
      OPT_EXP_ID=$tag PYTHONUNBUFFERED=1 \
      setsid .venv/bin/python historical_pipeline/train_itransformer.py > /tmp/${tag}.log 2>&1 < /dev/null &
  PIDS+=($!)
}

# GPU 保护门: 等残留训练结束
while true; do
  n=$(ps aux | grep "[t]rain_itransformer" | wc -l)
  [ "$n" -le 1 ] && break
  echo "[$(date +%H:%M:%S)] 等待中: 现存训练进程 $n" >> /tmp/o4_phaseA.log
  sleep 120
done

i=0
while [ $i -lt ${#ITEMS[@]} ]; do
  PIDS=()
  for j in 0 1 2 3 4 5 6 7; do
    idx=$((i+j))
    [ $idx -lt ${#ITEMS[@]} ] || break
    set -- ${ITEMS[$idx]}
    launch "$1" "$2" "$3" "$4" "$5"
  done
  echo "[$(date +%H:%M:%S)] wave $((i/8+1)) 启动 (${#PIDS[@]} 路)" >> /tmp/o4_phaseA.log
  for p in "${PIDS[@]}"; do wait "$p"; done
  echo "[$(date +%H:%M:%S)] wave $((i/8+1)) 结束" >> /tmp/o4_phaseA.log
  i=$((i+8))
done
echo "[$(date +%H:%M:%S)] O4 Phase A 全部 72 路训练完成" >> /tmp/o4_phaseA.log
echo "ALL_DONE_O4_PHASEA" >> /tmp/o4_phaseA.log
