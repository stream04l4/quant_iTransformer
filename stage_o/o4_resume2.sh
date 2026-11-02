#!/bin/bash
# O4 RESUME2（Phase A 72 路已完成）: val 评估(修复 B&H) → 选参(n>=8) → Phase B 24 路(修复八进制) → test 终评
cd ~/quant_iTransformer || exit 1
echo "[$(date '+%F %T')] O4 RESUME2 启动" >> /tmp/o4_all.log

.venv/bin/python /tmp/o4_eval.py --mode val --tags "AUTO:W1,W2,W3,W4" --out /tmp/O4_val_eval.json >> /tmp/o4_all.log 2>&1 || { echo "val_eval 失败" >> /tmp/o4_all.log; exit 1; }
echo "[$(date '+%F %T')] val_eval 完成" >> /tmp/o4_all.log

.venv/bin/python /tmp/o4_select.py >> /tmp/o4_all.log 2>&1 || { echo "select 失败" >> /tmp/o4_all.log; exit 1; }

bash /tmp/o4_phaseB.sh || { echo "PhaseB 失败" >> /tmp/o4_all.log; exit 1; }
echo "[$(date '+%F %T')] PhaseB 完成" >> /tmp/o4_all.log

TAGS="$(cat /tmp/O4_selected.txt | tr '\n' ',')$(cat /tmp/O4_phaseB_tags.txt | tr '\n' ',')"
echo "test 评估 tags: $(echo "$TAGS" | tr ',' ' ' | wc -w)" >> /tmp/o4_all.log
.venv/bin/python /tmp/o4_eval.py --mode test --tags "${TAGS%,}" --out /tmp/O4_test_eval.json >> /tmp/o4_all.log 2>&1 || { echo "test_eval 失败" >> /tmp/o4_all.log; exit 1; }

echo "[$(date '+%F %T')] O4 全部完成" >> /tmp/o4_all.log
echo "ALL_DONE_O4_ALL" >> /tmp/o4_all.log
