#!/usr/bin/env bash
# Ingilizce taban (ham ses) — uc hucrenin de 'none' kosulu.
# Kontrol 2 icin gereken veri budur: hucreler arasi WER siralamasi mantikli mi?
#
# Ayni anda tek cozme: hepsi ayni CSV'ye yaziyor.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE
CSV=paket3_asr/outputs/metrics/benchmark_results_en.csv

echo "[ZINCIR] PriMock 'none' bitisi bekleniyor...  $(date '+%H:%M:%S')"
until [ "$(python -c "
import csv
rows=list(csv.DictReader(open('$CSV',encoding='utf-8')))
print(len([r for r in rows if r['condition']=='none' and r['cell']=='en_clean_medical']))
" 2>/dev/null)" -ge 10 ]; do sleep 60; done

for CELL in en_noisy_sdm en_clean_ihm; do
  echo "######## $CELL | none  $(date '+%H:%M:%S')"
  python paket3_asr/scripts/benchmark_en.py --condition none --cell "$CELL" 2>&1 | grep -E "kosul|OK\]|EKSIK"
done
echo "[BITTI] $(date '+%H:%M:%S')"
