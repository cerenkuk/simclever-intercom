#!/usr/bin/env bash
# Ingilizce Turbo — KENDI ONGORUMUZU sinayan deney.
#
# ONGORU (Turkce veriden, Ingilizce sonuc gorulmeden):
#   Birlestirme kazanci taban hatayla olcekleniyordu:
#       Large-v3 %28,4 -> -0,40    Medium %34,7 -> -1,61    Turbo %34,8 -> -1,99
#   Ingilizce Large-v3 cok guclu (%16,8) ve orada kazanc YOK.
#   Oyleyse Ingilizcede ZAYIF bir modelde kazanc CIKMALI.
#
# Cikarsa: yasa dogrulanir, pozitif sonuc ongorulen kosul altinda iki dile cikar.
# Cikmazsa: sonuc Turkce'ye ozgudur ve bunu kesin biliriz.
# Iki durumda da raporlanir.
#
# Kosullar: taban + donmus uc uye (H2 icin gerekli olanlar).
# Yapilandirma yeniden ayarlanmaz.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE
CSV=paket3_asr/outputs/metrics/benchmark_results_en.csv

echo "[ZINCIR] pcen bitisi bekleniyor...  $(date '+%H:%M:%S')"
until [ "$(python -c "
import csv
rows=list(csv.DictReader(open('$CSV',encoding='utf-8')))
print(len([r for r in rows if r['condition']=='pcen']))
" 2>/dev/null)" -ge 30 ]; do sleep 120; done

echo "######## INGILIZCE TURBO  $(date '+%Y-%m-%d %H:%M:%S')"
for C in none oa_aggr_a020 sa_psm_aggr_a040 sa_psm_mild_a025; do
  echo "=== turbo | $C  $(date '+%H:%M:%S') ==="
  python paket3_asr/scripts/benchmark_en.py --condition "$C" \
    --model-id faster-whisper-large-v3-turbo-int8 --model-name large-v3-turbo \
    2>&1 | grep -E "kosul|OK\]|EKSIK|Error"
done
echo "[BITTI] $(date '+%Y-%m-%d %H:%M:%S')"
