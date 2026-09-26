#!/usr/bin/env bash
# Medium bitince Turbo'da pcen — 5 uyeli yapilandirmanin CAPRAZ MODEL sinamasi.
#
# Gerekce: 4 uye ile 5 uye arasindaki fark tek modelde anlamsiz (-0.153 pp,
# %95 GA sifiri kapsiyor) ve guc hesabina gore tek modelde anlamli cikmasi icin
# ~50 kayit gerekir. Ayni karsilastirma uc modelde tekrarlanip ucunde de ayni
# yone cikarsa, KORPUS BUYUTMEDEN elde edilebilecek en guclu ek kanit budur.
#
# Sirali kalmasi icin Medium'un 30/30 olmasi beklenir; ayni anda iki cozme
# CSV'ye escomanli yazma riski dogurur.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE

echo "[ZINCIR] Medium 30/30 bekleniyor...  $(date '+%H:%M:%S')"
until [ "$(python -c "
import csv
rows=list(csv.DictReader(open('paket3_asr/outputs/metrics/benchmark_results.csv',encoding='utf-8')))
print(len([r for r in rows if r['model_id']=='faster-whisper-medium-int8'
           and r['condition'] in ('oa_aggr_a020','sa_psm_aggr_a040','sa_psm_mild_a025')]))
" 2>/dev/null)" -ge 30 ]; do sleep 120; done

echo "######## Turbo | pcen  $(date '+%H:%M:%S')"
for B in saha_01_img_3463 saha_02_img_3492 saha_03_img_3602 saha_04_vid_20220913_142911 video_01_genel_anamnez video_02_psikiyatrik_anamnez video_03_pediatrik_anamnez video_04_simule_hasta_gorusmesi video_05_poliklinik_anamnez_fizik_muayene video_06_gogus_hastaliklari_oyku_alma; do
  echo "=== turbo | pcen | $B ==="
  python paket3_asr/scripts/benchmark.py --basename "$B" --model-id faster-whisper-large-v3-turbo-int8 \
    --condition pcen --run-purpose main_benchmark 2>&1 | grep -E "KAYDED|MUKERRER|HATA" | head -1
done
echo "[BITTI] $(date '+%H:%M:%S')"
