#!/usr/bin/env bash
# D7 — sradan bozmalarla kontrol deneyi (Ek-2 Rev B, Bolum 10), Turkce korpus.
# 3 kosul x 10 kayit, Large-v3. Tamamlanan hucreleri benchmark.py atlar.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE
ALL="saha_01_img_3463 saha_02_img_3492 saha_03_img_3602 saha_04_vid_20220913_142911 video_01_genel_anamnez video_02_psikiyatrik_anamnez video_03_pediatrik_anamnez video_04_simule_hasta_gorusmesi video_05_poliklinik_anamnez_fizik_muayene video_06_gogus_hastaliklari_oyku_alma"
echo "######## D7 sradan bozmalar  $(date '+%H:%M:%S')"
for C in pert_speed pert_clip pert_gain; do
  for B in $ALL; do
    echo "=== $C | $B ==="
    python paket3_asr/scripts/benchmark.py --basename "$B" --model-id faster-whisper-large-v3-int8 \
      --condition "$C" --run-purpose main_benchmark 2>&1 | grep -E "KAYDED|MUKERRER|HATA" | head -1
  done
done
echo "[BITTI] $(date '+%H:%M:%S')"
