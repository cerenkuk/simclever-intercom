#!/usr/bin/env bash
# Medium'da pcen — 4 uye vs 5 uye karsilastirmasinin UCUNCU noktasi.
#
# Iki modelde (Large-v3, Turbo) havuzlanmis fark -0.118 pp, %95 GA [-0.227, -0.018]:
# sifiri kil payi disliyor. Elde ucuncu bir model varken iki modelle yetinip
# "anlamli" demek, testi anlamli oldugu yerde durdurmak olur. Medium eklenince
# sonuc ya guclenir ya duser; her iki durumda da raporlanacak olan budur.
#
# pcen sesleri tam 10 kayit icin zaten uretilmis; yalnizca cozme gerekiyor.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE

echo "######## Medium | pcen  $(date '+%H:%M:%S')"
for B in saha_01_img_3463 saha_02_img_3492 saha_03_img_3602 saha_04_vid_20220913_142911 video_01_genel_anamnez video_02_psikiyatrik_anamnez video_03_pediatrik_anamnez video_04_simule_hasta_gorusmesi video_05_poliklinik_anamnez_fizik_muayene video_06_gogus_hastaliklari_oyku_alma; do
  echo "=== medium | pcen | $B ==="
  python paket3_asr/scripts/benchmark.py --basename "$B" --model-id faster-whisper-medium-int8 \
    --condition pcen --run-purpose main_benchmark 2>&1 | grep -E "KAYDED|MUKERRER|HATA" | head -1
done
echo "[BITTI] $(date '+%H:%M:%S')"
