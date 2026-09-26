#!/usr/bin/env bash
# Kaldigi yerden devam kuyrugu — 14 Eylul 2026, PC yeniden baslatmasi sonrasi.
#
# Tum tamamlanmis cozmeler benchmark_results.csv'de; hicbir sey kaybolmadi.
# benchmark.py ayni hucreyi yeniden cozmeye calisirsa MUKERRER deyip atlar,
# bu yuzden listeyi oldugu gibi birakmak guvenlidir.
#
# Kullanim (proje kokunden):
#     bash paket3_asr/scripts/resume_queue.sh > paket3_asr/outputs/metrics/resume.log 2>&1 &
#
# SIRA ONEMLI: pcen once. Gerekce, pcen'in mevcut uc birlestirme uyesini de
# iki eksende birden gecmesi (kalite -0.01 vs +0.38..+1.35; ham sese benzerlik
# %81.1 vs %85.8..%87.2) — Y3/Y3b/Y5 boyunca aranan profil. Medium ise SABIT bir
# yapilandirmanin dogrulamasi; yapilandirma kesinlesmeden kosturmak erken olur.

set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE

LV3=faster-whisper-large-v3-int8
MED=faster-whisper-medium-int8
ALL="saha_01_img_3463 saha_02_img_3492 saha_03_img_3602 saha_04_vid_20220913_142911 video_01_genel_anamnez video_02_psikiyatrik_anamnez video_03_pediatrik_anamnez video_04_simule_hasta_gorusmesi video_05_poliklinik_anamnez_fizik_muayene video_06_gogus_hastaliklari_oyku_alma"

# --- 1) pcen tam-10'a tamamlama  (kalan 3 kayit, ~1.3 sa) ---
echo "######## pcen tam-10 tamamlama  $(date '+%H:%M:%S')"
for B in video_02_psikiyatrik_anamnez video_03_pediatrik_anamnez video_06_gogus_hastaliklari_oyku_alma; do
  echo "=== pcen | $B ==="
  python paket3_asr/scripts/benchmark.py --basename "$B" --model-id "$LV3" \
    --condition pcen --run-purpose main_benchmark 2>&1 | grep -E "KAYDED|MUKERRER|HATA" | head -1
done

# --- 2) Medium ucuncu model dogrulamasi (28 cozme kaldi, ~5 sa) ---
echo "######## Medium ucuncu model dogrulamasi  $(date '+%H:%M:%S')"
for B in $ALL; do
  for C in oa_aggr_a020 sa_psm_aggr_a040 sa_psm_mild_a025; do
    echo "=== medium | $B | $C ==="
    python paket3_asr/scripts/benchmark.py --basename "$B" --model-id "$MED" \
      --condition "$C" --run-purpose main_benchmark 2>&1 | grep -E "KAYDED|MUKERRER|HATA" | head -1
  done
done
echo "[BITTI] $(date '+%H:%M:%S')"
