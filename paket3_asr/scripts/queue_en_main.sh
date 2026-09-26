#!/usr/bin/env bash
# Ingilizce asil kosu — 5 kosul x 30 kayit, Large-v3 (Ek-2 Rev B, Bolum 7).
#
# SIRA GEREKCESI: once mild/aggressive (H1: negatif sonucun Ingilizcede de
# gorulmesi, kiyas calismayla dogrudan karsilastirma), sonra uc birlestirme
# uyesi (H2), en son opsiyonel pcen. Kosu yarida kalirsa en kritik sonuclar
# elde kalir.
#
# Yapilandirma Ingilizce icin YENIDEN AYARLANMADI (Bolum 7 Kural 2).
# Ayni anda tek cozme: hepsi ayni CSV'ye yaziyor.
# Tamamlanan hucreleri benchmark_en.py atlar.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
export KMP_DUPLICATE_LIB_OK=TRUE
echo "######## INGILIZCE ASIL KOSU  $(date '+%Y-%m-%d %H:%M:%S')"
for C in mild aggressive oa_aggr_a020 sa_psm_aggr_a040 sa_psm_mild_a025 pcen; do
  echo "=== $C  $(date '+%H:%M:%S') ==="
  python paket3_asr/scripts/benchmark_en.py --condition "$C" 2>&1 | grep -E "kosul|OK\]|EKSIK|Error"
done
echo "[BITTI] $(date '+%Y-%m-%d %H:%M:%S')"
