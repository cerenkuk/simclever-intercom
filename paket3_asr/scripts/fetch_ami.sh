#!/usr/bin/env bash
# AMI indirme — test bolunmesinin ilk 10 toplantisi (Ek-2 Rev B, Bolum 4).
#
# Secim kurali: resmi test bolunmesindeki toplantilar kimlik sirasina dizilir,
# ilk 10 alinir. Test bolunmesi kaynaktan dogrulandi: 16 toplanti.
#
# Her toplanti icin: Array1-01 (tek uzak mikrofon = SDM) ve Headset-0..3 (IHM).
# Kirpma ve karistirma AYRI adimda; burada yalniz ham dosyalar cekilir.
set -u
cd "c:/Users/kukem/.gemini/antigravity-ide/scratch/simclever-intercom-acceptance"
BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
OUT="paket3_asr/data/corpora/ami"
MEETINGS="EN2002a EN2002b EN2002c EN2002d ES2004a ES2004b ES2004c ES2004d IS1009a IS1009b"
mkdir -p "$OUT"
echo "######## AMI indirme  $(date '+%H:%M:%S')"
for M in $MEETINGS; do
  mkdir -p "$OUT/$M"
  for F in Array1-01 Headset-0 Headset-1 Headset-2 Headset-3; do
    P="$OUT/$M/$M.$F.wav"
    if [ -s "$P" ]; then echo "[ATLANDI] $M.$F"; continue; fi
    curl -sL --retry 3 --retry-delay 5 -o "$P" "$BASE/$M/audio/$M.$F.wav"
    SZ=$(stat -c%s "$P" 2>/dev/null || echo 0)
    if [ "$SZ" -lt 100000 ]; then echo "[HATA] $M.$F  ($SZ bayt)"; rm -f "$P"; else
      echo "[OK] $M.$F  $(echo "scale=1; $SZ/1048576" | bc) MB"; fi
  done
done
echo "[BITTI] $(date '+%H:%M:%S')  toplam: $(du -sh $OUT | cut -f1)"
