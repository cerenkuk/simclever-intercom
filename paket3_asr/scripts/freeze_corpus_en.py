"""
Ingilizce korpusu DONDURMA (Ek-2 Rev B, Bolum 4 Kural 1 ve Bolum 12).

Bu dosya yazildiktan sonra kayit listesi DEGISTIRILEMEZ. Sonuc gorulduk ten
sonra kayit eklemek veya cikarmak calismanin butun istatistigini gecersiz kilar.

ICERIK
------
Her kayit icin: kaynak, kimlik, hucre, sure, pencere (varsa), SHA-256, referans
kelime sayisi ve transkript kapsamasi.

PENCERE KURALI — degistirilmedi
--------------------------------
AMI pencereleri "ilk konusmadan itibaren 10 dakika" kuralindan gelir (Bolum 4).
Hazirlik sirasinda alternatif bir kural (transkript yogunlugu esigi) olculdu:
4 toplantida baslangici 45-50 sn kaydirip kapsamayi 5-8 puan artiriyordu.
DEGISTIRILMEDI — kazanc mutevazi, bedeli ise onceden ilan edilmis bir kurali
veriye baktiktan sonra degistirmek olurdu. Kurulum asamasi gercek toplanti
sesidir ve tum kosullar ayni sesi gordugu icin farkta sadelesir.

Kullanim:
    python paket3_asr/scripts/freeze_corpus_en.py
"""

import csv
import collections
import hashlib
import os
import sys

import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from oa_conditions import PKG_ROOT  # noqa: E402

AUDIO = os.path.join(PKG_ROOT, "data", "audio_16k_en")
REF = os.path.join(PKG_ROOT, "data", "references_en")
META = os.path.join(PKG_ROOT, "data", "corpora", "ami_meta", "segments_sdm.csv")
AMI_MAN = os.path.join(PKG_ROOT, "data", "corpora", "manifest_ami.csv")
OUT = os.path.join(PKG_ROOT, "config", "corpus_en_frozen.yaml")
SR = 16000


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ami_coverage():
    segs = collections.defaultdict(list)
    with open(META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segs[r["meeting_id"]].append((float(r["begin_time"]), float(r["end_time"])))
    win = {}
    with open(AMI_MAN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            win[r["meeting"]] = (float(r["window_start_s"]), float(r["window_end_s"]))
    out = {}
    for m, (t0, t1) in win.items():
        tot = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in segs[m])
        out[m] = (t0, t1, tot / (t1 - t0) * 100)
    return out


def main():
    if os.path.exists(OUT):
        print(f"[DUR] {os.path.relpath(OUT, PKG_ROOT)} zaten var.")
        print("      Dondurulmus liste degistirilemez. Yeniden uretmek icin once")
        print("      dosyayi elle silmeniz gerekir; bu bilincli bir karar olmalidir.")
        return 1

    cov = ami_coverage()
    L = ["# Ingilizce korpus — DONDURULMUS kayit listesi",
         "# Ek-2 Rev B, Bolum 4 Kural 1. Bu dosya bir daha degistirilmez.",
         "# Uretim: freeze_corpus_en.py",
         "",
         "window_rule: \"AMI: ilk konusmadan itibaren 10 dakika (Bolum 4); degistirilmedi\"",
         "sample_rate: 16000",
         "cells:"]

    cells = [
        ("en_clean_medical", "PriMock57", "CC BY 4.0",
         [f"day1_consultation{i:02d}" for i in range(1, 11)]),
        ("en_noisy_sdm", "AMI (Array1-01, tek uzak mikrofon)", "CC BY 4.0",
         [f"{m}_sdm" for m in sorted(cov)]),
        ("en_clean_ihm", "AMI (Headset-0..3 karisimi)", "CC BY 4.0",
         [f"{m}_ihm" for m in sorted(cov)]),
    ]

    n_tot, s_tot = 0, 0.0
    for cell, src, lic, ids in cells:
        L += [f"  - id: {cell}", f"    source: \"{src}\"", f"    license: \"{lic}\"",
              "    records:"]
        for rid in ids:
            ap = os.path.join(AUDIO, f"{rid}.wav")
            rp = os.path.join(REF, f"{rid}.txt")
            dur = sf.info(ap).frames / SR
            nw = len(open(rp, encoding="utf-8").read().split())
            L += [f"      - id: {rid}",
                  f"        duration_s: {dur:.2f}",
                  f"        ref_words: {nw}",
                  f"        sha256: {sha256(ap)}"]
            if rid[:-4] in cov:
                t0, t1, c = cov[rid[:-4]]
                L += [f"        window_s: [{t0:.2f}, {t1:.2f}]",
                      f"        transcript_coverage_pct: {c:.1f}"]
            n_tot += 1; s_tot += dur
        L.append("")

    L += [f"# toplam: {n_tot} kayit, {s_tot/3600:.2f} saat"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"DONDURULDU: {n_tot} kayit, {s_tot/3600:.2f} saat")
    print(f"  -> {os.path.relpath(OUT, PKG_ROOT)}")
    print("\nBu andan itibaren kayit listesi degistirilemez.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
