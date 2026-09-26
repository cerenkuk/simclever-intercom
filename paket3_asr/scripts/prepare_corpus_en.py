"""
Ingilizce korpus hazirlama (Ek-2 Rev B, Bolum 5-6).

KAPSAM
------
PriMock57 (EN temiz, tibbi)  — 10 konsultasyon, iki kanal birlestirilir
AMI SDM   (EN gurultulu)     — 10 toplantinin ilk 10 dk, tek uzak mikrofon
AMI IHM-mix (EN temiz, eslestirilmis) — ayni toplantilar, kulaklik kanallari toplami

TASARIM KARARLARI — hepsi Bolum 5-6'dan
---------------------------------------
* Ses duzeyi esitleme disinda HICBIR iyilestirme/filtre uygulanmaz. Taban her
  zaman ham sestir.
* PriMock57'de doktor ve hasta ayri kanallarda. Tek mikrofonlu senaryomuzu
  yansitmak icin esit kazancla toplanir, sonra tepe normalize edilir.
* Referans: utterance duzeyi transkriptler ZAMAN SIRASINA gore birlestirilir.
  PriMock57'de iki kanalin utterance'lari tek zaman ekseninde harmanlanir.

ETIKET ISLEME — dogrulanmasi gereken karar
------------------------------------------
PriMock57 transkriptlerinde iki etiket var:
  <UNSURE>metin</UNSURE> — yazici emin degil ama metni vermis  -> METIN KORUNUR
  <UNIN/>                — anlasilmayan bolum, metin yok       -> SILINIR

<UNIN/> silindiginde ASR o bolumde bir sey uretirse ekleme (insertion) hatasi
sayilir. Alternatif, o utterance'i tamamen atmakti; referans surekliligini
bozdugu icin secilmedi. Bu karar rapora yazilir.

Kullanim:
    python paket3_asr/scripts/prepare_corpus_en.py --cell primock
    python paket3_asr/scripts/prepare_corpus_en.py --cell ami_sdm --cell ami_ihm
"""

import argparse
import hashlib
import os
import re
import sys

import numpy as np
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from oa_conditions import PKG_ROOT  # noqa: E402

CORPORA = os.path.join(PKG_ROOT, "data", "corpora")
OUT_AUDIO = os.path.join(PKG_ROOT, "data", "audio_16k_en")
OUT_REF = os.path.join(PKG_ROOT, "data", "references_en")
SR = 16000

PRIMOCK_IDS = [f"day1_consultation{i:02d}" for i in range(1, 11)]


def parse_textgrid(path):
    """(baslangic, bitis, metin) listesi; bos araliklar atlanir."""
    s = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(
            r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"(.*?)\"\s*",
            s, re.S):
        txt = m.group(3).strip()
        if txt:
            out.append((float(m.group(1)), float(m.group(2)), txt))
    return out


def clean_tags(t):
    """<UNSURE> sarmalayicisi kaldirilir (metin kalir), <UNIN/> silinir."""
    t = re.sub(r"</?UNSURE>", " ", t)
    t = re.sub(r"<UNIN/>", " ", t)
    t = re.sub(r"<[^>]*>", " ", t)          # baska bir etiket kalirsa
    return re.sub(r"\s+", " ", t).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mix_equal(paths, peak=0.98):
    """Kanallari esit kazancla topla, tepe normalize et. Uzunluk farki sifirla doldurulur."""
    sigs = []
    for p in paths:
        x, sr = sf.read(p, dtype="float64")
        assert sr == SR, f"{p}: {sr} Hz (16000 bekleniyordu)"
        if x.ndim > 1:
            x = x.mean(axis=1)
        sigs.append(x)
    n = max(len(x) for x in sigs)
    y = np.zeros(n)
    for x in sigs:
        y[:len(x)] += x
    pk = float(np.max(np.abs(y)))
    if pk > 0:
        y *= peak / pk
    return y


def prepare_primock():
    src = os.path.join(CORPORA, "primock57")
    rows = []
    for cid in PRIMOCK_IDS:
        wavs = [os.path.join(src, "audio", f"{cid}_{r}.wav") for r in ("doctor", "patient")]
        tgs = [os.path.join(src, "transcripts", f"{cid}_{r}.TextGrid") for r in ("doctor", "patient")]

        y = mix_equal(wavs)
        op = os.path.join(OUT_AUDIO, f"{cid}.wav")
        sf.write(op, y.astype(np.float32), SR, subtype="PCM_16")

        # Iki kanalin utterance'lari tek zaman ekseninde harmanlanir
        utts = []
        for t in tgs:
            utts.extend(parse_textgrid(t))
        utts.sort(key=lambda u: u[0])
        ref = clean_tags(" ".join(u[2] for u in utts))
        with open(os.path.join(OUT_REF, f"{cid}.txt"), "w", encoding="utf-8") as f:
            f.write(ref)

        rows.append(dict(cell="en_clean_medical", source="PriMock57", rec_id=cid,
                         duration_s=round(len(y) / SR, 2), n_utt=len(utts),
                         n_words=len(ref.split()), sha256=sha256(op)))
        print(f"[OK] {cid:<26}{len(y)/SR/60:6.2f} dk  {len(utts):4d} utterance  "
              f"{len(ref.split()):5d} kelime")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ingilizce korpus hazirlama")
    ap.add_argument("--cell", action="append", default=None,
                    choices=["primock", "ami_sdm", "ami_ihm"])
    args = ap.parse_args()
    cells = args.cell or ["primock"]

    os.makedirs(OUT_AUDIO, exist_ok=True)
    os.makedirs(OUT_REF, exist_ok=True)

    rows = []
    if "primock" in cells:
        print("=== PriMock57 (EN temiz, tibbi) ===")
        rows += prepare_primock()
    if "ami_sdm" in cells or "ami_ihm" in cells:
        print("[BEKLIYOR] AMI hucreleri ayri adimda hazirlanacak")

    if rows:
        import csv
        man = os.path.join(PKG_ROOT, "data", "corpora", "manifest_en.csv")
        new = not os.path.exists(man)
        with open(man, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nmanifest -> {os.path.relpath(man, PKG_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
