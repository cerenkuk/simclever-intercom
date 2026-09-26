"""
AMI hucreleri hazirlama (Ek-2 Rev B, Bolum 4-6).

IKI HUCRE, AYNI ICERIK
----------------------
ami_sdm      EN gurultulu  — tek uzak mikrofon (Array1-01), ilk 10 dk
ami_ihm      EN temiz      — AYNI toplantinin AYNI 10 dakikasi, kulaklik
                             kanallari esit kazancla toplanmis

Bu ciftin degeri: ayni konusma, iki akustik hal. Tek degisken akustik; referans
ayni oldugu icin ek transkripsiyon maliyeti de yok.

SECIM KURALI (Bolum 4)
----------------------
Resmi test bolunmesindeki toplantilar kimlik sirasina dizilir, ilk 10 alinir.
Test bolunmesi kaynaktan dogrulandi: 16 toplanti. Ilk 10: EN2002a-d, ES2004a-d,
IS1009a-b. Her birinin ILK KONUSMADAN itibaren ilk 10 dakikasi kirpilir.

"ILK KONUSMADAN ITIBAREN" — onemli
-----------------------------------
AMI kayitlari konusma baslamadan once uzun sessizlikler icerebilir. Pencereyi
0. saniyeden baslatmak, 10 dakikanin bir kismini sessizlige harcamak olurdu.
Pencere, transkriptteki ilk segmentin baslangicindan baslatilir ve ses ile
transkript AYNI pencereden kesilir.

Kullanim:
    python paket3_asr/scripts/prepare_ami.py
"""

import csv
import hashlib
import os
import sys

import numpy as np
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from oa_conditions import PKG_ROOT  # noqa: E402

AMI = os.path.join(PKG_ROOT, "data", "corpora", "ami")
META = os.path.join(PKG_ROOT, "data", "corpora", "ami_meta", "segments_sdm.csv")
OUT_AUDIO = os.path.join(PKG_ROOT, "data", "audio_16k_en")
OUT_REF = os.path.join(PKG_ROOT, "data", "references_en")
SR = 16000
WIN = 600.0          # 10 dakika
HEADSETS = [f"Headset-{i}" for i in range(4)]

MEETINGS = ["EN2002a", "EN2002b", "EN2002c", "EN2002d",
            "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b"]


def load_segments():
    segs = {}
    with open(META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segs.setdefault(r["meeting_id"], []).append(
                (float(r["begin_time"]), float(r["end_time"]), r["speaker_id"], r["text"]))
    for m in segs:
        segs[m].sort(key=lambda t: t[0])
    return segs


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_window(path, t0, t1):
    info = sf.info(path)
    assert info.samplerate == SR, f"{path}: {info.samplerate} Hz"
    start = int(t0 * SR)
    frames = min(int((t1 - t0) * SR), info.frames - start)
    x, _ = sf.read(path, start=start, frames=frames, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def main():
    os.makedirs(OUT_AUDIO, exist_ok=True)
    os.makedirs(OUT_REF, exist_ok=True)
    segs = load_segments()
    rows = []

    print(f"{'toplanti':<10}{'pencere':>18}{'segment':>9}{'kelime':>8}{'kel/dk':>8}  cikti")
    print("-" * 78)
    for m in MEETINGS:
        ss = segs[m]
        t0 = ss[0][0]                       # ilk konusmanin baslangici
        t1 = t0 + WIN
        win = [s for s in ss if s[1] > t0 and s[0] < t1]
        ref = " ".join(s[3] for s in win).strip()
        nw = len(ref.split())

        with open(os.path.join(OUT_REF, f"{m}_sdm.txt"), "w", encoding="utf-8") as f:
            f.write(ref)
        with open(os.path.join(OUT_REF, f"{m}_ihm.txt"), "w", encoding="utf-8") as f:
            f.write(ref)                    # ayni icerik -> ayni referans

        # SDM: tek uzak mikrofon, karistirma yok
        sdm = read_window(os.path.join(AMI, m, f"{m}.Array1-01.wav"), t0, t1)
        p_sdm = os.path.join(OUT_AUDIO, f"{m}_sdm.wav")
        sf.write(p_sdm, sdm.astype(np.float32), SR, subtype="PCM_16")

        # IHM-mix: kulaklik kanallari esit kazancla toplanir, tepe normalize
        chans = [read_window(os.path.join(AMI, m, f"{m}.{hs}.wav"), t0, t1) for hs in HEADSETS]
        n = max(len(c) for c in chans)
        mix = np.zeros(n)
        for c in chans:
            mix[:len(c)] += c
        pk = float(np.max(np.abs(mix)))
        if pk > 0:
            mix *= 0.98 / pk
        p_ihm = os.path.join(OUT_AUDIO, f"{m}_ihm.wav")
        sf.write(p_ihm, mix.astype(np.float32), SR, subtype="PCM_16")

        dur = len(sdm) / SR
        print(f"{m:<10}{f'{t0:.1f}-{t1:.1f} sn':>18}{len(win):>9}{nw:>8}"
              f"{nw/(dur/60):>8.0f}  {os.path.basename(p_sdm)}, {os.path.basename(p_ihm)}")

        for cell, p in (("en_noisy_sdm", p_sdm), ("en_clean_ihm", p_ihm)):
            rows.append(dict(cell=cell, source="AMI", rec_id=f"{m}_{cell[-3:]}",
                             meeting=m, window_start_s=round(t0, 2), window_end_s=round(t1, 2),
                             duration_s=round(sf.info(p).frames / SR, 2),
                             n_utt=len(win), n_words=nw, sha256=sha256(p)))

    man = os.path.join(PKG_ROOT, "data", "corpora", "manifest_ami.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("-" * 78)
    print(f"{len(rows)} kayit -> {os.path.relpath(man, PKG_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
