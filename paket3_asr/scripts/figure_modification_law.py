"""
BIRLESIK SEKIL — gurultu seviyesi ekseni uzerinde hasar ve cesitlilik.

NE GOSTERIYOR
-------------
Iki dil, bes akustik kosul, 17,6-63,6 dB'lik bir gurultu ekseni. Her noktada
bastirmanin iki sonucu birlikte olculuyor:
    hasar      = aggressive kosulunun ham sese gore WER farki
    cesitlilik = ayni kosulun ham transkriptten askin farkliligi

Bulgu: ikisi de gurultuyle birlikte artiyor ve birbirini takip ediyor. Yani
"bastirma ASR'i bozar" degil, "bastirmanin etkisi bastirilacak gurultunun
miktariyla olceklenir" — ve tam bozdugu yerde cesitlilik de uretir.

Bu, calismanin en genellenebilir sekli: dil degil AKUSTIK belirliyor.
TR saha (19,3 dB) ile EN AMI uzak mikrofon (17,6 dB) yan yana duser.

Kullanim:
    python paket3_asr/scripts/figure_snr_law.py
"""

import collections
import csv
import os
import statistics
import sys

import numpy as np
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from rapidfuzz.distance import Levenshtein  # noqa: E402

from analyze_fusion import load_hypotheses  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT  # noqa: E402

FIG = os.path.join(PKG_ROOT, "outputs", "figures", "modification_law.png")
W = int(0.05 * 16000)
_V = {}


def ids(t):
    o = []
    for w in t:
        i = _V.get(w)
        if i is None:
            i = len(_V); _V[w] = i
        o.append(i)
    return o


def degisim(raw_p, den_p, lag=96):
    """
    Bastiricinin sesi FIILEN ne kadar degistirdigi. Surucu 96 orneklik gecikme
    getirdigi icin telafi edilir; aksi halde olcum sifira yakin cikar ve
    "hicbir sey yapmamis" gibi gorunur.
    """
    x, _ = sf.read(raw_p, dtype="float64")
    y, _ = sf.read(den_p, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if y.ndim > 1:
        y = y[:, 0]
    n = min(len(x), len(y)) - lag
    seg = min(16000 * 30, n)
    st = max(range(0, max(n - seg, 1), max(seg, 1)), key=lambda k: np.mean(x[k:k + seg] ** 2))
    return (1 - float(np.corrcoef(x[st:st + seg], y[st + lag:st + lag + seg])[0, 1])) * 100


def snr(path):
    x, _ = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    n = len(x) // W * W
    e = np.sqrt(np.mean(x[:n].reshape(-1, W) ** 2, axis=1))
    db = 20 * np.log10(e + 1e-12)
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def main():
    pts = []

    # --- Turkce: saha ve video ayri ---
    h = load_hypotheses()
    M = "faster-whisper-large-v3-int8"
    TR = {"TR saha": ["saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
                      "saha_04_vid_20220913_142911"],
          "TR video": ["video_01_genel_anamnez", "video_02_psikiyatrik_anamnez",
                       "video_03_pediatrik_anamnez", "video_04_simule_hasta_gorusmesi",
                       "video_05_poliklinik_anamnez_fizik_muayene",
                       "video_06_gogus_hastaliklari_oyku_alma"]}
    for ad, recs in TR.items():
        s = statistics.mean(degisim(
            os.path.join(PKG_ROOT, "data", "audio_16k", f"{b}.wav"),
            os.path.join(PKG_ROOT, "data", "audio_denoised_aggressive", f"{b}.wav")) for b in recs)
        q, d = [], []
        for b in recs:
            ref = normalize_turkish_asr(open(os.path.join(PKG_ROOT, "data", "references", f"{b}.txt"),
                                             encoding="utf-8").read().strip())
            rl = max(len(ref.split()), 1)
            raw, rw = h[(M, b)]["none"]
            cur, cw = h[(M, b)]["aggressive"]
            q.append((cw - rw) * 100)
            d.append(Levenshtein.distance(ids(raw.split()), ids(cur.split())) / rl * 100)
        mq = statistics.mean(q)
        pts.append((ad, s, mq, statistics.mean(d) - statistics.mean(abs(x) for x in q), "tr"))

    # --- Ingilizce: uc hucre ---
    H = collections.defaultdict(dict); RF = {}
    with open(os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results_en.csv"),
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            H[(r["cell"], r["basename"])][r["condition"]] = (r["transcription"], float(r["WER"]) * 100)
            RF[r["basename"]] = r["reference_text"]
    EN = {"en_clean_medical": "EN PriMock", "en_clean_ihm": "EN AMI kulaklik",
          "en_noisy_sdm": "EN AMI uzak mik."}
    for cell, ad in EN.items():
        keys = [k for k in H if k[0] == cell]
        s = statistics.mean(degisim(
            os.path.join(PKG_ROOT, "data", "audio_16k_en", f"{k[1]}.wav"),
            os.path.join(PKG_ROOT, "data", "audio_denoised_aggressive_en", f"{k[1]}.wav")) for k in keys)
        q, d = [], []
        for k in keys:
            rl = max(len(RF[k[1]].split()), 1)
            raw, rw = H[k]["none"]
            cur, cw = H[k]["aggressive"]
            q.append(cw - rw)
            d.append(Levenshtein.distance(ids(raw.split()), ids(cur.split())) / rl * 100)
        pts.append((ad, s, statistics.mean(q),
                    statistics.mean(d) - statistics.mean(abs(x) for x in q), "en"))

    pts.sort(key=lambda p: p[1])
    print(f"{'hucre':<20}{'degisim':>12}{'hasar':>9}{'askin cesitlilik':>19}")
    print("-" * 64)
    for ad, s, q, d, _ in pts:
        print(f"{ad:<20}{s:>11.2f}%{q:>+9.2f}{d:>18.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=160)
    ax2 = ax.twinx()
    xs = [p[1] for p in pts]
    ax.plot(xs, [p[2] for p in pts], "o-", color="#c0392b", lw=2, ms=9, label="hasar (WER farkı, puan)")
    ax2.plot(xs, [p[3] for p in pts], "s--", color="#2980b9", lw=2, ms=8, label="aşkın çeşitlilik (%)")
    for ad, s, q, d, lang in pts:
        ax.annotate(ad, (s, q), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.5,
                    fontweight="bold" if lang == "tr" else "normal")
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xlabel("Bastırıcının sesi değiştirme miktarı (%)   → daha çok müdahale", fontsize=10)
    ax.set_ylabel("Bastırmanın hasarı (puan)", color="#c0392b", fontsize=10)
    ax2.set_ylabel("Aşkın çeşitlilik (%)", color="#2980b9", fontsize=10)
    ax.tick_params(axis="y", colors="#c0392b")
    ax2.tick_params(axis="y", colors="#2980b9")
    ax.set_title("Bastırmanın etkisi gürültü miktarıyla ölçekleniyor\n"
                 "iki dil · beş akustik koşul · kalın = Türkçe", fontsize=11.5, pad=12)
    ax.grid(alpha=0.25, lw=0.6)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG); plt.close(fig)
    print(f"\nsekil -> {os.path.relpath(FIG, PKG_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
