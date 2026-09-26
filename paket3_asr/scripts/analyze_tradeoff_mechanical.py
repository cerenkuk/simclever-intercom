"""
D2 — Takas egrisindeki MEKANIK bilesenin ayrilmasi (Ek-2 Rev B).

ELESTIRI
--------
Takas egrisi iki eksen kullaniyordu: kalite (ham sese gore WER farki) ve ham
transkriptle benzerlik. Bu iki eksen BAGIMSIZ DEGIL. Ucgen esitsizligi geregi,
duzenleme mesafesi d icin:

    d(ham, kosul) >= | d(ham, referans) - d(kosul, referans) |

Referans uzunluguna bolunce:

    farklilik(ham, kosul) >= | WER_kosul - WER_ham |  =  kalite farki

Yani kalitesi 10 puan bozulan bir kosul, ham transkriptten EN AZ %10 farkli
olmak ZORUNDADIR. Korelasyonun bir kismi tanim geregidir; olculen -0,92'nin ne
kadari gercek bir takas, ne kadari bu tabandir?

UC AYRI SINAMA
--------------
1. ASKIN CESITLILIK: gozlenen farklilik eksi mekanik taban. Taban cikarildiktan
   sonra ilinti hala guclu mu?
2. HAM'DAN BAGIMSIZ EKSEN: ham sese benzerlik yerine KOSULLAR ARASI ortalama
   benzerlik. Bu eksen ham transkripte capalanmadigi icin mekanik bag zayiflar.
3. ESIT KALITE KATMANI: neredeyse ayni WER'e sahip kosullar arasinda cesitlilik
   degisiyor mu? Degisiyorsa cesitlilik kaliteden bagimsiz bir boyuttur.

Kullanim:
    python paket3_asr/scripts/analyze_tradeoff_mechanical.py
"""

import csv
import itertools
import math
import os
import statistics
import sys

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from analyze_fusion import load_hypotheses  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT  # noqa: E402
from evaluate_fusion_final import REC10  # noqa: E402

REPORT = os.path.join(PKG_ROOT, "outputs", "metrics", "tradeoff_mechanical_report.csv")


_VOCAB = {}


def _ids(tokens):
    """Kelimeleri tamsayiya esler; rapidfuzz dizi uzerinde C++ hizinda calisir."""
    out = []
    for t in tokens:
        i = _VOCAB.get(t)
        if i is None:
            i = len(_VOCAB)
            _VOCAB[t] = i
        out.append(i)
    return out


def edit_distance(a, b):
    """
    Kelime duzeyi Levenshtein.

    Saf Python (ve satir satir numpy) 2000 kelimelik dizilerde cok yavas: bir cift
    ~4 milyon hucre, 300+ cift gerekiyor. rapidfuzz ayni hesabi C++'ta yapar.
    """
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(_ids(a), _ids(b))


def main():
    h = load_hypotheses()
    M = "faster-whisper-large-v3-int8"
    refs = {b: normalize_turkish_asr(
        open(os.path.join(PKG_ROOT, "data", "references", f"{b}.txt"), encoding="utf-8").read().strip())
        for b in REC10}
    reflen = {b: len(refs[b].split()) for b in REC10}

    # Tam 10 kayitta kapsami olan kosullar (guvenilir kayit ortalamasi icin)
    conds = [c for c in h[(M, REC10[0])] if all(c in h.get((M, b), {}) for b in REC10)]
    conds = [c for c in conds if c != "none"]

    rows = {}
    for c in conds:
        q, obs, floor = [], [], []
        for b in REC10:
            raw_t = h[(M, b)]["none"][0].split()
            cur_t = h[(M, b)][c][0].split()
            d = edit_distance(raw_t, cur_t) / reflen[b] * 100.0
            dq = (h[(M, b)][c][1] - h[(M, b)]["none"][1]) * 100.0
            q.append(dq); obs.append(d); floor.append(abs(dq))
        rows[c] = (statistics.mean(q), statistics.mean(obs), statistics.mean(floor))

    def pearson(xs, ys):
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs)) * math.sqrt(sum((b - my) ** 2 for b in ys))
        return num / den if den else float("nan")

    print("=" * 96)
    print("D2 — TAKAS EGRISINDEKI MEKANIK BILESENIN AYRILMASI   |   Large-v3, tam 10 kayit")
    print("=" * 96)
    print(f"{'kosul':<22}{'kalite':>9}{'gozlenen':>11}{'mekanik':>10}{'askin':>9}{'taban payi':>12}")
    print(f"{'':<22}{'(pp)':>9}{'farklilik%':>11}{'taban%':>10}{'cesitlilik%':>9}{'':>12}")
    print("-" * 96)
    for c, (q, o, f) in sorted(rows.items(), key=lambda kv: kv[1][0]):
        print(f"{c:<22}{q:>+9.2f}{o:>11.1f}{f:>10.1f}{o - f:>9.1f}{f / o * 100 if o else 0:>11.0f}%")
    print("-" * 96)

    qs = [v[0] for v in rows.values()]
    obs = [v[1] for v in rows.values()]
    exc = [v[1] - v[2] for v in rows.values()]
    print(f"1) kalite  vs  GOZLENEN farklilik : r = {pearson(obs, qs):+.3f}")
    print(f"   kalite  vs  ASKIN cesitlilik   : r = {pearson(exc, qs):+.3f}   <- mekanik taban cikarilmis")
    print(f"   mekanik tabanin gozlenen farkliliktaki ortalama payi: "
          f"%{statistics.mean(v[2] / v[1] * 100 for v in rows.values()):.0f}")

    # 2) Ham'dan bagimsiz eksen: kosullar arasi ortalama benzerlik
    print()
    print("2) HAM'DAN BAGIMSIZ EKSEN — kosullar arasi ortalama farklilik")
    inter = {}
    for c in conds:
        ds = []
        for b in REC10:
            for other in conds:
                if other == c:
                    continue
                ds.append(edit_distance(h[(M, b)][c][0].split(),
                                        h[(M, b)][other][0].split()) / reflen[b] * 100.0)
        inter[c] = statistics.mean(ds)
    iv = [inter[c] for c in rows]
    print(f"   kalite  vs  kosullar-arasi farklilik : r = {pearson(iv, qs):+.3f}")

    # 3) Esit kalite katmani
    print()
    print("3) ESIT KALITE KATMANI — WER farki <= 1.0 pp olan kosullar")
    near = {c: v for c, v in rows.items() if abs(v[0]) <= 1.0}
    if len(near) >= 3:
        print(f"{'kosul':<22}{'kalite':>9}{'askin cesitlilik':>18}")
        for c, (q, o, f) in sorted(near.items(), key=lambda kv: kv[1][1] - kv[1][2]):
            print(f"{c:<22}{q:>+9.2f}{o - f:>18.1f}")
        sp = [v[1] - v[2] for v in near.values()]
        print(f"   {len(near)} kosul neredeyse ayni kalitede (|dWER| <= 1.0 pp) ama askin")
        print(f"   cesitlilikleri %{min(sp):.1f} ile %{max(sp):.1f} arasinda degisiyor"
              f" ({max(sp) - min(sp):.1f} puan yayilim).")
    else:
        print("   yeterli kosul yok")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "quality_delta_pp", "observed_dissim_pct",
                    "mechanical_floor_pct", "excess_diversity_pct", "inter_condition_dissim_pct"])
        for c, (q, o, f) in sorted(rows.items(), key=lambda kv: kv[1][0]):
            w.writerow([c, f"{q:.4f}", f"{o:.4f}", f"{f:.4f}", f"{o-f:.4f}", f"{inter[c]:.4f}"])
    print(f"\nrapor -> {os.path.relpath(REPORT, PKG_ROOT)}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
