"""
H2 — Donmus birlestirmenin Ingilizce korpusta degerlendirilmesi (Ek-2 Rev B).

YAPILANDIRMA YENIDEN AYARLANMADI (Bolum 7 Kural 2)
--------------------------------------------------
    uyeler : none + oa_aggr_a020 + sa_psm_aggr_a040 + sa_psm_mild_a025
    pivot  : ham capali
    marj   : 1
Turkce'de ne ise aynisi. Ingilizce icin uye/alfa/marj denenmis olsaydi iddia
"yontem genelleniyor" degil "her dilde yeniden ayar yapilabiliyor" olurdu.

ONCEDEN ILAN EDILEN BEKLENTI
----------------------------
Birlestirme cesitlilikten beslenir. Uye cesitliligi olculdu:
    TR (kiyas)            %14.1-14.6
    EN AMI uzak mikrofon  %11.7
    EN AMI kulaklik       % 7.4
    EN PriMock            % 5.2
Bu nedenle temiz hucrelerde kazanc BEKLENMIYOR; adil sinama AMI uzak mikrofon
hucresidir. Bu beklenti sonuc gorulmeden yazildi.

HUCRELER HAVUZLANMAZ
--------------------
Uc hucre uc farkli akustik kosul. Ayri raporlanir; toplam yalnizca tamamlayici
bilgi olarak verilir. Turkce ile hicbir sekilde havuzlanmaz (Bolum 13).

Kullanim:
    python paket3_asr/scripts/evaluate_fusion_en.py
"""

import collections
import csv
import os
import random
import statistics
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from rapidfuzz.distance import Levenshtein  # noqa: E402

from analyze_fusion import rover_fuse  # noqa: E402
from oa_conditions import PKG_ROOT  # noqa: E402

CSV_EN = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results_en.csv")
MEMBERS = ["none", "oa_aggr_a020", "sa_psm_aggr_a040", "sa_psm_mild_a025"]
CELLS = [("en_clean_medical", "PriMock (temiz, tibbi)"),
         ("en_clean_ihm", "AMI kulaklik (temiz)"),
         ("en_noisy_sdm", "AMI uzak mikrofon (gurultulu)")]

_V = {}


def _ids(t):
    o = []
    for w in t:
        i = _V.get(w)
        if i is None:
            i = len(_V); _V[w] = i
        o.append(i)
    return o


def wer(ref, hyp):
    r = _ids(ref.split())
    return Levenshtein.distance(r, _ids(hyp.split())) / max(len(r), 1)


def ci(vals, iters=10000, seed=20260910):
    rng = random.Random(seed)
    n = len(vals)
    m = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return statistics.mean(vals), m[int(0.025 * iters)], m[int(0.975 * iters) - 1]


def sign_p(vals):
    from scipy import stats
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    m = pos + neg
    return float("nan") if m == 0 else min(1.0, 2.0 * stats.binom.cdf(min(pos, neg), m, 0.5))


def main():
    H = collections.defaultdict(dict)
    REF = {}
    for r in csv.DictReader(open(CSV_EN, encoding="utf-8")):
        H[(r["cell"], r["basename"])][r["condition"]] = (r["transcription"], float(r["WER"]))
        REF[r["basename"]] = r["reference_text"]

    print("=" * 94)
    print("H2 — DONMUS BIRLESTIRME, INGILIZCE KORPUS   (yapilandirma yeniden ayarlanmadi)")
    print(f"uyeler: {', '.join(MEMBERS)}   pivot: ham capali   marj: 1")
    print("=" * 94)
    print(f"{'hucre':<32}{'ham':>8}{'birlesik':>10}{'fark':>8}{'%95 GA':>18}"
          f"{'iyi/esit/kotu':>15}{'isaret p':>10}")
    print("-" * 94)

    tum = []
    for cell, ad in CELLS:
        keys = [k for k in H if k[0] == cell]
        if any(m not in H[k] for k in keys for m in MEMBERS):
            print(f"{ad:<32}  EKSIK")
            continue
        d, base = [], []
        for k in keys:
            f = rover_fuse([H[k][m][0] for m in MEMBERS],
                           [f"{k[1]}|{m}" for m in MEMBERS],
                           pivot_index=0, override_margin=1)
            raw = H[k]["none"][1]
            d.append((wer(REF[k[1]], f) - raw) * 100.0)
            base.append(raw * 100.0)
        tum.extend(d)
        mu, lo, hi = ci(d)
        print(f"{ad:<32}{statistics.mean(base):>7.2f}%{statistics.mean(base)+mu:>9.2f}%{mu:>+8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
              f"{f'{sum(1 for x in d if x<0)}/{sum(1 for x in d if x==0)}/{sum(1 for x in d if x>0)}':>15}"
              f"{sign_p(d):>10.4f}")

    if len(tum) == 30:
        mu, lo, hi = ci(tum)
        print("-" * 94)
        print(f"{'TUM 30 KAYIT (tamamlayici)':<32}{'':>8}{'':>10}{mu:>+8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
              f"{f'{sum(1 for x in tum if x<0)}/{sum(1 for x in tum if x==0)}/{sum(1 for x in tum if x>0)}':>15}"
              f"{sign_p(tum):>10.4f}")
    print("=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(main())
