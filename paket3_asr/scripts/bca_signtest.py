"""
BCa guven araligi + isaret testi (Ek-2 Rev B, D5).

NEDEN BCa
---------
Simdiye kadar yuzdelik (percentile) bootstrap kullandik. Yuzdelik yontem,
dagilim carpiksa veya tahmin edici yanliysa gercek kapsama oraninin altinda
kalir. BCa iki duzeltme uygular:
  z0 — yanlilik duzeltmesi (bootstrap dagiliminin gozlenen degere gore konumu)
  a  — hizlanma (jackknife ile olculen carpiklik)
Kucuk orneklemde (n=10) fark onemli olabilir; ikisi de raporlanir.

NEDEN ISARET TESTI
------------------
"10 kaydin 10'unda iyilesme" ifadesi tek basina bir p degeri tasimaz. Isaret
testi bunu dogrudan sinar: etki yoksa her kaydin yonu yazi-tura olur, dolayisiyla
10/10 ayni yon p = 2 * 0.5^10 = 0.00195 verir. Bu, WER buyuklugunden BAGIMSIZ
bir kanit katmanidir — buyuk bir kayittan gelen kazanca dayanmaz.

Kullanim:
    python paket3_asr/scripts/bca_signtest.py
"""

import os
import random
import statistics
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from scipy import stats  # noqa: E402

from analyze_fusion import load_hypotheses, rover_fuse  # noqa: E402
from calculate_metrics import calculate_wer  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402
from evaluate_fusion_final import REC10  # noqa: E402


def bca_ci(vals, iters=10000, seed=20260910, alpha=0.05):
    """BCa araligi. Donen: (yuzdelik_lo, yuzdelik_hi, bca_lo, bca_hi)."""
    n = len(vals)
    obs = statistics.mean(vals)
    rng = random.Random(seed)
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))

    lo_p = boots[int(alpha / 2 * iters)]
    hi_p = boots[int((1 - alpha / 2) * iters) - 1]

    # z0: gozlenen degerin altinda kalan bootstrap orani
    n_less = sum(1 for b in boots if b < obs)
    prop = min(max(n_less / iters, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop)

    # a: jackknife carpikligi
    jack = [statistics.mean([vals[j] for j in range(n) if j != i]) for i in range(n)]
    jm = statistics.mean(jack)
    num = sum((jm - x) ** 3 for x in jack)
    den = 6.0 * (sum((jm - x) ** 2 for x in jack) ** 1.5)
    a = num / den if den != 0 else 0.0

    def adj(q):
        z = stats.norm.ppf(q)
        p = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return boots[min(max(int(p * iters), 0), iters - 1)]

    return lo_p, hi_p, adj(alpha / 2), adj(1 - alpha / 2)


def sign_test(vals):
    """Iki yonlu isaret testi; esitlikler (fark = 0) atilir (standart yaklasim)."""
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    ties = sum(1 for v in vals if v == 0)
    m = pos + neg
    if m == 0:
        return pos, neg, ties, float("nan")
    k = min(pos, neg)
    p = min(1.0, 2.0 * stats.binom.cdf(k, m, 0.5))
    return pos, neg, ties, p


def main():
    h = load_hypotheses()
    refs = {b: normalize_turkish_asr(
        open(os.path.join(PKG_ROOT, "data", "references", f"{b}.txt"), encoding="utf-8").read().strip())
        for b in REC10}
    F4 = ["none", "oa_aggr_a020", "sa_psm_aggr_a040", "sa_psm_mild_a025"]
    F5 = F4 + ["pcen"]
    models = [("faster-whisper-large-v3-int8", "Large-v3"),
              ("faster-whisper-medium-int8", "Medium"),
              ("faster-whisper-large-v3-turbo-int8", "Turbo")]

    def deltas(M, members):
        out = []
        for b in REC10:
            f = rover_fuse([h[(M, b)][m][0] for m in members],
                           [f"{M}|{b}|{m}" for m in members],
                           pivot_index=0, override_margin=1)
            out.append((calculate_wer(refs[b], f) - h[(M, b)]["none"][1]) * 100.0)
        return out

    print("=" * 104)
    print("BCa GUVEN ARALIGI + ISARET TESTI   (ham sese karsi, tam 10 kayit)")
    print("=" * 104)
    print(f"{'model':<12}{'uye':<6}{'fark':>8}{'%95 yuzdelik':>20}{'%95 BCa':>20}"
          f"{'iyi/kotu/esit':>15}{'isaret p':>11}")
    print("-" * 104)
    for M, lbl in models:
        for tag, mem in (("4", F4), ("5", F5)):
            if any(m not in h.get((M, REC10[0]), {}) for m in mem):
                continue
            d = deltas(M, mem)
            lo_p, hi_p, lo_b, hi_b = bca_ci(d)
            pos, neg, ties, p = sign_test(d)
            print(f"{lbl:<12}{tag:<6}{statistics.mean(d):>+8.2f}"
                  f"{f'[{lo_p:+.2f}, {hi_p:+.2f}]':>20}{f'[{lo_b:+.2f}, {hi_b:+.2f}]':>20}"
                  f"{f'{neg}/{pos}/{ties}':>15}{p:>11.4f}")
    print("-" * 104)
    print("isaret testi: WER buyuklugunden bagimsiz; yalniz kac kaydin hangi yone gittigine bakar.")
    print("10/10 tek yon -> p = 0.0020;  8/0 (2 esit) -> p = 0.0078")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
