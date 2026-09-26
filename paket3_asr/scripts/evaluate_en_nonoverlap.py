"""
AMI hucrelerinin ORTUSMESIZ kelimeler uzerinde yeniden degerlendirilmesi.

NEDEN
-----
Denetimde bulundu: AMI referansi esZAMANLI konusan kisilerin cumlelerini ardisik
yaziyor, Whisper ise tek akis uretiyor. Olculdu: ortusen kelimelerde hata orani
%57,3, ortusmeyenlerde %17,1 — 3,4 kat. O kelimeler yapisal olarak ulasilamaz.

Sonucu: kelimelerin %26,7'si sabit bir hata tabani olusturuyor ve olculebilir
hareket alanini daraltiyor. Kosullar arasi FARK gecerli kalir (taban hepsinde
ayni) ama DUYARLILIK duser.

Bu betik ayni karsilastirmalari yalnizca ORTUSMEYEN referans kelimeleri uzerinde
tekrarlar. Ekleme (insertion) hatalari bir referans kelimesine baglanamadigi icin
disarida birakilir; olculen sey "degistirme + silme orani"dir.

DURUSTLUK KAYDI
---------------
Bu olcut, H1/H2 sonuclari GORULDUKTEN SONRA, denetim ortusme sorununu ortaya
cikardigi icin eklendi. Bu nedenle BIRINCIL sonuc degildir; duyarlilik analizi
olarak raporlanir. Birincil sonuclar Bolum 7b'deki tam-pencere olcumleridir.

Kullanim:
    python paket3_asr/scripts/evaluate_en_nonoverlap.py
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
from whisper.normalizers import EnglishTextNormalizer  # noqa: E402

from analyze_fusion import rover_fuse  # noqa: E402
from oa_conditions import PKG_ROOT  # noqa: E402

CSV_EN = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results_en.csv")
META = os.path.join(PKG_ROOT, "data", "corpora", "ami_meta", "segments_sdm.csv")
MAN = os.path.join(PKG_ROOT, "data", "corpora", "manifest_ami.csv")
MEMBERS = ["none", "oa_aggr_a020", "sa_psm_aggr_a040", "sa_psm_mild_a025"]
N = EnglishTextNormalizer()
_V = {}


def ids(t):
    o = []
    for w in t:
        i = _V.get(w)
        if i is None:
            i = len(_V); _V[w] = i
        o.append(i)
    return o


def ref_labels(meeting, t0, t1, segs):
    """(kelimeler, ortusen_mi) — cumlenin ortusen sure orani >0.5 ise ortusen."""
    ss = sorted([(max(a, t0), min(b, t1), t) for a, b, t in segs[meeting] if b > t0 and a < t1])
    toks, lab = [], []
    for i, (a, b, t) in enumerate(ss):
        par = []
        for j, (c, d, _) in enumerate(ss):
            if j == i:
                continue
            x0, x1 = max(a, c), min(b, d)
            if x1 > x0:
                par.append((x0, x1))
        u = []
        for x0, x1 in sorted(par):
            if u and x0 <= u[-1][1]:
                u[-1] = (u[-1][0], max(u[-1][1], x1))
            else:
                u.append((x0, x1))
        oran = sum(y - x for x, y in u) / max(b - a, 1e-9)
        w = N(t).split()
        toks += w
        lab += [oran > 0.5] * len(w)
    return toks, lab


def sub_del_rate(ref_toks, lab, hyp_text, only_nonoverlap=True):
    """Ortusmeyen (veya tum) referans kelimeleri uzerinde degistirme+silme orani."""
    ri, hi = ids(ref_toks), ids(N(hyp_text).split())
    err = [0] * len(ri)
    for tag, i0, i1, _j0, _j1 in Levenshtein.opcodes(ri, hi):
        if tag in ("replace", "delete"):
            for k in range(i0, i1):
                err[k] = 1
    sel = [err[k] for k in range(len(ri)) if (not lab[k]) or (not only_nonoverlap)]
    return statistics.mean(sel) * 100 if sel else float("nan")


def ci(v, iters=10000, seed=20260910):
    rng = random.Random(seed)
    n = len(v)
    m = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return statistics.mean(v), m[int(0.025 * iters)], m[int(0.975 * iters) - 1]


def main():
    segs = collections.defaultdict(list)
    with open(META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segs[r["meeting_id"]].append((float(r["begin_time"]), float(r["end_time"]), r["text"]))
    win = {}
    with open(MAN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            win[r["meeting"]] = (float(r["window_start_s"]), float(r["window_end_s"]))

    H = collections.defaultdict(dict)
    with open(CSV_EN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            H[(r["cell"], r["basename"])][r["condition"]] = r["transcription"]

    print("=" * 92)
    print("DUYARLILIK ANALIZI — AMI hucreleri, YALNIZ ORTUSMEYEN kelimeler")
    print("(degistirme+silme orani; ekleme hatalari referans kelimesine baglanamadigi icin disarida)")
    print("=" * 92)

    for cell, ad in (("en_clean_ihm", "AMI kulaklik"), ("en_noisy_sdm", "AMI uzak mikrofon")):
        keys = sorted(k for k in H if k[0] == cell)
        print(f"\n### {ad}")
        print(f"{'kosul':<20}{'tam pencere':>14}{'ORTUSMESIZ':>13}{'fark (ham-a gore)':>20}{'%95 GA':>18}")
        print("-" * 88)
        base_full, base_no = {}, {}
        for k in keys:
            m = k[1].rsplit("_", 1)[0]
            toks, lab = ref_labels(m, *win[m], segs)
            base_full[k] = sub_del_rate(toks, lab, H[k]["none"], only_nonoverlap=False)
            base_no[k] = sub_del_rate(toks, lab, H[k]["none"], only_nonoverlap=True)
        print(f"{'none (taban)':<20}{statistics.mean(base_full.values()):>13.2f}%"
              f"{statistics.mean(base_no.values()):>12.2f}%")

        for cond in ("mild", "aggressive"):
            d = []
            for k in keys:
                m = k[1].rsplit("_", 1)[0]
                toks, lab = ref_labels(m, *win[m], segs)
                d.append(sub_del_rate(toks, lab, H[k][cond]) - base_no[k])
            mu, lo, hi = ci(d)
            print(f"{cond:<20}{'':>14}{'':>13}{mu:>+20.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}")

        # birlestirme
        d = []
        for k in keys:
            m = k[1].rsplit("_", 1)[0]
            toks, lab = ref_labels(m, *win[m], segs)
            f = rover_fuse([H[k][x] for x in MEMBERS], [f"{k[1]}|{x}" for x in MEMBERS],
                           pivot_index=0, override_margin=1)
            d.append(sub_del_rate(toks, lab, f) - base_no[k])
        mu, lo, hi = ci(d)
        print(f"{'BIRLESTIRME (4 uye)':<20}{'':>14}{'':>13}{mu:>+20.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
