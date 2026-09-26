"""
D3 — Kayit-birakmali secim: "secim dahil" durust tahmin (Ek-2 Rev B).

SORUN
-----
Simdiye kadar bildirdigimiz -0,40 pp, YAPILANDIRMA ZATEN BELLIYKEN olculmus bir
degerdir. Ama gercek hayatta yapilandirmayi da secmek gerekir ve secim bedavaya
gelmez: 6e'de olctuk ki bu korpusta secim gurultu uretiyor.

Durust soru sudur: **yapilandirmayi kendim secmek zorunda olsaydim, gormedigim
bir kayitta ne kazanirdim?**

YONTEM — ic ice (nested) degerlendirme
--------------------------------------
Her kayit r icin:
  1. DIGER 9 kayit uzerinde tum aday yapilandirmalar degerlendirilir
  2. En iyisi secilir (r'ye hic bakilmadan)
  3. O yapilandirma r uzerinde uygulanir ve sonuc kaydedilir
10 katmanin ortalamasi, SECIM MALIYETI DAHIL durust tahmindir.

HIZ NOTU
--------
(yapilandirma, kayit) ciftinin birlesik WER'i katmandan bagimsizdir; bir kez
hesaplanip onbellege alinir. Aksi halde ayni hesap 10 kez tekrarlanirdi.
WER icin rapidfuzz kullanilir: saf Python Levenshtein 2000 kelimelik dizilerde
cift basina saniyeler suruyor ve bu analiz binlerce cift gerektiriyor.

Kullanim:
    python paket3_asr/scripts/analyze_nested_selection.py
"""

import csv
import itertools
import os
import statistics
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from rapidfuzz.distance import Levenshtein  # noqa: E402

from analyze_fusion import load_hypotheses, rover_fuse  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT  # noqa: E402
from evaluate_fusion_final import REC10  # noqa: E402

REPORT = os.path.join(PKG_ROOT, "outputs", "metrics", "nested_selection_report.csv")
FROZEN = ("none", "oa_aggr_a020", "sa_psm_aggr_a040", "sa_psm_mild_a025")

_V = {}


def _ids(toks):
    out = []
    for t in toks:
        i = _V.get(t)
        if i is None:
            i = len(_V); _V[t] = i
        out.append(i)
    return out


def wer(ref, hyp):
    r = _ids(ref.split())
    return Levenshtein.distance(r, _ids(hyp.split())) / max(len(r), 1)


def main():
    M = "faster-whisper-large-v3-int8"
    h = load_hypotheses()
    refs = {b: normalize_turkish_asr(
        open(os.path.join(PKG_ROOT, "data", "references", f"{b}.txt"), encoding="utf-8").read().strip())
        for b in REC10}
    raw = {b: h[(M, b)]["none"][1] for b in REC10}

    pool = [c for c in h[(M, REC10[0])]
            if c != "none" and all(c in h.get((M, b), {}) for b in REC10)]
    # Aday havuzu yapisal olarak tum tam-10 kapsamli kosullardir. KALITE ESIGI
    # ise HER KATMANDA O KATMANIN 9 KAYDINDAN hesaplanir (asagida): esigi tum 10
    # kayittan hesaplamak, birakilan kaydin secime sizmasi olurdu.
    cand = sorted(pool, key=lambda c: statistics.mean(
        (h[(M, b)][c][1] - raw[b]) * 100 for b in REC10))

    def quality_on(records, c):
        return statistics.mean((h[(M, b)][c][1] - raw[b]) * 100 for b in records)

    configs = []
    for k in (2, 3, 4):
        for combo in itertools.combinations(cand, k):
            for mg in (0, 1, 2):
                configs.append((("none",) + combo, mg))

    print("=" * 92)
    print("D3 — KAYIT-BIRAKMALI SECIM (secim maliyeti dahil durust tahmin)")
    print("=" * 92)
    print(f"aday kosul: {len(cand)}   yapilandirma: {len(configs)}   kayit: {len(REC10)}")
    print(f"havuz: {', '.join(cand)}")
    print("hesaplaniyor...", flush=True)

    # (yapilandirma, kayit) -> fark (pp). Katmandan bagimsiz, bir kez hesaplanir.
    cache = {}
    for idx, (mem, mg) in enumerate(configs, 1):
        for b in REC10:
            f = rover_fuse([h[(M, b)][m][0] for m in mem],
                           [f"{M}|{b}|{m}" for m in mem], pivot_index=0, override_margin=mg)
            cache[(idx, b)] = (wer(refs[b], f) - raw[b]) * 100.0
        if idx % 40 == 0:
            print(f"  {idx}/{len(configs)}", flush=True)

    # Kayit-birakmali secim. Kalite esigi de katman ICINDE uygulanir.
    rows, picks = [], []
    for b in REC10:
        others = [x for x in REC10 if x != b]
        izinli = {c for c in cand if quality_on(others, c) <= 4.5}
        uygun = [i for i, (mem, mg) in enumerate(configs, 1)
                 if all(m == "none" or m in izinli for m in mem)]
        best = min(uygun, key=lambda i: statistics.mean(cache[(i, o)] for o in others))
        mem, mg = configs[best - 1]
        rows.append(cache[(best, b)])
        picks.append((b, "+".join(m for m in mem if m != "none"), mg,
                      statistics.mean(cache[(best, o)] for o in others), cache[(best, b)]))

    # Uyeler kalite sirasina gore uretildigi icin siralama FROZEN ile ayni degil;
    # karsilastirma kume olarak yapilir.
    frozen_idx = None
    for i, (mem, mg) in enumerate(configs, 1):
        if set(mem) == set(FROZEN) and mg == 1:
            frozen_idx = i
    frozen = [cache[(frozen_idx, b)] for b in REC10] if frozen_idx else None
    oracle_i = min(range(1, len(configs) + 1),
                   key=lambda i: statistics.mean(cache[(i, b)] for b in REC10))

    print()
    print(f"{'birakilan kayit':<45}{'secilen (9 kayitta)':<44}{'marj':>5}{'ic':>8}{'dis':>8}")
    print("-" * 92)
    for b, mem, mg, inn, out in picks:
        print(f"{b:<45}{mem[:42]:<44}{mg:>5}{inn:>+8.2f}{out:>+8.2f}")
    print("-" * 92)
    print(f"SECIM DAHIL durust tahmin      : {statistics.mean(rows):+.3f} pp")
    if frozen:
        print(f"donmus yapilandirma (secim yok) : {statistics.mean(frozen):+.3f} pp")
    om, og = configs[oracle_i - 1]
    print(f"oracle (10 kayitta en iyi)      : "
          f"{statistics.mean(cache[(oracle_i, b)] for b in REC10):+.3f} pp"
          f"   [{'+'.join(m for m in om if m != 'none')}, marj {og}]")
    print(f"secimin bedeli (durust - oracle) : "
          f"{statistics.mean(rows) - statistics.mean(cache[(oracle_i, b)] for b in REC10):+.3f} pp")
    print()
    farkli = len({(p[1], p[2]) for p in picks})
    print(f"10 katmanda {farkli} FARKLI yapilandirma secildi.")
    if farkli > 1:
        print("  -> secim kararsiz: birakilan kayit degisince kazanan degisiyor.")
        print("     6e'deki bulgunun ic ice degerlendirmedeki karsiligi budur.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["held_out_record", "selected_members", "margin",
                    "inner_9rec_delta_pp", "outer_heldout_delta_pp"])
        for b, mem, mg, inn, out in picks:
            w.writerow([b, mem, mg, f"{inn:.4f}", f"{out:.4f}"])
    print(f"\nrapor -> {os.path.relpath(REPORT, PKG_ROOT)}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
