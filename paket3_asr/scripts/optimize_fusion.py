"""
Y4b — Birlestirmenin kendisinin optimizasyonu: agirlikli oylama + uye aramasi.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Sonuclar         : docs/MAKALE_SONUCLAR_2026-09-13.md

GEREKCE
-------
Pozitif sonucu ureten mekanizma birlestirme, ama mekanizmanin kendisi hic
optimize edilmedi: duz cogunluk oyu + sabit marj kullanildi ve uye kumesi elle
secildi. ROVER literaturunde oylarin GUVENLE agirliklandirilmasi standarttir.

Bu betik YENI COZME GEREKTIRMEZ; eldeki transkriptlerle calisir.

NOT — guven verisi yok
----------------------
Whisper'in segment duzeyi avg_logprob degerleri CSV'de saklanmiyor (yalniz tam
metin var), dolayisiyla gercek guven agirliklandirmasi su an yapilamaz. Onun
yerine uyenin dev-5'te olculmus KALITESI agirlik olarak kullanilir. Bu daha
zayif ama tamamen ucretsiz bir yaklasimdir.

ASIRI UYUM DISIPLINI — en onemli nokta
--------------------------------------
Uye kumesi ve agirlik semasi YALNIZ dev-5'te secilir. Secilen yapilandirma
sonra DOKUNULMAMIS 5 kayitta degerlendirilir. Rapor edilen kazanc iddiasi
held-out sonuctur; dev-5 sonucu secim aracidir, kanit degil.

Bu ayrim olmadan 10 kayit uzerinde 200+ yapilandirma denemek korpusa asiri uyum
uretir ve sonucu gecersiz kilar.

Kullanim:
    python paket3_asr/scripts/optimize_fusion.py
    python paket3_asr/scripts/optimize_fusion.py --max-members 5 --quality-cap 4.0
"""

import argparse
import csv
import difflib
import itertools
import os
import statistics
import sys
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from analyze_fusion import load_hypotheses  # noqa: E402
from calculate_metrics import calculate_wer  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT, development_subset, load_experiment_config  # noqa: E402

REPORT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "fusion_optimization_report.csv")
REC10 = [
    "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
    "saha_04_vid_20220913_142911", "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez", "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi", "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
]
# Donmus yapilandirma — kiyas noktasi (docs/MAKALE_SONUCLAR bolum 6)
FROZEN = ("none", "oa_aggr_a020", "sa_psm_aggr_a040", "sa_psm_mild_a025")


def weighted_rover(hyps, weights, pivot_index=0, override_margin=0.0):
    """
    Agirlikli ROVER. analyze_fusion.rover_fuse ile ayni hizalama ve ayni ham
    capalama kurali; tek fark oylarin 1 yerine weights[i] ile sayilmasi.

    Agirliklar hepsi 1.0 verildiginde davranis duz oylamayla BIREBIR aynidir
    (--self-test bunu dogrular).
    """
    if not hyps:
        return ""
    if len(hyps) == 1:
        return hyps[0]

    pivot = hyps[pivot_index].split()
    if not pivot:
        return hyps[pivot_index]

    votes = [Counter() for _ in pivot]
    for i, h in enumerate(hyps):
        w = weights[i]
        cur = h.split()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pivot, cur).get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    votes[i1 + k][pivot[i1 + k]] += w
            elif tag == "replace":
                for k in range(i2 - i1):
                    rep = cur[j1 + k] if (j1 + k) < j2 else ""
                    votes[i1 + k][rep] += w
            elif tag == "delete":
                for k in range(i1, i2):
                    votes[k][""] += w

    out = []
    for k, c in enumerate(votes):
        if not c:
            out.append(pivot[k]); continue
        best_n = max(c.values())
        tied = [w_ for w_, n in c.items() if n == best_n]
        pv = c.get(pivot[k], 0.0)
        if pivot[k] in tied or best_n <= pv + override_margin:
            pick = pivot[k]
        else:
            pick = tied[0]
        if pick:
            out.append(pick)
    return " ".join(out)


def weight_schemes(members, quality):
    """
    Agirlik semalari. Hepsinde ham ses ('none') 0. sirada ve pivot olarak kalir.

    uniform : mevcut davranis (kiyas)
    quality : w = 1 / (1 + dev5_hasar_pp)  — daha iyi uye daha cok oy
    rawheavy: ham sese 2x agirlik, digerleri kaliteye gore
    """
    q = [max(0.0, quality.get(m, 0.0)) for m in members]
    out = {"uniform": [1.0] * len(members)}
    out["quality"] = [1.0 / (1.0 + v) for v in q]
    out["rawheavy"] = [(2.0 if m == "none" else 1.0 / (1.0 + v))
                       for m, v in zip(members, q)]
    return out


def fused_delta(hyp, model, records, refs, members, weights, margin):
    """Makro WER farki (birlesik - ham), puan. Pivot her zaman uyeler[0]='none'."""
    ds = []
    for b in records:
        cell = hyp[(model, b)]
        f = weighted_rover([cell[m][0] for m in members], weights,
                           pivot_index=0, override_margin=margin)
        ds.append(calculate_wer(refs[b], f) - cell["none"][1])
    return statistics.mean(ds) * 100.0


def main():
    ap = argparse.ArgumentParser(description="Y4b — birlestirme optimizasyonu")
    ap.add_argument("--model", default="faster-whisper-large-v3-int8")
    ap.add_argument("--max-members", type=int, default=5)
    ap.add_argument("--quality-cap", type=float, default=4.5,
                    help="dev-5 hasari bu degerden buyuk kosullar aday degildir")
    ap.add_argument("--top-n", type=int, default=20,
                    help="dev-5'teki ilk N yapilandirma held-out'ta da olculur")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config()
    hyp = load_hypotheses()
    M = args.model
    dev5 = development_subset(cfg)
    held = [b for b in REC10 if b not in dev5]

    refs = {b: normalize_turkish_asr(
        open(os.path.join(PKG_ROOT, "data", "references", f"{b}.txt"), encoding="utf-8").read().strip())
        for b in REC10}

    # Aday havuzu: TUM 10 kayitta mevcut kosullar (held-out dogrulamasi sart).
    pool = [c for c in hyp[(M, REC10[0])]
            if all(c in hyp.get((M, b), {}) for b in REC10)]
    quality = {c: statistics.mean(
        (hyp[(M, b)][c][1] - hyp[(M, b)]["none"][1]) * 100.0 for b in dev5) for c in pool}
    cand = sorted([c for c in pool if c != "none" and quality[c] <= args.quality_cap],
                  key=lambda c: quality[c])

    print("=" * 88)
    print(f"BIRLESTIRME OPTIMIZASYONU   |   {M}")
    print(f"secim: dev-5 ({len(dev5)} kayit)   dogrulama: DOKUNULMAMIS {len(held)} kayit")
    print("=" * 88)
    print(f"aday havuzu ({len(cand)} kosul, dev-5 hasari <= {args.quality_cap:+.1f} pp):")
    for c in cand:
        print(f"   {c:<26}{quality[c]:+6.2f} pp")

    if args.self_test:
        from analyze_fusion import rover_fuse
        b = dev5[0]
        cell = hyp[(M, b)]
        ms = list(FROZEN)
        a = rover_fuse([cell[m][0] for m in ms], [f"{M}|{b}|{m}" for m in ms],
                       pivot_index=0, override_margin=1)
        c_ = weighted_rover([cell[m][0] for m in ms], [1.0] * len(ms),
                            pivot_index=0, override_margin=1.0)
        print(f"\n[OZ-TEST] duz oylama ile birebir ayni mi: {'EVET' if a == c_ else 'HAYIR'}")
        return 0

    # --- Kiyas: donmus yapilandirma ---
    base_dev = fused_delta(hyp, M, dev5, refs, list(FROZEN), [1.0] * 4, 1.0)
    base_held = fused_delta(hyp, M, held, refs, list(FROZEN), [1.0] * 4, 1.0)
    print(f"\nKIYAS — donmus 4 uye, duz oylama, marj 1")
    print(f"   dev-5   {base_dev:+.2f} pp        held-out {base_held:+.2f} pp")

    # --- Arama: dev-5 uzerinde ---
    rows = []
    print(f"\nARAMA (yalniz dev-5 uzerinde secim yapiliyor)...")
    for k in range(2, args.max_members):
        for combo in itertools.combinations(cand, k):
            members = ["none"] + list(combo)
            for wname, w in weight_schemes(members, quality).items():
                # Marj kurali: k iyilestirilmis uye icin ~ (k-1)/2
                for margin in sorted({0.0, (len(combo) - 1) / 2.0, len(combo) / 2.0}):
                    d = fused_delta(hyp, M, dev5, refs, members, w, margin)
                    rows.append((d, tuple(members), wname, margin))
    rows.sort(key=lambda r: r[0])

    print(f"toplam {len(rows)} yapilandirma denendi. dev-5'te en iyi 10:")
    print(f"{'dev-5':>8}  {'agirlik':<9}{'marj':>5}   uyeler")
    print("-" * 88)
    for d, ms, wn, mg in rows[:10]:
        print(f"{d:>+8.2f}  {wn:<9}{mg:>5.1f}   {', '.join(m for m in ms if m != 'none')}")

    # --- Dogrulama: ilk N yapilandirmanin HEPSI dokunulmamis 5 kayitta ---
    #
    # Neden yalniz 1. degil: donmus yapilandirma zamaninda TUM 10 kayit gorulerek
    # secilmisti, dolayisiyla onun held-out sayisi temiz degil ve tek bir
    # karsilastirma yanli olur. Ilk N'in tamami olculunce asil soru cevaplanir:
    # dev-5 siralamasi held-out basarisini ONGORUYOR MU?
    topn = rows[:args.top_n]
    print("-" * 88)
    print(f"DOGRULAMA — dev-5'teki ilk {len(topn)} yapilandirma, DOKUNULMAMIS {len(held)} kayitta")
    print(f"{'dev-5':>8}{'held-out':>10}  {'agirlik':<9}{'marj':>5}   uyeler")
    print("-" * 88)
    hv = []
    for d, ms, wn, mg in topn:
        w = weight_schemes(list(ms), quality)[wn]
        hd = fused_delta(hyp, M, held, refs, list(ms), w, mg)
        hv.append((d, hd, ms, wn, mg))
        print(f"{d:>+8.2f}{hd:>+10.2f}  {wn:<9}{mg:>5.1f}   {', '.join(m for m in ms if m != 'none')}")

    ds = [x[0] for x in hv]
    hs = [x[1] for x in hv]
    mdev, mhel = statistics.mean(ds), statistics.mean(hs)
    num = sum((a - mdev) * (b - mhel) for a, b in zip(ds, hs))
    den = (sum((a - mdev) ** 2 for a in ds) ** 0.5) * (sum((b - mhel) ** 2 for b in hs) ** 0.5)
    r = num / den if den > 0 else float("nan")

    best_dev = hv[0]
    best_held = min(hv, key=lambda x: x[1])

    print("-" * 88)
    print(f"dev-5 <-> held-out korelasyonu : r = {r:+.3f}   (n={len(hv)})")
    print(f"dev-5'in 1.'si held-out'ta     : {best_dev[1]:+.2f} pp")
    print(f"held-out'un gercek en iyisi    : {best_held[1]:+.2f} pp  "
          f"(dev-5'te {best_held[0]:+.2f} pp ile {ds.index(best_held[0]) + 1}. sirada)")
    print(f"donmus yapilandirma            : dev-5 {base_dev:+.2f}  held-out {base_held:+.2f}")
    print()
    if r > 0.5:
        print("SONUC: dev-5 siralamasi held-out'u ONGORUYOR — arama guvenilir.")
    else:
        print("SONUC: dev-5 siralamasi held-out'u ONGORMUYOR.")
        print("       5 kayitlik bir gelistirme kumesi uzerinde yapilandirma aramak,")
        print("       genellesen bir kazanc uretmiyor; secilen sey gurultudur.")
        print("       Bu, korpus buyutmenin neden en etkili adim oldugunu dogrudan gosterir.")
    print("=" * 88)
    best_d, best_ms, best_wn, best_mg = rows[0]
    held_d = best_dev[1]

    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["model_id", "members", "weight_scheme", "margin",
                     "dev5_delta_pp", "heldout_delta_pp", "is_selected", "is_frozen_baseline"])
        wr.writerow([M, "+".join(FROZEN), "uniform", 1.0,
                     f"{base_dev:.4f}", f"{base_held:.4f}", 0, 1])
        hmap = {(x[2], x[3], x[4]): x[1] for x in hv}
        for d, ms, wn, mg in rows[:50]:
            sel = 1 if (ms, wn, mg) == (best_ms, best_wn, best_mg) else 0
            h_ = hmap.get((ms, wn, mg))
            wr.writerow([M, "+".join(ms), wn, mg, f"{d:.4f}",
                         f"{h_:.4f}" if h_ is not None else "", sel, 0])
    print(f"rapor -> {os.path.relpath(REPORT_CSV, PKG_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
