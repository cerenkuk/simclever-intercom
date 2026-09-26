"""
Birlestirme yapılandirmasinin NIHAI degerlendirmesi — dort kabul olcutu + katman.

Makale protokolu: docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md (Bolum 7)
Sonuclar        : docs/MAKALE_SONUCLAR_2026-09-13.md

Secilen yapilandirma (tam 10 kayit, Large-v3'te dogrulandi):
    uyeler : none + oa_aggr_a020 + sa_psm_aggr_a040 + sa_psm_mild_a025
    pivot  : ham capali ('none' omurga)
    marj   : 1   (kural: k iyilestirilmis uye icin marj ~ (k-1)/2)

Bu betik ayni yapilandirmayi herhangi bir modelde degerlendirir ve dort olcutu
uygular. Turbo icin KATMANLI rapor kritiktir: o modelin iki dogrulanmis patolojik
dongu kaydi vardir ve kazancin oradan mi yoksa saglikli kayitlardan mi geldigi
ayrilmalidir (Olcut d).

Kullanim:
    python paket3_asr/scripts/evaluate_fusion_final.py
    python paket3_asr/scripts/evaluate_fusion_final.py --model faster-whisper-large-v3-turbo-int8
    python paket3_asr/scripts/evaluate_fusion_final.py --margin 0 --detection-pivot
"""

import argparse
import csv
import os
import random
import statistics
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from analyze_fusion import load_hypotheses, rover_fuse  # noqa: E402
from calculate_metrics import calculate_wer  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402

REC10 = [
    "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
    "saha_04_vid_20220913_142911", "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez", "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi", "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
]
MEMBERS = ["none", "oa_aggr_a020", "sa_psm_aggr_a040", "sa_psm_mild_a025"]
DETECT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "twopass_detection_report.csv")


def loop_records(cfg, model_id):
    """Insan dinlemesiyle dogrulanmis patolojik dongu kayitlari (model basina)."""
    blk = (cfg.get("pathological_loop_records") or {}).get(model_id) or {}
    out = set()
    for _c, names in blk.items():
        out.update(names or [])
    return out


def detected_collapse(model_id):
    """Y2'nin cokus tespit ettigi (model, kayit) ciftleri."""
    out = set()
    if not os.path.exists(DETECT_CSV):
        return out
    with open(DETECT_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("model_id") == model_id and int(r.get("n_collapsed", 0)) > 0:
                out.add(r["basename"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Birlestirmenin nihai degerlendirmesi")
    ap.add_argument("--model", default="faster-whisper-large-v3-int8")
    ap.add_argument("--members", nargs="*", default=MEMBERS)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--detection-pivot", action="store_true",
                    help="Cokus tespit edilen kayitta merkez pivot kullan (Y2 ⊕ Y4)")
    args = ap.parse_args()

    cfg = load_experiment_config()
    s = cfg.get("statistics") or {}
    iters = int(s.get("bootstrap_iterations", 10000))
    seed = int(s.get("random_seed", 20260910))

    hyp = load_hypotheses()
    M = args.model
    missing = [(b, c) for b in REC10 for c in args.members if c not in hyp.get((M, b), {})]
    if missing:
        print(f"[EKSIK] {len(missing)} hucre yok — degerlendirme yapilamaz:")
        for b, c in missing:
            print(f"   {b:<45}{c}")
        return 1

    refs = {b: normalize_turkish_asr(
        open(os.path.join(PKG_ROOT, "data", "references", f"{b}.txt"), encoding="utf-8").read().strip())
        for b in REC10}

    cok = detected_collapse(M) if args.detection_pivot else set()
    loops = loop_records(cfg, M)

    d, base = {}, {}
    for b in REC10:
        h = hyp[(M, b)]
        pi = None if b in cok else 0
        f = rover_fuse([h[c][0] for c in args.members],
                       [f"{M}|{b}|{c}" for c in args.members],
                       pivot_index=pi, override_margin=args.margin)
        base[b] = h["none"][1]
        d[b] = calculate_wer(refs[b], f) - base[b]

    vals = [d[b] for b in REC10]
    ham = statistics.mean(base.values()) * 100
    macro = statistics.mean(vals) * 100

    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo, hi = means[int(0.025 * iters)] * 100, means[int(0.975 * iters) - 1] * 100
    loo = [statistics.mean([d[x] for x in REC10 if x != dr]) for dr in REC10]

    clean = [b for b in REC10 if b not in loops]
    cd = statistics.mean(d[b] for b in clean) * 100 if clean else float("nan")
    ld = statistics.mean(d[b] for b in loops) * 100 if loops else float("nan")

    print("=" * 86)
    print(f"BIRLESTIRME — NIHAI DEGERLENDIRME   |   {M}")
    print(f"Uyeler: {', '.join(args.members)}")
    print(f"Pivot : {'tespit gudumlu' if args.detection_pivot else 'ham capali'}   marj: {args.margin}")
    print("=" * 86)
    print(f"{'kayit':<45}{'ham':>8}{'birlesik':>10}{'fark':>9}")
    print("-" * 86)
    for b in sorted(REC10, key=lambda x: d[x]):
        tag = " [dongu]" if b in loops else ""
        print(f"{b + tag:<45}{base[b]*100:>7.2f}%{(base[b]+d[b])*100:>9.2f}%{d[b]*100:>+9.2f}")

    print("-" * 86)
    print(f"ham taban      : %{ham:.2f}")
    print(f"birlesik       : %{ham + macro:.2f}   ({macro:+.2f} pp)")
    print(f"%95 GA         : [{lo:+.2f}, {hi:+.2f}]")
    print(f"leave-one-out  : [{min(loo)*100:+.2f}, {max(loo)*100:+.2f}]  yonu koruyan {sum(1 for v in loo if v < 0)}/{len(loo)}")
    print(f"katman         : dongusuz {len(clean)} = {cd:+.2f} pp" +
          (f"   |   dongulu {len(loops)} = {ld:+.2f} pp" if loops else "   (bu modelde dogrulanmis dongu yok)"))
    print(f"kayit sayimi   : iyilesen {sum(1 for v in vals if v < 0)}  esit {sum(1 for v in vals if v == 0)}  kotulesen {sum(1 for v in vals if v > 0)}")

    a = macro < 0
    b_ = hi < 0
    c_ = all(v < 0 for v in loo)
    d_ = (cd < 0) if cd == cd else False
    print()
    print("DORT KABUL OLCUTU")
    print(f"   (a) tabani geciyor              : {'GECTI' if a else 'KALDI'}")
    print(f"   (b) GA sifiri kapsamiyor        : {'GECTI' if b_ else 'KALDI'}")
    print(f"   (c) leave-one-out kararli       : {'GECTI' if c_ else 'KALDI'}")
    print(f"   (d) dongusuz katmanda kazanc    : {'GECTI' if d_ else 'KALDI'}")
    print(f"   -> SONUC: {'DORT OLCUT DE GECTI' if (a and b_ and c_ and d_) else 'KABUL EDILMEDI'}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.exit(main())
