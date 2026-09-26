"""
Alfa egrileri ve oracle boslugu analizi.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Literatur taramasi: docs/MAKALE_LITERATUR_TARAMASI_ASR_ON_ISLEME_2026-09-10.md

Uc soruyu yanitlar:

  S1  Alfa egrisi nerede en iyi? Literatur Whisper icin U bicimli egri ve
      alpha* ~= 0.25 bildiriyor (arXiv 2607.11157). Bizim Turkce/tibbi
      verimizde optimum nerede ve ham sesi geciyor mu?

  S2  Sinyal alani karisimi (Y1 oa) ile fazi koruyan spektral alan karisimi
      (Y1b psm/mex) AYNI alfada nasil kiyaslaniyor? (alpha=0.20 ve 0.40
      her iki ailede de uretildigi icin dogrudan karsilastirilabilir.)

  S3  ORACLE BOSLUGU — makalenin ana katki olcutu. Alfa kayit basina
      secilebilse ne kazanilir? Literatur %20-30 bagil kazanc oldugunu ama
      gerceklenemedigini bildiriyor. Sabit-en-iyi ile oracle arasindaki fark,
      Y2'nin (kesit bazli uyarlamali secim) hedefledigi boslugun ta kendisidir.

Kullanim:
    python paket3_asr/scripts/analyze_alpha_curves.py \
        --extra-csv paket3_asr/outputs/metrics_baseline/benchmark_results_baseline_refv2.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_conditional_gain import load_results, merge_supplement  # noqa: E402
from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402

DEFAULT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results.csv")
REPORT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "alpha_curve_report.csv")

# oa_mild_a020 / sa_psm_aggr_a025
RE_OA = re.compile(r"^oa_(?P<enh>[a-z]+)_a(?P<pct>\d{3})$")
RE_SA = re.compile(r"^sa_(?P<variant>psm|mex)_(?P<enh>[a-z]+)_a(?P<pct>\d{3})$")

BASE_ANCHORS = {"mild": "mild", "aggr": "aggressive"}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def parse_family(condition: str):
    """Kosul kimligini (aile, varyant, iyilestirici, alfa) olarak ayristirir."""
    m = RE_OA.match(condition)
    if m:
        return ("oa", "mix", m.group("enh"), int(m.group("pct")) / 100.0)
    m = RE_SA.match(condition)
    if m:
        return ("sa", m.group("variant"), m.group("enh"), int(m.group("pct")) / 100.0)
    return None


def collect_curves(results: dict):
    """
    (model, aile, varyant, iyilestirici) -> {alfa: {kayit: WER}}
    alpha=0.0 capasi 'none' kosulundan eklenir.
    """
    curves = defaultdict(lambda: defaultdict(dict))
    for (model_id, condition), per_rec in results.items():
        parsed = parse_family(condition)
        if parsed is None:
            continue
        family, variant, enh, alpha = parsed
        curves[(model_id, family, variant, enh)][alpha].update(per_rec)

    # alpha=0.0 = ham ses ('none'), her egriye capa olarak eklenir
    for key in list(curves):
        model_id = key[0]
        none_rows = results.get((model_id, "none"), {})
        if none_rows:
            curves[key][0.0].update(none_rows)
    return curves


def analyse_curve(alpha_map: dict):
    """
    Bir egri icin esit kapsamli analiz.
    Doner: (kayitlar, alfalar, makro{alfa}, oracle_makro, oracle_alfa{kayit})
    """
    alphas = sorted(alpha_map)
    if not alphas:
        return [], [], {}, float("nan"), {}
    # Tum alfalarda ortak olan kayitlar (esit kapsam sarti)
    shared = set(alpha_map[alphas[0]])
    for a in alphas[1:]:
        shared &= set(alpha_map[a])
    shared = sorted(shared)
    if not shared:
        return [], alphas, {}, float("nan"), {}

    macro = {a: _mean([alpha_map[a][r] for r in shared]) for a in alphas}
    best_alpha_per_record = {}
    oracle_vals = []
    for r in shared:
        pairs = [(alpha_map[a][r], a) for a in alphas]
        v, a = min(pairs)
        best_alpha_per_record[r] = a
        oracle_vals.append(v)
    return shared, alphas, macro, _mean(oracle_vals), best_alpha_per_record


def main():
    ap = argparse.ArgumentParser(description="Alfa egrileri ve oracle boslugu")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--extra-csv", default=None,
                    help="Tamamlayici CSV (yalniz eksik anahtarlar icin)")
    ap.add_argument("--run-purpose", default="main_benchmark")
    args = ap.parse_args()

    load_experiment_config()  # dogrulama amacli
    results = load_results(args.csv, run_purpose=args.run_purpose)
    if args.extra_csv:
        extra = load_results(args.extra_csv, run_purpose=args.run_purpose)
        results, n_add = merge_supplement(results, extra)
        print(f"Tamamlayici CSV: {n_add} eksik anahtar eklendi")

    curves = collect_curves(results)
    if not curves:
        print("[UYARI] Alfa kosulu bulunamadi (oa_* / sa_* satiri yok).")
        return 0

    rows = []
    print("=" * 122)
    print("ALFA EGRILERI — makro WER (%), esit kapsamli")
    print("alpha=0.00 = ham ses ('none'). * = egrinin en iyisi.")
    print("=" * 122)

    for key in sorted(curves):
        model_id, family, variant, enh = key
        shared, alphas, macro, oracle, best_per_rec = analyse_curve(curves[key])
        if not shared:
            continue

        label = f"{family}/{variant}/{enh}"
        base = macro.get(0.0, float("nan"))
        best_a = min(macro, key=lambda a: macro[a])
        best_v = macro[best_a]

        print(f"\n### {model_id}   |   {label}   |   N={len(shared)} kayit")
        hdr = "".join(f"{a:>9.2f}" for a in alphas)
        print(f"    {'alpha':<10}{hdr}{'ORACLE':>11}")
        cells = "".join(
            f"{macro[a]*100:>8.2f}" + ("*" if a == best_a else " ") for a in alphas
        )
        print(f"    {'makroWER':<10}{cells}{oracle*100:>10.2f}%")

        d_base = "".join(
            f"{(macro[a]-base)*100:>+9.2f}" if a != 0.0 else f"{'—':>9}" for a in alphas
        )
        print(f"    {'ham fark':<10}{d_base}{(oracle-base)*100:>+10.2f}")

        gap = best_v - oracle
        rel = (gap / best_v * 100.0) if best_v > 0 else float("nan")
        print(f"    en iyi sabit alpha = {best_a:.2f} ({best_v*100:.2f}%) | "
              f"ham ses = {base*100:.2f}% | "
              f"{'HAM SESI GECIYOR' if best_v < base else 'ham sesi gecemiyor'}")
        print(f"    ORACLE BOSLUGU     = {gap*100:+.2f} pp (bagil %{rel:.1f}) "
              f"-> Y2'nin hedefledigi bosluk")
        # Kayit basina en iyi alfa dagilimi: sabit secimin neden kaybettigini gosterir
        dist = defaultdict(list)
        for r, a in best_per_rec.items():
            dist[a].append(r)
        parts = [f"alpha={a:.2f}: {len(v)}" for a, v in sorted(dist.items())]
        print(f"    kayit basina en iyi alpha dagilimi -> {', '.join(parts)}")

        rows.append({
            "model_id": model_id, "family": family, "variant": variant, "enhancer": enh,
            "n_records": len(shared),
            "raw_macro_wer": f"{base:.6f}",
            "best_fixed_alpha": f"{best_a:.2f}",
            "best_fixed_macro_wer": f"{best_v:.6f}",
            "beats_raw": "evet" if best_v < base else "hayir",
            "oracle_macro_wer": f"{oracle:.6f}",
            "oracle_gap_pp": f"{gap*100:.4f}",
            "oracle_gap_rel_pct": f"{rel:.2f}",
            "n_distinct_best_alphas": len(dist),
        })

    if rows:
        os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
        with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print("\n" + "=" * 122)
        beats = [r for r in rows if r["beats_raw"] == "evet"]
        print(f"Egri sayisi                   : {len(rows)}")
        print(f"Sabit alfayla ham sesi gecen  : {len(beats)}")
        for r in sorted(beats, key=lambda x: float(x["best_fixed_macro_wer"])):
            print(f"   {r['model_id']} | {r['family']}/{r['variant']}/{r['enhancer']}: "
                  f"ham %{float(r['raw_macro_wer'])*100:.2f} -> "
                  f"alpha={r['best_fixed_alpha']} %{float(r['best_fixed_macro_wer'])*100:.2f}")
        if not beats:
            print("   (Hicbir egri sabit alfayla ham sesi gecmedi.)")
        gaps = [float(r["oracle_gap_pp"]) for r in rows]
        if gaps:
            print(f"Oracle boslugu (pp)           : ort {sum(gaps)/len(gaps):+.2f} | "
                  f"en buyuk {max(gaps):+.2f}")
        print(f"Rapor: {os.path.relpath(REPORT_CSV, os.path.dirname(PKG_ROOT))}")
        print("=" * 122)
    return 0


if __name__ == "__main__":
    sys.exit(main())
