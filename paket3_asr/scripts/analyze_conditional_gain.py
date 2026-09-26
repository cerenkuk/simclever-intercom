"""
Kosullu kazanc analizi — Bolum 7 kabul olcutlerinin uygulanmasi.

Makale protokolu: docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md

Her (model, kosul) icin 'none' tabanina karsi kazanci degerlendirir ve
protokolun DORT kabul olcutunu ayri ayri raporlar:

  (a) Makro WER'de tabani gecmek
  (b) Eslestirilmis bootstrap %95 GA'sinin sifiri KAPSAMAMASI
  (c) Dosya-birakmali (leave-one-out) 10 katmanda da yonun korunmasi
  (d) Kazancin en az bir kisminin DONGUSUZ kayitlardan gelmesi

Dorduncu olcut kritiktir: Turbo/mild vakasi (makro -2.56 pp 'kazanc', ama
dongusuz 7 kayitta +2.41 pp KAYIP) tam bu testte duser.

Kullanim:
    python paket3_asr/scripts/analyze_conditional_gain.py
    python paket3_asr/scripts/analyze_conditional_gain.py --csv <yol> --baseline-condition none
    python paket3_asr/scripts/analyze_conditional_gain.py --model faster-whisper-large-v3-turbo-int8
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402

DEFAULT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results.csv")
REPORT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "conditional_gain_report.csv")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load_results(csv_path: str, run_purpose: str = "main_benchmark") -> dict:
    """(model_id, condition) -> {basename: WER}"""
    out = defaultdict(dict)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sonuc CSV bulunamadi: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("run_purpose") != run_purpose:
                continue
            try:
                wer = float(r["WER"])
            except (KeyError, TypeError, ValueError):
                continue
            out[(r["model_id"], r["condition"])][r["basename"]] = wer
    return out


def merge_supplement(primary: dict, extra: dict) -> tuple:
    """
    'extra' satirlarini YALNIZCA primary'de bulunmayan (model, kosul, kayit)
    anahtarlari icin tamamlayici olarak ekler. Primary satirlari asla ezilmez.

    Mesru kullanim: Kademe 0 kapisi hattin baseline'i BIREBIR yeniden urettigini
    dogruladiktan sonra (24/24 satir, maks |dWER|=0.000 pp), temel kosullari
    yeniden cozmek yerine dogrulanmis baseline satirlari kullanilir.
    Doner: (birlesik_sozluk, tamamlanan_anahtar_sayisi)
    """
    merged = {k: dict(v) for k, v in primary.items()}
    added = 0
    for key, per_record in extra.items():
        target = merged.setdefault(key, {})
        for basename, wer in per_record.items():
            if basename not in target:
                target[basename] = wer
                added += 1
    return merged, added


def paired_bootstrap_ci(deltas_by_record: dict, iterations: int, conf: float, seed: int):
    """
    Kayit duzeyinde eslestirilmis bootstrap.
    deltas_by_record: {basename: (wer_kosul - wer_taban)}
    Doner: (nokta_tahmin, alt_sinir, ust_sinir)
    Nokta tahmin, makro ortalama delta (kayit basina esit agirlik).
    """
    names = sorted(deltas_by_record)
    vals = [deltas_by_record[n] for n in names]
    n = len(vals)
    point = _mean(vals)
    if n < 2:
        return point, float("nan"), float("nan")

    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_i = int((1.0 - conf) / 2.0 * iterations)
    hi_i = int((1.0 + conf) / 2.0 * iterations) - 1
    lo_i = max(0, min(lo_i, iterations - 1))
    hi_i = max(0, min(hi_i, iterations - 1))
    return point, means[lo_i], means[hi_i]


def leave_one_out_deltas(deltas_by_record: dict):
    """
    Her kaydi sirayla disarida birakip makro delta'yi yeniden hesaplar.
    Doner: (min_delta, max_delta, yonu_koruyan_katman_sayisi, toplam_katman)
    """
    names = sorted(deltas_by_record)
    n = len(names)
    if n < 2:
        return float("nan"), float("nan"), 0, 0
    full = _mean([deltas_by_record[x] for x in names])
    sign = -1 if full < 0 else 1
    loo = []
    for drop in names:
        kept = [deltas_by_record[x] for x in names if x != drop]
        loo.append(_mean(kept))
    consistent = sum(1 for d in loo if (d < 0 and sign < 0) or (d > 0 and sign > 0))
    return min(loo), max(loo), consistent, n


def loop_records_for_model(cfg: dict, model_id: str) -> set:
    """Model icin dogrulanmis patolojik dongu kayitlarinin birlesimi."""
    block = (cfg.get("pathological_loop_records") or {}).get(model_id) or {}
    out = set()
    for _cond, names in block.items():
        out.update(names or [])
    return out


def evaluate(cfg, results, model_id, condition, baseline_condition, stats_cfg):
    """Bir (model, kosul) icin dort olcutu degerlendirir. Yetersiz veri -> None."""
    base = results.get((model_id, baseline_condition), {})
    cand = results.get((model_id, condition), {})
    shared = sorted(set(base) & set(cand))
    if len(shared) < 2:
        return None

    deltas = {n: cand[n] - base[n] for n in shared}
    base_macro = _mean([base[n] for n in shared])
    cand_macro = _mean([cand[n] for n in shared])

    point, lo, hi = paired_bootstrap_ci(
        deltas,
        iterations=stats_cfg["bootstrap_iterations"],
        conf=stats_cfg["confidence_level"],
        seed=stats_cfg["random_seed"],
    )
    loo_min, loo_max, loo_ok, loo_n = leave_one_out_deltas(deltas)

    loops = loop_records_for_model(cfg, model_id) & set(shared)
    clean = [n for n in shared if n not in loops]
    loop_delta = _mean([deltas[n] for n in sorted(loops)]) if loops else float("nan")
    clean_delta = _mean([deltas[n] for n in clean]) if clean else float("nan")

    # Anlamlilik siniflandirmasi (makale tablolarinda dogrudan raporlanabilir)
    if lo != lo or hi != hi:  # NaN
        significance = "BELIRSIZ"
    elif hi < 0:
        significance = "IYILESME"      # GA tamamen negatif: anlamli iyilesme
    elif lo > 0:
        significance = "KOTULESME"     # GA tamamen pozitif: anlamli kotulesme
    else:
        significance = "ANLAMSIZ"      # GA sifiri kapsiyor

    # Dort olcut
    c_a = cand_macro < base_macro
    c_b = significance == "IYILESME"
    c_c = (loo_ok == loo_n) and loo_n > 0
    # (d): dongusuz katmanda da iyilesme olmali (kazanc yalniz dongu baskilamadan gelmemeli)
    c_d = (clean_delta < 0) if clean_delta == clean_delta else False

    return {
        "model_id": model_id,
        "condition": condition,
        "n": len(shared),
        "baseline_macro_wer": base_macro,
        "macro_wer": cand_macro,
        "delta": point,
        "ci_low": lo,
        "ci_high": hi,
        "significance": significance,
        "loo_min": loo_min,
        "loo_max": loo_max,
        "loo_consistent": loo_ok,
        "loo_total": loo_n,
        "n_loop": len(loops),
        "n_clean": len(clean),
        "loop_delta": loop_delta,
        "clean_delta": clean_delta,
        "crit_a_beats_baseline": c_a,
        "crit_b_ci_excludes_zero": c_b,
        "crit_c_loo_stable": c_c,
        "crit_d_clean_stratum_gain": c_d,
        "accepted": c_a and c_b and c_c and c_d,
    }


def _fmt(x, pct=True, signed=False, width=8, nd=2):
    if x != x:  # NaN
        return f"{'—':>{width}}"
    v = x * 100 if pct else x
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return f"{s:>{width}}"


def main():
    ap = argparse.ArgumentParser(description="Kosullu kazanc analizi (Bolum 7 kabul olcutleri)")
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"Sonuc CSV (varsayilan: {DEFAULT_CSV})")
    ap.add_argument("--baseline-condition", default="none", help="Taban kosul (varsayilan: none)")
    ap.add_argument("--model", action="append", default=None,
                    help="Yalniz bu model(ler); tekrarlanabilir. Varsayilan: CSV'deki tum modeller")
    ap.add_argument("--run-purpose", default="main_benchmark")
    ap.add_argument("--extra-csv", default=None,
                    help="Tamamlayici CSV: yalniz ana CSV'de EKSIK olan (model, kosul, kayit) "
                         "anahtarlari icin kullanilir; mevcut satirlar ezilmez")
    args = ap.parse_args()

    cfg = load_experiment_config()
    s = cfg.get("statistics") or {}
    stats_cfg = {
        "bootstrap_iterations": int(s.get("bootstrap_iterations", 10000)),
        "confidence_level": float(s.get("confidence_level", 0.95)),
        "random_seed": int(s.get("random_seed", 20260910)),
    }

    results = load_results(args.csv, run_purpose=args.run_purpose)
    supplemented = 0
    if args.extra_csv:
        extra = load_results(args.extra_csv, run_purpose=args.run_purpose)
        results, supplemented = merge_supplement(results, extra)
    if not results:
        print(f"[HATA] '{args.csv}' icinde run_purpose={args.run_purpose} satiri yok.", file=sys.stderr)
        return 1

    models = args.model or sorted({m for (m, _c) in results})
    conf_pct = int(round(stats_cfg["confidence_level"] * 100))

    print("=" * 128)
    print("KOSULLU KAZANC ANALIZI — Bolum 7 kabul olcutleri")
    print(f"CSV      : {os.path.relpath(args.csv, os.path.dirname(PKG_ROOT))}")
    print(f"Taban    : '{args.baseline_condition}'  |  bootstrap={stats_cfg['bootstrap_iterations']} "
          f"GA=%{conf_pct} seed={stats_cfg['random_seed']}")
    if args.extra_csv:
        print(f"Tamamlayici: {os.path.relpath(args.extra_csv, os.path.dirname(PKG_ROOT))} "
              f"({supplemented} eksik anahtar tamamlandi)")
    print("Olcutler : (a) tabani gecti  (b) GA sifiri kapsamiyor  "
          "(c) leave-one-out kararli  (d) dongusuz katmanda kazanc")
    print("=" * 128)

    all_rows = []
    for model_id in models:
        conds = sorted({c for (m, c) in results if m == model_id and c != args.baseline_condition})
        if not conds:
            continue
        base_n = len(results.get((model_id, args.baseline_condition), {}))
        loops = loop_records_for_model(cfg, model_id)
        print(f"\n### {model_id}   (taban N={base_n}, dogrulanmis dongu kaydi={len(loops)})")
        if base_n < 2:
            print(f"    [ATLANDI] '{args.baseline_condition}' tabani icin yeterli kayit yok.")
            continue
        print(f"    {'kosul':<16}{'N':>3}{'makroWER':>10}{'dWER':>9}"
              f"{'  GA%'+str(conf_pct)+' alt':>11}{'ust':>9}"
              f"{'dongusuz':>10}{'dongulu':>9}  {'anlamlilik':<11}{'a':>1} {'b':>1} {'c':>1} {'d':>1}  KARAR")
        print("    " + "-" * 131)
        for cond in conds:
            r = evaluate(cfg, results, model_id, cond, args.baseline_condition, stats_cfg)
            if r is None:
                print(f"    {cond:<16}{'—':>3}  (taban ile ortak kayit < 2, atlandi)")
                continue
            all_rows.append(r)
            mark = lambda b: "+" if b else "."  # noqa: E731
            print(
                f"    {cond:<16}{r['n']:>3}"
                f"{_fmt(r['macro_wer'], width=9)}%"
                f"{_fmt(r['delta'], signed=True, width=8)}pp"
                f"{_fmt(r['ci_low'], signed=True, width=11)}"
                f"{_fmt(r['ci_high'], signed=True, width=9)}"
                f"{_fmt(r['clean_delta'], signed=True, width=10)}"
                f"{_fmt(r['loop_delta'], signed=True, width=9)}"
                f"  {r['significance']:<11}"
                f"{mark(r['crit_a_beats_baseline'])} {mark(r['crit_b_ci_excludes_zero'])} "
                f"{mark(r['crit_c_loo_stable'])} {mark(r['crit_d_clean_stratum_gain'])}"
                f"  {'KABUL' if r['accepted'] else 'RED'}"
            )

    if not all_rows:
        print("\n[UYARI] Degerlendirilecek (model, kosul) cifti bulunamadi.")
        return 0

    fields = list(all_rows[0].keys())
    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    accepted = [r for r in all_rows if r["accepted"]]
    print("\n" + "=" * 128)
    print(f"Degerlendirilen (model, kosul) : {len(all_rows)}")
    print(f"DORT olcutu birlikte gecen     : {len(accepted)}")
    for r in sorted(accepted, key=lambda x: x["delta"]):
        print(f"   KABUL  {r['model_id']} | {r['condition']}: "
              f"{r['baseline_macro_wer']*100:.2f}% -> {r['macro_wer']*100:.2f}% "
              f"({r['delta']*100:+.2f} pp, GA [{r['ci_low']*100:+.2f}, {r['ci_high']*100:+.2f}])")
    if not accepted:
        print("   (Hicbir kosul dort olcutu birlikte gecmedi — H1 bu veri kumesinde desteklenmiyor.)")
    print(f"Rapor: {os.path.relpath(REPORT_CSV, os.path.dirname(PKG_ROOT))}")
    print("=" * 128)
    return 0


if __name__ == "__main__":
    sys.exit(main())
