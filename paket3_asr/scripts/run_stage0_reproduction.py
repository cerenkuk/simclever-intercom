"""
Kademe 0 — Hattin yeniden uretim dogrulamasi.

Makale protokolu: docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md, Bolum 9.1

experiment.yaml -> reproduction_check blogundaki kayitlari, confirmation_models
listesindeki modellerle ve 4 temel kosulda yeniden cozer; sonuclari baseline
CSV ile karsilastirir.

Bu, tarama oncesi KAPI kontroludur: hat baseline'i yeniden uretmiyorsa
(farkli faster-whisper/ctranslate2 surumu, farkli VAD davranisi vb.) sonraki
kademelerin baseline ile kiyaslanmasi gecersiz olur.

Kullanim:
    python paket3_asr/scripts/run_stage0_reproduction.py
    python paket3_asr/scripts/run_stage0_reproduction.py --compare-only
"""

import argparse
import csv
import os
import subprocess
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402

BENCHMARK_PY = os.path.join(_SCRIPT_DIR, "benchmark.py")
RESULTS_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results.csv")
REPORT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "stage0_reproduction_report.csv")

BASE_CONDITIONS = ["none", "mild", "aggressive", "loudnorm"]


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def load_wer_index(csv_path: str) -> dict:
    """(model_id, basename, condition) -> WER esleme tablosu (main_benchmark satirlari)."""
    idx = {}
    if not os.path.exists(csv_path):
        return idx
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("run_purpose") != "main_benchmark":
                continue
            try:
                wer = float(r["WER"])
            except (KeyError, TypeError, ValueError):
                continue
            idx[(r["model_id"], r["basename"], r["condition"])] = wer
    return idx


def run_decodes(basenames, models):
    total = len(basenames) * len(models)
    done = 0
    for model_id in models:
        for basename in basenames:
            done += 1
            print(f"\n{'='*78}\n[KADEME 0 {done}/{total}] {model_id} | {basename} | 4 kosul\n{'='*78}", flush=True)
            cmd = [
                sys.executable, BENCHMARK_PY,
                "--basename", basename,
                "--model-id", model_id,
                "--condition", "all",
                "--run-purpose", "main_benchmark",
            ]
            proc = subprocess.run(cmd, cwd=os.path.dirname(PKG_ROOT))
            if proc.returncode != 0:
                print(
                    f"[HATA] Cozme basarisiz (cikis kodu {proc.returncode}): {model_id} / {basename}",
                    file=sys.stderr,
                )
                return False
    return True


def compare(basenames, models, baseline_csv, max_delta):
    baseline = load_wer_index(baseline_csv)
    fresh = load_wer_index(RESULTS_CSV)

    if not fresh:
        print(f"[HATA] Yeni sonuc bulunamadi: {RESULTS_CSV}", file=sys.stderr)
        return 1

    rows = []
    print(f"\n{'='*98}")
    print("KADEME 0 — YENIDEN URETIM DOGRULAMASI")
    print(f"Kabul esigi: |dWER| <= {max_delta:.4f} ({max_delta*100:.2f} pp)")
    print(f"{'='*98}")
    print(f"{'model':<38}{'kayit':<32}{'kosul':<11}{'baseline':>9}{'yeni':>8}{'dWER':>10}  sonuc")
    print("-" * 98)

    missing = 0
    for model_id in models:
        for basename in basenames:
            for cond in BASE_CONDITIONS:
                key = (model_id, basename, cond)
                b = baseline.get(key)
                n = fresh.get(key)
                if n is None:
                    print(f"{model_id[:37]:<38}{basename[:31]:<32}{cond:<11}{'-':>9}{'YOK':>8}{'-':>10}  EKSIK")
                    missing += 1
                    continue
                if b is None:
                    print(f"{model_id[:37]:<38}{basename[:31]:<32}{cond:<11}{'YOK':>9}{n*100:>7.2f}%{'-':>10}  BASELINE-YOK")
                    rows.append({
                        "model_id": model_id, "basename": basename, "condition": cond,
                        "baseline_wer": "", "fresh_wer": f"{n:.6f}", "delta": "", "verdict": "BASELINE-YOK",
                    })
                    continue
                d = n - b
                ok = abs(d) <= max_delta
                print(
                    f"{model_id[:37]:<38}{basename[:31]:<32}{cond:<11}"
                    f"{b*100:>8.2f}%{n*100:>7.2f}%{d*100:>+9.2f}pp  {'GECTI' if ok else 'SAPMA'}"
                )
                rows.append({
                    "model_id": model_id, "basename": basename, "condition": cond,
                    "baseline_wer": f"{b:.6f}", "fresh_wer": f"{n:.6f}",
                    "delta": f"{d:.6f}", "verdict": "GECTI" if ok else "SAPMA",
                })

    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "basename", "condition",
                                          "baseline_wer", "fresh_wer", "delta", "verdict"])
        w.writeheader()
        w.writerows(rows)

    compared = [r for r in rows if r["verdict"] in ("GECTI", "SAPMA")]
    deviations = [r for r in compared if r["verdict"] == "SAPMA"]
    deltas = [abs(float(r["delta"])) for r in compared]

    print("-" * 98)
    print(f"Karsilastirilan  : {len(compared)}")
    print(f"Gecti            : {len(compared) - len(deviations)}")
    print(f"Sapma            : {len(deviations)}")
    print(f"Eksik (cozulmedi): {missing}")
    if deltas:
        print(f"Maks |dWER|      : {max(deltas)*100:.3f} pp")
        print(f"Ortalama |dWER|  : {_mean(deltas)*100:.3f} pp")
    print(f"Rapor            : {os.path.relpath(REPORT_CSV, os.path.dirname(PKG_ROOT))}")

    if missing:
        print("\n[SONUC] KAPI KAPALI — bazi kosullar cozulmedi.")
        return 1
    if deviations:
        print("\n[SONUC] KAPI KAPALI — hat baseline'i yeniden uretmiyor.")
        print("        Sonraki kademelerde baseline ile dogrudan kiyas GECERSIZ olur;")
        print("        tum kiyaslar bu repoda yeniden uretilmis satirlar uzerinden yapilmalidir.")
        return 2
    print("\n[SONUC] KAPI ACIK — hat baseline'i yeniden uretiyor. Kademe 1'e gecilebilir.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Kademe 0 — yeniden uretim dogrulamasi")
    ap.add_argument("--compare-only", action="store_true",
                    help="Cozme yapmadan yalniz mevcut sonuclari baseline ile karsilastir")
    args = ap.parse_args()

    cfg = load_experiment_config()
    rc = cfg.get("reproduction_check") or {}
    basenames = list(rc.get("basenames", []))
    models = list(cfg.get("confirmation_models", []))
    baseline_rel = rc.get("baseline_csv", "outputs/metrics_baseline/benchmark_results_baseline.csv")
    baseline_csv = os.path.join(PKG_ROOT, baseline_rel)
    max_delta = float(rc.get("max_abs_wer_delta", 0.005))

    if not basenames or not models:
        print("[HATA] experiment.yaml icinde reproduction_check.basenames veya confirmation_models bos.",
              file=sys.stderr)
        return 1
    if not os.path.exists(baseline_csv):
        print(f"[HATA] Baseline CSV bulunamadi: {baseline_csv}", file=sys.stderr)
        return 1

    print(f"Kayitlar : {basenames}")
    print(f"Modeller : {models}")
    print(f"Kosullar : {BASE_CONDITIONS}")
    print(f"Baseline : {baseline_rel}")

    if not args.compare_only:
        if not run_decodes(basenames, models):
            print("[HATA] Cozme asamasi tamamlanamadi; karsilastirma yapilmiyor.", file=sys.stderr)
            return 1

    return compare(basenames, models, baseline_csv, max_delta)


if __name__ == "__main__":
    sys.exit(main())
