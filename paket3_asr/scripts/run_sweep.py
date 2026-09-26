"""
Kademe 1 tarama kosucusu — Y1 (OA) ve Y1b (spektral alfa) icin.

Makale protokolu: docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md

experiment.yaml'dan okur:
  development_subset.basenames  -> hangi kayitlar
  sweep_models                  -> hangi modeller (tarama icin M1+M2)
  confirmation_models           -> nihai dogrulama icin (M1+M2+M3)

benchmark.py mukerrer satirlari kendisi atladigi icin bu kosucu
DEVAM ETTIRILEBILIR: yarida kesilirse tekrar calistirmak kaldigi yerden surdurur.

Kullanim:
    python paket3_asr/scripts/run_sweep.py --family sa
    python paket3_asr/scripts/run_sweep.py --family oa --models confirmation
    python paket3_asr/scripts/run_sweep.py --family sa --basenames video_06_gogus_hastaliklari_oyku_alma
    python paket3_asr/scripts/run_sweep.py --family sa --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from oa_conditions import (  # noqa: E402
    PKG_ROOT,
    development_subset,
    expand_oa_conditions,
    load_experiment_config,
)
from spectral_alpha import expand_sa_conditions  # noqa: E402
from adaptive_alpha import expand_aa_conditions  # noqa: E402

BENCHMARK_PY = os.path.join(_SCRIPT_DIR, "benchmark.py")
REPO_ROOT = os.path.dirname(PKG_ROOT)


def family_conditions(family: str, cfg: dict):
    """Aile adindan (benchmark.py --condition degeri, kosul sayisi) ureti."""
    if family == "oa":
        return "oa", len(expand_oa_conditions(cfg))
    if family == "sa":
        return "sa", len(expand_sa_conditions(cfg))
    if family == "aa":
        return "aa", len(expand_aa_conditions(cfg))
    if family == "base":
        return "all", 4
    raise ValueError(f"Bilinmeyen aile '{family}' (beklenen: oa, sa, base)")


def main():
    ap = argparse.ArgumentParser(description="Kademe 1 tarama kosucusu")
    ap.add_argument("--family", required=True, choices=["oa", "sa", "aa", "base"],
                    help="oa = Y1 sinyal alani, sa = Y1b spektral alan, "
                         "aa = Y2 uyarlamali zarf, base = 4 temel kosul")
    ap.add_argument("--models", default="sweep", choices=["sweep", "confirmation"],
                    help="sweep = sweep_models (M1+M2), confirmation = confirmation_models (M1+M2+M3)")
    ap.add_argument("--basenames", nargs="*", default=None,
                    help="Kayitlar; verilmezse development_subset kullanilir")
    ap.add_argument("--speech-pad-ms", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Yalniz ne kosulacagini yazdir")
    args = ap.parse_args()

    cfg = load_experiment_config()
    condition, n_cond = family_conditions(args.family, cfg)
    models = list(cfg.get(f"{args.models}_models", []))
    basenames = args.basenames or development_subset(cfg)

    if not models:
        print(f"[HATA] experiment.yaml icinde '{args.models}_models' bos.", file=sys.stderr)
        return 1
    if not basenames:
        print("[HATA] Kayit listesi bos (development_subset tanimli mi?).", file=sys.stderr)
        return 1

    total = len(models) * len(basenames)
    print(f"Aile     : {args.family}  (--condition {condition}, {n_cond} kosul)")
    print(f"Modeller : {models}")
    print(f"Kayitlar : {basenames}")
    print(f"Cagri    : {total} adet benchmark.py  ->  {total * n_cond} satir hedefi")
    if args.speech_pad_ms is not None:
        print(f"speech_pad_ms: {args.speech_pad_ms}")

    if args.dry_run:
        print("\n[DRY-RUN] Hicbir sey kosulmadi.")
        return 0

    t0 = time.time()
    done = 0
    failures = []
    for model_id in models:
        for basename in basenames:
            done += 1
            el = time.time() - t0
            print(f"\n{'='*78}")
            print(f"[TARAMA {done}/{total}] {model_id} | {basename} | {n_cond} kosul "
                  f"| gecen: {el/60:.1f} dk")
            print(f"{'='*78}", flush=True)
            cmd = [
                sys.executable, BENCHMARK_PY,
                "--basename", basename,
                "--model-id", model_id,
                "--condition", condition,
                "--run-purpose", "main_benchmark",
            ]
            if args.speech_pad_ms is not None:
                cmd += ["--speech-pad-ms", str(args.speech_pad_ms)]
            proc = subprocess.run(cmd, cwd=REPO_ROOT)
            if proc.returncode != 0:
                msg = f"{model_id} / {basename} (cikis {proc.returncode})"
                print(f"[HATA] {msg}", file=sys.stderr)
                failures.append(msg)

    el = time.time() - t0
    print(f"\n{'='*78}")
    print(f"TARAMA BITTI — {done}/{total} cagri, sure {el/3600:.2f} saat")
    if failures:
        print(f"BASARISIZ ({len(failures)}):")
        for f in failures:
            print(f"   {f}")
        return 1
    print("Tum cagrilar basarili.")
    print("\nSonraki adim:")
    print("   python paket3_asr/scripts/analyze_conditional_gain.py \\")
    print("       --extra-csv paket3_asr/outputs/metrics_baseline/benchmark_results_baseline_refv2.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
