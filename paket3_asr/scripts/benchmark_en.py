"""
Ingilizce korpus cozme hatti (Ek-2 Rev B, Bolum 7).

NEDEN AYRI HAT
--------------
Turkce benchmark.py 451 satirlik sonucu uretti ve Kademe 0 yeniden uretilebilirlik
kapisindan 0,000 puan farkla gecti. O dosyaya dokunmak butun Turkce sonuclari
risk altina atardi. Bu betik ayni cozme fonksiyonunu cagirir ama uc yerde ayrisir:

  dil          : "en"  (Turkce hatta "tr" sabit)
  referans     : data/references_en
  normalizasyon: Whisper Ingilizce normalizeri (Turkce normalizer Ingilizceyi bozar)

Cikti semasi Turkce CSV ile ayni tutulur; boylece analiz araclari ortak kalir.
Ayri dosyaya yazilir: Turkce ve Ingilizce sonuclar HAVUZLANMAZ (Bolum 13).

MUKERRER KORUMASI
-----------------
(model, kayit, kosul) ucluSU zaten varsa cozme atlanir. Kesintiden devam eden
kuyruklar icin sarttir.

Kullanim:
    python paket3_asr/scripts/benchmark_en.py --condition none
    python paket3_asr/scripts/benchmark_en.py --condition none --cell en_clean_medical
"""

import argparse
import csv
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

import yaml  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from oa_conditions import PKG_ROOT  # noqa: E402
from transcribe import (load_faster_whisper_model,  # noqa: E402
                        transcribe_audio_faster_whisper)

FROZEN = os.path.join(PKG_ROOT, "config", "corpus_en_frozen.yaml")
REF_EN = os.path.join(PKG_ROOT, "data", "references_en")
OUT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results_en.csv")

FIELDS = ["timestamp", "cell", "basename", "condition", "denoiser_method", "model_id",
          "model_name", "compute_type", "language", "WER", "n_ref_words", "n_hyp_words",
          "duration_seconds", "elapsed_seconds", "RTF", "reference_text", "transcription"]

# Kosul -> (ses dizini, surucu adi). Turkce hattaki adlandirma korunur.
CONDITIONS = {
    "none":             ("audio_16k_en", "none"),
    "mild":             ("audio_denoised_mild_en", "webrtc_mild"),
    "aggressive":       ("audio_denoised_aggressive_en", "webrtc_aggressive"),
    "oa_aggr_a020":     ("audio_oa_aggr_a020_en", "oa_aggressive_alpha0.20"),
    "sa_psm_aggr_a040": ("audio_sa_psm_aggr_a040_en", "sa_psm_aggressive_alpha0.40"),
    "sa_psm_mild_a025": ("audio_sa_psm_mild_a025_en", "sa_psm_mild_alpha0.25"),
    "pcen":             ("audio_pcen_en", "pcen_agc_a05_s012"),
}

_NORM = None


def normalize_en(t):
    """Whisper Ingilizce normalizeri — AMI'nin buyuk harfli formatiyla uyumu olculdu."""
    global _NORM
    if _NORM is None:
        from whisper.normalizers import EnglishTextNormalizer
        _NORM = EnglishTextNormalizer()
    return _NORM(t)


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


def corpus():
    with open(FROZEN, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return [(c["id"], r["id"]) for c in d["cells"] for r in c["records"]]


def done_set():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV, encoding="utf-8") as f:
        return {(r["model_id"], r["basename"], r["condition"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description="Ingilizce korpus cozme")
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--model-id", default="faster-whisper-large-v3-int8")
    ap.add_argument("--model-name", default="large-v3")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--cell", default=None)
    args = ap.parse_args()

    sub, method = CONDITIONS[args.condition]
    audio_dir = os.path.join(PKG_ROOT, "data", sub)
    recs = [(c, b) for c, b in corpus() if args.cell is None or c == args.cell]
    have = done_set()
    todo = [(c, b) for c, b in recs if (args.model_id, b, args.condition) not in have]

    print(f"kosul {args.condition} | model {args.model_id} | {len(todo)}/{len(recs)} cozulecek")
    if not todo:
        return 0

    model, load_s, _dev, load_mem = load_faster_whisper_model(
        args.model_name, device="cpu", compute_type=args.compute_type)

    new = not os.path.exists(OUT_CSV)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for cell, b in todo:
            ap_path = os.path.join(audio_dir, f"{b}.wav")
            if not os.path.exists(ap_path):
                print(f"[EKSIK] {ap_path}")
                continue
            t0 = time.time()
            res = transcribe_audio_faster_whisper(
                ap_path, model_size_or_path=args.model_name, language="en",
                temperature=0.0, device="cpu", compute_type=args.compute_type,
                model=model, load_seconds=load_s, load_mem_stats=load_mem)
            el = time.time() - t0
            hyp = normalize_en(res["text"])
            ref = normalize_en(open(os.path.join(REF_EN, f"{b}.txt"), encoding="utf-8").read().strip())
            e = wer(ref, hyp)
            dur = res.get("duration_seconds") or 0.0
            w.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "cell": cell, "basename": b,
                "condition": args.condition, "denoiser_method": method,
                "model_id": args.model_id, "model_name": args.model_name,
                "compute_type": args.compute_type, "language": "en",
                "WER": f"{e:.6f}", "n_ref_words": len(ref.split()),
                "n_hyp_words": len(hyp.split()), "duration_seconds": f"{dur:.2f}",
                "elapsed_seconds": f"{el:.2f}", "RTF": f"{el/dur:.4f}" if dur else "",
                "reference_text": ref, "transcription": hyp})
            fh.flush()
            print(f"[OK] {b:<26}{args.condition:<18}WER %{e*100:6.2f}  RTF {el/dur if dur else 0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
