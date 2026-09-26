"""
Y2 — Iki gecisli, cozucu-durumu gudumlu kosullu on isleme.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Literatur taramasi: docs/MAKALE_LITERATUR_TARAMASI_ASR_ON_ISLEME_2026-09-10.md (Bolum 5)

GEREKCE
-------
Y1 (dalga formu karisimi) ve Y1b (spektral karisim) ayni sonuca vardi: SAGLIKLI
kayitlarda hicbir sabit alfa ham sesi gecemiyor, en iyi ihtimalle esitleniyor
(olculdu: Y1b psm/aggr, Turbo, dongusuz 3 kayit -> +0.00 pp).

Ancak ayni mudahale, kod cozmenin COKTUGU kayitlarda WER'i %54.41 -> %33.70
indiriyor (-20.7 pp). Yani etkinin isareti kaydin durumuna gore degisiyor.

Bu nedenle dogru mudahale "her yere hafif denoise" degil, "yalniz coktugu yerde
denoise". Olculen ust sinir (dongu kayitlari BILINEREK secilirse):
    Turbo/aggr : %39.62 -> %31.34  (-8.28 pp)
    Turbo/mild : %39.62 -> %32.99  (-6.63 pp)

Bu betik o karari OTOMATIK vermeye calisir.

MIMARI
------
1. gecis : ham sesi coz, SEGMENT duzeyinde istatistikleri topla
           (compression_ratio, avg_logprob, no_speech_prob, tekrar orani)
2. tespit : cokus belirtisi gosteren segmentleri isaretle
3. karisim: yalniz o segmentlerde alfa'yi ac (psm, Mel alani esdegeri),
            geri kalan her yerde alfa = 0 (ham ses)
4. gecis  : uretilen sesi yeniden coz

ASIMETRI — neden tespit hassas olmak zorunda degil
--------------------------------------------------
Saglikli kayda gereksiz mudahale ~0.3-1.0 pp'ye mal olur (Y1b psm olcumu).
Coken kaydi kurtarmak ~20 pp kazandirir. Yani yanlis alarm maliyeti, kacirma
maliyetinin ~1/20'si. Tespit esikleri bu nedenle CESUR secilir: kacirmamak
hassas olmaktan onemlidir.

ON AYARLAR
----------
tpseg : segment duzeyinde — yalniz cokmus segmentlerde alfa acilir
tprec : kayit duzeyinde  — herhangi bir cokus varsa TUM kayda alfa uygulanir
        (olculen -8.28 pp ust sinirinin dogrudan karsiligi)

Kullanim:
    python paket3_asr/scripts/run_twopass.py --dry-run
    python paket3_asr/scripts/run_twopass.py
    python paket3_asr/scripts/run_twopass.py --alpha 0.40 --preset tpseg
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from adaptive_alpha import smooth_envelope  # noqa: E402
from calculate_metrics import evaluate_asr_pair  # noqa: E402
from medical_term_score import calculate_medical_term_recall  # noqa: E402
from oa_conditions import PKG_ROOT, development_subset, load_experiment_config  # noqa: E402
from spectral_alpha import WHISPER_HOP, WHISPER_N_FFT, WHISPER_SR, whisper_window  # noqa: E402
from transcribe import get_experiment_decoding_config, load_faster_whisper_model  # noqa: E402

RESULTS_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results.csv")
DETECT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "twopass_detection_report.csv")
AUDIO_16K = os.path.join(PKG_ROOT, "data", "audio_16k")
REF_DIR = os.path.join(PKG_ROOT, "data", "references")
TMP_DIR = os.path.join(PKG_ROOT, "data", "audio_twopass")

ENHANCER_DIRS = {
    "mild": os.path.join(PKG_ROOT, "data", "audio_denoised_mild"),
    "aggr": os.path.join(PKG_ROOT, "data", "audio_denoised_aggressive"),
}


# ---------------------------------------------------------------------------
# Cokus tespiti
# ---------------------------------------------------------------------------

def repetition_ratio(text: str, n: int = 3) -> float:
    """
    En sik tekrar eden n-gram'in oranı. Patolojik donguler ('hayir' x112 gibi)
    burada 1.0'a yakin deger uretir; normal konusma 0.05'in altinda kalir.
    """
    words = text.split()
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def is_collapsed(seg, thresholds: dict) -> tuple:
    """
    Bir segmentin cokup cokmedigine karar verir.
    Doner: (cokmus_mu, tetikleyen_gerekce_listesi)
    """
    reasons = []
    cr = getattr(seg, "compression_ratio", None)
    lp = getattr(seg, "avg_logprob", None)
    text = (getattr(seg, "text", "") or "").strip()

    if cr is not None and cr > thresholds["compression_ratio"]:
        reasons.append(f"compression_ratio={cr:.2f}")
    rr = repetition_ratio(text, n=thresholds["ngram"])
    if rr > thresholds["repetition_ratio"]:
        reasons.append(f"repetition={rr:.2f}")
    lp_thr = thresholds.get("avg_logprob")
    if lp_thr is not None and lp is not None and lp < lp_thr:
        reasons.append(f"avg_logprob={lp:.2f}")
    return (len(reasons) > 0), reasons


# ---------------------------------------------------------------------------
# Zarf ve karisim
# ---------------------------------------------------------------------------

def envelope_from_spans(spans, n_frames: int, alpha: float, hop: int = WHISPER_HOP,
                        sr: int = WHISPER_SR, pad_s: float = 0.5) -> np.ndarray:
    """
    Zaman araliklarindan (saniye) cerceve basina alfa zarfi uretir.
    Araliklarin kenarina pad_s kadar pay eklenir (cokus sinirlari kesin degildir).
    """
    env = np.zeros(n_frames, dtype=np.float64)
    fps = sr / hop
    for s, e in spans:
        i0 = max(0, int((s - pad_s) * fps))
        i1 = min(n_frames, int((e + pad_s) * fps) + 1)
        if i1 > i0:
            env[i0:i1] = alpha
    return env


def build_from_envelope(raw: np.ndarray, enh: np.ndarray, env: np.ndarray,
                        n_fft: int = WHISPER_N_FFT, hop: int = WHISPER_HOP,
                        block_frames: int = 4096) -> np.ndarray:
    """psm (guc alani) karisimi, cerceve basina alfa ile. Faz = ham sesin fazi."""
    w = whisper_window(n_fft)
    w2 = w ** 2
    pad = n_fft // 2
    length = len(raw)
    rp = np.pad(raw.astype(np.float64), (pad, pad), mode="reflect")
    ep = np.pad(enh.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(rp) - n_fft) // hop
    if len(env) < n_frames:
        env = np.pad(env, (0, n_frames - len(env)), mode="edge")
    env = env[:n_frames]

    total = (n_frames - 1) * hop + n_fft
    acc = np.zeros(total); wsum = np.zeros(total)
    offsets = np.arange(n_fft)
    for start in range(0, n_frames, block_frames):
        stop = min(start + block_frames, n_frames)
        idx = offsets[None, :] + hop * np.arange(start, stop)[:, None]
        S_raw = np.fft.rfft(rp[idx] * w, n=n_fft, axis=1)
        S_enh = np.fft.rfft(ep[idx] * w, n=n_fft, axis=1)
        A_raw = np.abs(S_raw); A_enh = np.abs(S_enh)
        phase = np.exp(1j * np.angle(S_raw))
        a = env[start:stop][:, None]
        A_out = np.sqrt(np.maximum((1.0 - a) * A_raw ** 2 + a * A_enh ** 2, 0.0))
        frames = np.fft.irfft(A_out * phase, n=n_fft, axis=1) * w
        for j, k in enumerate(range(start, stop)):
            o = k * hop
            acc[o:o + n_fft] += frames[j]
            wsum[o:o + n_fft] += w2
    nz = wsum > 1e-12
    acc[nz] /= wsum[nz]
    y = acc[pad:pad + length]
    return np.pad(y, (0, max(0, length - len(y))))[:length]


def finalize(y: np.ndarray, raw: np.ndarray, peak_guard: float = 0.999) -> np.ndarray:
    """Cikti seviyesini ham sesin RMS'ine sabitle, tepe korumasi uygula."""
    r = float(np.sqrt(np.mean(raw ** 2))); m = float(np.sqrt(np.mean(y ** 2)))
    if m > 1e-9 and r > 1e-9:
        y = y * (r / m)
    guard = max(peak_guard, float(np.max(np.abs(raw))))
    pk = float(np.max(np.abs(y)))
    if pk > guard:
        y = y * (guard / pk)
    return y


# ---------------------------------------------------------------------------
# Cozme
# ---------------------------------------------------------------------------

def decode(model, audio_path: str):
    """Hattin sozlesmesiyle birebir ayni cozme; segment listesi de dondurulur."""
    _pv, cfg = get_experiment_decoding_config()
    vad_kwargs = {"speech_pad_ms": cfg.get("speech_pad_ms", 400)} if cfg.get("vad_filter", True) else None
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language="tr", temperature=0.0,
        beam_size=cfg.get("beam_size", 5),
        vad_filter=cfg.get("vad_filter", True),
        vad_parameters=vad_kwargs,
        condition_on_previous_text=cfg.get("condition_on_previous_text", False),
        compression_ratio_threshold=cfg.get("compression_ratio_threshold", 2.4),
        log_prob_threshold=cfg.get("log_prob_threshold", -1.0),
        no_speech_threshold=cfg.get("no_speech_threshold", 0.6),
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, segs, info, round(time.time() - t0, 2)


def write_row(model_spec, basename, condition, method, text, ref, info,
              elapsed, load_seconds, decoding_cfg):
    """benchmark_results.csv semasiyla uyumlu satir yazar."""
    m = evaluate_asr_pair(ref, text)
    try:
        med = calculate_medical_term_recall(ref, text)
        recall = med.get("medical_term_accuracy") if isinstance(med, dict) else med
        found = ";".join(med.get("found_terms", [])) if isinstance(med, dict) else ""
        missed = ";".join(med.get("missed_terms", [])) if isinstance(med, dict) else ""
    except Exception:
        recall, found, missed = "", "", ""

    dur = round(info.duration, 2)
    row = {
        "protocol_version": decoding_cfg.get("protocol_version", "pilot-v2-antiloop"),
        "run_purpose": "main_benchmark", "timestamp": datetime.datetime.now().isoformat(),
        "basename": basename, "condition": condition, "denoiser_method": method,
        "model_id": model_spec["id"], "engine": "faster-whisper",
        "transcription_mode": "faster_whisper_native", "model_name": model_spec["model_name"],
        "device": model_spec.get("device", "cpu"), "compute_type": model_spec.get("compute_type", "int8"),
        "language": "tr", "temperature": 0.0, "transcription": text,
        "WER": m["wer"], "CER": m["cer"], "medical_term_recall": recall,
        "load_seconds": load_seconds, "elapsed_seconds": elapsed, "duration_seconds": dur,
        "RTF": round(elapsed / dur, 4) if dur > 0 else 0.0,
        "total_seconds": round(load_seconds + elapsed, 2),
        "load_peak_ram_mb": "", "load_ram_delta_mb": "", "inference_peak_ram_mb": "",
        "inference_ram_delta_mb": "", "peak_vram_mb": "", "vram_status": "N/A - CUDA unavailable",
        "memory_measurement_method": "not_tracked_twopass",
        "decoding_config": json.dumps(decoding_cfg, ensure_ascii=False),
        "unsupported_decoding_params": "[]", "found_terms": found, "missed_terms": missed,
        "reference_text": ref,
    }
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        fields = csv.DictReader(f).fieldnames
    with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, "") for k in fields})
    return m["wer"]


def already_done(model_id, basename, condition) -> bool:
    if not os.path.exists(RESULTS_CSV):
        return False
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("model_id") == model_id and r.get("basename") == basename
                    and r.get("condition") == condition and r.get("run_purpose") == "main_benchmark"):
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Y2 — iki gecisli kosullu on isleme")
    ap.add_argument("--alpha", type=float, default=0.40,
                    help="Cokmus bolgelerde uygulanacak alfa (varsayilan 0.40; psm'in Large-v3 optimumu)")
    ap.add_argument("--enhancer", default="aggr", choices=["mild", "aggr"])
    ap.add_argument("--preset", default=None, choices=["tpseg", "tprec"],
                    help="Yalniz bu on ayar; verilmezse ikisi de")
    ap.add_argument("--basenames", nargs="*", default=None)
    ap.add_argument("--models", default="sweep", choices=["sweep", "confirmation"])
    ap.add_argument("--pad-s", type=float, default=0.5)
    # Esikler pilot olcume gore kalibre edildi (saha_03):
    #   gercek cokus      -> compression_ratio=22.30, repetition=0.97
    #   yanlis alarmlar   -> compression_ratio<2.2,   repetition=0.20-0.25
    # Iki dagilim arasinda genis bir bosluk var; esikler o bosluga konur.
    ap.add_argument("--cr-threshold", type=float, default=5.0,
                    help="compression_ratio esigi (varsayilan 5.0; normal konusma ~2, cokus >20)")
    ap.add_argument("--rep-threshold", type=float, default=0.50,
                    help="tekrar orani esigi (varsayilan 0.50; dogal tekrar ~0.20, cokus ~0.97)")
    ap.add_argument("--logprob-threshold", type=float, default=None,
                    help="avg_logprob esigi; verilmezse bu tetikleyici KAPALI (zayif sinyal)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config()
    models = list(cfg.get(f"{args.models}_models", []))
    recs = args.basenames or development_subset(cfg)
    presets = [args.preset] if args.preset else ["tpseg", "tprec"]
    thresholds = {"compression_ratio": args.cr_threshold,
                  "repetition_ratio": args.rep_threshold,
                  "avg_logprob": args.logprob_threshold, "ngram": 3}

    print(f"Modeller : {models}")
    print(f"Kayitlar : {recs}")
    print(f"On ayar  : {presets}  | alfa={args.alpha}  iyilestirici={args.enhancer}")
    print(f"Esikler  : {thresholds}")
    print(f"Cozme    : {len(models)*len(recs)} x (1 tespit + {len(presets)} degerlendirme)")
    if args.dry_run:
        print("\n[DRY-RUN] Hicbir sey kosulmadi.")
        return 0

    os.makedirs(TMP_DIR, exist_ok=True)
    _pv, dcfg = get_experiment_decoding_config()
    detect_rows = []

    for model_id in models:
        spec = next((m for m in _models_yaml() if m["id"] == model_id), None)
        if spec is None:
            print(f"[HATA] models.yaml icinde '{model_id}' yok", file=sys.stderr)
            continue
        print(f"\n{'='*78}\n[MODEL] {model_id}\n{'='*78}", flush=True)
        model, load_s, _, _ = load_faster_whisper_model(
            model_size_or_path=spec["model_name"],
            device=spec.get("device", "cpu"),
            compute_type=spec.get("compute_type", "int8"),
        )

        for basename in recs:
            raw_path = os.path.join(AUDIO_16K, f"{basename}.wav")
            ref = open(os.path.join(REF_DIR, f"{basename}.txt"), encoding="utf-8").read().strip()

            # --- 1. gecis: tespit ---
            print(f"\n[1. GECIS] {basename} — ham ses ile cozuluyor...", flush=True)
            text1, segs, info, el1 = decode(model, raw_path)
            spans, reasons_all = [], []
            for s in segs:
                bad, reasons = is_collapsed(s, thresholds)
                if bad:
                    spans.append((s.start, s.end))
                    reasons_all.append(f"[{s.start:.1f}-{s.end:.1f}] " + ", ".join(reasons))
            wer1 = evaluate_asr_pair(ref, text1)["wer"]
            cok_sure = sum(e - s for s, e in spans)
            print(f"   segment={len(segs)}  cokmus={len(spans)}  "
                  f"cokmus_sure={cok_sure:.1f}s / {info.duration:.1f}s  ham_WER=%{wer1*100:.2f}")
            for r in reasons_all[:4]:
                print(f"     {r}")
            detect_rows.append({
                "model_id": model_id, "basename": basename, "n_segments": len(segs),
                "n_collapsed": len(spans), "collapsed_seconds": round(cok_sure, 2),
                "duration_seconds": round(info.duration, 2), "raw_wer": f"{wer1:.6f}",
                "reasons": " | ".join(reasons_all[:10]),
            })

            if not spans:
                print("   -> cokus yok; kosullu kural ham sesi aynen birakir")

            raw, _ = sf.read(raw_path, dtype="float64")
            enh, _ = sf.read(os.path.join(ENHANCER_DIRS[args.enhancer], f"{basename}.wav"), dtype="float64")
            n_frames = 1 + (len(raw) + 2 * (WHISPER_N_FFT // 2) - WHISPER_N_FFT) // WHISPER_HOP

            for preset in presets:
                cond = f"{preset}_{args.enhancer}_a{int(round(args.alpha*100)):03d}"
                if already_done(model_id, basename, cond):
                    print(f"   [ATLANDI] {cond} zaten var")
                    continue

                if not spans:
                    env = np.zeros(n_frames)
                elif preset == "tprec":
                    env = np.full(n_frames, args.alpha)      # tum kayda uygula
                else:
                    env = envelope_from_spans(spans, n_frames, args.alpha, pad_s=args.pad_s)
                env = np.clip(smooth_envelope(env, 12), 0.0, 1.0)

                out_path = os.path.join(TMP_DIR, f"{model_id}__{basename}__{cond}.wav")
                y = finalize(build_from_envelope(raw, enh, env), raw)
                sf.write(out_path, y.astype(np.float32), 16000, subtype="PCM_16")

                print(f"[2. GECIS] {cond}  alfa_ort={env.mean():.3f} "
                      f"etkilenen=%{(env > 0.01).mean()*100:.1f}", flush=True)
                text2, _s2, info2, el2 = decode(model, out_path)
                wer2 = write_row(spec, basename, cond, f"twopass_psm_{args.enhancer}",
                                 text2, ref, info2, el2, load_s, dcfg)
                print(f"   WER: ham %{wer1*100:.2f} -> {cond} %{wer2*100:.2f} "
                      f"({(wer2-wer1)*100:+.2f} pp)")
                try:
                    os.remove(out_path)
                except OSError:
                    pass

    if detect_rows:
        os.makedirs(os.path.dirname(DETECT_CSV), exist_ok=True)
        with open(DETECT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detect_rows[0].keys()))
            w.writeheader(); w.writerows(detect_rows)
        print(f"\nTespit raporu: {os.path.relpath(DETECT_CSV, os.path.dirname(PKG_ROOT))}")
    return 0


def _models_yaml():
    import yaml
    with open(os.path.join(PKG_ROOT, "config", "models.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


if __name__ == "__main__":
    sys.exit(main())
