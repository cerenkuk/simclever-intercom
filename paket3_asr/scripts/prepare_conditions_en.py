"""
Ingilizce korpus icin kosul uretimi (Ek-2 Rev B, Bolum 7).

NEDEN AYRI BETIK
----------------
Turkce hatti 451 satirlik sonucu uretti ve yeniden uretilebilirlik kapisindan
gecti. Ingilizce icin o hatti degistirmek yerine, ayni COZUM FONKSIYONLARINI
acik yollarla cagiran ayri bir betik yazildi. Boylece Turkce sonuclar risk
altina girmez.

KOSULLAR — Bolum 7'de ilan edilenler, yeniden ayar YOK
------------------------------------------------------
    mild, aggressive        WebRTC NS — negatif sonucun Ingilizcede de gorulmesi
    oa_aggr_a020            donmus birlestirme uyesi
    sa_psm_aggr_a040        donmus birlestirme uyesi
    sa_psm_mild_a025        donmus birlestirme uyesi
    pcen                    opsiyonel besinci uye

Alfa degerleri, marj ve uye kumesi Turkce'deki gibidir. Ingilizce icin yeniden
ayar yapilsaydi iddia "yontem genelleniyor" degil "her dilde yeniden ayar
yapilabiliyor" olurdu (Bolum 7, Kural 2).

Kullanim:
    python paket3_asr/scripts/prepare_conditions_en.py
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

import yaml  # noqa: E402

from nonsuppressive_filters import BUILDERS as NSF_BUILDERS, finalize as nsf_finalize  # noqa: E402
from oa_conditions import PKG_ROOT, get_mix_settings, load_experiment_config  # noqa: E402
from prepare_audio import (apply_observation_addition, apply_spectral_alpha,  # noqa: E402
                           process_single_denoise)
from spectral_alpha import get_sa_settings  # noqa: E402

RAW = os.path.join(PKG_ROOT, "data", "audio_16k_en")
FROZEN = os.path.join(PKG_ROOT, "config", "corpus_en_frozen.yaml")


def records():
    with open(FROZEN, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return [r["id"] for c in d["cells"] for r in c["records"]]


def outdir(cond):
    p = os.path.join(PKG_ROOT, "data", f"audio_{cond}_en")
    os.makedirs(p, exist_ok=True)
    return p


def main():
    recs = records()
    cfg = load_experiment_config()
    mix = get_mix_settings(cfg)
    sa = get_sa_settings(cfg)
    print(f"{len(recs)} kayit\n")

    # 1) WebRTC NS — alfa karisimlarinin girdisi de bunlar
    # Surucu YONTEM adi kosul adindan farklidir: kosul "mild", yontem "webrtc_mild".
    for method, cond in (("webrtc_mild", "denoised_mild"),
                         ("webrtc_aggressive", "denoised_aggressive")):
        d = outdir(cond)
        print(f"=== {method} ===")
        for b in recs:
            o = os.path.join(d, f"{b}.wav")
            if os.path.exists(o):
                continue
            process_single_denoise(method, os.path.join(RAW, f"{b}.wav"), o)
        print(f"  {len([f for f in os.listdir(d) if f.endswith('.wav')])}/{len(recs)}")

    # 2) Donmus birlestirme uyeleri
    jobs = [
        ("oa_aggr_a020", "oa", "denoised_aggressive", 0.20, None),
        ("sa_psm_aggr_a040", "sa", "denoised_aggressive", 0.40, "psm"),
        ("sa_psm_mild_a025", "sa", "denoised_mild", 0.25, "psm"),
    ]
    for cond, kind, src, alpha, variant in jobs:
        d = outdir(cond)
        print(f"=== {cond} ===")
        for b in recs:
            o = os.path.join(d, f"{b}.wav")
            if os.path.exists(o):
                continue
            raw_p = os.path.join(RAW, f"{b}.wav")
            enh_p = os.path.join(PKG_ROOT, "data", f"audio_{src}_en", f"{b}.wav")
            if kind == "oa":
                apply_observation_addition(
                    raw_p, enh_p, o, alpha,
                    rms_align=mix.get("rms_align", True),
                    output_rms_match=mix.get("output_rms_match", True),
                    peak_guard=mix.get("peak_guard", 0.999))
            else:
                apply_spectral_alpha(raw_p, enh_p, o, alpha, variant, sa)
        print(f"  {len([f for f in os.listdir(d) if f.endswith('.wav')])}/{len(recs)}")

    # 3) pcen (opsiyonel besinci uye)
    import numpy as np
    import soundfile as sf
    d = outdir("pcen")
    print("=== pcen ===")
    # nonsuppressive_filters.BUILDERS dogrudan fonksiyon tutar;
    # mundane_perturbations.BUILDERS ise (fonksiyon, rms_bayragi) demeti. Karistirilmisti.
    fn = NSF_BUILDERS["pcen"]
    for b in recs:
        o = os.path.join(d, f"{b}.wav")
        if os.path.exists(o):
            continue
        x, _ = sf.read(os.path.join(RAW, f"{b}.wav"), dtype="float64")
        if x.ndim > 1:
            x = x[:, 0]
        sf.write(o, nsf_finalize(fn(x), x).astype(np.float32), 16000, subtype="PCM_16")
    print(f"  {len([f for f in os.listdir(d) if f.endswith('.wav')])}/{len(recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
