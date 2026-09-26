"""
D7 — SIRADAN BOZMALARLA KONTROL DENEYI (Ek-2 Rev B, Bolum 10).

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Direktif         : Ek-2 Rev B, 19 Eylul 2026

SORU
----
Tezimiz "on isleme, CESITLILIK ureterek yararli olur" diyor. Hakemin soracagi
soru: cesitlilik icin ON ISLEME gerekli mi? Yoksa herhangi bir kucuk bozma da
ayni iSi gorur mu?

Bu betik, on isleme YAPMAYAN ama sinyali hafifce degistiren uc "sradan" bozma
uretir. Ayni oylama, ayni ham capa, ayni marj ile birlestirilir.

  Sonuc A — benzer kazanc cikarsa: tez "on isleme" degil "test-zamani bozma
            cesitliligi" olarak yeniden yazilir. Daha genel ve daha ucuz bulgu.
  Sonuc B — belirgin daha az kazanc cikarsa: on islemenin urettigi cesitliligin
            farkli oldugu kanitlanir; tez guclenir.

Hangi sonuc cikarsa ciksin makaleye girer.

KOSULLAR
--------
pert_speed  %+2 hiz (yeniden ornekleme; perde de kayar — kasitli, "sradan" olan bu)
pert_gain   -3 dB sabit kazanc; RMS ESITLEME YOK (aksi halde islem kimlik olurdu)
pert_clip   hafif kirpma: orneklerin ~%0.5'i yumusak doyuma girer

TASARIM NOTU — neden pert_gain'de RMS esitleme yok
--------------------------------------------------
Hattin diger tum kosullarinda cikis, ham sesin RMS'ine esitlenir; bu, kazanc ile
ses duzeyinin karismasini onler (Y1'de olculen tuzak). Ama sabit kazanc + RMS
esitleme = KIMLIK islemi olurdu, yani kosul hic var olmazdi. Bu yuzden burada
esitleme atlanir ve bu sapma acikca raporlanir.

Kullanim:
    python paket3_asr/scripts/mundane_perturbations.py --verify
    python paket3_asr/scripts/mundane_perturbations.py
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from oa_conditions import PKG_ROOT  # noqa: E402

AUDIO_16K = os.path.join(PKG_ROOT, "data", "audio_16k")
SR = 16000

REC10 = [
    "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
    "saha_04_vid_20220913_142911", "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez", "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi", "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
]


def speed_change(x, factor=1.02):
    """
    %+2 hiz: dogrusal yeniden ornekleme. Perde de kayar; bu kasitlidir —
    "sradan bozma" derken kastedilen tam olarak budur (zaman esneme gibi
    islenmis bir donusum degil).
    """
    n_out = int(round(len(x) / factor))
    t = np.linspace(0.0, len(x) - 1.0, n_out)
    return np.interp(t, np.arange(len(x)), x)


def gain_change(x, db=-3.0):
    """Sabit kazanc. RMS esitleme UYGULANMAZ (bkz. modul basligi)."""
    return x * (10.0 ** (db / 20.0))


def light_clip(x, frac=0.005):
    """
    Hafif kirpma: orneklerin ~%frac kadari yumusak doyuma girer. Esik,
    kaydin kendi genlik dagilimindan secilir; boylece her kayitta ayni
    ORAN kirpilir, ayni mutlak esik degil.
    """
    thr = float(np.quantile(np.abs(x), 1.0 - frac))
    if thr <= 1e-9:
        return x.copy()
    return thr * np.tanh(x / thr)


BUILDERS = {
    "pert_speed": (speed_change, True),   # (fonksiyon, rms_esitle)
    "pert_gain":  (gain_change, False),
    "pert_clip":  (light_clip, True),
}


def finalize(y, raw, rms_match=True, peak_guard=0.999):
    """Hattin ortak son islemesi. rms_match=False yalniz pert_gain icin."""
    if rms_match:
        r = float(np.sqrt(np.mean(raw ** 2)))
        m = float(np.sqrt(np.mean(y ** 2)))
        if m > 1e-9 and r > 1e-9:
            y = y * (r / m)
    g = max(peak_guard, float(np.max(np.abs(raw))))
    pk = float(np.max(np.abs(y)))
    if pk > g:
        y = y * (g / pk)
    return y


def verify(records):
    """
    Bozmalarin gercekten KUCUK oldugunu ve kimlik olmadigini gosterir.
    Ikisi de gerekli: kimlik olurlarsa kosul yok demektir; buyuk olurlarsa
    "sradan bozma" iddiasi cokerdi.
    """
    print(f"{'kayit':<40}{'kosul':<12}{'ilinti':>9}{'RMS dB':>9}{'sure %':>9}")
    print("-" * 80)
    for b in records:
        raw, _ = sf.read(os.path.join(AUDIO_16K, f"{b}.wav"), dtype="float64")
        if raw.ndim > 1:
            raw = raw[:, 0]
        for c, (fn, rms) in BUILDERS.items():
            y = finalize(fn(raw), raw, rms_match=rms)
            n = min(len(raw), len(y))
            corr = float(np.corrcoef(raw[:n], y[:n])[0, 1])
            rdb = 20.0 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(raw ** 2)))
            print(f"{b:<40}{c:<12}{corr:>+9.3f}{rdb:>+9.2f}{100.0*len(y)/len(raw):>9.1f}")
    print("-" * 80)
    print("beklenti: hicbiri kimlik degil (ilinti < 1.000) ve hicbiri yikici degil")


def main():
    ap = argparse.ArgumentParser(description="D7 — sradan bozmalar")
    ap.add_argument("--conditions", nargs="*", default=list(BUILDERS))
    ap.add_argument("--basenames", nargs="*", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    records = args.basenames or REC10
    if args.verify:
        verify(records[:2])
        return 0

    for c in args.conditions:
        fn, rms = BUILDERS[c]
        out_dir = os.path.join(PKG_ROOT, "data", f"audio_{c}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nKosul: {c}   (rms_esitle={rms})")
        for b in records:
            op = os.path.join(out_dir, f"{b}.wav")
            if os.path.exists(op) and not args.overwrite:
                print(f"[ATLANDI] {b}")
                continue
            raw, _ = sf.read(os.path.join(AUDIO_16K, f"{b}.wav"), dtype="float64")
            if raw.ndim > 1:
                raw = raw[:, 0]
            t0 = time.time()
            y = finalize(fn(raw), raw, rms_match=rms)
            sf.write(op, y.astype(np.float32), SR, subtype="PCM_16")
            print(f"[BASARILI] {b:<45}{len(raw)/SR/60:5.1f} dk  {time.time()-t0:5.1f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
