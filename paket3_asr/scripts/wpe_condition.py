"""
Y3 — WPE dereverberation: DEKORELE birlestirme uyesi.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Literatur taramasi: docs/MAKALE_LITERATUR_TARAMASI_ASR_ON_ISLEME_2026-09-10.md (Bolum 7)

NEDEN BU YONTEM — ILINTI SORUNU
-------------------------------
Birlestirme (Y4) su ana kadarki en iyi sonucu verdi, ancak uye ekleme kazanci
sinirli kaldi. Nedeni olculdu: TUM uyelerimiz birbirine %83-88 benziyor, cunku
hepsi AYNI WebRTC NS filtresinin farkli siddetleri/alanlari. Yani bagimsiz
bilgi katmiyorlar; ilintili oy katiyorlar ve ham capayi haksiz yere deviriyorlar.

WPE tamamen FARKLI bir mekanizmadir:
  - WebRTC NS  : toplamsal gurultuyu spektral kazanc maskesiyle bastirir
  - WPE        : dogrusal ongoru ile gec yankiyi (reverberation) giderir

Farkli bozulma modeli -> farkli artefakt profili -> farkli hatalar -> birlestirme
icin GERCEKTEN dekorele uye. Hedef, uyeler arasi benzerligi %85 bandinin altina
indirmek.

Literatur ayrica WPE'yi "hedef konusmaya cok az bozulma sokan" tek aile olarak
isaretliyor; yani tek basina da en azindan zarar vermemesi beklenir.

NOT — Paket 2'de WPE neden elenmisti
------------------------------------
%25+ CPU yuku ve cerceve gecikmesi nedeniyle CANLI AKIS icin elenmisti.
Bizim ASR hattimiz cevrimdisi oldugundan o kisit burada gecerli DEGILDIR.
Bu gerekce makalede acikca belirtilmelidir.

Kullanim:
    python paket3_asr/scripts/wpe_condition.py --dry-run
    python paket3_asr/scripts/wpe_condition.py
    python paket3_asr/scripts/wpe_condition.py --taps 5 --delay 3
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402
from spectral_alpha import WHISPER_HOP, WHISPER_N_FFT, whisper_window  # noqa: E402

AUDIO_16K = os.path.join(PKG_ROOT, "data", "audio_16k")

REC10 = [
    "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
    "saha_04_vid_20220913_142911", "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez", "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi", "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
]


def _wpe_core(x: np.ndarray, taps: int, delay: int, iterations: int,
              n_fft: int, hop: int) -> np.ndarray:
    """
    Tek kanalli WPE ile gec yanki giderme.

    WPE kendi cozumleme izgarasinda calisir (n_fft=512, hop=128 tipik degerler);
    Whisper'in izgarasi kullanilmaz cunku WPE dogrusal ongoru icin daha uzun
    pencere ister. Cikti dalga formuna dondurulur ve hattaki diger kosullar gibi
    islenir.

    Bellek kayit uzunluguyla buyur; parcalamayi cagiran wpe_dereverb yonetir.
    """
    from nara_wpe.wpe import wpe

    w = np.hanning(n_fft + 1)[:-1]
    pad = n_fft // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    S = np.fft.rfft(xp[idx] * w, n=n_fft, axis=1).T          # (freq, frames)

    # nara_wpe (kanal, frekans, zaman) bekler -> tek kanal ekseni ekle
    Y = S[None, :, :]
    Z = wpe(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full")
    S_out = Z[0]

    # Agirlikli ortusen-ekleme ile geri sentez
    frames = np.fft.irfft(S_out, n=n_fft, axis=0).T
    total = (n_frames - 1) * hop + n_fft
    acc = np.zeros(total); wsum = np.zeros(total)
    w2 = w ** 2
    for i in range(n_frames):
        o = i * hop
        acc[o:o + n_fft] += frames[i] * w
        wsum[o:o + n_fft] += w2
    nz = wsum > 1e-12
    acc[nz] /= wsum[nz]
    y = acc[pad:pad + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def wpe_dereverb(x: np.ndarray, taps: int, delay: int, iterations: int,
                 n_fft: int = 512, hop: int = 128,
                 block_s: float = 240.0, overlap_s: float = 10.0,
                 sr: int = 16000) -> np.ndarray:
    """
    Parcali (blokli) WPE.

    NEDEN: WPE dogrusal ongoru istatistiklerini tum zaman ekseninden kestirir ve
    korelasyon matrisleri kare sayisiyla buyur. OLCULDU: video_01 (17.1 dk) tek
    parcada ~8.8 GB tuketti; video_06 (44.9 dk) ~23 GB ister ve 32 GB'lik
    makinede 11.6 GB bos alanla CALISMAZ.

    Cozum: kayit block_s uzunlugunda, overlap_s ortusmeli parcalara bolunur;
    her parcaya WPE ayri uygulanir, ortusme bolgelerinde capraz gecisle
    birlestirilir. Istatistikler parca basina kestirildigi icin tam sinyal
    cozumunden kucuk farklar olur; yanki yerel bir olgu oldugundan bu kabul
    edilebilir ve uzun kayitlarda standart uygulamadir.
    """
    n = len(x)
    blk = int(block_s * sr)
    ov = int(overlap_s * sr)
    if n <= blk:
        return _wpe_core(x, taps, delay, iterations, n_fft, hop)

    out = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = min(start + blk, n)
        seg = x[start:stop]
        if len(seg) < n_fft * 4:
            y_seg = seg.copy()
        else:
            y_seg = _wpe_core(seg, taps, delay, iterations, n_fft, hop)
        win = np.ones(len(seg))
        f = min(ov, len(seg) // 2)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f)
            if start > 0:
                win[:f] = ramp
            if stop < n:
                win[-f:] = ramp[::-1]
        out[start:stop] += y_seg * win
        wsum[start:stop] += win
        if stop >= n:
            break
        start = stop - ov
    nz = wsum > 1e-9
    out[nz] /= wsum[nz]
    return out


def finalize(y: np.ndarray, raw: np.ndarray, peak_guard: float = 0.999) -> np.ndarray:
    """Hattin diger kosullariyla ayni son isleme: seviye sabitleme + tepe koruma."""
    r = float(np.sqrt(np.mean(raw ** 2))); m = float(np.sqrt(np.mean(y ** 2)))
    if m > 1e-9 and r > 1e-9:
        y = y * (r / m)
    guard = max(peak_guard, float(np.max(np.abs(raw))))
    pk = float(np.max(np.abs(y)))
    if pk > guard:
        y = y * (guard / pk)
    return y


def main():
    ap = argparse.ArgumentParser(description="Y3 — WPE dereverberation kosulu uretimi")
    ap.add_argument("--taps", type=int, default=10, help="Dogrusal ongoru tap sayisi")
    ap.add_argument("--delay", type=int, default=3, help="Ongoru gecikmesi (erken yankiyi korur)")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--block-s", type=float, default=240.0,
                    help="Parca uzunlugu (sn); bellek bu degerle olceklenir")
    ap.add_argument("--overlap-s", type=float, default=10.0,
                    help="Parcalar arasi ortusme (sn), capraz gecis icin")
    ap.add_argument("--basenames", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cond_id = f"wpe_t{args.taps:02d}d{args.delay}"
    out_dir = os.path.join(PKG_ROOT, "data", f"audio_{cond_id}")
    recs = args.basenames or REC10

    print(f"Kosul    : {cond_id}")
    print(f"WPE      : taps={args.taps} delay={args.delay} iterations={args.iterations}")
    print(f"Cozumleme: n_fft={args.n_fft} hop={args.hop}  (Whisper izgarasi DEGIL)")
    print(f"Dizin    : {os.path.relpath(out_dir, os.path.dirname(PKG_ROOT))}")
    print(f"Kayitlar : {len(recs)}")
    if args.dry_run:
        print("\n[DRY-RUN] Hicbir sey uretilmedi.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    for b in recs:
        out_path = os.path.join(out_dir, f"{b}.wav")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[ATLANDI] {b}")
            continue
        raw, sr = sf.read(os.path.join(AUDIO_16K, f"{b}.wav"), dtype="float64")
        if raw.ndim > 1:
            raw = raw[:, 0]
        t0 = time.time()
        try:
            y = wpe_dereverb(raw, args.taps, args.delay, args.iterations,
                             n_fft=args.n_fft, hop=args.hop,
                             block_s=args.block_s, overlap_s=args.overlap_s)
        except MemoryError:
            print(f"[HATA] Bellek yetmedi: {b} ({len(raw)/16000/60:.1f} dk)", file=sys.stderr)
            continue
        y = finalize(y, raw)
        sf.write(out_path, y.astype(np.float32), 16000, subtype="PCM_16")
        el = time.time() - t0
        dur = len(raw) / 16000
        print(f"[BASARILI] {b:<45}{dur/60:5.1f} dk  islem {el:6.1f} sn  RTF {el/dur:.3f}")

    print(f"\nSonraki adim:")
    print(f"   python paket3_asr/scripts/benchmark.py --basename <kayit> "
          f"--model-id faster-whisper-large-v3-int8 --condition {cond_id} --run-purpose main_benchmark")
    print(f"   (benchmark.py'nin bu kosulu tanimasi icin experiment.yaml'a eklenmeli)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
