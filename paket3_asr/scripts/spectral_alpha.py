"""
Y1b — Spektral alanda alfa kontrolu (fazi koruyarak).

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Literatur taramasi: docs/MAKALE_LITERATUR_TARAMASI_ASR_ON_ISLEME_2026-09-10.md (Bolum 4.2, 6)

NEDEN BU YONTEM
---------------
Whisper'in ozellik cikarici zinciri (faster_whisper/feature_extractor.py):

    stft       : n_fft=400, hop=160, periyodik Hann, center=True
    magnitudes = |STFT|**2                      <- GUC spektrumu
    mel_spec   = mel_filters @ magnitudes       <- DOGRUSAL matris carpimi
    log_spec   = log10(clip(mel_spec, 1e-10))

mel_spec, guc spektrogramininin DOGRUSAL bir fonksiyonudur. Dolayisiyla guc
alaninda dogrusal karisim, Mel alaninda karisima matematiksel olarak esdegerdir
(log'dan once):

    M @ [(1-a)*P_ham + a*P_iyi] = (1-a)*(M @ P_ham) + a*(M @ P_iyi)

Bu sayede faster-whisper'in API'sine dokunmadan, Whisper'in KENDI analiz
izgarasinda Mel alani karisimi gerceklenebilir. Faz hic degistirilmez; literatur
kestirilmis fazin ZARARLI oldugunu, gurultulu fazi korumanin (gamma=0) optimum
oldugunu gosteriyor (arXiv 2607.11157, p<0.001).

IKI VARYANT
-----------
psm (power spectral mixing):
    P_mix = (1-alpha)*P_ham + alpha*P_iyi ,  genlik = sqrt(P_mix), faz = ham faz
    -> Mel alani dogrusal karisiminin birebir karsiligi.

mex (mask exponent):
    A     = |S_iyi| / |S_ham|            (ortuk maske, ust siniri kirpilir)
    genlik = |S_ham| * A**alpha ,          faz = ham faz
    -> arXiv 2607.11157'nin M_{a,g} = A^a * e^{j*g*phi} formulasyonu, gamma=0.

Iki uc deger her iki varyantta da anlamlidir:
    alpha=0 -> maske/karisim etkisiz; cikti ham sesin STFT gidis-donus yeniden
               kurulumudur (hata olculur ve raporlanir)
    alpha=1 -> psm: iyilestirilmis guc + ham faz
               mex: iyilestirilmis genlik + ham faz
"""

import os
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402

# Whisper'in ozellik cikaricisinin sabitleri (faster_whisper ile birebir)
WHISPER_N_FFT = 400
WHISPER_HOP = 160
WHISPER_SR = 16000


def whisper_window(n_fft: int = WHISPER_N_FFT) -> np.ndarray:
    """faster_whisper ile birebir ayni periyodik Hann penceresi."""
    return np.hanning(n_fft + 1)[:-1].astype(np.float64)


def stft(x: np.ndarray, n_fft: int = WHISPER_N_FFT, hop: int = WHISPER_HOP) -> np.ndarray:
    """
    Whisper ile ayni cozumleme: center=True, 'reflect' dolgu, periyodik Hann.
    Doner: karmasik dizi, bicim (n_fft//2 + 1, kare_sayisi)
    """
    w = whisper_window(n_fft)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * w[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(S: np.ndarray, length: int, n_fft: int = WHISPER_N_FFT, hop: int = WHISPER_HOP) -> np.ndarray:
    """
    Agirlikli ortusen-ekleme (WOLA) ile ters cozumleme.
    Pencere karesiyle normalize edildigi icin NOLA saglandiginda kusursuz
    yeniden kurulum verir (Hann, n_fft=400, hop=160 -> %60 ortusme).
    """
    w = whisper_window(n_fft)
    n_frames = S.shape[1]
    frames = np.fft.irfft(S, n=n_fft, axis=0).T  # (kare, n_fft)

    total = (n_frames - 1) * hop + n_fft
    acc = np.zeros(total, dtype=np.float64)
    wsum = np.zeros(total, dtype=np.float64)
    w2 = w ** 2
    for i in range(n_frames):
        s = i * hop
        acc[s:s + n_fft] += frames[i] * w
        wsum[s:s + n_fft] += w2

    nz = wsum > 1e-12
    acc[nz] /= wsum[nz]

    pad = n_fft // 2
    y = acc[pad:pad + length]
    if len(y) < length:  # kenar guvenligi
        y = np.pad(y, (0, length - len(y)))
    return y


# Blok basina islenecek kare sayisi. Bellek sinirini belirler:
# 4096 kare ~ 41 sn ses; blok dizileri ~25 MB. Tam sinyal STFT'si yerine bloklu
# isleme zorunludur: video_06 (44.9 dk, 43.1M ornek) tek seferde ~2.6 GB/sinyal
# ister ve iki sinyal + ters cozumleme ile bellegi tuketir.
BLOCK_FRAMES = 4096


def build_spectral_alpha(
    raw: np.ndarray,
    enh: np.ndarray,
    alpha: float,
    variant: str,
    mask_clip_max: float = 1.0,
    n_fft: int = WHISPER_N_FFT,
    hop: int = WHISPER_HOP,
    block_frames: int = BLOCK_FRAMES,
) -> np.ndarray:
    """
    Spektral alanda alfa kontrollu karisimi uretir ve dalga formuna dondurur.
    Faz her zaman HAM sesin fazidir (gamma=0).

    Bloklu (streaming) uygulama: kareler block_frames'lik gruplar halinde
    islenir, ters cozumleme agirlikli ortusen-ekleme ile birikime yazilir.
    Bellek kullanimi kayit uzunlugunda dogrusal, blok boyutunda sabittir.
    """
    if variant not in ("psm", "mex"):
        raise ValueError(f"Bilinmeyen varyant '{variant}' (beklenen: psm, mex)")
    if len(raw) != len(enh):
        raise ValueError(f"Girdi uzunluklari esit degil: {len(raw)} != {len(enh)}")

    w = whisper_window(n_fft)
    w2 = w ** 2
    pad = n_fft // 2
    length = len(raw)

    rp = np.pad(raw.astype(np.float64), (pad, pad), mode="reflect")
    ep = np.pad(enh.astype(np.float64), (pad, pad), mode="reflect")

    n_frames = 1 + (len(rp) - n_fft) // hop
    total = (n_frames - 1) * hop + n_fft
    acc = np.zeros(total, dtype=np.float64)
    wsum = np.zeros(total, dtype=np.float64)
    offsets = np.arange(n_fft)

    for start in range(0, n_frames, block_frames):
        stop = min(start + block_frames, n_frames)
        idx = offsets[None, :] + hop * np.arange(start, stop)[:, None]

        S_raw = np.fft.rfft(rp[idx] * w, n=n_fft, axis=1)
        S_enh = np.fft.rfft(ep[idx] * w, n=n_fft, axis=1)

        A_raw = np.abs(S_raw)
        A_enh = np.abs(S_enh)
        phase = np.exp(1j * np.angle(S_raw))

        if variant == "psm":
            # Guc alaninda dogrusal karisim (Mel alani karisiminin esdegeri)
            P_mix = (1.0 - alpha) * (A_raw ** 2) + alpha * (A_enh ** 2)
            A_out = np.sqrt(np.maximum(P_mix, 0.0))
        else:
            # Ortuk maske ussu: A^alpha
            with np.errstate(divide="ignore", invalid="ignore"):
                mask = np.where(A_raw > 1e-12, A_enh / A_raw, 1.0)
            mask = np.clip(
                np.nan_to_num(mask, nan=1.0, posinf=mask_clip_max), 0.0, mask_clip_max
            )
            A_out = A_raw * (mask ** alpha)

        frames = np.fft.irfft(A_out * phase, n=n_fft, axis=1) * w
        for j, k in enumerate(range(start, stop)):
            o = k * hop
            acc[o:o + n_fft] += frames[j]
            wsum[o:o + n_fft] += w2

    nz = wsum > 1e-12
    acc[nz] /= wsum[nz]

    y = acc[pad:pad + length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y


def roundtrip_error(x: np.ndarray, n_fft: int = WHISPER_N_FFT, hop: int = WHISPER_HOP) -> float:
    """STFT gidis-donus hatasi (alpha=0 ozdesligi icin ust sinir)."""
    y = istft(stft(x, n_fft, hop), length=len(x), n_fft=n_fft, hop=hop)
    return float(np.max(np.abs(y - x)))


# ---------------------------------------------------------------------------
# experiment.yaml -> spectral_alpha blogunun kosullara genisletilmesi
# ---------------------------------------------------------------------------

def _alpha_pct(alpha: float) -> int:
    return int(round(float(alpha) * 100))


def expand_sa_conditions(cfg: dict = None) -> list:
    """spectral_alpha blogunu somut kosul kayitlarina genisletir."""
    if cfg is None:
        cfg = load_experiment_config()
    sa = cfg.get("spectral_alpha") or {}
    if not sa.get("enabled", False):
        return []

    id_tpl = sa.get("id_template", "sa_{variant}_{short}_a{alpha_pct:03d}")
    dir_tpl = sa.get("dir_template", "data/audio_sa_{variant}_{short}_a{alpha_pct:03d}")
    method_tpl = sa.get("method_template", "sa_{variant}_{enhancer}_alpha{alpha:.2f}")

    out = []
    for variant in sa.get("variants", []):
        vid = variant["id"] if isinstance(variant, dict) else str(variant)
        for enh in sa.get("base_enhancers", []):
            enh_id = enh["id"] if isinstance(enh, dict) else str(enh)
            short = enh.get("short", enh_id) if isinstance(enh, dict) else enh_id
            for alpha in sa.get("alphas", []):
                alpha = float(alpha)
                fields = {
                    "variant": vid, "short": short, "enhancer": enh_id,
                    "alpha": alpha, "alpha_pct": _alpha_pct(alpha),
                }
                out.append({
                    "id": id_tpl.format(**fields),
                    "variant": vid,
                    "short": short,
                    "enhancer": enh_id,
                    "alpha": alpha,
                    "alpha_pct": _alpha_pct(alpha),
                    "dir": os.path.abspath(os.path.join(PKG_ROOT, dir_tpl.format(**fields))),
                    "method": method_tpl.format(**fields),
                })
    return out


def sa_condition_map(cfg: dict = None) -> dict:
    """benchmark.py CONDITION_MAP bicimi."""
    return {
        c["id"]: {"dir": c["dir"], "denoiser_method": c["method"]}
        for c in expand_sa_conditions(cfg)
    }


def get_sa_settings(cfg: dict = None) -> dict:
    if cfg is None:
        cfg = load_experiment_config()
    sa = cfg.get("spectral_alpha") or {}
    st = sa.get("stft") or {}
    return {
        "n_fft": int(st.get("n_fft", WHISPER_N_FFT)),
        "hop_length": int(st.get("hop_length", WHISPER_HOP)),
        "mask_clip_max": float(sa.get("mask_clip_max", 1.0)),
        "output_rms_match": bool(sa.get("output_rms_match", True)),
        "peak_guard": float(sa.get("peak_guard", 0.999)),
        "verify_endpoints": bool(sa.get("verify_endpoints", True)),
    }


if __name__ == "__main__":
    cfg = load_experiment_config()
    conds = expand_sa_conditions(cfg)
    print(f"Genisletilen Y1b kosulu: {len(conds)}")
    for c in conds:
        print(f"  {c['id']:<26} varyant={c['variant']:<4} alpha={c['alpha']:.2f} "
              f"enhancer={c['enhancer']}")
    print(f"\nAyarlar: {get_sa_settings(cfg)}")
    print(f"Whisper izgarasi: n_fft={WHISPER_N_FFT}, hop={WHISPER_HOP}, sr={WHISPER_SR}")
