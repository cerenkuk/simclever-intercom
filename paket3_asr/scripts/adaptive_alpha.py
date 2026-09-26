"""
Y2 — Kesit bazli uyarlamali alfa (zamanla degisen siddet zarfi).

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Literatur taramasi: docs/MAKALE_LITERATUR_TARAMASI_ASR_ON_ISLEME_2026-09-10.md (Bolum 5)

NEDEN BU YONTEM — MAKALENIN ASIL KATKISI
----------------------------------------
Sabit alfa taramasi Whisper icin ZATEN YAYINLANMIS durumda (arXiv 2607.11157:
U bicimli egri, optimum alpha* ~= 0.25). Ayni makale kapatilmamis boslugu da
niceliyor: alfa KAYIT/KESIT BASINA secilebilse %20-30 bagil WER kazanci var,
ancak bu gerceklenememis. Y2 tam olarak bu boslugu hedefler.

Fikir: alfa bir skaler degil, cerceve basina degisen bir ZARF olsun.

    y[t] = ters_cozumleme( karisim(S_ham[t], S_iyi[t], alpha[t]) )

Sabit zarf (alpha[t] = sabit) Y1b'nin ta kendisidir. Yani Y2, Y1b'nin ve
dolayisiyla ham sesin (alpha=0) uzerine kurulu bir UST KUMEDIR; optimumda
tabandan kotu olmak yapisal olarak mumkun degildir.

IKI KARAR KURALI
----------------
vad  : Konusma / konusma-disi ayrimi.
       Hipotez: denoise ASR'i bozuyorsa bunun nedeni KONUSMAYI hasarlamasidir.
       Oyleyse iyilestirmeyi yalniz konusma-disi cerecevelere uygulayip konusmaya
       hic dokunmamak, gurultu azaltmanin faydasini hasar olmadan verir.
       Bu yanlislanabilir ve net bir hipotezdir: alpha_speech=0, alpha_nonspeech=1.
       Ayrica Whisper'in bilinen halusinasyon modu sessizlik/gurultu
       bolgelerinde tetiklenir; oralari temizlemek dogrudan o modu hedefler.

snr  : Cerceve basina kestirilen SNR'a gore surekli siddet.
       Dusuk SNR -> daha yuksek alfa (temizlemenin faydasi var)
       Yuksek SNR -> daha dusuk alfa (temiz konusmada iyilestirme yalniz zarar verir)
       Literaturun SNR-uyarlamali kopru modullerinin egitimsiz karsiligi.

ASIRI BASTIRMA KORUMASI
-----------------------
Whisper'in baskin hata modu asiri bastirmadir (arXiv 2607.11157). Iyilestiricinin
cok sey sildigi cercevelerde alfa ust sinirlanir; boylece kural ne derse desin
konusma hasari sinirlanir.

ZARF YUMUSATMA
--------------
Alfa cerceveler arasinda ani degisirse genlik modulasyonu artefakti dogar.
Zarf, hareketli ortalama ile yumusatilir (varsayilan ~120 ms).
"""

import os
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from oa_conditions import PKG_ROOT, load_experiment_config  # noqa: E402
from spectral_alpha import (  # noqa: E402
    BLOCK_FRAMES,
    WHISPER_HOP,
    WHISPER_N_FFT,
    WHISPER_SR,
    whisper_window,
)

EPS = 1e-12


# ---------------------------------------------------------------------------
# Cerceve ozellikleri
# ---------------------------------------------------------------------------

def frame_power(x: np.ndarray, n_fft: int = WHISPER_N_FFT, hop: int = WHISPER_HOP,
                block_frames: int = BLOCK_FRAMES) -> np.ndarray:
    """
    Cerceve basina toplam guc (bloklu; bellek kayit uzunlugundan bagimsiz).
    Whisper ile ayni cozumleme izgarasi kullanilir.
    """
    w = whisper_window(n_fft)
    pad = n_fft // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    offsets = np.arange(n_fft)
    out = np.empty(n_frames, dtype=np.float64)
    for start in range(0, n_frames, block_frames):
        stop = min(start + block_frames, n_frames)
        idx = offsets[None, :] + hop * np.arange(start, stop)[:, None]
        S = np.fft.rfft(xp[idx] * w, n=n_fft, axis=1)
        out[start:stop] = np.sum(np.abs(S) ** 2, axis=1)
    return out


def noise_floor_db(power: np.ndarray, percentile: float = 10.0) -> float:
    """
    Gurultu tabani kestirimi: cerceve guclerinin alt yuzdeligi.
    Konusmanin cerecevelerin cogunlugunu kaplamadigi varsayimina dayanir; tibbi
    gorusme kayitlarinda duraklamalar bol oldugu icin guvenlidir.
    """
    p = np.maximum(power, EPS)
    return float(10.0 * np.log10(np.percentile(p, percentile)))


def frame_snr_db(power: np.ndarray, floor_db: float) -> np.ndarray:
    """Cerceve basina gurultu tabanina gore SNR (dB)."""
    return 10.0 * np.log10(np.maximum(power, EPS)) - floor_db


def suppression_db(power_raw: np.ndarray, power_enh: np.ndarray) -> np.ndarray:
    """
    Iyilestiricinin her cercevede ne kadar enerji sildigi (dB, pozitif = bastirma).
    Asiri bastirma korumasinin girdisi budur.
    """
    return 10.0 * np.log10(np.maximum(power_raw, EPS) / np.maximum(power_enh, EPS))


# ---------------------------------------------------------------------------
# Alfa zarfi
# ---------------------------------------------------------------------------

def smooth_envelope(env: np.ndarray, win_frames: int) -> np.ndarray:
    """Hareketli ortalama ile yumusatma (kenarlar kenar degeriyle doldurulur)."""
    if win_frames <= 1 or len(env) == 0:
        return env
    k = int(win_frames)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(env, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(padded, kernel, mode="valid")


def alpha_envelope(rule: str, params: dict, snr_db: np.ndarray,
                   supp_db: np.ndarray) -> np.ndarray:
    """
    Karar kuralindan cerceve basina alfa uretir (yumusatma ve koruma oncesi).
    """
    if rule == "vad":
        thr = float(params.get("vad_threshold_db", 6.0))
        a_sp = float(params.get("alpha_speech", 0.0))
        a_ns = float(params.get("alpha_nonspeech", 1.0))
        is_speech = snr_db >= thr
        return np.where(is_speech, a_sp, a_ns)

    if rule == "snr":
        a_lo = float(params.get("alpha_lo", 0.05))
        a_hi = float(params.get("alpha_hi", 0.40))
        mid = float(params.get("snr_mid_db", 10.0))
        slope = max(float(params.get("snr_slope_db", 6.0)), 1e-6)
        # Dusuk SNR -> w ~ 1 (daha cok temizle), yuksek SNR -> w ~ 0
        w = 1.0 / (1.0 + np.exp((snr_db - mid) / slope))
        return a_lo + (a_hi - a_lo) * w

    if rule == "constant":
        # Dogrulama amacli: sabit zarf Y1b'yi birebir yeniden uretmelidir
        return np.full_like(snr_db, float(params.get("alpha", 0.25)))

    raise ValueError(f"Bilinmeyen kural '{rule}' (beklenen: vad, snr, constant)")


def apply_oversuppression_guard(env: np.ndarray, supp_db: np.ndarray,
                                params: dict) -> np.ndarray:
    """
    Iyilestiricinin cok sey sildigi cercevelerde alfayi ust sinirlar.
    guard_db esiginin uzerinde alfa guard_alpha degerini asamaz.
    """
    guard_db = params.get("oversuppression_guard_db")
    if guard_db is None:
        return env
    guard_alpha = float(params.get("oversuppression_guard_alpha", 0.25))
    return np.where(supp_db > float(guard_db), np.minimum(env, guard_alpha), env)


# ---------------------------------------------------------------------------
# Uyarlamali karisim
# ---------------------------------------------------------------------------

def build_adaptive_alpha(
    raw: np.ndarray,
    enh: np.ndarray,
    rule: str,
    params: dict,
    variant: str = "psm",
    smooth_ms: float = 120.0,
    mask_clip_max: float = 1.0,
    n_fft: int = WHISPER_N_FFT,
    hop: int = WHISPER_HOP,
    block_frames: int = BLOCK_FRAMES,
    return_envelope: bool = False,
):
    """
    Zamanla degisen alfa zarfiyla spektral karisim uretir.
    Faz her zaman HAM sesin fazidir (gamma=0).

    Sabit zarf verildiginde (rule='constant') cikti build_spectral_alpha ile
    birebir ayni olmalidir; bu, Y2'nin Y1b'yi kapsadiginin dogrulamasidir.
    """
    if variant not in ("psm", "mex"):
        raise ValueError(f"Bilinmeyen varyant '{variant}' (beklenen: psm, mex)")
    if len(raw) != len(enh):
        raise ValueError(f"Girdi uzunluklari esit degil: {len(raw)} != {len(enh)}")

    # --- 1. gecis: cerceve ozellikleri ---
    p_raw = frame_power(raw, n_fft, hop, block_frames)
    p_enh = frame_power(enh, n_fft, hop, block_frames)
    floor = noise_floor_db(p_raw, float(params.get("noise_floor_percentile", 10.0)))
    snr = frame_snr_db(p_raw, floor)
    supp = suppression_db(p_raw, p_enh)

    env = alpha_envelope(rule, params, snr, supp)
    env = apply_oversuppression_guard(env, supp, params)
    # Cerceve hizi = ornekleme / hop (16000/160 = 100 kare/sn)
    win = max(1, int(round(smooth_ms / 1000.0 * (WHISPER_SR / hop))))
    env = smooth_envelope(env, win)
    env = np.clip(env, 0.0, 1.0)

    # --- 2. gecis: karisim ---
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
        a = env[start:stop][:, None]  # cerceve basina alfa

        if variant == "psm":
            P_mix = (1.0 - a) * (A_raw ** 2) + a * (A_enh ** 2)
            A_out = np.sqrt(np.maximum(P_mix, 0.0))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                mask = np.where(A_raw > 1e-12, A_enh / A_raw, 1.0)
            mask = np.clip(
                np.nan_to_num(mask, nan=1.0, posinf=mask_clip_max), 0.0, mask_clip_max
            )
            A_out = A_raw * (mask ** a)

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

    if return_envelope:
        return y, {"envelope": env, "snr_db": snr, "suppression_db": supp,
                   "noise_floor_db": floor, "smooth_frames": win}
    return y


def envelope_summary(env: np.ndarray) -> dict:
    """Zarfin ozeti — raporlama ve akil saglligi kontrolu icin."""
    return {
        "alpha_mean": float(np.mean(env)),
        "alpha_min": float(np.min(env)),
        "alpha_max": float(np.max(env)),
        "alpha_p10": float(np.percentile(env, 10)),
        "alpha_p90": float(np.percentile(env, 90)),
        "frac_untouched": float(np.mean(env < 0.01)),   # ham birakilan cerceve orani
        "frac_full": float(np.mean(env > 0.99)),        # tam iyilestirilen oran
    }


# ---------------------------------------------------------------------------
# experiment.yaml -> adaptive_alpha blogunun kosullara genisletilmesi
# ---------------------------------------------------------------------------

def expand_aa_conditions(cfg: dict = None) -> list:
    """adaptive_alpha blogunu somut kosul kayitlarina genisletir."""
    if cfg is None:
        cfg = load_experiment_config()
    aa = cfg.get("adaptive_alpha") or {}
    if not aa.get("enabled", False):
        return []

    id_tpl = aa.get("id_template", "aa_{preset}_{short}")
    dir_tpl = aa.get("dir_template", "data/audio_aa_{preset}_{short}")
    method_tpl = aa.get("method_template", "aa_{preset}_{enhancer}")
    variant = aa.get("variant", "psm")
    smooth_ms = float(aa.get("smooth_ms", 120.0))

    out = []
    for preset in aa.get("presets", []):
        pid = preset["id"]
        params = {k: v for k, v in preset.items() if k not in ("id", "name", "rule")}
        for enh in aa.get("base_enhancers", []):
            enh_id = enh["id"] if isinstance(enh, dict) else str(enh)
            short = enh.get("short", enh_id) if isinstance(enh, dict) else enh_id
            fields = {"preset": pid, "short": short, "enhancer": enh_id}
            out.append({
                "id": id_tpl.format(**fields),
                "preset": pid,
                "name": preset.get("name", pid),
                "rule": preset["rule"],
                "params": params,
                "variant": variant,
                "smooth_ms": smooth_ms,
                "enhancer": enh_id,
                "short": short,
                "dir": os.path.abspath(os.path.join(PKG_ROOT, dir_tpl.format(**fields))),
                "method": method_tpl.format(**fields),
            })
    return out


def aa_condition_map(cfg: dict = None) -> dict:
    """benchmark.py CONDITION_MAP bicimi."""
    return {
        c["id"]: {"dir": c["dir"], "denoiser_method": c["method"]}
        for c in expand_aa_conditions(cfg)
    }


def get_aa_settings(cfg: dict = None) -> dict:
    if cfg is None:
        cfg = load_experiment_config()
    aa = cfg.get("adaptive_alpha") or {}
    return {
        "variant": aa.get("variant", "psm"),
        "smooth_ms": float(aa.get("smooth_ms", 120.0)),
        "mask_clip_max": float(aa.get("mask_clip_max", 1.0)),
        "output_rms_match": bool(aa.get("output_rms_match", True)),
        "peak_guard": float(aa.get("peak_guard", 0.999)),
        "verify_constant_equivalence": bool(aa.get("verify_constant_equivalence", True)),
    }


if __name__ == "__main__":
    cfg = load_experiment_config()
    conds = expand_aa_conditions(cfg)
    print(f"Genisletilen Y2 kosulu: {len(conds)}")
    for c in conds:
        print(f"  {c['id']:<22} kural={c['rule']:<9} iyilestirici={c['enhancer']}")
        print(f"      {c['name']}")
        print(f"      params={c['params']}")
    print(f"\nAyarlar: {get_aa_settings(cfg)}")
