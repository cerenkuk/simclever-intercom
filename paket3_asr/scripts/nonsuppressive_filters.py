"""
Y6 — BASTIRMA YAPMAYAN filtreleme ailesi.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Sonuclar         : docs/MAKALE_SONUCLAR_2026-09-13.md

NEDEN BU AILE
-------------
Simdiye kadar denenen dokuz yontemin TAMAMI bastirma temelli: gurultuyu kestirip
cikarmaya calisiyorlar (WebRTC NS, WPE, DTLN ve bunlarin alfa karisimlari).
Olculen takas — kalite ile ham sesten uzaklasma ters bagli, Pearson r = -0.741 —
tam olarak bunun bedeli: gurultu kestirimi hatalidir, hata konusmaya da bulasir.

Bu ailenin ayirt edici ozelligi: GURULTU KESTIRMEZ. Sinyalin hangi kisminin
konusma hangi kisminin gurultu oldugu hakkinda hicbir karar vermez; yalnizca
sabit ya da yavas degisen bir tayfsal bicimleme uygular. Dolayisiyla bastirma
artefakti uretmesi icin bir mekanizma yoktur.

Bu, takas egrisinden KACMA gerekcesi olan ilk aile. Diger dokuzunda boyle bir
gerekce yoktu.

HEDEF FARKI — onemli
--------------------
Y3/Y5 birlestirme UYESI ariyordu (dusuk ilinti + basabas kalite). Bu aile
DOGRUDAN IYILESME ariyor, yani calismanin ilk hedefi: ham sesi tek basina gecmek.
Beklenen konum takas egrisinin "sikici" kosesidir (cok yuksek benzerlik, cok
dusuk hasar); orada birlestirmeye katki olmaz ama dogrudan kazanc olabilir.

KOSULLAR
--------
hp80   Yuksek geciren, 80 Hz, sifir fazli (filtfilt).
       Gerekce: saha kayitlarinda klima/ugultu enerjisi konusma F0'inin altinda.
       Whisper'in mel suzgec bankasi 0 Hz'den baslar; bu enerji dinamik araligi
       yer. Konusma icerigine dokunmaz.

tilt   Nazik yuksek raf, 1 kHz uzeri +4 dB (STFT alaninda, faz korunur).
       Gerekce: oda ve mikrofon tepkisi yuksek frekansi soner; surtunmeli ve
       patlamali seslerin ayirt edici enerjisi 2-8 kHz bandindadir.

pcen   Bant basina enerji normalizasyonu (PCEN'in AGC bileseni), zaman degisken
       kazanc olarak STFT alaninda, faz korunur.
       Gerekce: ASR on yuzu icin tasarlanmis tek yontem budur (Wang ve ark.);
       log-mel'in yerine gecmek uzere onerildi. Kutuphaneyi degistirmemek icin
       kazanc bicimine cevrilip yeniden sentezlenir. Gurultu kestirmez, yalnizca
       her bandin kendi yavas ortalamasina gore olceklenir.

Hepsinde hattin ortak son islemesi uygulanir (ham RMS'e esitleme + tepe koruma),
boylece kazanc ile ses duzeyi karismaz — Y1'de bu karisiklik olculmustu.

Kullanim:
    python paket3_asr/scripts/nonsuppressive_filters.py --verify
    python paket3_asr/scripts/nonsuppressive_filters.py --conditions hp80 tilt pcen
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

from oa_conditions import PKG_ROOT, development_subset, load_experiment_config  # noqa: E402
from spectral_alpha import WHISPER_HOP, WHISPER_N_FFT, WHISPER_SR, istft, stft  # noqa: E402

AUDIO_16K = os.path.join(PKG_ROOT, "data", "audio_16k")

CONDITIONS = ("hp80", "tilt", "pcen")


# ---------------------------------------------------------------------------
# hp80 — sifir fazli yuksek geciren
# ---------------------------------------------------------------------------
def highpass(x: np.ndarray, cutoff_hz: float = 80.0, order: int = 4,
             sr: int = WHISPER_SR) -> np.ndarray:
    """
    Butterworth yuksek geciren, filtfilt ile SIFIR FAZLI uygulanir.

    Sifir faz onemli: Y1b'de olculdu ki faz bozulmasi ASR hasarinin buyuk
    kismini olusturuyor. Tek yonlu IIR faz kaydirir; filtfilt ileri-geri
    calisarak bunu tam olarak iptal eder (suzgec derecesi de iki katina cikar).
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_hz / (sr / 2.0), btype="highpass")
    return filtfilt(b, a, x)


# ---------------------------------------------------------------------------
# tilt — nazik yuksek raf, STFT alaninda (faz korunur)
# ---------------------------------------------------------------------------
def spectral_tilt(x: np.ndarray, pivot_hz: float = 1000.0, gain_db: float = 4.0,
                  sr: int = WHISPER_SR, n_fft: int = WHISPER_N_FFT,
                  hop: int = WHISPER_HOP) -> np.ndarray:
    """
    Sabit bir tayfsal zarf uygulanir: pivot altinda 0 dB, pivot ustunde gain_db'ye
    yumusak gecis. Zaman icinde DEGISMEZ ve sinyale BAKMAZ — bu yuzden bastirma
    degil, bicimlemedir.
    """
    S = stft(x, n_fft=n_fft, hop=hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # log-frekansta yumusak gecis: pivot'ta 0, 4*pivot'ta tam kazanc
    t = np.clip(np.log2(np.maximum(freqs, 1.0) / pivot_hz) / 2.0, 0.0, 1.0)
    shelf = 10.0 ** ((gain_db * t) / 20.0)
    return istft(S * shelf[:, None], length=len(x), n_fft=n_fft, hop=hop)


# ---------------------------------------------------------------------------
# pcen — bant basina uyarlamali kazanc (PCEN'in AGC bileseni)
# ---------------------------------------------------------------------------
def pcen_gain(x: np.ndarray, alpha: float = 0.5, s: float = 0.12,
              eps: float = 1e-6, gain_clip: tuple = (0.35, 2.0),
              n_fft: int = WHISPER_N_FFT, hop: int = WHISPER_HOP) -> np.ndarray:
    """
    PCEN'in uyarlamali kazanc bileseni, yeniden sentezlenebilir bicimde.

    M[f,t] = (1-s)*M[f,t-1] + s*|X[f,t]|        (bant basina yavas ortalama)
    G[f,t] = (eps + M[f,t]) ** (-alpha)          (kazanc)
    cikis   = |X| * G  ·  e^{j·faz(X)}           (FAZ KORUNUR)

    Gurultu kestirimi YOKTUR: hicbir bant "gurultu" diye isaretlenmez, yalnizca
    her bant kendi gecmis ortalamasina gore olceklenir. Sabit bir arka plan
    bileseni bu yuzden bastirilmaz, yalnizca sabit bir kazanca oturur.

    Kazanc medyanla normalize edilir ve sinirlanir; aksi halde sessizlik
    bolgelerinde kazanc patlar (bastirmanin tam tersi bir artefakt).

    PARAMETRE SECIMI — olculmus bir kisit
    -------------------------------------
    PCEN yavas ortalamayla calistigi icin ani baslangiclarda asim yapar; tepe
    koruma devreye girince cikis RMS'i duser ve kazanc SES DUZEYI ile karisir
    (Y1'de olculen tuzagin aynisi). dev-5 uzerinde 12 yapilandirma tarandi:

        s=0.025, alpha=0.8 (standart)  -> maks |dRMS| 2.64 dB   ilinti 0.990
        s=0.060, alpha=0.8             -> maks |dRMS| 0.71 dB   ilinti 0.993
        s=0.120, alpha=0.5  <- secilen -> maks |dRMS| 0.40 dB   ilinti 0.994

    "Seviye kaymasi < 0.5 dB" kosulunu saglayan her yapilandirmada ilinti 0.992
    ustunde kaliyor. Yani bu hatta PCEN'in gucu, seviye karisikligi yaratmadan
    ancak bu kadar artirilabilir — bu kisitin kendisi raporlanmasi gereken bir
    bulgudur, gizlenmis bir zayiflik degil.
    """
    from scipy.signal import lfilter, lfilter_zi

    S = stft(x, n_fft=n_fft, hop=hop)
    A = np.abs(S)

    # Bant basina birinci dereceden IIR duzlestirme. Cerceve dongusu yerine
    # lfilter: 40 dk'lik kayitta ~240 bin cerceve var, saf Python dongusu
    # dakikalar suruyordu. Baslangic durumu ilk cerceveye oturtulur (zi*A[:,0]),
    # boylece kayit basinda yapay bir gecici olusmaz.
    b_, a_ = [s], [1.0, -(1.0 - s)]
    zi = lfilter_zi(b_, a_)[0]
    M, _ = lfilter(b_, a_, A, axis=1, zi=(zi * A[:, :1]))

    G = (eps + M) ** (-alpha)
    # Konusma iceren cerceveler uzerinden medyan 1'e oturtulur: kazanc ile ses
    # duzeyi karismasin (Y1'de olculen tuzak).
    energy = A.sum(axis=0)
    voiced = energy > np.percentile(energy, 60.0)
    ref = np.median(G[:, voiced]) if voiced.any() else np.median(G)
    G = np.clip(G / max(ref, 1e-12), gain_clip[0], gain_clip[1])

    return istft(A * G * np.exp(1j * np.angle(S)), length=len(x), n_fft=n_fft, hop=hop)


BUILDERS = {
    "hp80": lambda x: highpass(x, cutoff_hz=80.0, order=4),
    "tilt": lambda x: spectral_tilt(x, pivot_hz=1000.0, gain_db=4.0),
    "pcen": lambda x: pcen_gain(x),
}


def finalize(y: np.ndarray, raw: np.ndarray, peak_guard: float = 0.999) -> np.ndarray:
    """Hattin ortak son islemesi — diger tum kosullarla ayni (bkz. oa/sa/dtln)."""
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
    Aile iddiasinin dogrulanmasi: bu filtreler BASTIRMA YAPMIYOR.

    Olcut, ham sesle sifir gecikmede ilinti. Bastirma temelli yontemler bu
    olcutte belirgin dusus gosteriyordu (DTLN 0.65, WPE daha da dusuk).
    Bicimleme yapan bir filtre yuksek ilinti korumali — korumazsa tasarim
    hatalidir ve karsilastirma yapmadan once bilinmelidir.
    """
    print(f"{'kayit':<45}{'kosul':<8}{'ilinti':>9}{'RMS dB':>9}{'tepe':>8}")
    print("-" * 80)
    for b in records:
        raw, _ = sf.read(os.path.join(AUDIO_16K, f"{b}.wav"), dtype="float64")
        if raw.ndim > 1:
            raw = raw[:, 0]
        for c in CONDITIONS:
            y = finalize(BUILDERS[c](raw), raw)
            n = min(len(raw), len(y))
            corr = float(np.corrcoef(raw[:n], y[:n])[0, 1])
            rdb = 20.0 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(raw ** 2)))
            print(f"{b:<45}{c:<8}{corr:>+9.3f}{rdb:>+9.2f}{np.max(np.abs(y)):>8.3f}")
    print("-" * 80)
    print("beklenti: ilinti > 0.9 (bicimleme), bastirma yontemlerinde 0.65 ve altiydi")


def main():
    ap = argparse.ArgumentParser(description="Y6 — bastirma yapmayan filtreleme")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--basenames", nargs="*", default=None)
    ap.add_argument("--verify", action="store_true", help="yalniz tani, dosya yazma")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    records = args.basenames or development_subset(load_experiment_config())

    if args.verify:
        verify(records[:2])
        return 0

    for c in args.conditions:
        if c not in BUILDERS:
            print(f"[HATA] bilinmeyen kosul: {c}")
            return 1
        out_dir = os.path.join(PKG_ROOT, "data", f"audio_{c}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nKosul: {c}")
        for b in records:
            op = os.path.join(out_dir, f"{b}.wav")
            if os.path.exists(op) and not args.overwrite:
                print(f"[ATLANDI] {b}")
                continue
            raw, _ = sf.read(os.path.join(AUDIO_16K, f"{b}.wav"), dtype="float64")
            if raw.ndim > 1:
                raw = raw[:, 0]
            t0 = time.time()
            y = finalize(BUILDERS[c](raw), raw)
            sf.write(op, y.astype(np.float32), 16000, subtype="PCM_16")
            dur = len(raw) / WHISPER_SR
            el = time.time() - t0
            print(f"[BASARILI] {b:<45}{dur/60:5.1f} dk  islem {el:6.1f} sn  RTF {el/dur:.3f}")

    print("\nexperiment.yaml -> extra_conditions: id=<kosul>, source_dir=data/audio_<kosul>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
