"""
Y5 — DTLN (Dual-signal Transformation LSTM Network): FARKLI KALITE-ILINTI EGRISI arayisi.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Sonuclar         : docs/MAKALE_SONUCLAR_2026-09-13.md

NEDEN BU YONTEM
---------------
Olculdu (29 kosul, dev-5, Large-v3): kalite ile ham sese benzerlik TERS bagli,
Pearson r = -0.742. Yani bir kosul ham sesten uzaklastikca (dekorelasyon artar)
kalitesi bozuluyor; "yuksek kalite + yuksek dekorelasyon" bolgesi BOS.

Ancak o 29 kosulun tamami AYNI aileden: WebRTC NS'in siddetleri, alfa karisimlari
ve WPE. Olculen sey bir ailenin egrisi olabilir.

DTLN farkli bir cins:
  - WebRTC NS : istatistiksel, spektral kazanc maskesi
  - WPE       : dogrusal ongoru, gec yanki
  - DTLN      : OGRENILMIS, iki asamali (STFT maskesi + zaman alani 1D konvolusyon)

Iki olasi sonuc, ikisi de degerli:
  (a) Egrinin DISINA duserse -> aradigimiz ideal uye; birlestirme kazancini
      buyutebilir.
  (b) Egrinin USTUNE duserse -> takas, belirli bir algoritmanin degil
      BASTIRMANIN KENDISININ ozelligi demektir. Daha genel ve guclu bir iddia.

PAKET 2 DONGUSU
---------------
DTLN, Paket 2 aday listesinde vardi ve "ONNX model dosyasi yoktu" gerekcesiyle
elenmisti. Model artik erisilebilir (breizhn/DTLN), yani o eleme kararini
kapatabiliyoruz.

MIMARI NOTU
-----------
DTLN'nin 1. asamasi GURULTULU FAZI korur (maskeyi yalniz genlige uygular).
Bu, bizim Y1b bulgumuzla ve literaturle (kestirilmis faz zararli, gamma=0
optimum) ortusuyor.

Kullanim:
    python paket3_asr/scripts/dtln_condition.py --basenames saha_03_img_3602
    python paket3_asr/scripts/dtln_condition.py
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

MODEL_DIR = os.path.join(PKG_ROOT, "models_onnx")
AUDIO_16K = os.path.join(PKG_ROOT, "data", "audio_16k")

BLOCK_LEN = 512
BLOCK_SHIFT = 128

REC10 = [
    "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
    "saha_04_vid_20220913_142911", "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez", "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi", "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
]


def dtln_enhance(x: np.ndarray, sess1, sess2) -> np.ndarray:
    """
    DTLN cerceve cerceve cikarim (breizhn/DTLN gercek-zamanli ornegiyle ayni akis).

    1. asama: STFT genligine maske, GURULTULU FAZ korunur
    2. asama: zaman alaninda ogrenilmis 1D konvolusyon kodlayici/cozucu

    LSTM durumlari cerceveler arasi tasinir; bu nedenle islem sirali olmak
    zorundadir (vektorlestirilemez).
    """
    n1_in = [i.name for i in sess1.get_inputs()]
    n2_in = [i.name for i in sess2.get_inputs()]

    audio = x.astype(np.float32)
    out = np.zeros(len(audio), dtype=np.float32)
    in_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
    s1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
    s2 = np.zeros((1, 2, 128, 2), dtype=np.float32)

    n_blocks = (len(audio) - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
    for i in range(n_blocks):
        in_buf[:-BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:]
        in_buf[-BLOCK_SHIFT:] = audio[i * BLOCK_SHIFT: i * BLOCK_SHIFT + BLOCK_SHIFT]

        spec = np.fft.rfft(in_buf)
        mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
        phase = np.angle(spec)

        mask, s1 = sess1.run(None, {n1_in[0]: mag, n1_in[1]: s1})
        est = (mag * mask).reshape(-1) * np.exp(1j * phase)
        blk = np.fft.irfft(est).reshape(1, 1, -1).astype(np.float32)

        outb, s2 = sess2.run(None, {n2_in[0]: blk, n2_in[1]: s2})
        outb = outb.reshape(-1)

        out_buf[:-BLOCK_SHIFT] = out_buf[BLOCK_SHIFT:]
        out_buf[-BLOCK_SHIFT:] = 0.0
        out_buf += outb
        out[i * BLOCK_SHIFT: i * BLOCK_SHIFT + BLOCK_SHIFT] = out_buf[:BLOCK_SHIFT]

    return out.astype(np.float64)


def finalize(y: np.ndarray, raw: np.ndarray, peak_guard: float = 0.999) -> np.ndarray:
    """Hattin diger kosullariyla ayni son isleme: seviye sabitleme + tepe koruma."""
    r = float(np.sqrt(np.mean(raw ** 2))); m = float(np.sqrt(np.mean(y ** 2)))
    if m > 1e-9 and r > 1e-9:
        y = y * (r / m)
    g = max(peak_guard, float(np.max(np.abs(raw))))
    pk = float(np.max(np.abs(y)))
    if pk > g:
        y = y * (g / pk)
    return y


def main():
    ap = argparse.ArgumentParser(description="Y5 — DTLN kosulu uretimi")
    ap.add_argument("--basenames", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2          # cerceve basina is kucuk; asiri is parcacigi zarar verir
    sess1 = ort.InferenceSession(os.path.join(MODEL_DIR, "model_1.onnx"),
                                 sess_options=so, providers=["CPUExecutionProvider"])
    sess2 = ort.InferenceSession(os.path.join(MODEL_DIR, "model_2.onnx"),
                                 sess_options=so, providers=["CPUExecutionProvider"])

    out_dir = os.path.join(PKG_ROOT, "data", "audio_dtln")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Kosul: dtln   (block_len={BLOCK_LEN}, block_shift={BLOCK_SHIFT}, faz=gurultulu)")

    for b in (args.basenames or REC10):
        op = os.path.join(out_dir, f"{b}.wav")
        if os.path.exists(op) and not args.overwrite:
            print(f"[ATLANDI] {b}")
            continue
        raw, _ = sf.read(os.path.join(AUDIO_16K, f"{b}.wav"), dtype="float64")
        if raw.ndim > 1:
            raw = raw[:, 0]
        t0 = time.time()
        y = finalize(dtln_enhance(raw, sess1, sess2), raw)
        sf.write(op, y.astype(np.float32), 16000, subtype="PCM_16")
        el = time.time() - t0
        dur = len(raw) / 16000
        print(f"[BASARILI] {b:<45}{dur/60:5.1f} dk  islem {el:7.1f} sn  RTF {el/dur:.3f}")

    print("\nexperiment.yaml -> extra_conditions: id=dtln, source_dir=data/audio_dtln")
    return 0


if __name__ == "__main__":
    sys.exit(main())
