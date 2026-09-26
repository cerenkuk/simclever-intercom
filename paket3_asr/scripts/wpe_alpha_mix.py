"""
Y3b — WPE'nin alfa ile seyreltilmesi (dekorele + basabas uye).

BULGU: Birlestirme uyeligi icin IKI kosul birden gerekiyor —
  dusuk ilinti VE basabas kalite.
Olculdu (tam 10 kayit, Large-v3):
  WebRTC turevleri : ilinti %84    kalite +0.17..+0.93 pp   -> kalite VAR, ilinti YOK
  WPE (tam)        : ilinti %78.7  kalite +3.21 pp          -> ilinti VAR, kalite YOK

Bu betik WPE'yi mevcut psm (guc/Mel alani) cercevesiyle ham sese karistirir.
Amac: WPE'nin dekorelasyonunu korurken kalitesini basabas bandina cekmek.
Faz her zaman ham sesin fazidir (gamma=0).
"""
import argparse, os, sys
import numpy as np, soundfile as sf
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path: sys.path.insert(0, _SD)
from oa_conditions import PKG_ROOT
from spectral_alpha import build_spectral_alpha

REC10 = ["saha_01_img_3463","saha_02_img_3492","saha_03_img_3602",
         "saha_04_vid_20220913_142911","video_01_genel_anamnez",
         "video_02_psikiyatrik_anamnez","video_03_pediatrik_anamnez",
         "video_04_simule_hasta_gorusmesi","video_05_poliklinik_anamnez_fizik_muayene",
         "video_06_gogus_hastaliklari_oyku_alma"]

def finalize(y, raw, peak_guard=0.999):
    r=float(np.sqrt(np.mean(raw**2))); m=float(np.sqrt(np.mean(y**2)))
    if m>1e-9 and r>1e-9: y=y*(r/m)
    g=max(peak_guard, float(np.max(np.abs(raw))))
    pk=float(np.max(np.abs(y)))
    if pk>g: y=y*(g/pk)
    return y

def main():
    ap=argparse.ArgumentParser(description="Y3b — WPE alfa karisimi")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--wpe-dir", default="audio_wpe_t10d3")
    ap.add_argument("--basenames", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a=ap.parse_args()
    cid=f"sa_psm_wpe_a{int(round(a.alpha*100)):03d}"
    out=os.path.join(PKG_ROOT,"data",f"audio_{cid}")
    os.makedirs(out,exist_ok=True)
    print(f"Kosul: {cid}  (alpha={a.alpha}, psm, faz=ham)")
    for b in (a.basenames or REC10):
        op=os.path.join(out,f"{b}.wav")
        if os.path.exists(op) and not a.overwrite:
            print(f"[ATLANDI] {b}"); continue
        raw,_=sf.read(os.path.join(PKG_ROOT,"data","audio_16k",f"{b}.wav"),dtype="float64")
        enh,_=sf.read(os.path.join(PKG_ROOT,"data",a.wpe_dir,f"{b}.wav"),dtype="float64")
        if raw.ndim>1: raw=raw[:,0]
        if enh.ndim>1: enh=enh[:,0]
        n=min(len(raw),len(enh)); raw,enh=raw[:n],enh[:n]
        y=finalize(build_spectral_alpha(raw,enh,alpha=a.alpha,variant="psm"),raw)
        sf.write(op,y.astype(np.float32),16000,subtype="PCM_16")
        print(f"[BASARILI] {b:<45}{n/16000/60:5.1f} dk")
    print(f"\nexperiment.yaml -> extra_conditions'a ekleyin: id={cid}, source_dir=data/audio_{cid}")


if __name__ == "__main__":
    main()
