"""
KALITE <-> DEKORELASYON TAKASI — olcum, korelasyon ve grafik.

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md
Sonuclar         : docs/MAKALE_SONUCLAR_2026-09-13.md

NE OLCULUYOR
------------
Birlestirmenin (Y4) ise yaramasi icin uyelerin AYNI ANDA iki ozelligi olmali:
  1. kalite   : ham sese yakin WER (kotu uye oylamayi bozar — olculdu: +2.53 pp)
  2. cesitlilik: hatalarin FARKLI yerlerde olmasi (ayni hata oylamayla duzelmez)

Bu betik her kosul icin ikisini de olcer:
  kalite      = makro WER farki (kosul - ham), puan; kucuk = iyi
  benzerlik   = transkriptin ham transkriptle hizalama orani (difflib), %;
                dekorelasyon = 100 - benzerlik

BULGU
-----
Iki eksen TERS bagli. Bir isleme ham sesten ne kadar uzaklasirsa (dekorelasyon
artar) kalitesi o kadar bozulur. "Yuksek kalite + yuksek cesitlilik" ceyregi BOS.
Bu, birlestirme kazancinin neden kucuk ama gercek oldugunu aciklar: mevcut on
isleme ailesi, birlestirmenin ihtiyac duydugu bagimsiz cesitliligi uretemiyor.

Alfa seyreltmesi bu egriyi DEGISTIRMEZ, uzerinde KAYDIRIR. WPE ornegi:
  alfa=1.00 -> kalite +3.21 pp, benzerlik %78.7
  alfa=0.25 -> kalite +0.30 pp, benzerlik %83.6
Kalite duzeldi ama aranan cesitlilik de kayboldu.

Kullanim:
    python paket3_asr/scripts/analyze_tradeoff_curve.py
    python paket3_asr/scripts/analyze_tradeoff_curve.py --records all --no-plot
"""

import argparse
import csv
import difflib
import math
import os
import statistics
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from analyze_fusion import load_hypotheses  # noqa: E402
from oa_conditions import PKG_ROOT, development_subset, load_experiment_config  # noqa: E402

REPORT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "tradeoff_curve_report.csv")
FIG_PATH = os.path.join(PKG_ROOT, "outputs", "figures", "tradeoff_curve.png")

REC10 = [
    "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602",
    "saha_04_vid_20220913_142911", "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez", "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi", "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
]

# Grafikte renk/isaret icin aile siniflandirmasi. Sira onemli: ilk eslesen kazanir.
FAMILIES = [
    ("none",       "Ham ses",                 lambda c: c == "none"),
    ("webrtc",     "WebRTC NS (sabit)",       lambda c: c in ("mild", "aggressive", "very_aggressive", "moderate")),
    ("loudnorm",   "Ses duzeyi (loudnorm)",   lambda c: c.startswith("loudnorm")),
    ("oa",         "Y1  dalga formu karisim", lambda c: c.startswith("oa_")),
    ("sa",         "Y1b spektral karisim",    lambda c: c.startswith("sa_")),
    ("aa",         "Y2  uyarlamali zarf",     lambda c: c.startswith("aa_")),
    ("wpe",        "Y3  yanki giderme",       lambda c: c.startswith("wpe") or "_wpe_" in c),
    ("dtln",       "Y5  ogrenilmis (DTLN)",   lambda c: c.startswith("dtln")),
    ("nsf",        "Y6  bastirmasiz filtre",  lambda c: c in ("hp80", "tilt", "pcen")),
]


def family_of(cond):
    for key, label, test in FAMILIES:
        if test(cond):
            return key, label
    return "other", "Diger"


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def collect(hyp, model, records):
    """
    Kosul -> (kalite_pp, benzerlik_yuzde, kapsanan_kayit_sayisi)

    Yalnizca SECILEN KAYITLARIN TAMAMINDA hem 'none' hem de kosul mevcutsa
    dahil edilir; eksik kapsamda makro ortalama karsilastirilabilir olmaz.
    """
    conds = set()
    for b in records:
        conds.update(hyp.get((model, b), {}).keys())

    rows = {}
    for c in sorted(conds):
        dws, sims = [], []
        for b in records:
            cell = hyp.get((model, b), {})
            if "none" not in cell or c not in cell:
                dws = None
                break
            raw_txt, raw_wer = cell["none"]
            txt, wer = cell[c]
            dws.append((wer - raw_wer) * 100.0)
            sims.append(difflib.SequenceMatcher(
                None, raw_txt.split(), txt.split(), autojunk=False).ratio() * 100.0)
        if dws is None:
            continue
        rows[c] = (statistics.mean(dws), statistics.mean(sims), len(records))
    return rows


def plot(rows, obs, model, records_label, r, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=160)

    style = {
        "none":     ("#111111", "*", 260),
        "webrtc":   ("#c0392b", "s", 70),
        "loudnorm": ("#8e44ad", "P", 70),
        "oa":       ("#2980b9", "o", 60),
        "sa":       ("#16a085", "^", 70),
        "aa":       ("#d68910", "v", 70),
        "wpe":      ("#7f8c8d", "D", 60),
        "dtln":     ("#e91e8c", "X", 130),
        "nsf":      ("#1f6fb4", "P", 150),
        "other":    ("#95a5a6", ".", 50),
    }

    # Ham sesle bit-birebir ayni kosullar (hic tetiklenmemis iki gecisli kurallar)
    # capanin tam ustune duser ve gorunmez; capa zaten cizildigi icin atlanir.
    groups = {}
    for c, (q, s, _n) in rows.items():
        if c != "none" and s >= 99.99:
            continue
        k, label = family_of(c)
        groups.setdefault(k, (label, []))[1].append((s, q, c))

    # Egilim dogrusu (en kucuk kareler) — ham ses capasi disarida birakilarak.
    xs = [s for (q, s, _n) in obs.values()]
    ys = [q for (q, s, _n) in obs.values()]
    if len(xs) > 2:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            x0, x1 = min(xs), max(xs)
            ax.plot([x0, x1], [my + m * (x0 - mx), my + m * (x1 - mx)],
                    color="#34495e", lw=1.4, ls="--", zorder=1,
                    label=f"egilim (Pearson r = {r:+.3f})")

    for k in ["webrtc", "loudnorm", "oa", "sa", "aa", "wpe", "dtln", "nsf", "other", "none"]:
        if k not in groups:
            continue
        label, pts = groups[k]
        col, mk, sz = style[k]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   c=col, marker=mk, s=sz, alpha=0.85, edgecolors="white",
                   linewidths=0.6, label=f"{label} (n={len(pts)})", zorder=3)

    ax.axhline(0.0, color="#111111", lw=1.0, alpha=0.6, zorder=2)

    # Aranan bolge: ham sese yakin kalite (y ~ 0) + dusuk benzerlik (sol taraf).
    lo_x = min(xs) - 1.0
    ax.axhspan(-1.0, 1.0, xmin=0.0, xmax=0.42, color="#27ae60", alpha=0.07, zorder=0)
    ax.text(lo_x + 0.4, 1.35, "aranan bolge: yuksek kalite + yuksek cesitlilik  (BOS)",
            fontsize=8.5, color="#1e8449", style="italic")

    ax.set_xlabel("Ham transkriptle benzerlik (%)   ← daha cok cesitlilik", fontsize=10)
    ax.set_ylabel("Kalite: makro WER farki (puan)   ↓ daha iyi", fontsize=10)
    ax.set_title(f"On isleme kosullarinda kalite ↔ cesitlilik takasi\n"
                 f"{model}  —  {records_label}  —  {len(obs)} olculen kosul",
                 fontsize=11.5, pad=12)
    ax.grid(alpha=0.22, lw=0.6)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.92)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Kalite-dekorelasyon takas egrisi")
    ap.add_argument("--model", default="faster-whisper-large-v3-int8")
    ap.add_argument("--records", choices=["dev5", "all"], default="dev5")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config()
    records = REC10 if args.records == "all" else development_subset(cfg)
    label = "tam 10 kayit" if args.records == "all" else f"dev-{len(records)}"

    rows = collect(load_hypotheses(), args.model, records)
    if len(rows) < 3:
        print(f"[EKSIK] yalnizca {len(rows)} kosul tam kapsamli — korelasyon anlamsiz")
        return 1

    # Korelasyon 'none' HARIC hesaplanir: ham ses tanim geregi (100%, 0.00)
    # noktasindadir, olculmus bir gozlem degil sabit capadir. Grafikte gosterilir
    # ama egilimi kendisi belirlememelidir.
    obs = {c: v for c, v in rows.items() if c != "none"}
    r = pearson([v[1] for v in obs.values()], [v[0] for v in obs.values()])

    # Saglamlik: hicbir zaman tetiklenmeyen iki gecisli kosullar ham sesle
    # bit-birebir aynidir (benzerlik %100, fark 0.00) — bunlar da capa kopyasidir.
    nd = {c: v for c, v in obs.items() if v[1] < 99.99}
    r_nd = pearson([v[1] for v in nd.values()], [v[0] for v in nd.values()]) if len(nd) > 2 else float("nan")

    print("=" * 82)
    print(f"KALITE <-> CESITLILIK TAKASI   |   {args.model}   |   {label}")
    print("=" * 82)
    print(f"{'kosul':<26}{'aile':<12}{'kalite (pp)':>13}{'benzerlik':>12}{'cesitlilik':>12}")
    print("-" * 82)
    for c, (q, s, _n) in sorted(rows.items(), key=lambda kv: kv[1][0]):
        print(f"{c:<26}{family_of(c)[0]:<12}{q:>+12.2f} {s:>11.1f}%{100-s:>11.1f}%")
    print("-" * 82)
    print(f"olculen kosul  : {len(obs)}   ('none' capa olarak haric)")
    print(f"Pearson r      : {r:+.3f}   (kalite farki  vs  ham sese benzerlik)")
    print(f"saglamlik      : {r_nd:+.3f}   (ham sesle bit-birebir ayni {len(obs)-len(nd)} kosul da haric, n={len(nd)})")
    print(f"yorum          : r<0 => ham sesten uzaklastikca kalite bozuluyor;")
    print(f"                 'yuksek kalite + yuksek cesitlilik' bolgesi bos.")

    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "records", "condition", "family",
                    "quality_delta_pp", "similarity_to_raw_pct", "decorrelation_pct",
                    "n_records", "is_anchor", "pearson_r", "pearson_r_nondegenerate"])
        for c, (q, s, n) in sorted(rows.items(), key=lambda kv: kv[1][0]):
            w.writerow([args.model, label, c, family_of(c)[0],
                        f"{q:.4f}", f"{s:.4f}", f"{100-s:.4f}", n,
                        1 if c == "none" else 0, f"{r:.4f}", f"{r_nd:.4f}"])
    print(f"\ntablo  -> {os.path.relpath(REPORT_CSV, PKG_ROOT)}")

    if not args.no_plot:
        plot(rows, obs, args.model, label, r, FIG_PATH)
        print(f"grafik -> {os.path.relpath(FIG_PATH, PKG_ROOT)}")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
