"""
Insan dogrulama paketi (Ek-2 Rev B, Bolum 4 oncesi kontrol).

NEDEN
-----
Liste dondurulduktan sonra degistirilemez (Kural 1). Dondurmadan once tek bir
sey dogrulanmali: SES ile REFERANS AYNI YERE denk geliyor mu?

Bu, transkripsiyon KALITESININ dogrulanmasi DEGILDIR — referanslar korpuslarla
birlikte, insan eliyle yazilmis olarak geliyor. Dogrulanan sey HIZALAMADIR.

Ozellikle AMI'de 60 dakikalik toplantidan ilk 10 dakika kesiliyor. Ses penceresi
ile transkript penceresi birkac saniye kayarsa WER cop cikar ve bu SAYIYA BAKARAK
FARK EDILEMEZ — "Ingilizce toplanti ASR'i zordur" gibi gorunur. Kulakla saniyeler
icinde yakalanir.

CIKTI
-----
outputs/verification/<kayit>/ klasorunde
    parca_1.wav … parca_3.wav   30 saniyelik kesitler
    OKUNACAK.md                 her kesitin referans metni

Kullanim:
    python paket3_asr/scripts/make_verification_pack.py --cell primock
"""

import argparse
import os
import re
import sys

import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from oa_conditions import PKG_ROOT  # noqa: E402
from prepare_corpus_en import (CORPORA, OUT_AUDIO, SR, clean_tags,  # noqa: E402
                               parse_textgrid)

OUT = os.path.join(PKG_ROOT, "outputs", "verification")
WIN = 30.0
POSITIONS = (0.10, 0.50, 0.85)   # kaydin basi / ortasi / sonu


def primock_utterances(cid):
    src = os.path.join(CORPORA, "primock57", "transcripts")
    utts = []
    for role in ("doctor", "patient"):
        for st, en, tx in parse_textgrid(os.path.join(src, f"{cid}_{role}.TextGrid")):
            utts.append((st, en, role, tx))
    utts.sort(key=lambda u: u[0])
    return utts


def build(cid, utts, label):
    audio_path = os.path.join(OUT_AUDIO, f"{cid}.wav")
    info = sf.info(audio_path)
    dur = info.frames / info.samplerate
    d = os.path.join(OUT, cid)
    os.makedirs(d, exist_ok=True)

    lines = [f"# Dogrulama — {cid}",
             "",
             f"**Hucre:** {label}  ·  **Toplam sure:** {dur/60:.1f} dk",
             "",
             "## Ne yapmaniz gerekiyor",
             "",
             "Her parcayi dinleyin ve altindaki metne bakin. Tek soru:",
             "**bu ses, bu metinle ayni seyden mi bahsediyor?**",
             "",
             "Kelime kelime ayni olmasi gerekmiyor — konu tutuyorsa hizalama dogrudur.",
             "Tutmuyorsa (ses baska sey, metin baska sey) hizalama bozuktur ve",
             "DONDURMADAN ONCE duzeltilmesi gerekir.",
             "",
             "**Iki liste var.** Ilk liste kesitin tamamen icinde kalan cumleler —",
             "asil kontrol onlar. Ikinci liste kesit sinirini asanlar; onlarin",
             "yalnizca bir kismini duyarsiniz, eksik gelmesi normaldir.",
             ""]

    for k, frac in enumerate(POSITIONS, 1):
        t0 = max(0.0, min(dur - WIN, dur * frac))
        t1 = t0 + WIN
        x, _ = sf.read(audio_path, start=int(t0 * SR), frames=int(WIN * SR), dtype="float32")
        sf.write(os.path.join(d, f"parca_{k}.wav"), x, SR, subtype="PCM_16")

        # Bu pencereyle kesisen utterance'lar, SINIR DURUMU isaretlenerek.
        #
        # Ilk surumde kesisen her utterance duz listeleniyordu ve bu YANLIS ALARM
        # uretti: kesitin sonuna denk gelen bir cumle 9 saniyeye kadar disari
        # tasabiliyor, dinleyen kisi onu duymuyor ve "metin tutmuyor" saniyor.
        # Insan dogrulamasinin degeri, GERCEK bir kaymanin goze carpmasindadir;
        # bilinen bir kusur o sinyali bogarsa arac ise yaramaz.
        hit = [u for u in utts if u[1] > t0 and u[0] < t1]
        tam = [u for u in hit if u[0] >= t0 and u[1] <= t1]
        lines += ["---", "",
                  f"## Parça {k} — `parca_{k}.wav`",
                  "",
                  f"Kaydın **{t0/60:.1f}–{t1/60:.1f}. dakikası** arası "
                  f"({t0:.0f}–{t1:.0f}. saniye).",
                  ""]
        if hit:
            lines.append("**Kesitin tamamen içinde olanlar — asıl kontrol bunlar:**")
            lines.append("")
            if tam:
                for st, en, role, tx in tam:
                    lines.append(f"- **[{st:7.1f}s {role}]** {clean_tags(tx)}")
            else:
                lines.append("- *(tamamen içeride kalan cümle yok)*")
            kismi = [u for u in hit if u not in tam]
            if kismi:
                lines += ["", "**Kesit sınırını aşanlar — yalnız bir kısmını duyarsınız, "
                              "eksik gelmesi NORMALDİR:**", ""]
                for st, en, role, tx in kismi:
                    if en > t1:
                        note = f"sonundan {en - t1:.1f} sn taşıyor"
                    else:
                        note = f"başından {t0 - st:.1f} sn önce başlıyor"
                    lines.append(f"- *[{st:7.1f}s {role}, {note}]* {clean_tags(tx)}")
        else:
            lines.append("- *(bu pencerede konuşma yok — sessizlikse normal, "
                         "konuşma duyuyorsanız HİZALAMA BOZUK)*")
        lines.append("")

    with open(os.path.join(d, "OKUNACAK.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return d



# ---------------------------------------------------------------------------
# AMI — iki hucre birden (SDM ve IHM-mix), ayni pencere
# ---------------------------------------------------------------------------
AMI_META = os.path.join(PKG_ROOT, "data", "corpora", "ami_meta", "segments_sdm.csv")
AMI_MANIFEST = os.path.join(PKG_ROOT, "data", "corpora", "manifest_ami.csv")


def ami_segments(meeting):
    import csv as _csv
    out = []
    with open(AMI_META, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r["meeting_id"] == meeting:
                out.append((float(r["begin_time"]), float(r["end_time"]),
                            r["speaker_id"], r["text"]))
    out.sort(key=lambda t: t[0])
    return out


def build_ami(meeting, t0_win):
    """
    AMI icin kesit paketi. PriMock'tan iki farki var:
      * Ayni anin HEM uzak mikrofon HEM kulaklik surumu verilir; ikisi ayni seyi
        anlatmiyorsa eslestirme bozuktur.
      * Zaman damgalari toplantinin kendi saatindedir; kesit icindeki konum
        pencere baslangicina gore cevrilir.
    """
    d = os.path.join(OUT, meeting)
    os.makedirs(d, exist_ok=True)
    utts = ami_segments(meeting)
    sdm_p = os.path.join(OUT_AUDIO, meeting + "_sdm.wav")
    ihm_p = os.path.join(OUT_AUDIO, meeting + "_ihm.wav")
    dur = sf.info(sdm_p).frames / SR

    L = []
    L.append("# Dogrulama — " + meeting + " (AMI)")
    L.append("")
    L.append("**Hucreler:** EN gurultulu (uzak mikrofon) + EN temiz (kulaklik karisimi)")
    L.append("**Pencere:** toplantinin %.0f.–%.0f. saniyesi  ·  kesit suresi %.1f dk"
             % (t0_win, t0_win + 600, dur / 60))
    L.append("")
    L.append("## Ne yapmaniz gerekiyor")
    L.append("")
    L.append("Her parca icin IKI ses dosyasi var: `_uzak.wav` ve `_kulaklik.wav`.")
    L.append("Ayni anin iki farkli mikrofonla kaydi.")
    L.append("")
    L.append("**Iki soru:**")
    L.append("1. Ses ile altindaki metin ayni seyden mi bahsediyor? (hizalama)")
    L.append("2. Uzak ve kulaklik surumleri ayni konusmayi mi tasiyor? (eslestirme)")
    L.append("")
    L.append("Uzak mikrofon kaydi gurultulu ve uzaktan gelir — bu normaldir, hata degil.")
    L.append("Anlasilmasi zorsa kulaklik surumunu olcut alin.")
    L.append("")
    L.append("**Iki liste var.** Ilki kesitin tamamen icinde kalan cumleler (asil kontrol),")
    L.append("ikincisi kesit sinirini asanlar — onlarin eksik gelmesi normaldir.")
    L.append("")

    for k, frac in enumerate(POSITIONS, 1):
        t0 = max(0.0, min(dur - WIN, dur * frac))
        t1 = t0 + WIN
        for tag, path in (("uzak", sdm_p), ("kulaklik", ihm_p)):
            x, _ = sf.read(path, start=int(t0 * SR), frames=int(WIN * SR), dtype="float32")
            sf.write(os.path.join(d, "parca_%d_%s.wav" % (k, tag)), x, SR, subtype="PCM_16")

        a0, a1 = t0_win + t0, t0_win + t1     # toplanti saatine cevir
        hit = [u for u in utts if u[1] > a0 and u[0] < a1]
        tam = [u for u in hit if u[0] >= a0 and u[1] <= a1]
        L.append("---")
        L.append("")
        L.append("## Parça %d — `parca_%d_uzak.wav` / `parca_%d_kulaklik.wav`" % (k, k, k))
        L.append("")
        L.append("Kesitin **%.1f–%.1f. dakikası** (toplantının %.0f.–%.0f. saniyesi)."
                 % (t0 / 60, t1 / 60, a0, a1))
        L.append("")
        L.append("**Kesitin tamamen içinde olanlar — asıl kontrol bunlar:**")
        L.append("")
        if tam:
            for st, en, spk, tx in tam:
                L.append("- **[%7.1fs %s]** %s" % (st - t0_win, spk, tx))
        else:
            L.append("- *(tamamen içeride kalan cümle yok)*")
        kismi = [u for u in hit if u not in tam]
        if kismi:
            L.append("")
            L.append("**Kesit sınırını aşanlar — eksik gelmesi NORMALDİR:**")
            L.append("")
            for st, en, spk, tx in kismi:
                note = ("sonundan %.1f sn taşıyor" % (en - a1)) if en > a1 \
                    else ("başından %.1f sn önce başlıyor" % (a0 - st))
                L.append("- *[%7.1fs %s, %s]* %s" % (st - t0_win, spk, note, tx))
        L.append("")

    with open(os.path.join(d, "OKUNACAK.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return d


def main():
    ap = argparse.ArgumentParser(description="Insan dogrulama paketi")
    ap.add_argument("--cell", default="primock", choices=["primock", "ami"])
    ap.add_argument("--n", type=int, default=2, help="hucre basina kac kayit")
    args = ap.parse_args()

    if args.cell == "primock":
        from prepare_corpus_en import PRIMOCK_IDS
        # Deterministik secim: ilk ve son kayit (uc noktalar en bilgilendirici)
        picks = [PRIMOCK_IDS[0], PRIMOCK_IDS[-1]][:args.n]
        for cid in picks:
            d = build(cid, primock_utterances(cid), "EN temiz, tibbi (PriMock57)")
            print(f"[HAZIR] {os.path.relpath(d, PKG_ROOT)}")
        print()
        print(f"{len(picks)} kayit x 3 parca = {len(picks)*3} kesit, "
              f"toplam {len(picks)*3*30/60:.0f} dk dinleme")
        return 0

    if args.cell == "ami":
        import csv as _csv
        win = {}
        with open(AMI_MANIFEST, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                win[r["meeting"]] = float(r["window_start_s"])
        ms = sorted(win)
        picks = [ms[0], ms[-1]][:args.n]      # deterministik: ilk ve son toplanti
        for m in picks:
            d = build_ami(m, win[m])
            print(f"[HAZIR] {os.path.relpath(d, PKG_ROOT)}")
        print()
        print(f"{len(picks)} toplanti x 3 parca x 2 mikrofon = {len(picks)*6} kesit, "
              f"toplam {len(picks)*6*30/60:.0f} dk dinleme")
        return 0

    print(f"bilinmeyen hucre: {args.cell}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
