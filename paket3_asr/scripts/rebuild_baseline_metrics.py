"""
Baseline metriklerini GUNCEL referans dosyalarina karsi yeniden hesaplar.

Neden gerekli
-------------
Devralinan baseline CSV'si, referans metinlerin FARKLI SURUMLERINE karsi
hesaplanmis satirlar iceriyor. Olculdu: satirlarin ~%17.7'si guncel olmayan
referansa karsi hesaplanmis ve sorun yalnizca saha_02 / saha_03 kayitlarinda —
yani en gurultulu, en yuksek WER'li iki saha kaydinda — yogunlasiyor.

Ornek (saha_03): eski referans surumu bas taraftaki 'cek,cek.' jetonlarini
icermiyordu ve 'QRS...' yazimi kullaniliyordu. Referans sonradan duzeltildi,
ancak yalnizca 'loudnorm' kosulu yeniden kosuldu; none/mild/aggressive satirlari
eski surume karsi hesaplanmis halde kaldi.

Sonuc: baseline'dan hesaplanan makro WER degerleri karisik referans surumleri
uzerine kuruludur. Bu, ozellikle 'denoise gurultulu saha kaydinda ise yarar mi'
sorusunu dogrudan etkiler.

Cozum
-----
Baseline CSV'si her satirda hem 'reference_text' hem 'transcription' saklıyor.
Bu nedenle ASR'i yeniden kosmaya GEREK YOKTUR: hipotez metni sabit tutulup
metrikler guncel referans dosyalarina karsi yeniden hesaplanir.

Kullanim:
    python paket3_asr/scripts/rebuild_baseline_metrics.py
    python paket3_asr/scripts/rebuild_baseline_metrics.py --dry-run
"""

import argparse
import csv
import glob
import hashlib
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from calculate_metrics import evaluate_asr_pair  # noqa: E402
from oa_conditions import PKG_ROOT  # noqa: E402

BASELINE_CSV = os.path.join(PKG_ROOT, "outputs", "metrics_baseline", "benchmark_results_baseline.csv")
REBUILT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics_baseline", "benchmark_results_baseline_refv2.csv")
CHANGELOG_CSV = os.path.join(PKG_ROOT, "outputs", "metrics_baseline", "baseline_rebuild_changelog.csv")
REF_DIR = os.path.join(PKG_ROOT, "data", "references")


def load_current_references() -> dict:
    refs = {}
    for p in glob.glob(os.path.join(REF_DIR, "*.txt")):
        b = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8") as f:
            refs[b] = f.read().strip()
    return refs


def main():
    ap = argparse.ArgumentParser(description="Baseline metriklerini guncel referanslara karsi yeniden hesapla")
    ap.add_argument("--dry-run", action="store_true", help="Dosya yazma; yalniz degisecek satirlari raporla")
    args = ap.parse_args()

    if not os.path.exists(BASELINE_CSV):
        print(f"[HATA] Baseline CSV bulunamadi: {BASELINE_CSV}", file=sys.stderr)
        return 1

    refs = load_current_references()
    if not refs:
        print(f"[HATA] Referans dosyasi bulunamadi: {REF_DIR}", file=sys.stderr)
        return 1
    print(f"Guncel referans dosyasi: {len(refs)}")

    with open(BASELINE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"Baseline satiri        : {len(rows)}")

    changes = []
    stale = 0
    unchanged = 0
    skipped_no_ref = 0
    skipped_no_hyp = 0

    for r in rows:
        b = r.get("basename", "")
        cur_ref = refs.get(b)
        if cur_ref is None:
            skipped_no_ref += 1
            continue
        hyp = r.get("transcription")
        if hyp is None or hyp == "":
            skipped_no_hyp += 1
            continue

        old_ref = (r.get("reference_text") or "").strip()
        ref_changed = old_ref != cur_ref
        if ref_changed:
            stale += 1

        try:
            old_wer = float(r["WER"])
            old_cer = float(r["CER"])
        except (KeyError, TypeError, ValueError):
            old_wer = old_cer = None

        # Referans degismemisse metrik tanim geregi aynidir; yeniden hesaplamak
        # gereksizdir. Ayrica jiwer'in CER hizalamasi uzun metinlerde O(n*m)
        # bellek harcar (video_06 ~30k karakter -> ~3.6 GB), bu yuzden bu atlama
        # hem hizli hem bellek acisindan zorunludur.
        if not ref_changed:
            unchanged += 1
            continue

        m = evaluate_asr_pair(cur_ref, hyp)
        new_wer, new_cer = m["wer"], m["cer"]

        r["reference_text"] = cur_ref
        r["WER"] = f"{new_wer}"
        r["CER"] = f"{new_cer}"

        if old_wer is not None and (abs(new_wer - old_wer) > 1e-9 or abs(new_cer - old_cer) > 1e-9):
            changes.append({
                "model_id": r.get("model_id", ""),
                "basename": b,
                "condition": r.get("condition", ""),
                "run_purpose": r.get("run_purpose", ""),
                "ref_changed": "evet" if ref_changed else "hayir",
                "old_wer": f"{old_wer:.6f}",
                "new_wer": f"{new_wer:.6f}",
                "wer_delta": f"{new_wer - old_wer:.6f}",
                "old_cer": f"{old_cer:.6f}",
                "new_cer": f"{new_cer:.6f}",
                "old_ref_sha1": hashlib.sha1(old_ref.encode()).hexdigest()[:12],
                "new_ref_sha1": hashlib.sha1(cur_ref.encode()).hexdigest()[:12],
            })

    print(f"Guncel olmayan referans : {stale} satir (yeniden hesaplandi)")
    print(f"Atlandi (referans ayni) : {unchanged} satir")
    print(f"Metrigi degisen         : {len(changes)} satir")
    if skipped_no_ref:
        print(f"Atlandi (referans yok)  : {skipped_no_ref}")
    if skipped_no_hyp:
        print(f"Atlandi (hipotez yok)   : {skipped_no_hyp}")

    if changes:
        print(f"\n{'model':<38}{'kayit':<32}{'kosul':<11}{'eski':>8}{'yeni':>8}{'fark':>9}  ref?")
        print("-" * 112)
        for c in sorted(changes, key=lambda x: -abs(float(x["wer_delta"])))[:40]:
            print(f"{c['model_id'][:37]:<38}{c['basename'][:31]:<32}{c['condition']:<11}"
                  f"{float(c['old_wer'])*100:>7.2f}%{float(c['new_wer'])*100:>7.2f}%"
                  f"{float(c['wer_delta'])*100:>+8.2f}pp  {c['ref_changed']}")
        if len(changes) > 40:
            print(f"... ve {len(changes)-40} satir daha (tam liste changelog CSV'sinde)")

    if args.dry_run:
        print("\n[DRY-RUN] Dosya yazilmadi.")
        return 0

    os.makedirs(os.path.dirname(REBUILT_CSV), exist_ok=True)
    with open(REBUILT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nYeniden hesaplanmis baseline: {os.path.relpath(REBUILT_CSV, os.path.dirname(PKG_ROOT))}")

    if changes:
        with open(CHANGELOG_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(changes[0].keys()))
            w.writeheader()
            w.writerows(changes)
        print(f"Degisim kaydi               : {os.path.relpath(CHANGELOG_CSV, os.path.dirname(PKG_ROOT))}")

    print("\n[NOT] Orijinal baseline DEGISTIRILMEDI; yeniden hesaplanmis surum ayri dosyada.")
    print("      Bundan sonraki tum kiyaslar '_refv2' surumu uzerinden yapilmalidir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
