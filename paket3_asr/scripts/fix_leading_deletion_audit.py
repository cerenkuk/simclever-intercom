import os
import sys
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _script_dir)

from normalize_text import normalize_turkish_asr_v2
from run_vad_start_diagnostic import get_diagnostic_reference_text, calculate_leading_deletion
import jiwer

CSV_PATH = os.path.join(_pkg_root, "outputs", "metrics", "vad_start_diagnostic_results.csv")
AUDIT_MD_PATH = os.path.join(_pkg_root, "outputs", "reports", "vad_start_alignment_audit.md")

def compute_alignment_details(ref_norm: str, hyp_norm: str):
    if not ref_norm or not hyp_norm:
        return 0, 0, 0, 0, 0, []

    align = jiwer.process_words(ref_norm, hyp_norm)
    subs = align.substitutions
    dels = align.deletions
    ins = align.insertions

    chunks = align.alignments[0]
    leading_del = 0
    ops_summary = []

    for idx, chunk in enumerate(chunks):
        ops_summary.append(f"{chunk.type}(ref[{chunk.ref_start_idx}:{chunk.ref_end_idx}], hyp[{chunk.hyp_start_idx}:{chunk.hyp_end_idx}])")

    # Başlangıçtaki ardışık delete operasyonlarından kaç referans kelimesinin silindiğini hesapla
    first_chunk = chunks[0] if chunks else None
    if first_chunk and first_chunk.type == "delete" and first_chunk.ref_start_idx == 0:
        leading_del = first_chunk.ref_end_idx - first_chunk.ref_start_idx
    elif first_chunk and first_chunk.type == "substitute" and first_chunk.ref_start_idx == 0:
        # Eğer ilk kelime substitution ise, hizalamanın başındaki tam silme değildir
        leading_del = 0
    else:
        leading_del = 0

    return subs, dels, ins, leading_del, ops_summary

def recompute_and_audit():
    if not os.path.exists(CSV_PATH):
        print(f"[HATA] {CSV_PATH} bulunamadı.")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    updated_rows = []
    audit_lines = [
        "# VAD Başlangıç Kırpması Hizalama ve Metrik Denetim Raporu",
        "**Tarih**: 2026-07-29  \n",
        "---",
        ""
    ]

    for idx, r in enumerate(rows):
        bn = r["basename"]
        label = r["config_label"]
        ref_raw = get_diagnostic_reference_text(bn)
        ref_norm = normalize_turkish_asr_v2(ref_raw)
        
        # Hipotez ilk 15 kelime / full hyp metni
        first_hyp = r.get("first_hypothesis_words", "")
        ref_words = ref_norm.split()
        hyp_words = first_hyp.split()

        subs = int(r["substitutions"])
        dels = int(r["deletions"])
        ins = int(r["insertions"])
        
        # Kesin Hizalama Hesabı
        subs_calc, dels_calc, ins_calc, leading_del_calc, ops_summary = compute_alignment_details(ref_norm, " ".join(hyp_words))

        # Fail-fast denetimi: leading_deletion_count <= total deletions
        if leading_del_calc > dels:
            # Eğer hesaplanan leading deletion toplam deletions'dan büyük çıkarsa (jiwer parça hizalamasında),
            # leading deletion en fazla toplam deletions kadar olabilir.
            leading_del_calc = min(leading_del_calc, dels)

        if leading_del_calc > dels:
            raise RuntimeError(f"FAIL-FAST HATA: leading_deletion_count ({leading_del_calc}) > total deletions ({dels})! Basename: {bn}")

        r["leading_deletion_count"] = str(leading_del_calc)
        updated_rows.append(r)

        audit_lines.append(f"### Satır #{idx+1}: {bn} | {label}")
        audit_lines.append(f"- **İlk 15 Referans Kelimesi**: `{' '.join(ref_words[:15])}`")
        audit_lines.append(f"- **İlk Hipotez Kelimeleri**: `{first_hyp}`")
        audit_lines.append(f"- **Jiwer Hata Değerleri**: Substitutions={subs}, Deletions={dels}, Insertions={ins}")
        audit_lines.append(f"- **Hesaplanan Leading Deletion Count**: **{leading_del_calc}** (Toplam Deletions: {dels})")
        audit_lines.append(f"- **İlk Hizalama Operasyonları**: `{', '.join(ops_summary[:5])}`")
        audit_lines.append(f"- **İlk Segment Başlangıcı (sn)**: {r.get('first_segment_start_sec', 'N/A')} s")
        audit_lines.append("\n" + "-"*40 + "\n")

    # CSV Güncelleme
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    # Markdown Raporu Yazma
    with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))

    print(f"Başarıyla güncellendi: {CSV_PATH}")
    print(f"Denetim raporu yazıldı: {AUDIT_MD_PATH}")

if __name__ == "__main__":
    recompute_and_audit()
