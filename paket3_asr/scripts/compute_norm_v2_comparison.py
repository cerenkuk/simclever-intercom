import os
import sys
import csv
import re

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, os.path.join(_pkg_root, "scripts"))
from normalize_text import normalize_turkish_asr, normalize_turkish_asr_v2, lowercase_tr, strip_punctuation_and_formatting, expand_numbers_in_text
from calculate_metrics import levenshtein_distance

BENCHMARK_CSV = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
COMPARISON_CSV = os.path.join(_pkg_root, "outputs", "metrics", "normalization_v2_comparison.csv")

def norm_dr_only(text: str) -> str:
    if not text: return ""
    text = lowercase_tr(text)
    text = re.sub(r'\bdr\.\b', 'doktor', text)
    text = strip_punctuation_and_formatting(text)
    text = expand_numbers_in_text(text)
    tokens = [('doktor' if tok == 'dr' else tok) for tok in text.split()]
    return ' '.join(tokens)

def norm_filler_only(text: str) -> str:
    if not text: return ""
    text = lowercase_tr(text)
    text = strip_punctuation_and_formatting(text)
    text = expand_numbers_in_text(text)
    tokens = []
    for tok in text.split():
        if re.fullmatch(r'ı+', tok): tok = 'ıı'
        elif re.fullmatch(r'e+', tok) and len(tok) > 1: tok = 'ee'
        elif tok in ['hım', 'hmm', 'hmmm']: tok = 'hım'
        tokens.append(tok)
    return ' '.join(tokens)

def compute_alignment_sdi(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    if n == 0: return 0, 0, m
    if m == 0: return 0, n, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins

def run_comparison_and_ablation():
    with open(BENCHMARK_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    comparison_rows = []
    
    # Impact counters
    impact = {
        "dr_only": {"ref_changed": 0, "hyp_changed": 0, "wer_diff_sum": 0.0, "count": 0},
        "filler_only": {"ref_changed": 0, "hyp_changed": 0, "wer_diff_sum": 0.0, "count": 0},
        "both_v2": {"ref_changed": 0, "hyp_changed": 0, "wer_diff_sum": 0.0, "count": 0}
    }

    video_old_wer, video_new_wer = [], []
    saha_old_wer, saha_new_wer = [], []

    recomputable_cnt = 0
    not_recomputable_cnt = 0

    for r in rows:
        model_id = r["model_id"]
        bn = r["basename"]
        cond = r["condition"]
        purpose = r.get("run_purpose", "main_benchmark")
        proto = r.get("protocol_version", "pilot-v2-antiloop")
        ref_raw = r.get("reference_text", "")
        hyp_raw = r.get("transcription", "")

        if not hyp_raw and not ref_raw:
            not_recomputable_cnt += 1
            comparison_rows.append({
                "model_id": model_id, "basename": bn, "condition": cond,
                "run_purpose": purpose, "protocol_version": proto,
                "old_normalizer_version": "turkish_asr_v1", "new_normalizer_version": "turkish_asr_verbatim_v2",
                "old_WER": r.get("WER", ""), "new_WER": "", "WER_delta_percentage_points": "",
                "old_CER": r.get("CER", ""), "new_CER": "", "CER_delta_percentage_points": "",
                "old_substitutions": r.get("substitutions", ""), "new_substitutions": "",
                "old_deletions": r.get("deletions", ""), "new_deletions": "",
                "old_insertions": r.get("insertions", ""), "new_insertions": "",
                "reference_words": "", "hypothesis_path": f"paket3_asr/data/references/{bn}.txt",
                "recompute_status": "not_recomputable", "recompute_reason": "Hypothesis or reference text missing"
            })
            continue

        recomputable_cnt += 1

        # v1 calculation
        ref_v1 = normalize_turkish_asr(ref_raw)
        hyp_v1 = normalize_turkish_asr(hyp_raw)
        ref_v1_words = ref_v1.split()
        hyp_v1_words = hyp_v1.split()
        N_v1 = len(ref_v1_words)
        s_v1, d_v1, i_v1 = compute_alignment_sdi(ref_v1_words, hyp_v1_words)
        wer_v1 = round((s_v1 + d_v1 + i_v1) / N_v1, 4) if N_v1 > 0 else 0.0
        cer_v1 = round(levenshtein_distance(list(ref_v1.replace(" ", "")), list(hyp_v1.replace(" ", ""))) / len(ref_v1.replace(" ", "")), 4) if ref_v1.replace(" ", "") else 0.0

        # v2 calculation
        ref_v2 = normalize_turkish_asr_v2(ref_raw)
        hyp_v2 = normalize_turkish_asr_v2(hyp_raw)
        ref_v2_words = ref_v2.split()
        hyp_v2_words = hyp_v2.split()
        N_v2 = len(ref_v2_words)
        s_v2, d_v2, i_v2 = compute_alignment_sdi(ref_v2_words, hyp_v2_words)
        wer_v2 = round((s_v2 + d_v2 + i_v2) / N_v2, 4) if N_v2 > 0 else 0.0
        cer_v2 = round(levenshtein_distance(list(ref_v2.replace(" ", "")), list(hyp_v2.replace(" ", ""))) / len(ref_v2.replace(" ", "")), 4) if ref_v2.replace(" ", "") else 0.0

        # Ablations
        ref_dr = norm_dr_only(ref_raw)
        hyp_dr = norm_dr_only(hyp_raw)
        if ref_dr != ref_v1: impact["dr_only"]["ref_changed"] += 1
        if hyp_dr != hyp_v1: impact["dr_only"]["hyp_changed"] += 1
        
        ref_fl = norm_filler_only(ref_raw)
        hyp_fl = norm_filler_only(hyp_raw)
        if ref_fl != ref_v1: impact["filler_only"]["ref_changed"] += 1
        if hyp_fl != hyp_v1: impact["filler_only"]["hyp_changed"] += 1

        if ref_v2 != ref_v1: impact["both_v2"]["ref_changed"] += 1
        if hyp_v2 != hyp_v1: impact["both_v2"]["hyp_changed"] += 1

        wer_delta_pp = round((wer_v2 - wer_v1) * 100, 2)
        cer_delta_pp = round((cer_v2 - cer_v1) * 100, 2)

        if purpose == "main_benchmark" and proto == "pilot-v2-antiloop":
            if bn.startswith("video_"):
                video_old_wer.append(wer_v1)
                video_new_wer.append(wer_v2)
            elif bn.startswith("saha_"):
                saha_old_wer.append(wer_v1)
                saha_new_wer.append(wer_v2)

        comparison_rows.append({
            "model_id": model_id,
            "basename": bn,
            "condition": cond,
            "run_purpose": purpose,
            "protocol_version": proto,
            "old_normalizer_version": "turkish_asr_v1",
            "new_normalizer_version": "turkish_asr_verbatim_v2",
            "old_WER": f"{wer_v1 * 100:.2f}%",
            "new_WER": f"{wer_v2 * 100:.2f}%",
            "WER_delta_percentage_points": f"{wer_delta_pp:+.2f} pp",
            "old_CER": f"{cer_v1 * 100:.2f}%",
            "new_CER": f"{cer_v2 * 100:.2f}%",
            "CER_delta_percentage_points": f"{cer_delta_pp:+.2f} pp",
            "old_substitutions": s_v1,
            "new_substitutions": s_v2,
            "old_deletions": d_v1,
            "new_deletions": d_v2,
            "old_insertions": i_v1,
            "new_insertions": i_v2,
            "reference_words": N_v2,
            "hypothesis_path": f"paket3_asr/data/references/{bn}.txt",
            "recompute_status": "recomputed_success",
            "recompute_reason": "Recomputed with turkish_asr_verbatim_v2"
        })

    fieldnames = list(comparison_rows[0].keys())
    with open(COMPARISON_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Generated {COMPARISON_CSV} with {len(comparison_rows)} rows.")
    print(f"Recomputable rows: {recomputable_cnt}, Not recomputable rows: {not_recomputable_cnt}")
    print("\n--- ABLATION IMPACT COUNTS ---")
    print(f"1. dr_only     -> Ref Changed: {impact['dr_only']['ref_changed']}, Hyp Changed: {impact['dr_only']['hyp_changed']}")
    print(f"2. filler_only -> Ref Changed: {impact['filler_only']['ref_changed']}, Hyp Changed: {impact['filler_only']['hyp_changed']}")
    print(f"3. both_v2     -> Ref Changed: {impact['both_v2']['ref_changed']}, Hyp Changed: {impact['both_v2']['hyp_changed']}")

    if video_old_wer and saha_old_wer:
        v_old_m = sum(video_old_wer) / len(video_old_wer)
        v_new_m = sum(video_new_wer) / len(video_new_wer)
        s_old_m = sum(saha_old_wer) / len(saha_old_wer)
        s_new_m = sum(saha_new_wer) / len(saha_new_wer)
        print("\n--- MACRO IMPACT SUMMARY (Main Benchmark Rows) ---")
        print(f"Video Group Macro WER: Old {v_old_m*100:.2f}% -> New {v_new_m*100:.2f}% ({(v_new_m - v_old_m)*100:+.2f} pp)")
        print(f"Saha Group Macro WER : Old {s_old_m*100:.2f}% -> New {s_new_m*100:.2f}% ({(s_new_m - s_old_m)*100:+.2f} pp)")

if __name__ == "__main__":
    run_comparison_and_ablation()
