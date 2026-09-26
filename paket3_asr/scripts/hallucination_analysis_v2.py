import os
import sys
import csv
from collections import Counter

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_script_dir)
from normalize_text import normalize_turkish_asr

_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
BENCHMARK_CSV = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
OUT_CSV = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_analysis_v2_results.csv")

TARGET_CONFIGS = [
    ("faster-whisper-large-v3-int8", "none"),
    ("faster-whisper-large-v3-turbo-int8", "none"),
    ("faster-whisper-large-v3-turbo-int8", "mild"),
    ("faster-whisper-medium-int8", "none")
]

def compute_alignment_sdi(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
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

def find_repeat_patterns(words):
    if not words:
        return 0, 0, ""

    consec_count = 0
    max_word_run = 0
    top_word_phrase = ""
    
    curr_w = words[0]
    curr_cnt = 1
    for w in words[1:]:
        if w == curr_w:
            curr_cnt += 1
        else:
            if curr_cnt > 1:
                consec_count += (curr_cnt - 1)
                if curr_cnt > max_word_run:
                    max_word_run = curr_cnt
                    top_word_phrase = f"'{curr_w}' (x{curr_cnt})"
            curr_w = w
            curr_cnt = 1
    if curr_cnt > 1:
        consec_count += (curr_cnt - 1)
        if curr_cnt > max_word_run:
            max_word_run = curr_cnt
            top_word_phrase = f"'{curr_w}' (x{curr_cnt})"

    max_phrase_run = 0
    top_ngram_phrase = ""
    for n in [2, 3, 4]:
        if len(words) >= n * 2:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)
            for ng, cnt in counts.items():
                if cnt >= 3 and cnt > max_phrase_run:
                    max_phrase_run = cnt
                    top_ngram_phrase = f"'{' '.join(ng)}' (x{cnt})"

    longest_run = max(max_word_run, max_phrase_run)
    repeated_info = []
    if top_word_phrase: repeated_info.append(top_word_phrase)
    if top_ngram_phrase: repeated_info.append(top_ngram_phrase)
    
    repeated_str = " | ".join(repeated_info) if repeated_info else "yok"
    return consec_count, longest_run, repeated_str

def detect_loop(consec_count, longest_run, ins_rate, hyp_ref_ratio):
    reasons = []
    if longest_run >= 5:
        reasons.append(f"en uzun tekrar zinciri yüksek ({longest_run})")
    if consec_count >= 15:
        reasons.append(f"toplam ardışık tekrar yüksek ({consec_count})")
    if hyp_ref_ratio > 1.4 and ins_rate > 0.2:
        reasons.append(f"aşırı metin uzaması (hyp_ref={hyp_ref_ratio}, ins_rate={ins_rate})")
    
    if reasons:
        return "EVET", "; ".join(reasons)
    return "HAYIR", "normal transkripsiyon"

with open(BENCHMARK_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

main_rows = [r for r in rows if r.get("run_purpose") == "main_benchmark" and r.get("protocol_version") == "pilot-v2-antiloop"]
analyzed_data = []

for model_id, cond in TARGET_CONFIGS:
    for r in main_rows:
        if r["model_id"] == model_id and r["condition"] == cond:
            bn = r["basename"]
            ref_raw = r.get("reference_text", "")
            hyp_raw = r.get("transcription", "")
            
            ref_norm = normalize_turkish_asr(ref_raw) if ref_raw else ""
            hyp_norm = normalize_turkish_asr(hyp_raw) if hyp_raw else ""
            
            ref_words = ref_norm.split() if ref_norm else []
            hyp_words = hyp_norm.split() if hyp_norm else []
            
            N = len(ref_words)
            M = len(hyp_words)
            
            subs, dels, ins = compute_alignment_sdi(ref_words, hyp_words)
            ins_rate = round(ins / N, 4) if N > 0 else 0.0
            hyp_ref_ratio = round(M / N, 4) if N > 0 else 0.0
            
            consec_count, longest_run, repeated_str = find_repeat_patterns(hyp_words)
            loop_detected, loop_reason = detect_loop(consec_count, longest_run, ins_rate, hyp_ref_ratio)
            
            analyzed_data.append({
                "model_id": model_id,
                "basename": bn,
                "condition": cond,
                "protocol_version": r["protocol_version"],
                "run_purpose": r["run_purpose"],
                "reference_word_count": N,
                "hypothesis_word_count": M,
                "hyp_ref_ratio": hyp_ref_ratio,
                "substitutions": subs,
                "deletions": dels,
                "insertions": ins,
                "insertion_rate": ins_rate,
                "consecutive_repeat_count": consec_count,
                "longest_repeat_run": longest_run,
                "repeated_token_or_phrase": repeated_str,
                "loop_detected": loop_detected,
                "loop_detection_reason": loop_reason,
                "WER": r.get("WER", ""),
                "CER": r.get("CER", ""),
                "RTF": r.get("RTF", "")
            })

fieldnames = [
    "model_id", "basename", "condition", "protocol_version", "run_purpose",
    "reference_word_count", "hypothesis_word_count", "hyp_ref_ratio",
    "substitutions", "deletions", "insertions", "insertion_rate",
    "consecutive_repeat_count", "longest_repeat_run", "repeated_token_or_phrase",
    "loop_detected", "loop_detection_reason", "WER", "CER", "RTF"
]

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(analyzed_data)

print(f"DONE! Wrote {len(analyzed_data)} rows to {OUT_CSV}")
