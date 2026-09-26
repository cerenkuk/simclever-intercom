import os
import sys
import csv
from collections import Counter

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, os.path.join(_pkg_root, "scripts"))
from normalize_text import normalize_turkish_asr

BENCHMARK_CSV = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
MANUAL_CSV = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_manual_review.csv")
OUT_CSV = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_analysis_v3_results.csv")

TARGET_CONFIGS = [
    ("faster-whisper-large-v3-int8", "none"),
    ("faster-whisper-large-v3-turbo-int8", "none"),
    ("faster-whisper-large-v3-turbo-int8", "mild"),
    ("faster-whisper-medium-int8", "none")
]

def analyze_repetition_v3(words):
    if not words:
        return 0, 0, 0, 0, 0, ""

    consec_token_count = 0
    longest_token_run = 1
    top_token_info = ""
    
    curr_w = words[0]
    curr_cnt = 1
    for w in words[1:]:
        if w == curr_w:
            curr_cnt += 1
        else:
            if curr_cnt > 1:
                consec_token_count += (curr_cnt - 1)
                if curr_cnt > longest_token_run:
                    longest_token_run = curr_cnt
                    top_token_info = f"'{curr_w}' (x{curr_cnt})"
            curr_w = w
            curr_cnt = 1
    if curr_cnt > 1:
        consec_token_count += (curr_cnt - 1)
        if curr_cnt > longest_token_run:
            longest_token_run = curr_cnt
            top_token_info = f"'{curr_w}' (x{curr_cnt})"
    if longest_token_run == 1:
        longest_token_run = 0

    consec_phrase_count = 0
    longest_phrase_run = 0
    top_phrase_info = ""

    for phrase_len in [2, 3, 4]:
        if len(words) < phrase_len * 3:
            continue
        i = 0
        while i <= len(words) - phrase_len * 3:
            phrase = tuple(words[i:i+phrase_len])
            run = 1
            j = i + phrase_len
            while j <= len(words) - phrase_len and tuple(words[j:j+phrase_len]) == phrase:
                run += 1
                j += phrase_len
            if run >= 3:
                consec_phrase_count += (run - 1)
                if run > longest_phrase_run:
                    longest_phrase_run = run
                    top_phrase_info = f"'{' '.join(phrase)}' (x{run})"
                i = j
            else:
                i += 1

    ngram_counts = Counter()
    for n in [2, 3, 4]:
        for i in range(len(words) - n + 1):
            ngram_counts[tuple(words[i:i+n])] += 1
    
    dist_count = sum(cnt for phrase, cnt in ngram_counts.items() if cnt >= 3)
    top_dist = [(phrase, cnt) for phrase, cnt in ngram_counts.items() if cnt >= 3]
    top_dist.sort(key=lambda x: -x[1])
    dist_info = f"dist:'{' '.join(top_dist[0][0])}' (x{top_dist[0][1]})" if top_dist else ""

    summary_phrases = []
    if top_token_info: summary_phrases.append(top_token_info)
    if top_phrase_info: summary_phrases.append(top_phrase_info)
    if dist_info and not top_phrase_info and longest_token_run < 5: summary_phrases.append(dist_info)
    rep_str = " | ".join(summary_phrases) if summary_phrases else "yok"

    return (
        consec_token_count,
        longest_token_run,
        consec_phrase_count,
        longest_phrase_run,
        dist_count,
        rep_str
    )

def classify_loop_v3(longest_token_run, consec_token_count, longest_phrase_run, ins_rate, hyp_ref_ratio):
    if longest_token_run >= 10 or longest_phrase_run >= 5 or (hyp_ref_ratio > 1.4 and ins_rate > 0.2):
        reasons = []
        if longest_token_run >= 10: reasons.append(f"patolojik ardışık kelime zinciri ({longest_token_run})")
        if longest_phrase_run >= 5: reasons.append(f"patolojik ardışık ifade zinciri ({longest_phrase_run})")
        if hyp_ref_ratio > 1.4 and ins_rate > 0.2: reasons.append(f"aşırı metin şişmesi (hyp_ref={hyp_ref_ratio}, ins_rate={ins_rate})")
        return "EVET", "pathological", "; ".join(reasons)
    elif (5 <= longest_token_run < 10) or (3 <= longest_phrase_run < 5):
        reasons = []
        if 5 <= longest_token_run < 10: reasons.append(f"şüpheli ardışık kelime tekrarı ({longest_token_run})")
        if 3 <= longest_phrase_run < 5: reasons.append(f"şüpheli ardışık ifade tekrarı ({longest_phrase_run})")
        return "EVET", "warning", "; ".join(reasons)
    else:
        reason = "normal transkripsiyon"
        if 2 <= longest_token_run <= 4:
            reason = f"doğal ikileme/vurgu tekrarı ({longest_token_run})"
        elif hyp_ref_ratio < 1.1:
            reason = "dağıtık doğal kelime geçişi; bitişik döngü yok"
        return "HAYIR", "none", reason

def run_v3_analysis():
    manual_map = {}
    if os.path.exists(MANUAL_CSV):
        with open(MANUAL_CSV, encoding="utf-8") as f:
            for mr in csv.DictReader(f):
                key = (mr["basename"], mr["model_id"], mr["condition"])
                manual_map[key] = mr

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

                ref_words = normalize_turkish_asr(ref_raw).split() if ref_raw else []
                hyp_words = normalize_turkish_asr(hyp_raw).split() if hyp_raw else []

                N = len(ref_words)
                M = len(hyp_words)

                ins_val = int(r.get("insertions", 0))
                ins_rate = round(ins_val / N, 4) if N > 0 else 0.0
                hyp_ref_ratio = round(M / N, 4) if N > 0 else 0.0

                (
                    consec_token_cnt,
                    longest_token_run,
                    consec_phrase_cnt,
                    longest_phrase_run,
                    dist_cnt,
                    rep_str
                ) = analyze_repetition_v3(hyp_words)

                key = (bn, model_id, cond)
                if key in manual_map:
                    mr = manual_map[key]
                    v_str = mr["manual_verdict"].lower()
                    manual_status = "completed"
                    manual_verdict = v_str
                    
                    if "real_speech_repetition" in v_str:
                        loop_detected = "HAYIR"
                        loop_severity = "none"
                        loop_reason = "CONFIRMED_REAL_SPEECH_REPETITION: seste gerçek tekrar var"
                    elif "substitution_with_over_repetition" in v_str:
                        loop_detected = "HAYIR"
                        loop_severity = "warning"
                        loop_reason = f"CONFIRMED_SUBSTITUTION_WITH_OVER_REPETITION: {mr['actual_spoken_term']} -> {mr['model_recognized_term']}"
                    else:
                        loop_detected = "EVET"
                        loop_severity = "pathological"
                        loop_reason = f"CONFIRMED_PATHOLOGICAL_LOOP: {mr['notes']}"
                else:
                    if longest_token_run >= 10 or longest_phrase_run >= 5 or (hyp_ref_ratio > 1.4 and ins_rate > 0.2):
                        loop_detected = "EVET"
                        loop_severity = "pathological"
                        loop_reason = f"patolojik ardışık zincir (token={longest_token_run}, phrase={longest_phrase_run})"
                        manual_status = "pending"
                        manual_verdict = ""
                    elif (5 <= longest_token_run < 10) or (3 <= longest_phrase_run < 5):
                        loop_detected = "EVET"
                        loop_severity = "warning"
                        loop_reason = f"şüpheli ardışık tekrar (token={longest_token_run}, phrase={longest_phrase_run})"
                        manual_status = "pending"
                        manual_verdict = ""
                    else:
                        loop_detected = "HAYIR"
                        loop_severity = "none"
                        loop_reason = "normal transkripsiyon"
                        manual_status = "not_required"
                        manual_verdict = "normal_transcription"

                analyzed_data.append({
                    "model_id": model_id,
                    "basename": bn,
                    "condition": cond,
                    "protocol_version": r["protocol_version"],
                    "run_purpose": r["run_purpose"],
                    "reference_word_count": N,
                    "hypothesis_word_count": M,
                    "hyp_ref_ratio": hyp_ref_ratio,
                    "substitutions": r.get("substitutions", 0),
                    "deletions": r.get("deletions", 0),
                    "insertions": ins_val,
                    "insertion_rate": ins_rate,
                    "consecutive_token_repeat_count": consec_token_cnt,
                    "longest_consecutive_token_run": longest_token_run,
                    "consecutive_phrase_repeat_count": consec_phrase_cnt,
                    "longest_consecutive_phrase_run": longest_phrase_run,
                    "distributed_occurrence_count": dist_cnt,
                    "repeated_token_or_phrase": rep_str,
                    "loop_detected": loop_detected,
                    "loop_severity": loop_severity,
                    "loop_detection_reason": loop_reason,
                    "manual_review_status": manual_status,
                    "manual_verdict": manual_verdict,
                    "WER": r.get("WER", ""),
                    "CER": r.get("CER", ""),
                    "RTF": r.get("RTF", "")
                })

    fieldnames = [
        "model_id", "basename", "condition", "protocol_version", "run_purpose",
        "reference_word_count", "hypothesis_word_count", "hyp_ref_ratio",
        "substitutions", "deletions", "insertions", "insertion_rate",
        "consecutive_token_repeat_count", "longest_consecutive_token_run",
        "consecutive_phrase_repeat_count", "longest_consecutive_phrase_run",
        "distributed_occurrence_count", "repeated_token_or_phrase",
        "loop_detected", "loop_severity", "loop_detection_reason",
        "manual_review_status", "manual_verdict",
        "WER", "CER", "RTF"
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analyzed_data)

    print(f"FAST SUCCESS: Wrote {len(analyzed_data)} rows to {OUT_CSV}")

if __name__ == "__main__":
    run_v3_analysis()
