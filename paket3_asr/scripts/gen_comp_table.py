import csv

bm_path = 'paket3_asr/outputs/metrics/benchmark_results.csv'
cp_path = 'paket3_asr/outputs/metrics/candidate_prescreen_results.csv'

target_basenames = ['video_01_genel_anamnez', 'video_04_simule_hasta_gorusmesi', 'saha_03_img_3602']

with open(bm_path, 'r', encoding='utf-8') as f:
    bm_rows = list(csv.DictReader(f))

with open(cp_path, 'r', encoding='utf-8') as f:
    cp_rows = list(csv.DictReader(f))

configs = [
    ('Large-v3 + none', 'faster-whisper-large-v3-int8', 'none', bm_rows),
    ('Turbo + none', 'faster-whisper-large-v3-turbo-int8', 'none', bm_rows),
    ('Turbo + mild', 'faster-whisper-large-v3-turbo-int8', 'mild', bm_rows),
    ('Medium + none', 'faster-whisper-medium-int8', 'none', bm_rows),
    ('Qwen3-ASR 0.6B + none', 'qwen3-asr-0.6b-hf', 'none', cp_rows),
    ('Qwen3-ASR 1.7B + none', 'qwen3-asr-1.7b-hf', 'none', cp_rows)
]

print(f"{'Konfigürasyon':<25} | {'Makro WER':<10} | {'Makro CER':<10} | {'Ort. RTF':<10} | {'Max Yük. RAM':<14} | {'Max Çıkarım RAM':<14}")
print('-' * 95)

for label, m_id, cond, source in configs:
    sub_rows = [r for r in source if r['model_id'] == m_id and r['condition'] == cond and r['basename'] in target_basenames]
    if len(sub_rows) == 0:
        print(f"{label:<25} | N/A        | N/A        | N/A        | N/A            | N/A")
        continue
    wers = [float(r['WER']) for r in sub_rows]
    cers = [float(r['CER']) for r in sub_rows]
    rtfs = [float(r['RTF']) for r in sub_rows]
    load_rams = [float(r['load_peak_ram_mb']) if r.get('load_peak_ram_mb') else None for r in sub_rows]
    inf_rams = [float(r['inference_peak_ram_mb']) if r.get('inference_peak_ram_mb') else None for r in sub_rows]
    
    m_wer = f"%{round(sum(wers)/len(wers)*100, 2)}"
    m_cer = f"%{round(sum(cers)/len(cers)*100, 2)}"
    m_rtf = f"{round(sum(rtfs)/len(rtfs), 4)}"
    
    valid_l_rams = [x for x in load_rams if x is not None]
    valid_i_rams = [x for x in inf_rams if x is not None]
    
    max_l = f"{round(max(valid_l_rams), 2)} MB" if valid_l_rams else 'N/A'
    max_i = f"{round(max(valid_i_rams), 2)} MB" if valid_i_rams else 'N/A'
    
    print(f"{label:<25} | {m_wer:<10} | {m_cer:<10} | {m_rtf:<10} | {max_l:<14} | {max_i:<14}")
