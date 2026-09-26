"""
Paket 3 - Aşama 6: Tıbbi Terim Analizi
1) Referans metinlerden aday tıbbi terimleri çıkarır -> medical_terms.yaml
2) Her model/kayıt/koşul çifti için tıbbi terim TP/FN/FP/Recall/Precision/F1 hesaplar
3) Sonuçları medical_term_analysis_results.csv'ye yazar

UYARI: Bu dosyada aday terimler yapay zekâ (LLM) tarafından önerilmiştir.
Nihai raporda kullanılmadan önce insan doğrulaması gerektirilmektedir.
"""

import csv
import os
import sys
import re

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_script_dir)
from normalize_text import normalize_turkish_asr

_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
CSV_IN      = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
CSV_OUT     = os.path.join(_pkg_root, "outputs", "metrics", "medical_term_analysis_results.csv")
YAML_OUT    = os.path.join(_pkg_root, "config", "medical_terms.yaml")

# ---------------------------------------------------------------------------
# 1) Kayıt bazlı tıbbi terim sözlüğü - YALNIZCA referans metinde gerçekten
#    geçen terimler eklendi. İnsan doğrulaması gerektirenler review_required: true
#    ile işaretlendi.
# ---------------------------------------------------------------------------
MEDICAL_TERMS_BY_RECORDING = {
    "video_01_genel_anamnez": {
        "terms": [
            {"term": "anamnez", "review_required": False, "category": "klinik_islem"},
            {"term": "demonstrasyon", "review_required": False, "category": "egitim"},
            {"term": "dahiliye", "review_required": False, "category": "uzmanlik"},
            {"term": "ic hastaliklari", "review_required": False, "category": "uzmanlik"},
            {"term": "sistemik sorgulama", "review_required": True, "category": "klinik_islem"},
            {"term": "solunum sistemi", "review_required": False, "category": "anatomik"},
            {"term": "kardiyovaskuler sistem", "review_required": False, "category": "anatomik"},
            {"term": "dispne", "review_required": False, "category": "semptom"},
            {"term": "nefes darligi", "review_required": False, "category": "semptom"},
            {"term": "gece terleme", "review_required": False, "category": "semptom"},
            {"term": "kilo kaybi", "review_required": False, "category": "semptom"},
            {"term": "ates", "review_required": False, "category": "semptom"},
            {"term": "oksuruk", "review_required": False, "category": "semptom"},
            {"term": "balgam", "review_required": False, "category": "semptom"},
            {"term": "muayene", "review_required": False, "category": "klinik_islem"},
            {"term": "tanikoyma", "review_required": True, "category": "klinik_islem"},
        ]
    },
    "video_02_psikiyatrik_anamnez": {
        "terms": [
            {"term": "psikiyatrik anamnez", "review_required": False, "category": "klinik_islem"},
            {"term": "ruhsal durum muayenesi", "review_required": False, "category": "klinik_islem"},
            {"term": "psikiyatrik", "review_required": False, "category": "uzmanlik"},
            {"term": "antidepresan", "review_required": False, "category": "ilac"},
            {"term": "hipotiroidi", "review_required": False, "category": "hastalik"},
            {"term": "euthyrox", "review_required": False, "category": "ilac"},
            {"term": "paroksetin", "review_required": True, "category": "ilac"},
            {"term": "fluoksetin", "review_required": True, "category": "ilac"},
            {"term": "sertralin", "review_required": True, "category": "ilac"},
            {"term": "depresyon", "review_required": False, "category": "hastalik"},
            {"term": "anksiyete", "review_required": False, "category": "hastalik"},
            {"term": "oryantasyon", "review_required": False, "category": "klinik_islem"},
            {"term": "affect", "review_required": True, "category": "klinik_kavram"},
            {"term": "kognitif", "review_required": False, "category": "klinik_kavram"},
            {"term": "intihar", "review_required": False, "category": "klinik_kavram"},
            {"term": "halusinasyon", "review_required": False, "category": "semptom"},
            {"term": "illuzyon", "review_required": True, "category": "semptom"},
        ]
    },
    "video_03_pediatrik_anamnez": {
        "terms": [
            {"term": "pediatrik", "review_required": False, "category": "uzmanlik"},
            {"term": "pediatri", "review_required": False, "category": "uzmanlik"},
            {"term": "cocuk hastaliklari", "review_required": False, "category": "uzmanlik"},
            {"term": "asi", "review_required": False, "category": "klinik_islem"},
            {"term": "gelisim takibi", "review_required": False, "category": "klinik_islem"},
            {"term": "besi aliskanligi", "review_required": False, "category": "klinik_kavram"},
            {"term": "ates", "review_required": False, "category": "semptom"},
            {"term": "kusma", "review_required": False, "category": "semptom"},
            {"term": "ishal", "review_required": False, "category": "semptom"},
            {"term": "konvulziyon", "review_required": True, "category": "semptom"},
            {"term": "febril konvulziyon", "review_required": True, "category": "hastalik"},
            {"term": "otitis media", "review_required": True, "category": "hastalik"},
            {"term": "fizik muayene", "review_required": False, "category": "klinik_islem"},
            {"term": "anne sutuu", "review_required": False, "category": "klinik_kavram"},
        ]
    },
    "video_04_simule_hasta_gorusmesi": {
        "terms": [
            {"term": "bas agrisi", "review_required": False, "category": "semptom"},
            {"term": "ense tutulmasi", "review_required": True, "category": "semptom"},
            {"term": "tansiyon", "review_required": False, "category": "vital"},
            {"term": "kan basinci", "review_required": False, "category": "vital"},
            {"term": "hipertansiyon", "review_required": False, "category": "hastalik"},
            {"term": "bulanti", "review_required": False, "category": "semptom"},
            {"term": "kusma", "review_required": False, "category": "semptom"},
            {"term": "gorme bozuklugu", "review_required": False, "category": "semptom"},
            {"term": "rontgen", "review_required": False, "category": "tani_yontemi"},
            {"term": "ekg", "review_required": False, "category": "tani_yontemi"},
        ]
    },
    "video_05_poliklinik_anamnez_fizik_muayene": {
        "terms": [
            {"term": "poliklinik", "review_required": False, "category": "saglik_birimi"},
            {"term": "fizik muayene", "review_required": False, "category": "klinik_islem"},
            {"term": "nefes darligi", "review_required": False, "category": "semptom"},
            {"term": "akcigerler", "review_required": False, "category": "anatomik"},
            {"term": "auskultasyon", "review_required": False, "category": "klinik_islem"},
            {"term": "solunum sesi", "review_required": False, "category": "klinik_bulgu"},
            {"term": "ronkus", "review_required": True, "category": "klinik_bulgu"},
            {"term": "wheeze", "review_required": True, "category": "klinik_bulgu"},
            {"term": "saturasyon", "review_required": False, "category": "vital"},
            {"term": "oksijen saturasyonu", "review_required": False, "category": "vital"},
            {"term": "spo2", "review_required": False, "category": "vital"},
            {"term": "nebul", "review_required": True, "category": "tedavi"},
        ]
    },
    "video_06_gogus_hastaliklari_oyku_alma": {
        "terms": [
            {"term": "gogus hastaliklari", "review_required": False, "category": "uzmanlik"},
            {"term": "propedetik", "review_required": True, "category": "egitim"},
            {"term": "anamnez", "review_required": False, "category": "klinik_islem"},
            {"term": "oyku alma", "review_required": False, "category": "klinik_islem"},
            {"term": "plevral efuzyon", "review_required": True, "category": "hastalik"},
            {"term": "pnomonit", "review_required": True, "category": "hastalik"},
            {"term": "koah", "review_required": False, "category": "hastalik"},
            {"term": "astim", "review_required": False, "category": "hastalik"},
            {"term": "dispne", "review_required": False, "category": "semptom"},
            {"term": "hemoptizi", "review_required": True, "category": "semptom"},
            {"term": "kanlibalgam", "review_required": True, "category": "semptom"},
            {"term": "oksuruk", "review_required": False, "category": "semptom"},
            {"term": "balgam", "review_required": False, "category": "semptom"},
            {"term": "brons", "review_required": False, "category": "anatomik"},
            {"term": "akciger", "review_required": False, "category": "anatomik"},
            {"term": "soluk boru", "review_required": False, "category": "anatomik"},
        ]
    },
    "saha_01_img_3463": {
        "terms": [
            {"term": "dogum", "review_required": False, "category": "klinik_islem"},
            {"term": "vajinal dogum", "review_required": True, "category": "klinik_islem"},
            {"term": "resusitasyon", "review_required": True, "category": "klinik_islem"},
            {"term": "yenidogan", "review_required": False, "category": "hasta_grubu"},
            {"term": "hipoksi", "review_required": True, "category": "hastalik"},
            {"term": "ambu", "review_required": True, "category": "tibbi_cihaz"},
            {"term": "entupasyon", "review_required": True, "category": "klinik_islem"},
        ]
    },
    "saha_02_img_3492": {
        "terms": [
            {"term": "svt", "review_required": False, "category": "hastalik"},
            {"term": "supraventrikuler tasikardi", "review_required": True, "category": "hastalik"},
            {"term": "tasikardi", "review_required": False, "category": "hastalik"},
            {"term": "p dalgasi", "review_required": False, "category": "klinik_bulgu"},
            {"term": "adenozin", "review_required": False, "category": "ilac"},
            {"term": "vagal manevra", "review_required": False, "category": "klinik_islem"},
            {"term": "karotis masaji", "review_required": True, "category": "klinik_islem"},
            {"term": "valsalva", "review_required": False, "category": "klinik_islem"},
            {"term": "damar yolu", "review_required": False, "category": "klinik_islem"},
            {"term": "nabiz", "review_required": False, "category": "vital"},
        ]
    },
    "saha_03_img_3602": {
        "terms": [
            {"term": "svt", "review_required": False, "category": "hastalik"},
            {"term": "kalp hizi", "review_required": False, "category": "vital"},
            {"term": "adenozin", "review_required": False, "category": "ilac"},
            {"term": "damar yolu", "review_required": False, "category": "klinik_islem"},
            {"term": "buz uygulamasi", "review_required": True, "category": "klinik_islem"},
            {"term": "valsalva", "review_required": False, "category": "klinik_islem"},
            {"term": "karotis masaji", "review_required": True, "category": "klinik_islem"},
            {"term": "kardiyoversiyon", "review_required": True, "category": "klinik_islem"},
        ]
    },
    "saha_04_vid_20220913_142911": {
        "terms": [
            {"term": "pace", "review_required": False, "category": "tibbi_cihaz"},
            {"term": "pacer", "review_required": False, "category": "tibbi_cihaz"},
            {"term": "pacemaker", "review_required": True, "category": "tibbi_cihaz"},
            {"term": "nabiz", "review_required": False, "category": "vital"},
            {"term": "kalp hizi", "review_required": False, "category": "vital"},
            {"term": "ekg", "review_required": False, "category": "tani_yontemi"},
        ]
    },
}


def normalize_for_matching(text):
    """
    Eşleştirme normalizasyonu:
    - Küçük harfe çevir
    - Türkçe karakterleri Latin karşılığına dönüştür (opsiyonel, esnek eşleştirme için)
    - Noktalama kaldır
    - normalize_turkish_asr kullan
    NOT: Aşırı gevşek substring eşleştirmesi yerine token tabanlı eşleştirme yapılır.
    """
    text = normalize_turkish_asr(text)
    return text


def term_in_text(term_norm, text_words_set, text_words_list):
    """
    Terim eşleştirme kuralı:
    - Tek kelimeli terim: kelime kümesinde tam kelime eşleşmesi.
    - Çok kelimeli terim: metin listesinde bitişik sıralı token eşleşmesi.
    - Ek almış kelimeler: term_norm'un ilk 5 karakteri token'ın ilk 5 karakteriyle eşleşmesi
      (minimum 5 karakter prefix, substring değil prefix kontrolü).
    Aşırı gevşek substring eşleştirmesi kullanılmaz.
    """
    term_tokens = term_norm.split()
    n = len(term_tokens)

    if n == 1:
        t = term_tokens[0]
        # Tam eşleşme
        if t in text_words_set:
            return True
        # Prefix (ek alma) - minimum 5 karakter
        if len(t) >= 5:
            for w in text_words_set:
                if len(w) >= len(t) and w[:len(t)] == t:
                    return True
        return False
    else:
        # Çok kelimeli: bitişik token eşleşmesi
        for i in range(len(text_words_list) - n + 1):
            window = text_words_list[i:i+n]
            match = True
            for t_tok, w_tok in zip(term_tokens, window):
                if t_tok != w_tok and not (len(t_tok) >= 5 and w_tok[:len(t_tok)] == t_tok):
                    match = False
                    break
            if match:
                return True
        return False


def analyze_medical_terms(row, terms_for_recording):
    """Bir CSV satırı için tıbbi terim analizi."""
    ref_raw = row.get("reference_text", "").strip()
    hyp_raw = row.get("transcription", "").strip()

    ref_norm = normalize_for_matching(ref_raw)
    hyp_norm = normalize_for_matching(hyp_raw)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    ref_word_set = set(ref_words)
    hyp_word_set = set(hyp_words)

    terms = terms_for_recording.get("terms", [])
    if not terms:
        return None

    ref_present = []    # gerçekten referansta bulunan terimler
    tp_terms = []       # model de bulmuş
    fn_terms = []       # model kaçırmış
    fp_terms = []       # modelde var ama referansta yok (aday)

    for term_entry in terms:
        t = term_entry["term"]
        t_norm = normalize_for_matching(t)

        in_ref = term_in_text(t_norm, ref_word_set, ref_words)
        in_hyp = term_in_text(t_norm, hyp_word_set, hyp_words)

        if in_ref:
            ref_present.append(t)
            if in_hyp:
                tp_terms.append(t)
            else:
                fn_terms.append(t)
        else:
            # Referansta yok ama hipotezde varsa FP
            if in_hyp:
                fp_terms.append(t)

    total_ref = len(ref_present)
    tp = len(tp_terms)
    fn = len(fn_terms)
    fp = len(fp_terms)

    recall = round(tp / total_ref, 4) if total_ref > 0 else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "model_id": row["model_id"],
        "basename": row["basename"],
        "condition": row["condition"],
        "protocol_version": row["protocol_version"],
        "run_purpose": row["run_purpose"],
        "medical_term_reference_count": total_ref,
        "medical_term_true_positive": tp,
        "medical_term_false_negative": fn,
        "medical_term_false_positive": fp,
        "medical_term_recall": recall,
        "medical_term_precision": precision,
        "medical_term_f1": f1,
        "tp_terms": "; ".join(tp_terms),
        "fn_terms": "; ".join(fn_terms),
        "fp_terms": "; ".join(fp_terms),
    }


def write_yaml():
    """medical_terms.yaml dosyasını oluştur."""
    os.makedirs(os.path.dirname(YAML_OUT), exist_ok=True)

    lines = [
        "# Paket 3 - Kayıt Bazlı Tıbbi Terim Sözlüğü",
        "# UYARI: Bu dosyadaki terimler yapay zekâ (LLM) tarafından aday olarak çıkarılmıştır.",
        "# Nihai raporda kullanılmadan önce insan doğrulaması gerektirilmektedir.",
        "# review_required: true olan terimler öncelikli doğrulama gerektirir.",
        "#",
        "# Eşleştirme kuralı:",
        "#   - Tek kelimeli terim: tam kelime eşleşmesi (büyük/küçük harf normalize)",
        "#   - Çok kelimeli terim: bitişik token sırası",
        "#   - Ek almış kelime desteği: minimum 5 karakter prefix kontrolü",
        "#   - Aşırı gevşek substring eşleştirmesi kullanılmaz",
        "#",
        "# Sütunlar: term, review_required, category",
        "# Kategoriler: hastalik, ilac, anatomik, semptom, vital, klinik_islem,",
        "#              klinik_bulgu, klinik_kavram, tani_yontemi, tibbi_cihaz,",
        "#              uzmanlik, saglik_birimi, hasta_grubu, tedavi, egitim",
        "",
    ]

    review_required_count = 0
    total_count = 0

    for recording, data in MEDICAL_TERMS_BY_RECORDING.items():
        lines.append(f"{recording}:")
        for term_entry in data["terms"]:
            total_count += 1
            rr = term_entry["review_required"]
            if rr:
                review_required_count += 1
            cat = term_entry.get("category", "diger")
            lines.append(f"  - term: \"{term_entry['term']}\"")
            lines.append(f"    review_required: {str(rr).lower()}")
            lines.append(f"    category: {cat}")
        lines.append("")

    with open(YAML_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"YAML yazıldı: {YAML_OUT}")
    print(f"Toplam terim: {total_count}")
    print(f"review_required: {review_required_count}")
    return total_count, review_required_count


def main():
    # YAML yaz
    total_terms, review_count = write_yaml()

    # CSV analizi
    with open(CSV_IN, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    target_rows = [
        r for r in all_rows
        if r["run_purpose"] == "main_benchmark"
        and r["protocol_version"] == "pilot-v2-antiloop"
        and r.get("transcription", "").strip()
        and r.get("reference_text", "").strip()
        and r["basename"] in MEDICAL_TERMS_BY_RECORDING
    ]

    # Mükerrer koruma
    existing_keys = set()
    existing_results = []
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
                existing_keys.add(key)
                existing_results.append(r)

    results = []
    for row in target_rows:
        key = (row["model_id"], row["basename"], row["condition"], row["protocol_version"], row["run_purpose"])
        if key in existing_keys:
            continue
        recording_terms = MEDICAL_TERMS_BY_RECORDING.get(row["basename"], {})
        result = analyze_medical_terms(row, recording_terms)
        if result:
            results.append(result)

    all_results = existing_results + results

    fieldnames = [
        "model_id", "basename", "condition", "protocol_version", "run_purpose",
        "medical_term_reference_count", "medical_term_true_positive",
        "medical_term_false_negative", "medical_term_false_positive",
        "medical_term_recall", "medical_term_precision", "medical_term_f1",
        "tp_terms", "fn_terms", "fp_terms"
    ]

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n=== TIBBİ TERİM ANALİZİ TAMAMLANDI ===")
    print(f"Yeni satır: {len(results)}")
    print(f"Toplam: {len(all_results)}")
    print(f"Çıktı CSV: {CSV_OUT}")

    # Sabit konfigürasyonlar için özet
    configs = [
        ("faster-whisper-large-v3-int8", "none", "A"),
        ("faster-whisper-large-v3-turbo-int8", "none", "B"),
        ("faster-whisper-large-v3-turbo-int8", "mild", "C"),
    ]
    print(f"\n=== SABİT KONFİGÜRASYON TIBBİ TERİM MAKRO ORTALAMALAR ===")
    print(f"{'Konfig':<8} {'model':<45} {'cond':<12} {'grup':<8} {'recall':>8} {'precision':>10} {'f1':>8} {'n':>4}")
    print("-" * 110)

    def get_group(bn):
        return "video" if bn.startswith("video_") else ("saha" if bn.startswith("saha_") else "diger")

    for model, cond, label in configs:
        for group_name in ["video", "saha", "genel"]:
            subset = [
                r for r in all_results
                if r.get("model_id") == model
                and r.get("condition") == cond
                and (group_name == "genel" or get_group(r.get("basename","")) == group_name)
                and float(r.get("medical_term_reference_count", 0) or 0) > 0
            ]
            if not subset:
                continue
            recalls = [float(r.get("medical_term_recall", 0) or 0) for r in subset]
            precs = [float(r.get("medical_term_precision", 0) or 0) for r in subset]
            f1s = [float(r.get("medical_term_f1", 0) or 0) for r in subset]
            print(f"  {label:<6} {model:<45} {cond:<12} {group_name:<8} {sum(recalls)/len(recalls):>8.3f} {sum(precs)/len(precs):>10.3f} {sum(f1s)/len(f1s):>8.3f} {len(subset):>4}")


if __name__ == "__main__":
    main()
