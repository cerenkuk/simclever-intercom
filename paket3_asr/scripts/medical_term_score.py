"""
Paket 3 - Tıbbi Terim İsabeti & Halüsinasyon Puanlayıcı (Bölüm 9)
SimClever Intercom - Medical Term Accuracy & Hallucination Counter
"""

import re
from typing import List, Dict

# Varsayılan Tıbbi Terim Listesi (Simülasyon İntercom için kritik kelimeler)
DEFAULT_MEDICAL_TERMS = [
    "anamnez", "tansiyon", "nabız", "sıkıntı", "öksürük", "solunum",
    "kardiyak", "stetoskop", "muayene", "entübasyon", "endotrakeal",
    "saturasyon", "hipoksi", "dispne", "taşikardi", "bradikardi",
    "göğüs", "batın", "nöroloji", "pediatrik", "fiziksel", "poliklinik"
]

def calculate_medical_term_recall(reference_text: str, hypothesis_text: str, medical_terms: List[str] = None) -> Dict:
    """
    Referans metinde geçen tıbbi terimlerin model çıktısında doğru yazılıp yazılmadığını tespit eder.
    """
    if medical_terms is None:
        medical_terms = DEFAULT_MEDICAL_TERMS
        
    ref_words = set(re.findall(r'\w+', reference_text.lower()))
    hyp_words = set(re.findall(r'\w+', hypothesis_text.lower()))
    
    # Referansta geçen tıbbi terimler
    terms_in_ref = [t for t in medical_terms if t in ref_words]
    if not terms_in_ref:
        return {
            "terms_in_ref_count": 0,
            "terms_found_count": 0,
            "medical_term_accuracy": 1.0,
            "missed_terms": []
        }
        
    found_terms = [t for t in terms_in_ref if t in hyp_words]
    missed_terms = [t for t in terms_in_ref if t not in hyp_words]
    
    accuracy = round(len(found_terms) / len(terms_in_ref), 4)
    
    return {
        "terms_in_ref_count": len(terms_in_ref),
        "terms_found_count": len(found_terms),
        "medical_term_accuracy": accuracy,
        "found_terms": found_terms,
        "missed_terms": missed_terms
    }

def count_hallucinations_in_silence(silence_hypothesis_text: str) -> int:
    """Sessizlik/gürültü kesitinde modelin ürettiği kelime sayısını (halüsinasyon) döndürür."""
    words = re.findall(r'\w+', silence_hypothesis_text.strip())
    return len(words)

if __name__ == "__main__":
    ref = "Hasta dispne ve solunum sıkıntısı ile acile geldi, tansiyon 120/80."
    hyp = "Hasta solunum sıkıntısı ile acile geldi tansiyon seksen."
    
    res = calculate_medical_term_recall(ref, hyp)
    print("Tıbbi Terim Analiz Testi:")
    print(f"Referanstaki Terim Sayısı: {res['terms_in_ref_count']}")
    print(f"Yakalama Oranı (Accuracy): %{res['medical_term_accuracy'] * 100:.2f}")
    print(f"Kaçırılan Terimler       : {res['missed_terms']}")
