"""
Y4 — Hipotez duzeyinde birlestirme (ROVER benzeri oylama).

Makale protokolu : docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md

GEREKCE
-------
Y1, Y1b ve Y2 hep ayni seyi denedi: on islemeyi kullanarak TEK bir daha iyi
sinyal uretmek. Sonuc: saglikli seste en iyi ihtimalle basabas.

Bu betik tamamen farkli bir mekanizmayi sinar: on islemeyi daha iyi bir sinyal
uretmek icin degil, CESITLILIK uretmek icin kullanmak. Farkli kosullar farkli
hatalar yapiyorsa, transkriptleri oylayarak birlestirmek her birinden iyi olabilir.

On kosulu olculdu: dev-5'te en iyi uc hipotezin benzerligi yalnizca %63-86.
Yani neredeyse ayni WER'e sahip transkriptler, jetonlarin ucte birinde FARKLI
hatalar yapiyor — birlestirmenin ise yaramasi icin aranan durum tam budur.

ADALET KURALI — en onemli nokta
-------------------------------
Hipotez secimi ve pivot secimi REFERANSA BAKILMADAN yapilir. Referansa bakarak
"en iyi 3'u sec" demek oracle olur ve sonucu gecersiz kilar. Burada:
  - Kosul kumesi ONCEDEN, sabit bir kuralla belirlenir
  - Pivot, digerlerine ORTALAMA EN BENZER hipotezdir (referanssiz sezgisel)
  - Oylama cogunluk esasina gore yapilir
Karsilastirma her zaman 'none' (ham ses) tabanina karsidir.

Kullanim:
    python paket3_asr/scripts/analyze_fusion.py
    python paket3_asr/scripts/analyze_fusion.py --strategy all
"""

import argparse
import csv
import difflib
import os
import statistics
import sys
from collections import Counter, defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from calculate_metrics import calculate_wer  # noqa: E402
from normalize_text import normalize_turkish_asr  # noqa: E402
from oa_conditions import PKG_ROOT, development_subset, load_experiment_config  # noqa: E402

RESULTS_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "benchmark_results.csv")
BASELINE_CSV = os.path.join(PKG_ROOT, "outputs", "metrics_baseline",
                            "benchmark_results_baseline_refv2.csv")
REF_DIR = os.path.join(PKG_ROOT, "data", "references")
REPORT_CSV = os.path.join(PKG_ROOT, "outputs", "metrics", "fusion_report.csv")

# Onceden belirlenmis kosul kumeleri (referansa bakilmadan secildi).
# Gerekce her kumede aciklanir.
STRATEGIES = {
    "raw_oa_sa": {
        "name": "Ham + sinyal alani + spektral alan (uc farkli isleme ailesi)",
        "conditions": ["none", "oa_aggr_a020", "sa_psm_aggr_a040"],
    },
    "raw_plus_light": {
        "name": "Ham + iki hafif mudahale (literatur optimumu cevresi)",
        "conditions": ["none", "oa_mild_a020", "sa_psm_mild_a025"],
    },
    "five_family": {
        "name": "Bes aile: ham, mild, aggressive, OA, spektral",
        "conditions": ["none", "mild", "aggressive", "oa_aggr_a020", "sa_psm_aggr_a040"],
    },
    "classic3_all10": {
        "name": "Ham + mild + aggressive — TUM 10 kayitta mevcut, ek cozme gerektirmez",
        "conditions": ["none", "mild", "aggressive"],
    },
    "all_light": {
        "name": "Ham + TUM hafif alfa kosullari (a<=0.25)",
        "conditions": ["none", "oa_mild_a020", "oa_aggr_a020",
                       "sa_psm_mild_a020", "sa_psm_mild_a025",
                       "sa_psm_aggr_a020", "sa_psm_aggr_a025"],
    },
}


def load_hypotheses():
    """
    (model_id, basename) -> {condition: (normalize edilmis transkript, WER)}

    WER CSV'de ZATEN kayitli; yeniden hesaplamak gereksizdir ve pahalidir
    (CER hizalamasi 2000 kelimelik kayitlarda saf Python'da cok yavas).
    Yalnizca BIRLESTIRILMIS hipotez icin yeni WER hesabi yapilir.
    """
    out = defaultdict(dict)
    for path in (RESULTS_CSV, BASELINE_CSV):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("run_purpose") != "main_benchmark":
                    continue
                t = r.get("transcription")
                if not t:
                    continue
                try:
                    wer = float(r["WER"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (r["model_id"], r["basename"])
                if r["condition"] not in out[key]:
                    out[key][r["condition"]] = (normalize_turkish_asr(t), wer)
    return out


_SIM_CACHE = {}


def _ratio(a_key, b_key, a_toks, b_toks):
    """Hizalama orani; ayni cift birden cok stratejide tekrar istendigi icin onbellekli."""
    k = (a_key, b_key) if a_key <= b_key else (b_key, a_key)
    if k not in _SIM_CACHE:
        _SIM_CACHE[k] = difflib.SequenceMatcher(None, a_toks, b_toks).ratio()
    return _SIM_CACHE[k]


def pick_pivot(hyps, keys=None):
    """
    Referanssiz pivot secimi: digerlerine ortalama en benzer hipotez.
    Fikir, uc noktalardan uzak 'konsensus merkezi'ni secmektir.
    """
    toks = [h.split() for h in hyps]
    # Anahtar verilmezse onbellek KULLANILMAZ. Aksi halde genel anahtarlar
    # ("0","1","2") farkli kayitlar arasinda carpisir ve bir kaydin hizalama
    # sonucu digerine sizar.
    use_cache = keys is not None
    best_i, best_s = 0, -1.0
    for i in range(len(toks)):
        sims = [(_ratio(keys[i], keys[j], toks[i], toks[j]) if use_cache
                 else difflib.SequenceMatcher(None, toks[i], toks[j]).ratio())
                for j in range(len(toks)) if j != i]
        s = statistics.mean(sims) if sims else 0.0
        if s > best_s:
            best_i, best_s = i, s
    return best_i, best_s


def rover_fuse(hyps, keys=None, pivot_index=None, override_margin=0):
    """
    ROVER benzeri oylama: pivot omurga olarak kullanilir, diger hipotezler
    ona hizalanir, her konumda cogunluk oyu alinir.

    Silme oylari ('' jetonu) da sayilir: hipotezlerin cogu bir jetonu
    icermiyorsa o jeton dusurulur. Bu, dongu tekrarlarini kirpma etkisi yapar.
    Beraberlikte pivot'un jetonu korunur (muhafazakar davranis).
    """
    if not hyps:
        return ""
    if len(hyps) == 1:
        return hyps[0]

    # pivot_index verilirse o hipotez omurga olur (orn. ham ses = 'none').
    # Gerekce: birlestirme bilgi URETMEZ, uzlasi secer. Tum alternatifler ham
    # sesten kotuyse uzlasi da kotudur; ham sese capalamak bu riski sinirlar.
    pivot_i = pivot_index if pivot_index is not None else pick_pivot(hyps, keys)[0]
    pivot = hyps[pivot_i].split()
    if not pivot:
        return hyps[pivot_i]

    votes = [Counter() for _ in pivot]
    for i, h in enumerate(hyps):
        cur = h.split()
        sm = difflib.SequenceMatcher(None, pivot, cur)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    votes[i1 + k][pivot[i1 + k]] += 1
            elif tag == "replace":
                # pivot[i1:i2] yerine cur[j1:j2]; konum konum esle
                for k in range(i2 - i1):
                    rep = cur[j1 + k] if (j1 + k) < j2 else ""
                    votes[i1 + k][rep] += 1
            elif tag == "delete":
                # bu hipotez pivot'un jetonlarini icermiyor -> silme oyu
                for k in range(i1, i2):
                    votes[k][""] += 1
            # insert: pivot omurgasinda karsiligi yok; v1'de yok sayilir

    out = []
    for k, c in enumerate(votes):
        if not c:
            out.append(pivot[k]); continue
        top = c.most_common()
        best_n = top[0][1]
        tied = [w for w, n in top if n == best_n]
        # Pivot'u degistirmek icin override_margin kadar FAZLA oy gerekir.
        # 0 = duz cogunluk; >0 = ham sese daha guclu capa.
        pivot_votes = c.get(pivot[k], 0)
        if pivot[k] in tied or best_n <= pivot_votes + override_margin:
            pick = pivot[k]
        else:
            pick = tied[0]
        if pick:
            out.append(pick)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description="Y4 — hipotez birlestirme (ROVER)")
    ap.add_argument("--strategy", default=None, choices=list(STRATEGIES) + ["all"])
    ap.add_argument("--basenames", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_experiment_config()
    recs = args.basenames or development_subset(cfg)
    strategies = list(STRATEGIES) if args.strategy in (None, "all") else [args.strategy]

    hyp = load_hypotheses()
    # Referanslar bir kez normalize edilir. Hipotezler load_hypotheses icinde
    # zaten normalize edildigi icin karsilastirma ayni uzayda yapilir.
    refs = {}
    for b in recs:
        p = os.path.join(REF_DIR, f"{b}.txt")
        if os.path.exists(p):
            refs[b] = normalize_turkish_asr(open(p, encoding="utf-8").read().strip())

    models = sorted({m for (m, b) in hyp if b in recs})
    rows = []

    print("=" * 108)
    print("HIPOTEZ BIRLESTIRME (ROVER benzeri oylama)")
    print("Kosul kumeleri ONCEDEN sabit; pivot referanssiz secilir. Taban = 'none' (ham ses).")
    print("=" * 108)

    for model_id in models:
        print(f"\n### {model_id}")
        for sname in strategies:
            spec = STRATEGIES[sname]
            conds = spec["conditions"]
            per_rec_base, per_rec_fuse, per_rec_bestsingle = [], [], []
            eksik = False
            for b in recs:
                h = hyp.get((model_id, b), {})
                if "none" not in h or b not in refs:
                    eksik = True; break
                have = [c for c in conds if c in h]
                if len(have) < 2:
                    eksik = True; break
                avail = [h[c][0] for c in have]
                keys = [f"{model_id}|{b}|{c}" for c in have]   # onbellek anahtari
                fused = rover_fuse(avail, keys)
                # Taban ve tek-kosul WER'leri CSV'den okunur (yeniden hesaplanmaz);
                # yalnizca birlestirilmis hipotez icin yeni hesap yapilir.
                per_rec_base.append(h["none"][1])
                # Yalniz WER hesaplanir; CER karakter duzeyi hizalama gerektirir
                # ve uzun kayitlarda (12 bin karakter) saf Python'da cok pahalidir.
                per_rec_fuse.append(calculate_wer(refs[b], fused))
                per_rec_bestsingle.append(min(h[c][1] for c in have))
            if eksik or not per_rec_base:
                print(f"    {sname:<16} (yeterli hipotez yok, atlandi)")
                continue

            base = statistics.mean(per_rec_base) * 100
            fuse = statistics.mean(per_rec_fuse) * 100
            orac = statistics.mean(per_rec_bestsingle) * 100
            verdict = "GECIYOR" if fuse < base else "gecemiyor"
            print(f"    {sname:<16} N={len(per_rec_base)} kosul={len(conds)}  "
                  f"ham=%{base:.2f}  BIRLESIK=%{fuse:.2f} ({fuse-base:+.2f} pp)  "
                  f"[oracle-tek=%{orac:.2f}]  {verdict}")
            rows.append({
                "model_id": model_id, "strategy": sname, "name": spec["name"],
                "n_records": len(per_rec_base), "n_conditions": len(conds),
                "raw_macro_wer": f"{base/100:.6f}", "fused_macro_wer": f"{fuse/100:.6f}",
                "delta_pp": f"{fuse-base:.4f}",
                "oracle_single_macro_wer": f"{orac/100:.6f}",
                "beats_raw": "evet" if fuse < base else "hayir",
            })

    if rows:
        os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
        with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        wins = [r for r in rows if r["beats_raw"] == "evet"]
        print("\n" + "=" * 108)
        print(f"Denenen (model, strateji): {len(rows)}   |   Ham sesi gecen: {len(wins)}")
        for r in sorted(wins, key=lambda x: float(x["delta_pp"])):
            print(f"   {r['model_id']} | {r['strategy']}: "
                  f"%{float(r['raw_macro_wer'])*100:.2f} -> %{float(r['fused_macro_wer'])*100:.2f} "
                  f"({r['delta_pp']} pp)")
        if not wins:
            print("   (Hicbir birlestirme ham sesi gecmedi.)")
        print(f"Rapor: {os.path.relpath(REPORT_CSV, os.path.dirname(PKG_ROOT))}")
        print("=" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
