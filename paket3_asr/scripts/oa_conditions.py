"""
Paket 3 — Observation Addition (Y1) kosul genisletici.

Makale protokolu: docs/MAKALE_DENEY_PROTOKOLU_ASR_ON_ISLEME_2026-09-10.md

experiment.yaml -> observation_addition blogunu okur ve somut kosul kayitlarina
genisletir. Hem prepare_audio.py (ses uretimi) hem benchmark.py (cozme) ayni
kaynaktan okur; kosul kimlikleri ve dizinleri koda gomulmez.

OA tanimi:
    y[n] = (1 - alpha) * x_ham[n] + alpha * x_iyilestirilmis[n]

alpha=0.0 ham sese, alpha=1.0 tam denoise'a esittir; bu uc degerler ayri kosul
olarak uretilmez, mevcut none/mild/aggressive kosullarina capa olarak eslenir.
"""

import os

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
EXPERIMENT_YAML = os.path.join(PKG_ROOT, "config", "experiment.yaml")


def load_experiment_config(path: str = EXPERIMENT_YAML) -> dict:
    """experiment.yaml'i okur. YAML yoksa veya okunamazsa bos sozluk doner."""
    if not _YAML_AVAILABLE or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _alpha_pct(alpha: float) -> int:
    """0.2 -> 20, 0.05 -> 5. Kimlik ve dizin adlarinda kullanilir."""
    return int(round(float(alpha) * 100))


def expand_oa_conditions(cfg: dict = None) -> list:
    """
    observation_addition blogunu somut kosul kayitlarina genisletir.

    Doner: [{id, short, enhancer, alpha, alpha_pct, dir, method}, ...]
    Blok yoksa veya enabled=false ise bos liste doner.
    """
    if cfg is None:
        cfg = load_experiment_config()

    oa = cfg.get("observation_addition") or {}
    if not oa.get("enabled", False):
        return []

    id_tpl = oa.get("id_template", "oa_{short}_a{alpha_pct:03d}")
    dir_tpl = oa.get("dir_template", "data/audio_oa_{short}_a{alpha_pct:03d}")
    method_tpl = oa.get("method_template", "oa_{enhancer}_alpha{alpha:.2f}")

    out = []
    for enh in oa.get("base_enhancers", []):
        enh_id = enh["id"] if isinstance(enh, dict) else str(enh)
        short = enh.get("short", enh_id) if isinstance(enh, dict) else enh_id
        for alpha in oa.get("alphas", []):
            alpha = float(alpha)
            pct = _alpha_pct(alpha)
            fields = {
                "short": short,
                "enhancer": enh_id,
                "alpha": alpha,
                "alpha_pct": pct,
            }
            rel_dir = dir_tpl.format(**fields)
            out.append({
                "id": id_tpl.format(**fields),
                "short": short,
                "enhancer": enh_id,
                "alpha": alpha,
                "alpha_pct": pct,
                "dir": os.path.abspath(os.path.join(PKG_ROOT, rel_dir)),
                "method": method_tpl.format(**fields),
            })
    return out


def oa_condition_map(cfg: dict = None) -> dict:
    """
    benchmark.py CONDITION_MAP formatinda sozluk doner:
        {kosul_id: {"dir": <abs>, "denoiser_method": <str>}}
    """
    return {
        c["id"]: {"dir": c["dir"], "denoiser_method": c["method"]}
        for c in expand_oa_conditions(cfg)
    }


def oa_params_by_id(cfg: dict = None) -> dict:
    """Kosul kimliginden uretim parametrelerine (enhancer, alpha) esleme."""
    return {c["id"]: c for c in expand_oa_conditions(cfg)}


def get_mix_settings(cfg: dict = None) -> dict:
    """Karisim davranisi ayarlari (RMS esitleme, tepe koruma, uc dogrulama)."""
    if cfg is None:
        cfg = load_experiment_config()
    oa = cfg.get("observation_addition") or {}
    return {
        "rms_align": bool(oa.get("rms_align", True)),
        "output_rms_match": bool(oa.get("output_rms_match", True)),
        "peak_guard": float(oa.get("peak_guard", 0.999)),
        "verify_endpoints": bool(oa.get("verify_endpoints", True)),
    }


def extra_condition_map(cfg: dict = None) -> dict:
    """
    experiment.yaml -> extra_conditions blogunu CONDITION_MAP bicimine cevirir.
    Sablonla uretilmeyen, tekil kosullar icindir (orn. WPE).
    """
    if cfg is None:
        cfg = load_experiment_config()
    out = {}
    for c in (cfg.get("extra_conditions") or []):
        out[c["id"]] = {
            "dir": os.path.abspath(os.path.join(PKG_ROOT, c["source_dir"])),
            "denoiser_method": c.get("denoiser_method", c["id"]),
        }
    return out


def development_subset(cfg: dict = None) -> list:
    """Dondurulmus gelistirme alt kumesinin kok adlari."""
    if cfg is None:
        cfg = load_experiment_config()
    return list((cfg.get("development_subset") or {}).get("basenames", []))


def reproduction_check_basenames(cfg: dict = None) -> list:
    """Kademe 0 yeniden uretim dogrulamasinda kullanilan kok adlar."""
    if cfg is None:
        cfg = load_experiment_config()
    return list((cfg.get("reproduction_check") or {}).get("basenames", []))


if __name__ == "__main__":
    cfg = load_experiment_config()
    conds = expand_oa_conditions(cfg)
    print(f"Genisletilen OA kosulu sayisi: {len(conds)}")
    for c in conds:
        print(f"  {c['id']:<20} alpha={c['alpha']:.2f}  enhancer={c['enhancer']:<20} -> {os.path.relpath(c['dir'], PKG_ROOT)}")
    print(f"\nKarisim ayarlari : {get_mix_settings(cfg)}")
    print(f"Gelistirme alt kumesi ({len(development_subset(cfg))}): {development_subset(cfg)}")
    print(f"Kademe 0 kayitlari    : {reproduction_check_basenames(cfg)}")
