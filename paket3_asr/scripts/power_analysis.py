"""
Guc analizi — eslestirilmis t testi, t dagilimiyla (Ek-2 Rev B, D4).

ONCEKI HESABIN HATASI
---------------------
Ilk surum normal yaklasimini kullaniyordu:  n = (z_{1-a/2} + z_{1-b})^2 * s^2 / d^2
Bu, kucuk orneklemde gereken n'i SISTEMATIK OLARAK AZ tahmin eder; cunku
eslestirilmis t testinin kritik degeri z degil, n-1 serbestlik dereceli t'dir ve
t > z'dir. Dogru hesap n'e bagli oldugu icin yinelemeli cozulur.

Ayrica guc, merkezi olmayan (noncentral) t dagilimiyla hesaplanir:
    guc = P( T'(df = n-1, ncp = d/s * sqrt(n)) > t_{1-a/2, n-1} )
tek yonlu kuyruk ihmal edilebilir oldugundan iki yonlu testin ust kuyrugu alinir.

Kullanim:
    python paket3_asr/scripts/power_analysis.py
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from scipy import stats  # noqa: E402


def power_at(n, d, s, alpha=0.05):
    """Eslestirilmis t testinin n orneklemde gucu (merkezi olmayan t)."""
    if n < 2:
        return 0.0
    df = n - 1
    ncp = abs(d) / s * (n ** 0.5)
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def required_n(d, s, alpha=0.05, target=0.80, nmax=100000):
    """Hedef gucu saglayan en kucuk n (yinelemeli; t dagilimi n'e bagli)."""
    for n in range(2, nmax):
        if power_at(n, d, s, alpha) >= target:
            return n
    return None


def main():
    # Olculmus degerler (Large-v3, tam 10 kayit) — docs/MAKALE_SONUCLAR bolum 6h
    rows = [
        ("ham -> 4 uyeli birlestirme", -0.397, 0.254),
        ("ham -> 5 uyeli birlestirme", -0.550, 0.466),
        ("4 uye -> 5 uye (yapilandirma secimi)", -0.153, 0.384),
    ]
    print("=" * 96)
    print("GUC ANALIZI — eslestirilmis t testi, merkezi olmayan t dagilimi (alpha=0.05, hedef guc 0.80)")
    print("=" * 96)
    print(f"{'karsilastirma':<40}{'etki':>8}{'std':>8}{'gereken n':>11}{'n=10 guc':>10}{'n=6 guc':>9}{'(eski z)':>10}")
    print("-" * 96)
    for lbl, d, s in rows:
        n_t = required_n(d, s)
        # Eski (hatali) normal yaklasimi — karsilastirma icin
        z = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) ** 2 * s ** 2 / d ** 2
        import math
        print(f"{lbl:<40}{d:>+8.3f}{s:>8.3f}{n_t:>11}{power_at(10, d, s):>10.3f}"
              f"{power_at(6, d, s):>9.3f}{math.ceil(z):>10}")
    print("-" * 96)
    print("Not: 'eski z' sutunu normal yaklasimidir ve gereken n'i az tahmin eder.")
    print("     Rapor edilecek olan t sutunudur.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
