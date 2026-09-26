"""
AMI transkriptlerini cekme (Ek-2 Rev B, Bolum 6).

NEDEN BU YOL
------------
Parquet dosyalari 13,7 GB — cunku ses baytlarini da tasiyorlar. Bize yalniz
metin ve zaman damgalari lazim. datasets-server /rows ucu sesi URL olarak
dondurdugu icin sayfa basina ~83 KB; 12.643 satir ~10 MB eder.

/filter ucu daha dogrudan olurdu ama sunucu tarafi indeksin hazirlanmasini
bekliyor; sayfalama indekse ihtiyac duymaz.

CIKTI
-----
data/corpora/ami_meta/segments_sdm.csv
    meeting_id, begin_time, end_time, speaker_id, text
"""

import csv
import json
import os
import sys
import time
import urllib.request

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)

from oa_conditions import PKG_ROOT  # noqa: E402

OUT = os.path.join(PKG_ROOT, "data", "corpora", "ami_meta")
URL = ("https://datasets-server.huggingface.co/rows"
       "?dataset=edinburghcstr%2Fami&config={cfg}&split=test&offset={off}&length=100")


PAUSE = 2.0          # sayfalar arasi nezaket beklemesi
BACKOFF = (30, 60, 120, 240)   # 429 sonrasi bekleme merdiveni


def _page(cfg, off):
    """Tek sayfa; 429'da artan bekleme ile yeniden dener."""
    for i, wait in enumerate((0,) + BACKOFF):
        if wait:
            print(f"    [429] {wait} sn bekleniyor...", flush=True)
            time.sleep(wait)
        try:
            req = urllib.request.Request(
                URL.format(cfg=cfg, off=off),
                headers={"User-Agent": "simclever-paket3-research/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == len(BACKOFF):
                raise
        except Exception:
            if i == len(BACKOFF):
                raise
            time.sleep(5)
    raise RuntimeError("sayfa alinamadi")


def fetch(cfg="sdm"):
    """Kesintiden devam eder: her 1000 satirda ara dosya yazilir."""
    os.makedirs(OUT, exist_ok=True)
    part = os.path.join(OUT, f"_partial_{cfg}.json")
    rows, off, total = [], 0, None
    if os.path.exists(part):
        st = json.load(open(part, encoding="utf-8"))
        rows, off = [tuple(x) for x in st["rows"]], st["off"]
        print(f"  [devam] {off}. satirdan", flush=True)

    while total is None or off < total:
        d = _page(cfg, off)
        total = d["num_rows_total"]
        for x in d["rows"]:
            r = x["row"]
            rows.append((r["meeting_id"], float(r["begin_time"]), float(r["end_time"]),
                         r["speaker_id"], r["text"]))
        off += 100
        if off % 1000 == 0:
            json.dump({"off": off, "rows": rows}, open(part, "w", encoding="utf-8"))
            print(f"  {off}/{total}", flush=True)
        time.sleep(PAUSE)
    if os.path.exists(part):
        os.remove(part)
    rows.sort(key=lambda t: (t[0], t[1]))
    p = os.path.join(OUT, f"segments_{cfg}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["meeting_id", "begin_time", "end_time", "speaker_id", "text"])
        w.writerows(rows)
    print(f"{len(rows)} segment -> {os.path.relpath(p, PKG_ROOT)}")
    return rows


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "sdm")
