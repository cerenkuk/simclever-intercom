import os
import openpyxl
import re

xlsx_path = 'paket3_asr/outputs/manual_review/paket3_tibbi_terim_insan_kontrol_sonuclari.xlsx'
wb = openpyxl.load_workbook(xlsx_path, data_only=True)
s2 = wb['Eklenen Terimler']
rows2 = list(s2.iter_rows(values_only=True))
h2 = [str(c).strip() if c else '' for c in rows2[0]]

def norm_tr(text):
    if text is None: return ''
    t = str(text).strip().lower().replace('i̇', 'i').replace('I', 'ı').replace('İ', 'i')
    t = re.sub(r'[^\w\s]', '', t)
    return ' '.join(t.split())

seen = {}
duplicates = []

for idx, r in enumerate(rows2[1:], start=2):
    if not any(r): continue
    rd = dict(zip(h2, r))
    bname = rd.get('Basename', '')
    uterm = rd.get('Eklenen Terim', '')
    nterm = norm_tr(uterm)
    key = (bname, nterm)

    if key in seen:
        duplicates.append({
            'first_row': seen[key]['row'],
            'duplicate_row': idx,
            'basename': bname,
            'user_term': uterm,
            'norm_term': nterm
        })
    else:
        seen[key] = {'row': idx, 'user_term': uterm, 'norm_term': nterm}

print(f'Toplam mükerrer sayısı: {len(duplicates)}')
for d in duplicates:
    print(f"İlk Satır: {d['first_row']} | Tekrar Satırı: {d['duplicate_row']} | Basename: {d['basename']} | Terim: {d['user_term']}")
