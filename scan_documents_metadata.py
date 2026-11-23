"""
Scan and Analyze Metadata for 2,270+ NIT Jamshedpur Legacy Documents
====================================================================
Extracts:
1. Filename prefixes & category distribution
2. File extensions and size statistics
3. PDF page counts and digital vs scanned (OCR-needed) classification
4. Internal PDF metadata (CreationDate, Producer, Author)
5. Content keyword clusters from extractable text
"""

import os
import re
import time
from collections import Counter, defaultdict
import fitz  # PyMuPDF
import pandas as pd
import numpy as np

DOCS_DIR = r"c:\Users\enup1\OneDrive\Desktop\projects\captone\documents"

print(f"Scanning directory: {DOCS_DIR}")
t0 = time.time()

files = os.listdir(DOCS_DIR)
print(f"Total entries found: {len(files)}")

# 1. Basic File Stats
ext_counts = Counter()
sizes = []
file_records = []

# Filename prefix extractor
def extract_prefix(fname):
    name, ext = os.path.splitext(fname)
    if ext.lower() != ".pdf":
        return "non_pdf"
    # Match patterns like dean_office_notices, department_notices_chem, etc.
    # Strip trailing numbers
    prefix = re.sub(r"_\d+$", "", name)
    return prefix

for fname in files:
    fpath = os.path.join(DOCS_DIR, fname)
    if os.path.isdir(fpath):
        continue
    sz = os.path.getsize(fpath)
    _, ext = os.path.splitext(fname)
    ext_lower = ext.lower()
    ext_counts[ext_lower] += 1
    sizes.append(sz)
    prefix = extract_prefix(fname)
    file_records.append({
        "filename": fname,
        "extension": ext_lower,
        "size_bytes": sz,
        "size_kb": round(sz / 1024, 1),
        "size_mb": round(sz / (1024 * 1024), 2),
        "prefix": prefix
    })

df_files = pd.DataFrame(file_records)

# 2. PDF Deep Scan (Page counts, text presence, scanned vs digital)
pdf_records = []
total_pdfs = len(df_files[df_files["extension"] == ".pdf"])
print(f"\nPerforming deep PDF inspection on {total_pdfs} PDF files using PyMuPDF...")

scanned_count = 0
digital_count = 0
encrypted_count = 0
corrupted_count = 0
total_pages = 0

sample_step = 1  # Check ALL PDFs! PyMuPDF is extremely fast

for idx, row in df_files[df_files["extension"] == ".pdf"].iterrows():
    fpath = os.path.join(DOCS_DIR, row["filename"])
    try:
        doc = fitz.open(fpath)
        if doc.is_encrypted:
            encrypted_count += 1
            is_scanned = None
            char_count = 0
            n_pages = len(doc)
        else:
            n_pages = len(doc)
            total_pages += n_pages
            
            # Sample text from up to 3 pages
            char_count = 0
            for p_num in range(min(3, n_pages)):
                page = doc[p_num]
                txt = page.get_text()
                char_count += len(txt.strip())
            
            # If average chars per page < 40, likely a scanned image PDF
            avg_chars = char_count / max(1, min(3, n_pages))
            if avg_chars < 40:
                scanned_count += 1
                is_scanned = True
            else:
                digital_count += 1
                is_scanned = False
                
        meta = doc.metadata or {}
        doc.close()
        
        pdf_records.append({
            "filename": row["filename"],
            "prefix": row["prefix"],
            "size_kb": row["size_kb"],
            "size_mb": row["size_mb"],
            "pages": n_pages,
            "char_count_sample": char_count,
            "is_scanned": is_scanned,
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "creation_date": meta.get("creationDate", "")
        })
    except Exception as e:
        corrupted_count += 1
        pdf_records.append({
            "filename": row["filename"],
            "prefix": row["prefix"],
            "size_kb": row["size_kb"],
            "size_mb": row["size_mb"],
            "pages": 0,
            "char_count_sample": 0,
            "is_scanned": None,
            "creator": "ERROR",
            "producer": "ERROR",
            "creation_date": ""
        })

df_pdfs = pd.DataFrame(pdf_records)
scan_duration = time.time() - t0
print(f"Deep scan completed in {scan_duration:.1f} seconds!")

# 3. Aggregate Reports
print("\n" + "=" * 70)
print("NIT JAMSHEDPUR DOCUMENT CORPUS METADATA REPORT")
print("=" * 70)

print(f"\n1. Overall Corpus Size & Storage:")
print(f"   - Total files: {len(df_files)}")
print(f"   - Total disk footprint: {df_files['size_mb'].sum():.2f} MB ({df_files['size_mb'].sum()/1024:.2f} GB)")
print(f"   - Average file size: {df_files['size_kb'].mean():.1f} KB (Median: {df_files['size_kb'].median():.1f} KB)")
print(f"   - Min size: {df_files['size_kb'].min():.1f} KB | Max size: {df_files['size_mb'].max():.2f} MB")

print(f"\n2. File Formats:")
for ext, count in ext_counts.items():
    print(f"   - {ext or '[No Extension]'}: {count} files")

print(f"\n3. PDF Page & Text Extraction Audit ({total_pdfs} PDFs):")
print(f"   - Total Pages: {total_pages:,} pages")
print(f"   - Average Pages/Doc: {df_pdfs['pages'].mean():.1f} (Median: {df_pdfs['pages'].median():.0f})")
print(f"   - Digital Searchable PDFs (Direct Text): {digital_count} ({digital_count/total_pdfs*100:.1f}%)")
print(f"   - Scanned Image PDFs (Requires OCR):     {scanned_count} ({scanned_count/total_pdfs*100:.1f}%)")
if encrypted_count:
    print(f"   - Encrypted / Password-Protected:        {encrypted_count}")
if corrupted_count:
    print(f"   - Corrupted / Unreadable:                {corrupted_count}")

# Page count distribution
p_1 = (df_pdfs["pages"] == 1).sum()
p_2_3 = ((df_pdfs["pages"] >= 2) & (df_pdfs["pages"] <= 3)).sum()
p_4_10 = ((df_pdfs["pages"] >= 4) & (df_pdfs["pages"] <= 10)).sum()
p_11_50 = ((df_pdfs["pages"] >= 11) & (df_pdfs["pages"] <= 50)).sum()
p_50_plus = (df_pdfs["pages"] > 50).sum()

print(f"\n4. Document Length Distribution:")
print(f"   - 1 Page Notices:         {p_1} ({p_1/total_pdfs*100:.1f}%) -> Fast single notices/circulars")
print(f"   - 2-3 Pages:              {p_2_3} ({p_2_3/total_pdfs*100:.1f}%)")
print(f"   - 4-10 Pages:             {p_4_10} ({p_4_10/total_pdfs*100:.1f}%)")
print(f"   - 11-50 Pages:            {p_11_50} ({p_11_50/total_pdfs*100:.1f}%) -> Detailed reports/ordinances")
print(f"   - 50+ Pages (Long Books): {p_50_plus} ({p_50_plus/total_pdfs*100:.1f}%) -> Handbooks/curricula")

print(f"\n5. Top 20 Filename Category Prefixes:")
prefix_summary = df_pdfs.groupby("prefix").agg(
    count=("filename", "count"),
    digital=("is_scanned", lambda s: (s == False).sum()),
    scanned=("is_scanned", lambda s: (s == True).sum()),
    total_mb=("size_mb", "sum"),
    avg_pages=("pages", "mean")
).sort_values(by="count", ascending=False).reset_index()

print(prefix_summary.head(25).to_string(index=False))

# Save summary to CSV for deeper inspection
prefix_summary.to_csv("documents_prefix_summary.csv", index=False)
df_pdfs.to_csv("documents_metadata_detailed.csv", index=False)
print("\nSaved documents_prefix_summary.csv and documents_metadata_detailed.csv")
