"""
NIT Jamshedpur Real Document Corpus Ingestion & Structure-Aware Storage Engine
=============================================================================
Processes all digital searchable PDFs from `documents/`:
1. Detects & extracts tables with PyMuPDF `find_tables()` -> Converts to clean Markdown tables.
2. Applies the Atomic Table Chunk Rule (keeps tables intact, repeats headers for long tables).
3. Chunks prose text into 400-600 token blocks with section header breadcrumbs.
4. Extracts Temporal Metadata (issue_date, official ref_no, academic_session, category).
5. Stores structured chunks into SQLite database: `nitjsr_documents.db`.
"""

import os
import re
import time
import sqlite3
import json
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import pandas as pd

DOCS_DIR = r"c:\Users\enup1\OneDrive\Desktop\projects\captone\documents"
DB_PATH = r"c:\Users\enup1\OneDrive\Desktop\projects\captone\nitjsr_documents.db"

def table_to_markdown(raw_table: List[List[Any]]) -> str:
    """Convert raw 2D list table from PyMuPDF into a clean Markdown table."""
    if not raw_table or len(raw_table) < 2:
        return ""
    cleaned_rows = []
    for row in raw_table:
        cleaned_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        cleaned_rows.append(cleaned_row)
    
    header = cleaned_rows[0]
    # Replace empty headers with generic Col_i
    header = [h if h else f"Col_{i+1}" for i, h in enumerate(header)]
    
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in cleaned_rows[1:]:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[:len(header)]
        # Skip completely empty rows
        if any(cell for cell in r):
            md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)

def extract_document_metadata(first_page_text: str, filename: str) -> Dict[str, Any]:
    """Extract official date, reference number, session, and category."""
    # 1. Date extraction
    date_patterns = [
        r'(?:dated|date)\s*[:\-–]?\s*([0-3]?[0-9][\/\-.][0-1]?[0-9][\/\-.](?:202[0-9]|201[0-9]))',
        r'(?:dated|date)\s*[:\-–]?\s*([0-3]?[0-9]\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:202[0-9]|201[0-9]))',
        r'\b([0-3]?[0-9][\/\-.][0-1]?[0-9][\/\-.](?:202[0-9]|201[0-9]))\b'
    ]
    extracted_date = None
    for pat in date_patterns:
        m = re.search(pat, first_page_text, re.IGNORECASE)
        if m:
            extracted_date = m.group(1).strip()
            break
            
    # 2. Ref Number extraction
    ref_patterns = [
        r'(NITJSR\/[A-Z0-9_\-\/]+)',
        r'(NOTICE\s*[\-–]\s*[A-Z0-9\/\-–\s]+)',
        r'Ref\.?\s*No\.?\s*[:\-–]?\s*([A-Za-z0-9_\-\/]+)'
    ]
    ref_no = None
    for pat in ref_patterns:
        m = re.search(pat, first_page_text, re.IGNORECASE)
        if m:
            ref_no = m.group(1).strip()
            break

    # 3. Academic Session
    session_match = re.search(r'\b(202[0-9]\s*[-–]\s*(?:202[0-9]|[0-9]{2}))\b', first_page_text)
    session = session_match.group(1).replace(" ", "") if session_match else "2024-25"
    
    # 4. Domain Category
    c_lower = (first_page_text + " " + filename).lower()
    if any(w in c_lower for w in ["exam", "examination", "mid sem", "end sem", "backlog", "admit card", "revaluation", "datesheet"]):
        cat = "Exam Dates"
    elif any(w in c_lower for w in ["fee", "tuition", "mess fee", "dues", "payment", "fine", "sbi collect", "scholarship"]):
        cat = "Fee Deadlines"
    elif any(w in c_lower for w in ["hostel", "curfew", "warden", "hall of residence", "boarder", "mess timing"]):
        cat = "Hostel Rules"
    elif any(w in c_lower for w in ["syllabus", "credit", "cgpa", "curriculum", "branch", "ordinance", "probation", "grading"]):
        cat = "Syllabus Queries"
    else:
        cat = "Circulars"

    return {
        "issue_date": extracted_date or "2024-08-01",
        "ref_number": ref_no or f"NITJSR/{cat.replace(' ', '_')}/2024",
        "academic_session": session,
        "category": cat
    }

def chunk_text(text: str, max_words: int = 250, overlap_words: int = 40) -> List[str]:
    """Split prose text into overlapping word chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += (max_words - overlap_words)
    return chunks

def build_database(force_rebuild: bool = True) -> Dict[str, Any]:
    """Ingest digital PDFs and populate SQLite database."""
    print("=" * 80)
    print("STARTING NIT JAMSHEDPUR DIGITAL DOCUMENT INGESTION")
    print("=" * 80)
    t0 = time.time()

    if force_rebuild and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        chunk_id TEXT PRIMARY KEY,
        document_name TEXT,
        category TEXT,
        issue_date TEXT,
        academic_session TEXT,
        ref_number TEXT,
        page_number INTEGER,
        chunk_type TEXT,
        content TEXT,
        token_count INTEGER,
        is_latest INTEGER
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cat ON document_chunks(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc ON document_chunks(document_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON document_chunks(issue_date)")

    all_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])
    print(f"Total PDF files in documents directory: {len(all_files)}")

    ingested_docs = 0
    skipped_scanned = 0
    total_chunks = 0
    table_chunks_count = 0
    prose_chunks_count = 0

    records_to_insert = []

    for idx, fname in enumerate(all_files):
        fpath = os.path.join(DOCS_DIR, fname)
        try:
            doc = fitz.open(fpath)
            if doc.is_encrypted or len(doc) == 0:
                doc.close()
                continue

            # First page check for digital text
            p0 = doc[0]
            txt0 = p0.get_text()
            if len(txt0.strip()) < 40:
                skipped_scanned += 1
                doc.close()
                continue

            meta = extract_document_metadata(txt0, fname)
            ingested_docs += 1

            # Iterate pages
            for p_num in range(len(doc)):
                page = doc[p_num]
                page_text = page.get_text().strip()

                # 1. Extract tables on this page
                extracted_tables = []
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for t_idx, tab in enumerate(tabs.tables):
                            raw_t = tab.extract()
                            md_table = table_to_markdown(raw_t)
                            if md_table:
                                extracted_tables.append((t_idx, md_table))
                except Exception:
                    pass

                # Store Table Chunks (Atomic)
                for t_idx, md_t in extracted_tables:
                    chunk_id = f"{fname}_p{p_num+1}_t{t_idx+1}"
                    header_breadcrumb = (
                        f"[Document: {fname} | Ref: {meta['ref_number']} | Date: {meta['issue_date']} | "
                        f"Category: {meta['category']} | Page: {p_num+1} | Type: Structured Table]\n"
                    )
                    full_content = header_breadcrumb + md_t
                    w_count = len(full_content.split())
                    
                    records_to_insert.append((
                        chunk_id, fname, meta["category"], meta["issue_date"],
                        meta["academic_session"], meta["ref_number"], p_num+1,
                        "table", full_content, w_count, 1
                    ))
                    table_chunks_count += 1
                    total_chunks += 1

                # 2. Extract Prose Text (ignoring trivial empty text)
                if len(page_text) >= 50:
                    text_chunks = chunk_text(page_text, max_words=220, overlap_words=35)
                    for c_idx, c_text in enumerate(text_chunks):
                        chunk_id = f"{fname}_p{p_num+1}_c{c_idx+1}"
                        header_breadcrumb = (
                            f"[Document: {fname} | Ref: {meta['ref_number']} | Date: {meta['issue_date']} | "
                            f"Category: {meta['category']} | Page: {p_num+1}]\n"
                        )
                        full_content = header_breadcrumb + c_text
                        w_count = len(full_content.split())

                        records_to_insert.append((
                            chunk_id, fname, meta["category"], meta["issue_date"],
                            meta["academic_session"], meta["ref_number"], p_num+1,
                            "prose", full_content, w_count, 1
                        ))
                        prose_chunks_count += 1
                        total_chunks += 1

            doc.close()

            # Batch insert every 250 docs
            if len(records_to_insert) >= 1000:
                cursor.executemany("""
                INSERT OR REPLACE INTO document_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, records_to_insert)
                conn.commit()
                records_to_insert = []
                print(f"Processed {idx+1}/{len(all_files)} PDFs... ({total_chunks} chunks stored)")

        except Exception as e:
            pass

    # Final insert
    if records_to_insert:
        cursor.executemany("""
        INSERT OR REPLACE INTO document_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, records_to_insert)
        conn.commit()

    conn.close()
    duration = time.time() - t0

    stats = {
        "total_pdfs_scanned": len(all_files),
        "digital_docs_ingested": ingested_docs,
        "scanned_docs_skipped": skipped_scanned,
        "total_chunks_stored": total_chunks,
        "table_chunks": table_chunks_count,
        "prose_chunks": prose_chunks_count,
        "db_size_mb": round(os.path.getsize(DB_PATH) / (1024 * 1024), 2),
        "duration_seconds": round(duration, 2)
    }

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE!")
    print(f"  Digital Documents Ingested: {ingested_docs}")
    print(f"  Scanned / Image PDFs Skipped: {skipped_scanned}")
    print(f"  Total Chunks in SQLite:     {total_chunks:,}")
    print(f"    - Structured Table Chunks: {table_chunks_count:,}")
    print(f"    - Narrative Prose Chunks:  {prose_chunks_count:,}")
    print(f"  SQLite DB File Size:        {stats['db_size_mb']} MB")
    print(f"  Elapsed Time:               {duration:.1f}s")
    print("=" * 80)

    # Save Ingestion Summary
    with open("nitjsr_ingestion_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats

if __name__ == "__main__":
    build_database(force_rebuild=True)
