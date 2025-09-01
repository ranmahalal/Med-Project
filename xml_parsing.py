"""
Title: PubMed XML Parser and Database Loader

Note for us: currently only parses this file (filepath): C:/med/xml_articles/pubmed25n0001.xml

Description:
    This script parses PubMed XML files, extracts structured metadata for each
    article (PMID, PMCID, section title, article title, abstract, revision and
    completion dates, and MeSH terms), and stores the results in a local SQLite
    database (`pubmed.db`).
Functionality:
    - Parse XML files from PubMed and extract article fields.
    - Optionally limit the number of records inserted (default = 30).
    - Create or update the `pubmed_articles` table in SQLite.
    - Store MeSH terms as JSON strings for structural preservation.
    - Allow clearing the table interactively before insertion.
    - Print all stored records for inspection.
Outputs:
    - SQLite database file: `pubmed.db` with table `pubmed_articles`.
Use Case:
    Provides a persistent, queryable local store of PubMed article metadata for
    later use in text mining, embeddings, or search pipelines.
Dependencies:
    - lxml (for XML parsing)
    - sqlite3 (for database storage)
    - json (for serializing MeSH terms)
"""

import sys
from lxml import etree
import sqlite3
from typing import List, Dict
import json

def parse_pubmed_xml(file_path: str, top_n: int = 30) -> List[Dict]:
    # Load and parse the file into an XML tree.
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Find all PubmedArticle elements in the XML tree.
    articles = root.findall(".//PubmedArticle")

    # Initialize an empty list to store article data.
    data = []

    for article in articles:
        # PMID
        pmid = article.findtext(".//PMID")

        # Article Section Title (Journal Title)
        section_title = article.findtext(".//Journal//Title")

        # Article Title
        article_title = article.findtext(".//ArticleTitle")

        # Abstract Text
        abstract_parts = article.findall(".//AbstractText")
        abstract_text = " ".join([
            etree.tostring(part, method="text", encoding="unicode").strip()
            for part in abstract_parts
        ])

        # Date Revised
        revised_elem = article.find(".//DateRevised")
        date_revised = None
        if revised_elem is not None:
            y = revised_elem.findtext("Year", default="")
            m = revised_elem.findtext("Month", default="")
            d = revised_elem.findtext("Day", default="")
            date_revised = f"{y}-{m}-{d}"

        # Date Completed
        completed_elem = article.find(".//DateCompleted")
        date_completed = None
        if completed_elem is not None:
            y = completed_elem.findtext("Year", default="")
            m = completed_elem.findtext("Month", default="")
            d = completed_elem.findtext("Day", default="")
            date_completed = f"{y}-{m}-{d}"

        # Mesh Headings
        mesh_elems = article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        mesh_terms = [elem.text for elem in mesh_elems if elem.text]

        # PMCID (if present)
        pmcid_elem = article.find(".//ArticleIdList/ArticleId[@IdType='pmc']")
        pmcid = pmcid_elem.text.strip() if pmcid_elem is not None else None

        data.append({
            "pmid": pmid,
            "pmcid": pmcid,
            "section_title": section_title,
            "article_title": article_title,
            "abstract": abstract_text,
            "date_revised": date_revised,
            "date_completed": date_completed,
            "mesh_terms": mesh_terms
        })

    print(len(data))

    return data

def setup_db() -> tuple:
    conn = sqlite3.connect('pubmed.db')
    cur = conn.cursor()
    
    # Create the table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid TEXT PRIMARY KEY,
            pmcid TEXT,
            section_title TEXT,
            article_title TEXT,
            abstract TEXT,
            date_revised TEXT,
            date_completed TEXT,
            mesh_terms TEXT
        );
    """)
    
    return conn, cur

def clear_table(cur) -> None:
    clear = input("Clear pubmed_articles table? (y/n): ").strip().lower()
    if clear == 'y':
        cur.execute("DELETE FROM pubmed_articles;")
        print("Cleared pubmed_articles table.")
    else:
        print("Skipped clearing table.")

def insert_records(cur, data: List[Dict]) -> None:
    insert_sql = """
        INSERT INTO pubmed_articles
        (pmid, pmcid, section_title, article_title, abstract, date_revised, date_completed, mesh_terms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    for record in data:
        # Store mesh_terms as JSON to preserve list structure
        mesh_terms_json = json.dumps(record["mesh_terms"]) if record["mesh_terms"] else "[]"
        
        cur.execute(insert_sql, (
            record["pmid"],
            record["pmcid"],
            record["section_title"],
            record["article_title"],
            record["abstract"],
            record["date_revised"],
            record["date_completed"],
            mesh_terms_json
        ))

def print_records(cur, num_records: int = 10) -> None:
    cur.execute("""
        SELECT
            pmid,
            pmcid,
            section_title,
            article_title,
            abstract,
            date_revised,
            date_completed,
            mesh_terms
        FROM pubmed_articles;
    """)
    all_rows = cur.fetchmany(num_records)
    print(f"\nCurrent top {num_records} contents of pubmed_articles:")
    for row in all_rows:
        pmid, pmcid, section_title, article_title, abstract, date_revised, date_completed, mesh_terms = row
        print(
            f"\nPMID:           {pmid}\n"
            f"PMCID:          {pmcid}\n"
            f"Section Title:  {section_title}\n"
            f"Article Title:  {article_title}\n"
            f"Abstract:       {abstract}\n"
            f"Date Revised:   {date_revised}\n"
            f"Date Completed: {date_completed}\n"
            f"MeSH Terms:     {mesh_terms}\n"
            
        )

def main():
    file_path = r"C:\med\xml_articles/pubmed25n0001.xml"
    data = parse_pubmed_xml(file_path)

    conn, cur = setup_db()

    clear_table(cur)

    insert_records(cur, data)

    conn.commit()
    print(f"Inserted {len(data)} records into pubmed_articles.")

    print_records(cur)

    cur.close()
    conn.close()
    sys.exit(0)

if __name__ == "__main__":
    main()