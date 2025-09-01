"""
Semantic search engine for medical literature using neural embeddings.
Loads pre-computed sentence embeddings from PubMed abstracts stored in .npy files.
Uses FAISS for efficient similarity search with cosine distance.
Returns ranked PMIDs based on query relevance scores.
Fetches full-text articles from PMC when PMCIDs are available in the database.
Interactive command-line interface for real-time literature discovery.

"""


import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import requests
from lxml import etree
import sqlite3
from dotenv import load_dotenv
from typing import List
load_dotenv()

def search_pmids_from_npy(
    top_k: int = 10
) -> List[int]:
    """
    Given a text query entered by the user, load embeddings and PMIDs from .npy files,
    build a cosine-similarity FAISS index, and return the top_k PMIDs.
    """
    query = input("Enter your search query: ").strip()

    # Load and normalize embeddings (N×D → unit‐length rows)
    embeddings = np.load('data/vectors/abstracts.npy')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    # Build FAISS inner‐product index (cosine on normalized vectors)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Load and cast the PMID array
    pmids = np.load('data/vectors/pmid_index.npy', allow_pickle=True)
    if not np.issubdtype(pmids.dtype, np.integer):
        pmids = pmids.astype(int)

    # Encode and normalize the query
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    q_vec = model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)
    q_vec = q_vec.reshape(1, -1).astype('float32')

    # Search FAISS for the top_k most‐similar embeddings + Map back to PMIDs and return

    scores, indices = index.search(q_vec, top_k)
    top_pmids = pmids[indices[0]].tolist()
    top_scores = scores[0].tolist()  # similarity scores for each result
    for pmid, score in zip(top_pmids, top_scores):
        print(f"PMID: {pmid}, similarity: {score:.4f}")
    return top_pmids


# Full-text fetcher
def fetch_and_print_full_texts(pmids: List[int]) -> None:
    """
    For each PMID in pmids, checks the DB for a PMCID.
    If found, fetches full-text JATS XML from PMC and prints all <p> text under <body>.
    Otherwise, notes that no full text is available.
    """
    # Connect to your database
    conn = sqlite3.connect('pubmed.db')
    cur = conn.cursor()
    for pmid in pmids:
        # Lookup PMCID in DB
        cur.execute("SELECT pmcid FROM pubmed_articles WHERE pmid = ?;", (str(pmid),))
        row = cur.fetchone()
        pmcid = row[0] if row and row[0] else None
        if not pmcid:
            print(f"No PMCID for PMID {pmid}; skipping full text.")
            continue

        # Strip "PMC" prefix for EFetch ID, if present
        fetch_id = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid

        print(f"\nFetching full text for PMID {pmid} (PMCID {pmcid}, fetch ID {fetch_id})...\n")
        # Fetch full-text JATS XML from PMC
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db":      "pmc",
            "id":      fetch_id,
            "rettype": "docsum",
            "retmode": "xml",
            "api_key": os.getenv("NCBI_API_KEY"),
            "tool":    os.getenv("NCBI_TOOL", "MedFlow")
        }
        email = os.getenv("NCBI_EMAIL")
        if email:
            params["email"] = email

        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        root = etree.fromstring(resp.content)

        # Print the full PMC XML response
        print("\n--- Full PMC XML ---\n")
        print(resp.text)
    cur.close()
    conn.close()


# Example
if __name__ == '__main__':
    top_pmids = search_pmids_from_npy(top_k=10)
    print("Top 10 matching PMIDs (10 is the default):", top_pmids)

    # Fetch and print full text for any results with PMCID in DB
    fetch_and_print_full_texts(top_pmids)