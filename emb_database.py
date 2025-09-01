"""
PubMed Abstract Embeddings Generator
Description:
    This script connects to a local SQLite database of PubMed articles, fetches
    abstracts and associated metadata (section title, article title, MeSH terms),
    formats them into strings, and generates embeddings using a
    SentenceTransformer model. The embeddings are normalized and saved to disk,
    along with a corresponding array of PMIDs.
Purpose:
    Provides precomputed vector representations of PubMed abstracts for
    downstream tasks such as semantic search, similarity comparison, or
    machine learning applications.
Outputs:
    - data/vectors/abstracts.npy : N×D NumPy array of normalized embeddings
    - data/vectors/pmid_index.npy : NumPy array of corresponding PMIDs
Notes:
    - Fetches database rows in batches to handle large datasets efficiently.
    - Embeddings are generated in batches to reduce memory usage.
    - Normalization ensures cosine similarity can be computed directly.
"""

import sqlite3
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from pathlib import Path

# Connect to database
conn = sqlite3.connect('pubmed.db')

# Create a server‐side cursor so we can fetch in chunks
cur = conn.cursor()
#returns all the rows where the abstract is not null
cur.execute("""
    SELECT pmid,
           section_title,
           article_title,
           abstract,
           mesh_terms
      FROM pubmed_articles
     WHERE abstract IS NOT NULL;
""")

all_texts = []  #empty list to hold formatted strings
all_pmids  = [] #empty list to hold pmids
# Fetch 100 rows at a time, until none remain
while True:
    rows = cur.fetchmany(100)
    if not rows:
        break
    for pmid, section, title, abstract, mesh in rows:
        formatted = (
            f"Section Title: {section}, "
            f"Article Title: {title}, "
            f"Abstract: {abstract}, "
            f"MeSH Terms: {mesh}"
        )
        all_texts.append(formatted)
        all_pmids.append(pmid)

#clean up
cur.close()
conn.close()

print("length of texts: " + str(len(all_texts)))
print("example: "+ all_texts[0])
print("length of pmids: " + str(len(all_pmids)))
print("example of pmid: " + all_pmids[0])

# all_texts now contains one formatted string per record


# embedding part
#load the model
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device='cpu')
user_input = input("Enter the number of texts to embed (default 100): ")
to_embed = int(user_input) if user_input.strip() else 100

# Use a smaller batch size that won't exceed the requested amount
batch_size = min(10, to_embed)
embeddings_batches = []

for start in range(0, to_embed, batch_size):
    end = min(start + batch_size, to_embed)  # Ensure we don't exceed to_embed
    batch_texts = all_texts[start : end]
    batch_vecs = model.encode(
        batch_texts,
        batch_size=len(batch_texts),  # Use actual batch size
        convert_to_numpy=True,
        show_progress_bar=False
    )
    embeddings_batches.append(batch_vecs)

# Combine all batches into a single NumPy array
all_embeddings = np.vstack(embeddings_batches)
print(f"Processed {to_embed} texts into embeddings of shape {all_embeddings.shape}")

#combine all batches into a single numpy array
print("first 3 embeddings:")
print(all_embeddings[:3])  # Print first 3 articles to see the embeddings

#convert the pmids to a numpy array - only include the ones that were actually embedded
pmid_array = np.array(all_pmids[:to_embed])

# Normalize embeddings to unit length for cosine similarity
norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
all_embeddings = all_embeddings / np.where(norms == 0, 1, norms)

# Prepare the output directory
project_root = Path(__file__).resolve().parent
file_dir     = project_root / 'data' / 'vectors'
file_dir.mkdir(parents=True, exist_ok=True)

#Save abstracts.npy
emb_file = file_dir / 'abstracts.npy'
# Delete old embeddings file if it exists
if os.path.exists(emb_file):
    os.remove(emb_file)
# Write fresh embeddings file
np.save(emb_file, all_embeddings)
print(f"Saved embeddings matrix (shape={all_embeddings.shape}) to {emb_file}")

# Save pmid_index.npy
pmid_file = file_dir / 'pmid_index.npy'
if pmid_file.exists():
    pmid_file.unlink()
np.save(pmid_file, pmid_array)
print(f"Saved PMID index (length={len(pmid_array)}) to {pmid_file}")