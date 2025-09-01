"""
Given a PMID, it verifies presence in pmid_index.npy, re-embeds the corresponding 
record, and checks for an identical vector in abstracts.npy, reporting the matching index.

Purpose: Confirms end-to-end integrity by ensuring that database records consistently 
         map to their saved embeddings.

"""

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Load & normalize your arrays
embeddings = np.load('data/vectors/abstracts.npy')
pmid_index = np.load('data/vectors/pmid_index.npy', allow_pickle=True)


# Convert from strings (if needed) to integer dtype
if not np.issubdtype(pmid_index.dtype, np.integer):
    pmid_index = pmid_index.astype(int)
    print("converted into int")

print(pmid_index)

# Ask for the PMID and locate it 
pmid_to_check = int(input("Enter PMID to check: ").strip())
positions = np.where(pmid_index == pmid_to_check)[0]

if positions.size == 0:
    print(f"PMID {pmid_to_check} not found in pmid_index.npy")
    exit()

print(f"Found PMID {pmid_to_check} at positions {positions.tolist()} in pmid_index.npy")

# Fetch the corresponding formatted text from the database
conn = sqlite3.connect('pubmed.db')
cur = conn.cursor()
cur.execute(
    """
    SELECT section_title,
           article_title,
           abstract,
           mesh_terms
      FROM pubmed_articles
     WHERE pmid = ?;
    """,
    (str(pmid_to_check),)
)
row = cur.fetchone()
cur.close()
conn.close()

if not row:
    print(f"No record found in database for PMID {pmid_to_check}")
    exit()

section, title, abstract, mesh = row
formatted = (
    f"Section Title: {section}, "
    f"Article Title: {title}, "
    f"Abstract: {abstract}, "
    f"MeSH Terms: {mesh}"
)

#  Encode the single formatted string
model = SentenceTransformer(
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    device='cpu'
)
vec = model.encode(
    formatted,
    convert_to_numpy=True
)

# Step 4: Check for identical vectors in embeddings.npy  
# Use np.allclose rather than == to tolerate floating-point minutiae
matches = np.where(np.allclose(embeddings, vec, atol=1e-6), True, False)


# Elementwise compare each embedding to vec
elementwise = np.isclose(embeddings, vec, atol=1e-6)

# For each row, check that *all* D positions match
mask = np.all(elementwise, axis=1)

# Extract the indices of True entries
identical_positions = np.nonzero(mask)[0]

if identical_positions.size > 0:
    print(f"Found identical embedding at positions {identical_positions.tolist()}")
else:
    print("No identical embedding found.")