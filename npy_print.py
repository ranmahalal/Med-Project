"""
Loads the .npy files and reports their dimensions and a sample PMID.

Purpose: Provides quick sanity checks to ensure embeddings and PMIDs align 
         and have expected shapes.

Status: Complete - accurately reports shapes and sample entries without errors.
"""

import numpy as np

# Load the files
embeddings = np.load('data/vectors/abstracts.npy')
pmids      = np.load('data/vectors/pmid_index.npy')

# Print only what you asked for
print(f"abstracts.npy shape: {embeddings.shape}")   # (N, D)
print(f"pmid_index.npy length: {pmids.shape[0]}")   # N
print(f"First PMID (index 0): {pmids[0]}")
print(f"First 5 PMIDs: {pmids[:5]}")