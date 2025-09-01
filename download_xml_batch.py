"""
This file downloads a specified number of XMLs from PubMed's published database.
Each XML contains content and metadata for a large number of articles.
The XMLs are saved locally in the specified destination folder.

NCBI: The National Center for Biotechnology Information provides access to
biomedical and genomic databases.
PubMed: PubMed is a database of biomedical literature maintained by NCBI,
containing abstracts, citations, and metadata for millions of articles.
"""

import requests
import os
import gzip
import shutil
from typing import List

#Configuratin
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
DESTINATION_FOLDER = r"C:\med\xml_articles"


def generate_pubmed_filenames(start: int, end: int) -> List[str]:
    """Generate a list of PubMed baseline XML file names."""
    return [f"pubmed25n{str(i).zfill(4)}.xml.gz" for i in range(start, end + 1)]


def download_and_decompress(file_name: str, destination_folder: str) -> None:
    """Download a .gz file and decompress it to .xml."""
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    gz_path = os.path.join(destination_folder, file_name)
    xml_path = gz_path[:-3]  # Remove ".gz" to get .xml

    # Download the file
    print(f"Downloading {file_name} ...")
    response = requests.get(BASE_URL + file_name, stream=True)
    if response.status_code == 200:
        with open(gz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to: {gz_path}")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")
        return

    # Decompress the file
    print(f"Decompressing {file_name} ...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(xml_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed to: {xml_path}")


def main():
    file_list = generate_pubmed_filenames(1, 2)  # Example: download first 2 files
    for file_name in file_list:
        download_and_decompress(file_name, DESTINATION_FOLDER)


if __name__ == "__main__":
    main()
