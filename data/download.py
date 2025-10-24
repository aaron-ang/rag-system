"""
Data download script for RAG system.
Downloads PubMed abstracts and Semantic Scholar papers.
"""

import json
import os
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_pubmed_abstracts():
    try:
        print("Downloading PubMed abstracts...")
        df = pd.read_json(
            "hf://datasets/datajuicer/the-pile-pubmed-abstracts-refined-by-data-juicer/the-pile-pubmed-abstract-refine-result-preview.jsonl",
            lines=True,
        )
        df.to_csv(os.path.join(SCRIPT_DIR, "pubmed_abstracts.csv"), index=False)
        print(f"✓ Downloaded {len(df)} PubMed abstracts")
    except Exception as e:
        raise Exception(f"Error downloading PubMed abstracts: {e}")


def download_semantic_scholar_papers():
    query = "medicine"
    fields = [
        "url",
        "title",
        "abstract",
        "year",
        "referenceCount",
        "citationCount",
        "influentialCitationCount",
        "authors",
        "citations",
        "references",
        # "embedding.specter_v2",
        "tldr",
    ]
    fields_of_study = ["Medicine", "Biology"]
    year_range = "2020-2025"
    limit = 100

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": ",".join(fields),
        "fieldsOfStudy": ",".join(fields_of_study),
        "year": year_range,
        "limit": limit,
        "openAccessPdf": "",
    }

    try:
        print("Downloading Semantic Scholar papers...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        with open(os.path.join(SCRIPT_DIR, "semanticscholar.json"), "w") as f:
            json.dump(data, f, indent=2)

        if "data" in data:
            print(f"✓ Downloaded {len(data['data'])} Semantic Scholar papers")
        else:
            print("✓ Semantic Scholar API response saved")

    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading from Semantic Scholar: {e}")
        raise Exception(f"Error downloading from Semantic Scholar: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")


def main():
    print("Starting data download process...")
    download_pubmed_abstracts()
    download_semantic_scholar_papers()
    print("✓ Data download completed successfully!")


if __name__ == "__main__":
    main()
