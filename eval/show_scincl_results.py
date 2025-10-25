"""
Script to show detailed SciNCL retrieval results with document IDs
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import RAG system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_scincl import SciNCLMedicalRAG


def show_detailed_scincl_results():
    """
    Show detailed SciNCL retrieval results with document IDs for each query.
    """
    # Load test queries
    df = pd.read_csv("eval/retrieval_queries.csv")
    
    # Initialize RAG system
    rag = SciNCLMedicalRAG()
    
    print("🔍 SciNCL Detailed Retrieval Results")
    print("=" * 80)
    
    for _, row in df.iterrows():
        query_id = row["query_id"]
        query = row["query_text"]
        gt_doc_ids = set(str(row["target_paper_ids"]).split(","))
        
        print(f"\n📋 Query {query_id}: '{query}'")
        print("-" * 80)
        print(f"🎯 Ground Truth IDs: {list(gt_doc_ids)}")
        
        try:
            # Get results
            result = rag.query(query, limit=10, generate_answer=False)
            
            if result and result['documents']:
                print(f"\n📄 Retrieved {len(result['documents'])} documents:")
                print("-" * 40)
                
                retrieved_ids = []
                for i, doc in enumerate(result['documents'], 1):
                    paper_id = doc.get('paper_id', 'N/A')
                    retrieved_ids.append(paper_id)
                    
                    # Check if this is a relevant document
                    is_relevant = "✅ RELEVANT" if paper_id in gt_doc_ids else "❌ Not relevant"
                    
                    print(f"{i:2d}. ID: {paper_id}")
                    print(f"    Title: {doc['title'][:70]}...")
                    print(f"    Score: {doc['score']:.3f} | {is_relevant}")
                    print(f"    Source: {doc['source']} | Year: {doc.get('year', 'N/A')}")
                    print()
                
                # Show matches
                matches = set(retrieved_ids) & gt_doc_ids
                print(f"🎯 Matches found: {len(matches)}/{len(gt_doc_ids)}")
                if matches:
                    print(f"✅ Relevant IDs found: {list(matches)}")
                else:
                    print("❌ No relevant documents found")
                    
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 80)


if __name__ == "__main__":
    show_detailed_scincl_results()
