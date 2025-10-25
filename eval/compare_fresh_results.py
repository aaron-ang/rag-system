"""
Compare Fresh TF-IDF vs Fresh SciNCL Results
Shows detailed ID matching for both systems
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_with_persistence import PersistentMedicalRAG
from rag_scincl import SciNCLMedicalRAG


def get_detailed_results(rag_system, system_name):
    """Get detailed results for a RAG system."""
    df = pd.read_csv("eval/retrieval_queries.csv")
    
    results = []
    
    print(f"\n🔍 {system_name} Detailed Results")
    print("=" * 80)
    
    for _, row in df.iterrows():
        query_id = row["query_id"]
        query = row["query_text"]
        gt_doc_ids = set(str(row["target_paper_ids"]).split(","))
        
        print(f"\n📋 Query {query_id}: '{query}'")
        print("-" * 60)
        print(f"🎯 Ground Truth IDs: {list(gt_doc_ids)}")
        
        try:
            result = rag_system.query(query, limit=10, generate_answer=False)
            
            if result and result['documents']:
                retrieved_paper_ids = []
                retrieved_scores = []
                
                print(f"\n📄 Retrieved {len(result['documents'])} documents:")
                print("-" * 40)
                
                for i, doc in enumerate(result['documents'], 1):
                    paper_id = doc.get('paper_id', 'N/A')
                    score = doc['score']
                    retrieved_paper_ids.append(paper_id)
                    retrieved_sim_scores.append(score)
                    
                    # Check if relevant
                    is_relevant = "✅ RELEVANT" if paper_id in gt_doc_ids else "❌ Not relevant"
                    
                    print(f"{i:2d}. ID: {paper_id}")
                    print(f"    Title: {doc['title'][:70]}...")
                    print(f"    Score: {score:.3f} | {is_relevant}")
                    print(f"    Source: {doc['source']} | Year: {doc.get('year', 'N/A')}")
                    print()
                
                # Calculate matches
                matches = set(retrieved_paper_ids) & gt_doc_ids
                missing = gt_doc_ids - set(retrieved_paper_ids)
                wrong = set(retrieved_paper_ids) - gt_doc_ids
                
                print(f"🎯 Matches found: {len(matches)}/{len(gt_doc_ids)}")
                if matches:
                    print(f"✅ Relevant IDs found: {list(matches)}")
                if missing:
                    print(f"❌ Missing IDs: {list(missing)}")
                if wrong:
                    print(f"⚠️  Wrong IDs (first 3): {list(wrong)[:3]}")
                
                results.append({
                    'query_id': query_id,
                    'query': query,
                    'ground_truth': list(gt_doc_ids),
                    'retrieved': retrieved_paper_ids,
                    'matches': list(matches),
                    'missing': list(missing),
                    'wrong': list(wrong),
                    'num_matches': len(matches),
                    'total_relevant': len(gt_doc_ids)
                })
                
            else:
                print("❌ No results found.")
                results.append({
                    'query_id': query_id,
                    'query': query,
                    'ground_truth': list(gt_doc_ids),
                    'retrieved': [],
                    'matches': [],
                    'missing': list(gt_doc_ids),
                    'wrong': [],
                    'num_matches': 0,
                    'total_relevant': len(gt_doc_ids)
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'query_id': query_id,
                'query': query,
                'ground_truth': list(gt_doc_ids),
                'retrieved': [],
                'matches': [],
                'missing': list(gt_doc_ids),
                'wrong': [],
                'num_matches': 0,
                'total_relevant': len(gt_doc_ids)
            })
    
    return results


def compare_results(tfidf_results, scincl_results):
    """Compare results between TF-IDF and SciNCL."""
    print("\n" + "=" * 100)
    print("📊 DETAILED COMPARISON: Fresh TF-IDF vs Fresh SciNCL")
    print("=" * 100)
    
    for i, (tfidf, scincl) in enumerate(zip(tfidf_results, scincl_results), 1):
        print(f"\n🔍 Query {i}: {tfidf['query'][:60]}...")
        print("-" * 80)
        
        print(f"🎯 Ground Truth: {tfidf['ground_truth']}")
        print()
        
        # TF-IDF Results
        print(f"📊 Fresh TF-IDF Results:")
        print(f"   Matches: {tfidf['num_matches']}/{tfidf['total_relevant']}")
        if tfidf['matches']:
            print(f"   ✅ Found: {tfidf['matches']}")
        if tfidf['missing']:
            print(f"   ❌ Missing: {tfidf['missing']}")
        print()
        
        # SciNCL Results
        print(f"📊 Fresh SciNCL Results:")
        print(f"   Matches: {scincl['num_matches']}/{scincl['total_relevant']}")
        if scincl['matches']:
            print(f"   ✅ Found: {scincl['matches']}")
        if scincl['missing']:
            print(f"   ❌ Missing: {scincl['missing']}")
        print()
        
        # Comparison
        tfidf_matches = set(tfidf['matches'])
        scincl_matches = set(scincl['matches'])
        
        both_found = tfidf_matches & scincl_matches
        only_tfidf = tfidf_matches - scincl_matches
        only_scincl = scincl_matches - tfidf_matches
        
        print(f"🔍 Comparison:")
        if both_found:
            print(f"   🤝 Both found: {list(both_found)}")
        if only_tfidf:
            print(f"   🏆 Only TF-IDF: {list(only_tfidf)}")
        if only_scincl:
            print(f"   🏆 Only SciNCL: {list(only_scincl)}")
        
        # Winner
        if tfidf['num_matches'] > scincl['num_matches']:
            print(f"   🏆 Winner: Fresh TF-IDF ({tfidf['num_matches']} vs {scincl['num_matches']})")
        elif scincl['num_matches'] > tfidf['num_matches']:
            print(f"   🏆 Winner: Fresh SciNCL ({scincl['num_matches']} vs {tfidf['num_matches']})")
        else:
            print(f"   🤝 Tie: Both found {tfidf['num_matches']} matches")
        
        print("=" * 80)


def main():
    """Main comparison function."""
    print("🏥 Fresh Embeddings Detailed Comparison")
    print("=" * 50)
    
    # Initialize systems
    print("🔄 Initializing systems...")
    tfidf_rag = PersistentMedicalRAG(
        collection_name="medical_papers_tfidf_fresh",
        vectorizer_path="vectorizer_fresh.pkl"
    )
    scincl_rag = SciNCLMedicalRAG(collection_name="medical_papers_scincl_fresh")
    
    # Get detailed results
    print("\n📊 Getting TF-IDF results...")
    tfidf_results = get_detailed_results(tfidf_rag, "Fresh TF-IDF")
    
    print("\n📊 Getting SciNCL results...")
    scincl_results = get_detailed_results(scincl_rag, "Fresh SciNCL")
    
    # Compare results
    compare_results(tfidf_results, scincl_results)
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("📈 SUMMARY STATISTICS")
    print("=" * 100)
    
    tfidf_total_matches = sum(r['num_matches'] for r in tfidf_results)
    scincl_total_matches = sum(r['num_matches'] for r in scincl_results)
    total_possible = sum(r['total_relevant'] for r in tfidf_results)
    
    print(f"Fresh TF-IDF Total Matches: {tfidf_total_matches}/{total_possible} ({tfidf_total_matches/total_possible:.1%})")
    print(f"Fresh SciNCL Total Matches: {scincl_total_matches}/{total_possible} ({scincl_total_matches/total_possible:.1%})")
    
    # Per-query wins
    tfidf_wins = sum(1 for tfidf, scincl in zip(tfidf_results, scincl_results) if tfidf['num_matches'] > scincl['num_matches'])
    scincl_wins = sum(1 for tfidf, scincl in zip(tfidf_results, scincl_results) if scincl['num_matches'] > tfidf['num_matches'])
    ties = len(tfidf_results) - tfidf_wins - scincl_wins
    
    print(f"\n🏆 Query-by-Query Wins:")
    print(f"   Fresh TF-IDF: {tfidf_wins} queries")
    print(f"   Fresh SciNCL: {scincl_wins} queries")
    print(f"   Ties: {ties} queries")
    
    if tfidf_wins > scincl_wins:
        print(f"\n🎉 Overall Winner: Fresh TF-IDF ({tfidf_wins} vs {scincl_wins} query wins)")
    elif scincl_wins > tfidf_wins:
        print(f"\n🎉 Overall Winner: Fresh SciNCL ({scincl_wins} vs {tfidf_wins} query wins)")
    else:
        print(f"\n🤝 Overall Result: Tie ({tfidf_wins} vs {scincl_wins} query wins)")


if __name__ == "__main__":
    main()
