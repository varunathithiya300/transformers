import pandas as pd
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import os

def read_and_clean_data(input_path):
    """
    Reads a CSV file, removes unused fields, and returns a list of business names.
    """
    data = pd.read_csv(input_path, encoding="ISO-8859-1")
    return data['Business Name'].dropna().tolist()  # Drop NaN values

def compute_cosine_similarity(vendor_list, customer_list, model):
    """
    Computes cosine similarity between customer and vendor field names using the specified model.
    """
    vendor_embeddings = model.encode(vendor_list, convert_to_tensor=True)
    customer_embeddings = model.encode(customer_list, convert_to_tensor=True)
    return util.pytorch_cos_sim(customer_embeddings, vendor_embeddings)  # Similarity matrix

def find_best_matches(customer_list, vendor_list, sim_matrix_1, sim_matrix_2):
    """
    Finds the best match for each customer field from the vendor list using two similarity matrices.
    """
    best_matches = []
    
    for i, customer_field in enumerate(customer_list):
        best_match_idx_1 = sim_matrix_1[i].argmax().item()
        best_match_target_1 = vendor_list[best_match_idx_1]
        best_match_score_1 = round(sim_matrix_1[i][best_match_idx_1].item(), 4)
        
        best_match_idx_2 = sim_matrix_2[i].argmax().item()
        best_match_target_2 = vendor_list[best_match_idx_2]
        best_match_score_2 = round(sim_matrix_2[i][best_match_idx_2].item(), 4)
        
        best_matches.append((customer_field, 
                             best_match_target_1, best_match_score_1, 
                             best_match_target_2, best_match_score_2))
    
    return best_matches

def save_results_to_csv(results, output_filename):
    """
    Writes the best match results to a CSV file with Cosine Similarity scores for both models.
    """
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Customer Field", "Vendor Field (MiniLM)", "Cosine Similarity (MiniLM)", 
                         "Vendor Field (MPNet)", "Cosine Similarity (MPNet)"])
        writer.writerows(results)
    print(f"Results saved to {output_filename}")

def run_similarity_analysis(source_path, target_path, output_filename):
    """
    Runs similarity analysis using both models and compares results.
    """
    models = {
        "MiniLM": SentenceTransformer("all-MiniLM-L6-v2"),
        "MPNet": SentenceTransformer("all-mpnet-base-v2")
    }
    
    vendor_fields = read_and_clean_data(source_path)
    customer_fields = read_and_clean_data(target_path)
    
    print("Computing similarity with MiniLM...")
    sim_matrix_1 = compute_cosine_similarity(vendor_fields, customer_fields, models["MiniLM"])
    
    print("Computing similarity with MPNet...")
    sim_matrix_2 = compute_cosine_similarity(vendor_fields, customer_fields, models["MPNet"])
    
    best_matches = find_best_matches(customer_fields, vendor_fields, sim_matrix_1, sim_matrix_2)
    save_results_to_csv(best_matches, output_filename)
    
    return best_matches

if __name__ == "__main__":
    source_path = Path(r"D:\knowledge_sharing\demo\sentence_transformer\vendor_input_format.csv")
    target_path = Path(r"D:\knowledge_sharing\demo\sentence_transformer\customer_standard_format.csv")
    output_filename = os.path.join(r"D:\knowledge_sharing\demo\sentence_transformer", "comparison_results.csv")
    
    run_similarity_analysis(source_path, target_path, output_filename)
