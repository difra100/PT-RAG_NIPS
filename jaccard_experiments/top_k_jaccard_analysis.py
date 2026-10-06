#!/usr/bin/env python3
"""
Top K Perturbations Jaccard Analysis

This script computes Jaccard similarity between the top K most retrieved perturbations
for each gene across different cell types.

Usage:
    python top_k_jaccard_analysis.py dir1 dir2 dir3 dir4 --top-k 10

Output:
- JSON file with top K perturbations per gene per cell type
- Jaccard similarity matrices for each gene
- Summary statistics
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')


def extract_cell_type_from_path(path):
    """Extract cell type name from directory path."""
    path_str = str(path).lower()
    
    if 'jurkat' in path_str:
        return 'Jurkat'
    elif 'hepg2' in path_str:
        return 'HepG2'
    elif 'k562' in path_str:
        return 'K562'
    elif 'rpe1' in path_str:
        return 'RPE1'
    else:
        # Fallback to directory name
        return Path(path).name.replace('_rag_diff_sparsity0.1', '').replace('eval_last.ckpt', '').strip('_.')


def parse_selected_perturbations(selected_perts_str):
    """Parse pipe-separated perturbations string into a list."""
    if pd.isna(selected_perts_str) or selected_perts_str == "none" or selected_perts_str == "":
        return []
    return selected_perts_str.split('|')


def load_gene_perturbation_data(directory, filename="retrieval_info.csv"):
    """
    Load gene -> retrieved perturbations mapping from directory.
    
    Returns:
        dict: {gene_name: Counter_of_retrieved_perturbations}
    """
    gene_perturbations = defaultdict(Counter)
    
    # Find CSV file
    dir_path = Path(directory)
    csv_file = dir_path / filename
    
    if not csv_file.exists():
        csv_files = list(dir_path.glob("**/*.csv"))
        if csv_files:
            csv_file = csv_files[0]
        else:
            print(f"Warning: No CSV files found in {directory}")
            return {}
    
    print(f"Loading {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['cell_type', 'pert_name', 'selected_perturbations']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns {required_cols}")
            return {}
        
        # Process each row
        processed_genes = set()
        for _, row in df.iterrows():
            pert_name = str(row['pert_name'])
            
            # Skip non-targeting perturbations
            if pert_name.lower() == 'non-targeting':
                continue
                
            selected_perts = parse_selected_perturbations(row['selected_perturbations'])
            
            # Count each retrieved perturbation for this gene
            gene_perturbations[pert_name].update(selected_perts)
            processed_genes.add(pert_name)
        
        print(f"  Processed {len(processed_genes)} unique genes")
        
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return {}
    
    return dict(gene_perturbations)


def get_top_k_perturbations(gene_perturbations, k=10):
    """
    Get top K most retrieved perturbations for each gene.
    
    Args:
        gene_perturbations: {gene: Counter_of_perturbations}
        k: number of top perturbations to return
    
    Returns:
        dict: {gene: [top_k_perturbations_ordered]}
    """
    top_k_data = {}
    
    for gene, pert_counter in gene_perturbations.items():
        # Get top K most common perturbations
        top_k = pert_counter.most_common(k)
        top_k_list = [pert for pert, count in top_k]
        top_k_data[gene] = top_k_list
    
    return top_k_data


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    set1, set2 = set(set1), set(set2)
    
    if not set1 and not set2:
        return 1.0  # Both empty
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def compute_jaccard_matrices(all_top_k_data, cell_types):
    """
    Compute Jaccard similarity matrices for each gene across cell types.
    
    Args:
        all_top_k_data: {cell_type: {gene: [top_k_perturbations]}}
        cell_types: list of cell type names
    
    Returns:
        dict: {gene: jaccard_similarity_matrix}
    """
    # Find genes common to all cell types
    gene_sets = [set(data.keys()) for data in all_top_k_data.values()]
    common_genes = set.intersection(*gene_sets) if gene_sets else set()
    
    print(f"Found {len(common_genes)} genes common to all cell types")
    
    jaccard_matrices = {}
    n_types = len(cell_types)
    
    for gene in common_genes:
        # Create Jaccard matrix for this gene
        jaccard_matrix = np.zeros((n_types, n_types))
        
        for i, cell_type_1 in enumerate(cell_types):
            for j, cell_type_2 in enumerate(cell_types):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                else:
                    top_k_1 = all_top_k_data[cell_type_1][gene]
                    top_k_2 = all_top_k_data[cell_type_2][gene]
                    
                    jaccard_score = jaccard_similarity(top_k_1, top_k_2)
                    jaccard_matrix[i, j] = jaccard_score
        
        jaccard_matrices[gene] = jaccard_matrix
    
    return jaccard_matrices, common_genes


def compute_average_jaccard_matrix(jaccard_matrices, cell_types):
    """Compute average Jaccard similarity matrix across all genes."""
    n_types = len(cell_types)
    avg_matrix = np.zeros((n_types, n_types))
    n_genes = len(jaccard_matrices)
    
    # Average across all gene matrices
    for gene, matrix in jaccard_matrices.items():
        avg_matrix += matrix
    
    avg_matrix /= n_genes
    
    return avg_matrix


def plot_jaccard_heatmap(similarity_matrix, cell_types, output_file, top_k, n_genes):
    """Create and save Jaccard similarity heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set publication-ready style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.4,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create figure
    figsize = max(6, len(cell_types) * 1.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='plasma',
        square=True,
        xticklabels=cell_types,
        yticklabels=cell_types,
        cbar_kws={
            'label': 'Average Jaccard Similarity',
            'shrink': 0.8,
            'aspect': 20
        },
        linewidths=0.5,
        linecolor='white',
        annot_kws={
            'size': 14,
            'weight': 'bold',
            'color': 'white'
        },
        ax=ax
    )
    
    # Enhance title and labels
    # ax.set_title(f'Average Jaccard Similarity (Top {top_k} perturbations, {n_genes} genes)', 
    #             fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cell Type', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cell Type', fontsize=18, fontweight='bold')
    
    # Improve tick formatting
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', rotation=0, labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight', 
               pad_inches=0.1, facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Jaccard similarity heatmap saved to: {output_file}")
    plt.rcdefaults()


def save_results(all_top_k_data, jaccard_matrices, common_genes, cell_types, top_k, output_dir):
    """Save only JSON data and PDF heatmap."""
    
    # 1. Save top K perturbations data as JSON
    json_file = os.path.join(output_dir, f"top_{top_k}_perturbations.json")
    
    # Convert to desired structure: {gene: {cell_type: [perturbations]}}
    json_data = {}
    for gene in common_genes:
        json_data[gene] = {}
        for cell_type in cell_types:
            if gene in all_top_k_data[cell_type]:
                json_data[gene][cell_type] = all_top_k_data[cell_type][gene]
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Top K perturbations data saved to: {json_file}")
    
    # 2. Compute average Jaccard matrix and save as PDF
    avg_jaccard_matrix = compute_average_jaccard_matrix(jaccard_matrices, cell_types)
    pdf_file = os.path.join(output_dir, f"jaccard_similarity_matrix.pdf")
    
    plot_jaccard_heatmap(avg_jaccard_matrix, cell_types, pdf_file, top_k, len(common_genes))
    
    # Print summary statistics
    off_diagonal = avg_jaccard_matrix[np.triu_indices_from(avg_jaccard_matrix, k=1)]
    print(f"\nAverage Jaccard Similarity Statistics:")
    print(f"  Mean: {np.mean(off_diagonal):.3f}")
    print(f"  Std:  {np.std(off_diagonal):.3f}")
    print(f"  Min:  {np.min(off_diagonal):.3f}")
    print(f"  Max:  {np.max(off_diagonal):.3f}")
    
    return json_file, pdf_file


def main():
    parser = argparse.ArgumentParser(
        description="Compute Jaccard similarity on top K retrieved perturbations per gene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python top_k_jaccard_analysis.py jurkat/ hepg2/ k562/ rpe1/ --top-k 10
    python top_k_jaccard_analysis.py experiments/*/eval_* --top-k 5 --output-dir results/
        """
    )
    
    parser.add_argument(
        'directories',
        nargs='+',
        help='Input directories containing CSV files with predictions'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top retrieved perturbations to consider for each gene (default: 10)'
    )
    
    parser.add_argument(
        '--csv-filename',
        default='retrieval_info.csv',
        help='Name of CSV file to look for in each directory (default: retrieval_info.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for results (default: auto-generated based on cell types)'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    valid_directories = []
    for directory in args.directories:
        if os.path.isdir(directory):
            valid_directories.append(directory)
        else:
            print(f"Warning: {directory} is not a valid directory")
    
    if len(valid_directories) < 2:
        print("Error: At least 2 valid directories are required")
        return
    
    print(f"Processing {len(valid_directories)} directories:")
    
    # Load data from all directories
    all_top_k_data = {}
    cell_types = []
    
    for directory in valid_directories:
        cell_type = extract_cell_type_from_path(directory)
        cell_types.append(cell_type)
        
        print(f"\nProcessing {cell_type}: {directory}")
        
        # Load gene -> perturbations data
        gene_perturbations = load_gene_perturbation_data(directory, args.csv_filename)
        
        if not gene_perturbations:
            print(f"  Skipping {directory} - no valid data found")
            continue
        
        # Get top K perturbations for each gene
        top_k_data = get_top_k_perturbations(gene_perturbations, args.top_k)
        all_top_k_data[cell_type] = top_k_data
        
        print(f"  Extracted top {args.top_k} perturbations for {len(top_k_data)} genes")
    
    if len(all_top_k_data) < 2:
        print("Error: Need at least 2 directories with valid data")
        return
    
    # Compute Jaccard matrices
    print(f"\nComputing Jaccard similarity matrices (top {args.top_k})...")
    jaccard_matrices, common_genes = compute_jaccard_matrices(all_top_k_data, cell_types)
    
    if not jaccard_matrices:
        print("Error: No common genes found across all cell types")
        return
    
    # Create output directory based on cell types
    if args.output_dir is None:
        cell_types_clean = [ct.lower() for ct in cell_types]
        dir_name = "_".join(cell_types_clean)
        output_dir = os.path.join("jaccard_experiments", dir_name)
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results (only JSON and PDF)
    json_file, pdf_file = save_results(
        all_top_k_data, jaccard_matrices, common_genes, 
        cell_types, args.top_k, output_dir
    )
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved in: {output_dir}")
    print(f"  1. JSON data: {os.path.basename(json_file)}")
    print(f"  2. PDF heatmap: {os.path.basename(pdf_file)}")
    print(f"Analyzed {len(common_genes)} genes common to all {len(cell_types)} cell types")


if __name__ == "__main__":
    main()