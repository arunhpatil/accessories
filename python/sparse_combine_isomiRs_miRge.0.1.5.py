#!/usr/bin/python
# Author: Arun H Patil, Lieber Institute for Brain Development
# Code: Join multiple CSV files using Dask, Pandas and sparse matrix, filter rows by sum (filter out < 30)
# To monitor the progress put the following link in the browser: http://10.17.9.178:43365/status

# Import libraries 
import dask.dataframe as dd
import pandas as pd
import dask
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import gc
import sys
from dask.delayed import delayed
from dask.distributed import Client
from scipy.sparse import csr_matrix, vstack, hstack, find
from pathlib import Path

walk_dir = sys.argv[1] #../complete

'''
uniqueSequence,match,seqType,miRNA,variantType,change
AAATGACCCAGGAGAC,isomiR miRNA,isomiR,hsa-miR-4724-5p,iso_snv_seed;iso_snv_central_supp;iso_3p:-9;iso_add3p:2,iso_snv_seed:3=C>A|7=A>C|;iso_snv_central_supp:15=T>A|16=G>C|
AAATGACCCGATCACTCCCGTTGAGT,isomiR miRNA,isomiR,hsa-miR-425-5p,iso_snv_seed;iso_add5p:1;iso_3p:+2,iso_snv_seed:7=A>C|;iso_add5p:-1=G>A|
AAATGACCCTTCTGTAT,isomiR miRNA,isomiR,hsa-miR-514a-3p,iso_snv_seed;iso_snv_central_supp;iso_add5p:1;iso_3p:-7;iso_add3p:2,iso_snv_seed:2=T>A|7=A>C|;iso_snv_central_supp:15=G>A|16=A>T|;iso_add5p:-1=G>A|
AAATGACCTTCAACCTCC,isomiR miRNA,isomiR,hsa-miR-6855-3p,iso_snv_seed;iso_snv_central_supp;iso_5p:+1;iso_3p:-4;iso_add3p:1,iso_snv_seed:2=G>A|4=C>A|;iso_snv_central_supp:17=C>T|;iso_add3p:19=A>C|
'''


# Function to read CSV and convert to sparse matrix
@delayed
def read_and_convert_to_sparse(index, file):
    index+=1
    print(f"{index}. Reading input file {file}")
    df = pd.read_csv(file, dtype={'uniqueSequence': 'object', 'match' : 'object', 'seqType': 'object', 'miRNA': 'object','variantType': 'object','change': 'object'}, low_memory=False)

    concat_columns = ['uniqueSequence', 'match', 'seqType', 'miRNA', 'variantType', 'change']
    new_key = 'Sequence#match#seqType#miRNA#variantType#change'
    df[new_key] = df[concat_columns].astype(str).agg('#'.join, axis=1)
    
    df = df.drop(concat_columns, axis=1)
    df = df.set_index(new_key).sort_index()
    
    sparse_matrix = csr_matrix(df.values)
    #client.run(gc.collect)

    return df.index, df.columns, sparse_matrix

@delayed
def align_sparse_matrix_chunk(index, sparse_matrix, all_indices):
    # Convert sparse matrix to DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=index)
    # Reindex to align with all_indices, fill missing values with 0
    aligned_df = df.reindex(all_indices, fill_value=0)
    # Convert back to sparse matrix
    aligned_sparse_matrix = csr_matrix(aligned_df.values)
    return aligned_sparse_matrix

if __name__ == '__main__':
    # Getting file list
    file_list = list()
    for root, subdirs, files in os.walk(walk_dir):
        for subdir in subdirs:
            if not "tRFs" in subdir:
                curDir = Path(Path(root).absolute()/subdir)
                miRC = str(Path(curDir/"isomiRs_mirglmm.csv"))
                file_list.append(miRC)
    #print(file_list)
    global client  
    client = Client(
            n_workers=20,
            memory_limit='40GB',
            processes=True,
            threads_per_worker=5,
            #dashboard_address=':44395'
            #dashboard_address=':44396'
            dashboard_address=':43365'
            )

    #delayed_results = [read_and_convert_to_sparse(index, file, nan_check_columns, common_key) for index, file in enumerate(file_list[1:3])]
    delayed_results = [read_and_convert_to_sparse(index, file) for index, file in enumerate(file_list)]
    results = dask.compute(*delayed_results)
    #client.amm.stop()
    
    print("1. Done with reading files\n")
    print("2. Now extracting indices and column names\n")
    # Extract indices, column names, and sparse matrices
    indices, column_names, sparse_matrices = zip(*results)
 
    print("3. creating union of all indices from files\n")

    # Function to merge two indices
    @delayed
    def merge_indices(idx1, idx2):
        return idx1.union(idx2)

    # Create a delayed list of indices unions
    delayed_indices = indices

    while len(delayed_indices) > 1:
        merged_indices = []
        for i in range(0, len(delayed_indices), 2):
            if i + 1 < len(delayed_indices):
                merged_indices.append(merge_indices(delayed_indices[i], delayed_indices[i+1]))
            else:
                merged_indices.append(delayed_indices[i])
        delayed_indices = merged_indices

    # Compute the final union of all indices
    all_indices = dask.compute(delayed_indices[0])[0]
     
    print("4. Done with union of all indices from files\n")
    print(len(all_indices))
    # Print total indices of isomiR and exactmiRNA sequences
    print()

    # Process matrices in chunks
    chunk_size =  40 # Adjust based on memory limits and matrix size, i.e., it takes 40 matrices at once. Like wise for our data, it is 40+40+40+13 = 133. It took about approximately one hour to get to this stage. 
    aligned_matrices = []

    print("5. Aligning each sparse matrix to the union of all indices \n")
    for i in range(0, len(sparse_matrices), chunk_size):
        #chunk_results = [align_sparse_matrix_chunk(index, sparse_matrix, scattered_indices) 
        chunk_results = [align_sparse_matrix_chunk(index, sparse_matrix, all_indices) 
                for index, sparse_matrix in zip(indices[i:i+chunk_size], sparse_matrices[i:i+chunk_size])]
        aligned_matrices.extend(dask.compute(*chunk_results))

    print("6. Done with aligning all sparse matrix to the union of all indices.\nNow combining all matricies by hstack\n")
    # Stack all aligned sparse matrices horizontally
    combined_sparse_matrix = hstack(aligned_matrices)
     
    print("7. Done hstack of aligned matrices\n")
    # Convert to CSR format if not already
    if not isinstance(combined_sparse_matrix, csr_matrix):
        combined_sparse_matrix = combined_sparse_matrix.tocsr()

    print("8. Computing row sums across aligned matrices\n")
    # Calculate the row sums of the sparse matrix
    row_sums = combined_sparse_matrix.sum(axis=1).A1  # .A1 converts it to a flat array
    
    print("9. Ensuring row sums is properly alinged with combined sparse matrix \n")
    # Ensure row_sums is properly aligned with combined_sparse_matrix
    assert len(row_sums) == combined_sparse_matrix.shape[0], "Mismatch between row_sums and combined_sparse_matrix rows."

    print("10. Find rows whose row sums >= 30, this is to filter noise from data\n")
    # Find rows with sum >= 30
    rows_to_keep = row_sums >= 30
    #rows_to_keep = row_sums >= 1000

    # Verify dimensions before filtering
    if len(rows_to_keep) != combined_sparse_matrix.shape[0]:
        raise ValueError("Dimension mismatch between rows_to_keep and combined_sparse_matrix.")

    print("11. Filter sparse matrix by rows whose row sums >= 30\n")
    # Filter the sparse matrix by these rows
    filtered_sparse_matrix = combined_sparse_matrix[rows_to_keep, :]
    
    print("12. Now filter indices to just keep rows whose row sums >= 30\n")
    # Ensure the filtered indices are correct
    filtered_indices = all_indices[rows_to_keep]

    print("13. Join corresponding columns for the combined sparse matrix with row sums >= 30\n")
    # Join corresponding columns from the original CSV files
    filtered_dataframes = []
    for sparse_matrix, col_names in zip(aligned_matrices, column_names):
        df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=all_indices, columns=col_names)
        filtered_df = df.loc[filtered_indices]
        filtered_dataframes.append(filtered_df)
    
    print("14. Now concatenate all individual dataframes with union of indices\n")
    # Concatenate all DataFrames along columns
    combined_dataframe = pd.concat(filtered_dataframes, axis=1)
    print(f"\n15. The union of all indices after filtering for rowSums 30 is : {combined_dataframe.shape}\n")
    # Ensure the DataFrame is dense
    print("16. Now convert sparse matrix to dense from the combined dataframe\n")
    dense_combined_df = combined_dataframe.sparse.to_dense()
    print("17. Give a name to row index\n")
    dense_combined_df.index.names = ['match']
    print("18. Finally save the complete combined dataframe to csv\n")
    dense_combined_df.to_csv("combined_denseMat_112924.csv")


