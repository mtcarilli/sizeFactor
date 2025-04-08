
readarray -t count_path_list <<EOF
nonzero_gene_counts_25p_.5f.txt  
nonzero_gene_counts_75p_.5f.txt
nonzero_gene_counts_25p.txt      
nonzero_gene_counts_75p.txt
nonzero_gene_counts_10p_.5f.txt  
nonzero_gene_counts_50p_.5f.txt  
nonzero_gene_counts_90p_.5f.txt
nonzero_gene_counts_10p.txt      
nonzero_gene_counts_50p.txt      
nonzero_gene_counts.txt
EOF


readarray -t results_path_list <<EOF
nonzero_gene_counts_25p_.5f
nonzero_gene_counts_75p_.5f
nonzero_gene_counts_25p  
nonzero_gene_counts_75p
nonzero_gene_counts_10p_.5f 
nonzero_gene_counts_50p_.5f 
nonzero_gene_counts_90p_.5f
nonzero_gene_counts_10p     
nonzero_gene_counts_50p    
nonzero_gene_counts
EOF

for i in "${!count_path_list[@]}"; do
    count_path="../data/bottomly/${count_path_list[$i]}"
    results_path="../results/bottomly/${results_path_list[$i]}"
    echo $count_path
    Rscript runDESeq2EdgeR.R --count_path $count_path --results_path $results_path --reps1 10 --reps2 11
    python runGLMfit.py --count_path $count_path --results_path $results_path
done