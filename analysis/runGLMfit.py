import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--count_path")
parser.add_argument("--results_path")
parser.add_argument("--ridge_factor",default=1e-6,type=np.float)
args = parser.parse_args()
count_path = args.count_path
results_path = args.results_path


import numpy as np
import pandas as pd
import scipy

import glmFit


min_mu = 0.5
beta_tol = 1e-8
min_bat = -30
max_beta = 30
maxiter = 250
optimizer = "L-BFGS-B"


# counts
counts = pd.read_csv(count_path,sep="\t",header=None)

R_list = [10,11]
S = 21
C = 2
G = counts.shape[0]-1 # genes

design = glmFit.getDesignExtended(C,R_list,G)

dispersions = pd.read_csv(f'{results_path}_deseq2_dispersions.csv').dispGeneEst.values # per gene 
disp = np.repeat(dispersions,S)


counts_flattened = counts.values.T.flatten()   # need the format to be GENE BY SAMPLE
X = design

# do the fitting
beta, mu, converged  =  glmFit.irls_solver(
                                counts = counts_flattened,
                                design_matrix = design,
                                disp = disp,
                                )



# save beta and mu 
np.savetxt(f'{results_path}_glmFit_weights.txt',beta, delimiter='\t')
np.savetxt(f'{results_path}_glmFit_mu.txt',mu, delimiter='\t')
