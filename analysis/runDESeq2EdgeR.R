suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("glue"))
suppressPackageStartupMessages(library("DESeq2"))
suppressPackageStartupMessages(library("edgeR"))

parser <- ArgumentParser()

parser$add_argument("--count_path")
parser$add_argument("--results_path")
parser$add_argument("--reps1")
parser$add_argument("--reps2")

# parse arguments
args <- parser$parse_args()
count_path <- args$count_path
results_path <- args$results_path
reps1 <- as.numeric(args$reps1)
reps2 <- as.numeric(args$reps2)

# read in counts
counts <- read.delim(count_path, header = FALSE)

# create simple colData and design matrix
group <- c( rep("A",reps1), rep("B",reps2) )
colData <- data.frame(group=group)
design <- model.matrix(~1 + group, data = colData)

# run edgeR and save results
y <- DGEList(counts=counts)  
y <- normLibSizes(y)
y <- calcNormFactors(y)
y <- estimateDisp(y, design)


# lib.size (total counts) and norm.factors (normalization factors calculated using TMM) 
libsize <- y$samples$lib.size
normfactors <- y$samples$norm.factors
efflibsize <- libsize*normfactors

# run edgeR and save size factors and weights
fit <- glmFit(y, design)
lrt <- glmLRT(fit, coef=2)

coefficients_df <- as.data.frame(fit$coefficients)
predicted_counts <- fitted(fit)

write.csv(coefficients_df, glue("{results_path}_edgeR_defaultSF_weights.csv"),row.names=FALSE)
write.csv(predicted_counts, glue("{results_path}_edgeR_defaultSF_mu.csv"),row.names=FALSE)
write.csv(data.frame(libsize = libsize, normfactors = normfactors, efflibsize = efflibsize), glue("{results_path}_edgeR_SF_df.csv"), row.names = TRUE)
write.csv(as.data.frame(lrt$table), glue("{results_path}_edgeR_defaultSF_resLRT_df.csv"),row.names=FALSE)


# run DESeq2 and save size factors and weights
dds <- DESeqDataSetFromMatrix(countData = counts, colData = colData, design = ~1 + group)
dds <- DESeq(dds)
dds_lrt <- DESeq(dds, test = "LRT", reduced = ~ 1)
res_lrt <- results(dds_lrt)

sizefactors <- sizeFactors(dds)
coef_matrix <- coef(dds)  
fitted_counts <- fitted(dds)  


write.csv(coef_matrix, glue("{results_path}_deseq2_defaultSF_weights.csv"),row.names=FALSE)
write.csv(as.data.frame(fitted_counts), "{results_path}_deseq2_mu.csv", row.names = TRUE)
write.csv(sizefactors, glue("{results_path}_deseq2_sizefactors.csv"),row.names=FALSE)
disp_table <- as.data.frame(mcols(dds)[, c("dispGeneEst", "dispFit", "dispMAP", "dispersion")])
write.csv(disp_table,glue("{results_path}_deseq2_dispersions.csv"),row.names=FALSE)
write.csv(as.data.frame(res_lrt), file = glue("{results_path}_deseq2_defaultSF_resLRT_df.csv"), row.names = FALSE)  # write LRT results
