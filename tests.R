if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("Biobase")


library(Biobase)

data <- data(sample.ExpressionSet)
data = assayData(sample.ExpressionSet)$exprs
#lmQCM(data)
print(data)




data <- read.csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/RPPA_for_r.csv", stringsAsFactors=TRUE)
data = subset(data, select= -c(X)) # remove added X column 
matrix <- as.matrix(data)
class(matrix) <- "numeric"
print(matrix)
eset <- new("ExpressionSet", expr = matrix)
data_new = assayData(eset)$exprs

print(data_new)

install.packages("matrixStats")

library("matrixStats")
print(data_new)

a = matrix(rnorm(1e4), nrow=10)
rowSds(a)








data_mRNA <- read.csv("/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/mRNA_for_r.csv")
data_mRNA = subset(data_mRNA, select= -c(X)) # remove added X column
matrix_mRNA <- t(as.matrix(data_mRNA))
class(matrix_mRNA) <- "numeric"
eset_mRNA <- new("ExpressionSet", expr = matrix_mRNA)
data_new_mRNA = assayData(eset_mRNA)$exprs

eigengene_mRNA = lmQCM(data_new_mRNA)

eigengene_mRNA = t(eigengene_mRNA)


data_test_mRNA <- read.csv("/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/mRNA_test_for_r.csv")
data_test_mRNA = subset(data_test_mRNA, select= -c(X)) # remove added X column
matrix_test_mRNA <- t(as.matrix(data_test_mRNA))
class(matrix_test_mRNA) <- "numeric"
eset_test_mRNA <- new("ExpressionSet", expr = matrix_test_mRNA)
data_test_new_mRNA = assayData(eset_test_mRNA)$exprs

eigengene_test_mRNA = lmQCM(data_test_new_mRNA)

eigengene_test_mRNA = t(eigengene_test_mRNA)



data_microRNA <- read.csv("/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/microRNA_for_r.csv")
data_microRNA = subset(data_microRNA, select= -c(X)) # remove added X column
matrix_microRNA <- t(as.matrix(data_microRNA))
class(matrix_microRNA) <- "numeric"
eset_microRNA <- new("ExpressionSet", expr = matrix_microRNA)
data_new_microRNA = assayData(eset_microRNA)$exprs

eigengene_microRNA = lmQCM(data_new_microRNA)
eigengene_microRNA = t(eigengene_microRNA)

data_test_microRNA <- read.csv("/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/microRNA_test_for_r.csv")
data_test_microRNA = subset(data_test_microRNA, select= -c(X)) # remove added X column
matrix_test_microRNA <- t(as.matrix(data_test_microRNA))
class(matrix_test_microRNA) <- "numeric"
eset_test_microRNA <- new("ExpressionSet", expr = matrix_test_microRNA)
data_test_new_microRNA = assayData(eset_test_microRNA)$exprs

eigengene_test_microRNA = lmQCM(data_test_new_microRNA)
eigengene_test_microRNA = t(eigengene_test_microRNA)


write.csv(eigengene_DNA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/DNA_eigengene_matrix.csv", row.names = TRUE)
write.csv(eigengene_mRNA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/mRNA_eigengene_matrix.csv", row.names = TRUE)
write.csv(eigengene_microRNA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/microRNA_eigengene_matrix.csv", row.names = TRUE)
write.csv(eigengene_RPPA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/RPPA_eigengene_matrix.csv", row.names = TRUE)

write.csv(eigengene_test_DNA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/DNA_test_eigengene_matrix.csv", row.names = TRUE)
write.csv(eigengene_test_mRNA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/mRNA_test_eigengene_matrix.csv", row.names = TRUE)
write.csv(eigengene_test_microRNA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/microRNA_test_eigengene_matrix.csv", row.names = TRUE)
write.csv(eigengene_test_RPPA, "/Users/marlon/DataspellProjectsForSAMO/SAMO/TCGAData/PRAD/RPPA_test_eigengene_matrix.csv", row.names = TRUE)
