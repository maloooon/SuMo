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