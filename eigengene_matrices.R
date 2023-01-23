#' localMaximumQCM: Subroutine for Creating Gene Clusters
#'
#' Author: Zhi Huang
#' @param cMatrix a correlation matirx
#' @param gamma gamma value (default = 0.55)
#' @param t t value (default = 1)
#' @param lambda lambda value (default = 1)
#' @return An unmerged clusters group 'C'
#' @import genefilter
#' @import Biobase
#' @import progress
#' @import stats
#' @export
#' fastFilter: Subroutine for filtering expression matrix
#'
#' Author: Zhi Huang
#' @param RNA an expression matrix (rows: genes; columns: samples)
#' @param lowest_percentile_mean a float value range 0-1
#' @param lowest_percentile_variance a float value range 0-1
#' @param var.func specify variance function
#' @return An filtered expression matrix
#' @import genefilter
#' @import Biobase
#' @import stats



#' merging_lmQCM: Subroutine for Merging Gene Clusters
#'
#' Author: Zhi Huang
#' @param C Resulting clusters
#' @param beta beta value (default = 0.4)
#' @param minClusterSize minimum length of cluster to retain (default = 10)
#' @return mergedCluster - An merged clusters group
#' @import genefilter
#' @import Biobase
#' @import stats


options(repos = list(CRAN="http://cran.rstudio.com/"))

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("Biobase")
library(Biobase)


install.packages("matrixStats")

library("matrixStats")


.onAttach = function(libname, pkgname)
{
  ourVer = try( gsub("[^0-9_.-]", "", utils::packageVersion("lmQCM"), fixed = FALSE) );

  if (inherits(ourVer, "try-error")) ourVer = "";
  packageStartupMessage(paste("  Package lmQCM", ourVer, "loaded.\n"))
  packageStartupMessage("==========================================================================\n");
  packageStartupMessage(paste0(
    " If you benefit from this package, please cite:\n",
    " \n",
    "Huang Z, Han Z, Wang T, Shao W, Xiang S, Salama P, Rizkalla M, Huang K, Zhang J.\n",
    "TSUNAMI: Translational Bioinformatics Tool Suite For Network Analysis And Mining.\n",
    "Genomics, Proteomics & Bioinformatics, Volume 19, Issue 6, December 2021, Pages 1023-1031\n",
    "https://doi.org/10.1016/j.gpb.2019.05.006."))
  packageStartupMessage("==========================================================================\n\n");
}

merging_lmQCM <- function(C, beta=0.4, minClusterSize=10){
  # Merge the overlapped networks
  sizeC <- matrix(0, nrow = 0, ncol = length(C))

  for (i in 1:length(C)){
    sizeC[i] <- length(C[[i]])
  }

  res <- sort.int(sizeC, decreasing = TRUE, index.return=TRUE)
  sortC <- res$x
  sortInd <- res$ix

  C <- C[sortInd] # Still C, but sorted based on number of elements in each cell


  ind <- which(sortC >= minClusterSize)

  mergedCluster <- C[ind]

  #  print(C[sortInd])
  mergeOccur <- 1
  currentInd <- 0

  message(sprintf(" %d Modules before merging.", length(C)))

  while (mergeOccur == 1) {
    mergeOccur <- 0
    while (currentInd < length(mergedCluster)){

      currentInd <- currentInd + 1
      if (currentInd < length(mergedCluster)){
        keepInd <- 1:currentInd
        for (j in (currentInd+1) : length(mergedCluster)) {
          interCluster <- intersect(mergedCluster[[currentInd]], mergedCluster[[j]]);
          if (length(interCluster) >= beta*min(length(mergedCluster[[j]]), length(mergedCluster[[currentInd]]))) {
            mergedCluster[currentInd] <- list(union(mergedCluster[[currentInd]], mergedCluster[[j]]))
            mergeOccur <- 1
          }
          else {
            keepInd <- c(keepInd, j)
          }
        }
        mergedCluster <- mergedCluster[keepInd]
        message(sprintf("The length of merged Cluster: %d", length(mergedCluster)))
      }
    }

    sizeMergedCluster <- matrix(0, nrow = 0, ncol = length(mergedCluster))

    if (length(mergedCluster) != 0) { # ADDED

    for (i in 1 : length(mergedCluster)) {
      #  print(i)
      #  print(length(mergedCluster[[i]])) #

      sizeMergedCluster[i] <- length(mergedCluster[[i]]) # ERROR HERE
    }

    }




    res <- sort.int(sizeMergedCluster, decreasing = TRUE, index.return=TRUE)
    sortSize <- res$x
    sortMergedInd <- res$ix
    mergedCluster <- mergedCluster[sortMergedInd]
    currentInd <- 0


  }
  if (length(mergedCluster) != 0) { # ADDED
  for (i in 1:length(mergedCluster)){

    mergedCluster[[i]] <- unname(mergedCluster[[i]]) # Ohne das Error!


  }


  }

  message(sprintf(" %d Modules remain after merging.", length(mergedCluster)))

  if (length(mergedCluster) == 0) { #ADDED
    return(mergedCluster)

  }

  else {
    return (C) # Without this, we get an error !
  }

}


fastFilter <- function (RNA, lowest_percentile_mean = 0.2, lowest_percentile_variance = 0.2, var.func = "var"){
  RNA = as.matrix(RNA)
  rowIQRs <- function(eSet) {
    numSamp <- ncol(eSet)
    lowQ <- rowQ(eSet, floor(0.25 * numSamp))
    upQ <- rowQ(eSet, ceiling(0.75 * numSamp))
    upQ - lowQ
  }
  varFilter <- function (eset, var.cutoff = 0.5, filterByQuantile = TRUE, var.func = var.func) {
    if (deparse(substitute(var.func)) == "IQR") {
      message("Using row-wise IQR for calculating the variances.")
      vars <- rowIQRs(eset)
    } else {
      message("Calculating the variances.")
      vars <- apply(eset, 1, var.func)
    }
    if (filterByQuantile) {
      if (0 < var.cutoff && var.cutoff < 1) {
        quant = quantile(vars, probs = var.cutoff)
        selected = !is.na(vars) & vars > quant
      } else stop("Cutoff Quantile has to be between 0 and 1.")
    } else {
      selected <- !is.na(vars) & vars > var.cutoff
    }
    return(selected)
  }

  # Remove data with lowest m% mean exp value shared by all samples
  message("Note: For RNA data, we suppose input matrix (data frame) is with:")
  message("      Row: Genes;    Columns: Samples.")
  geneID = rownames(RNA)
  percentile = lowest_percentile_mean
  if (percentile > 0){
    RNAmean = apply(RNA, 1, mean)
    RNA_filtered1 = RNA[RNAmean > quantile(RNAmean, percentile), ]
    geneID_filtered1 = geneID[RNAmean > quantile(RNAmean, percentile)]
  } else {
    RNA_filtered1 = RNA
    geneID_filtered1 = geneID
  }
  message(sprintf("(%d genes, %d samples) after removing lowest %.2f%% mean expression value.",
                  dim(RNA_filtered1)[1], dim(RNA_filtered1)[2], percentile*100))

  # Remove data with lowest 10% variance across samples
  percentile = lowest_percentile_variance
  if (percentile > 0){
    if (dim(RNA_filtered1)[2] > 3){
      index <- varFilter(eset = RNA_filtered1, var.cutoff = percentile, var.func = var.func)
      RNA_filtered2 = RNA_filtered1[index, ]
      geneID_filtered2 = geneID_filtered1[index]
    } else{
      message("Cannot calculate order statistic on object with less than 3 columns, will not remove data based on variance.")
      RNA_filtered2 = RNA_filtered1
      geneID_filtered2 = geneID_filtered1
    }
  } else {
    RNA_filtered2 = RNA_filtered1
    geneID_filtered2 = geneID_filtered1
  }

  message(sprintf("(%d genes, %d samples) after removing lowest %.2f%% variance expression value.",
                  dim(RNA_filtered2)[1], dim(RNA_filtered2)[2], lowest_percentile_variance*100))
  return(RNA_filtered2)
}




localMaximumQCM <- function (cMatrix, gamma = 0.55, t = 1, lambda = 1){
  C <- list()
  nRow <- nrow(cMatrix)
  maxV <- apply(cMatrix, 2, max) # max correlation value in columns of correlation matrix (represented as vector)
  maxInd <- apply(cMatrix, 2, which.max) # several diferrences comparing with Matlab results ; # index of max


  # Step 1 - find the local maximal edges
  # maxEdges <- matrix(0, nrow = 0, ncol = 2)
  # maxW <- matrix(0, nrow = 0, ncol = 1)
  # for (i in 1:nRow){
  #   if (maxV[i] == max(cMatrix[maxInd[i], ])) {
  #     maxEdges <- rbind(maxEdges, c(maxInd[i], i))
  #     maxW <- rbind(maxW, maxV[i])
  #   }
  # }
  lm.ind <- which(maxV == sapply(maxInd, function(x) max(cMatrix[x,]))) # sapply() function takes list,
  #vector or data frame as input and gives output in vector or matrix. It is useful for operations on list objects and returns a list object of same length of original set
  maxEdges <- cbind(maxInd[lm.ind], lm.ind) # Take a sequence of vector, matrix or data-frame arguments and combine by columns or rows, respectively.
  maxW <- maxV[lm.ind]

  res <- sort.int(maxW, decreasing = TRUE, index.return=TRUE)
  sortMaxV <- res$x
  sortMaxInd <- res$ix
  sortMaxEdges <- maxEdges[sortMaxInd,]
  message(sprintf("Number of Maximum Edges: %d", length(sortMaxInd)))

  currentInit <- 1
  noNewInit <- 0

  nodesInCluster <- matrix(0, nrow = 0, ncol = 1)




  while ((currentInit <= length(sortMaxInd)) & (noNewInit == 0)) {

    if (sortMaxV[currentInit] < (gamma * sortMaxV[1]) ) {
      noNewInit <- 1
    }
    else {
      if ( (is.element(sortMaxEdges[currentInit, 1], nodesInCluster) == FALSE) & is.element(sortMaxEdges[currentInit, 2], nodesInCluster) == FALSE) {
        newCluster <- sortMaxEdges[currentInit, ]
        addingMode <- 1
        currentDensity <- sortMaxV[currentInit]
        nCp <- 2
        totalInd <- 1:nRow
        remainInd <- setdiff(totalInd, newCluster)
        # C = setdiff(A,B) for vectors A and B, returns the values in A that
        # are not in B with no repetitions. C will be sorted.
        while (addingMode == 1) {
          neighborWeights <- colSums(cMatrix[newCluster, remainInd])
          maxNeighborWeight <- max(neighborWeights)
          maxNeighborInd <- which.max(neighborWeights)
          c_v = maxNeighborWeight/nCp;
          alphaN = 1 - 1/(2*lambda*(nCp+t));
          if (c_v >= alphaN * currentDensity) {
            newCluster <- c(newCluster, remainInd[maxNeighborInd])
            nCp <- nCp + 1
            currentDensity <- (currentDensity*((nCp-1)*(nCp-2)/2)+maxNeighborWeight)/(nCp*(nCp-1)/2)
            remainInd <- setdiff(remainInd, remainInd[maxNeighborInd]);
          }
          else {
            addingMode <- 0
          }
        }
        nodesInCluster <- c(nodesInCluster, newCluster)
        C <- c(C, list(newCluster))
      }
    }
    currentInit <- currentInit + 1
  }
  message(" Calculation Finished.")
  return(C)
}




setClass("QCMObject", representation(clusters.id = "list", clusters.names = "list",
                                     eigengene.matrix = "data.frame"))
#' lmQCM: Main Routine for Gene Co-expression Analysis
#'
#' Author: Zhi Huang
#' @param data_in real-valued expression matrix with rownames indicating gene ID or gene symbol
#' @param gamma gamma value (default = 0.55)
#' @param t t value (default = 1)
#' @param lambda lambda value (default = 1)
#' @param beta beta value (default = 0.4)
#' @param minClusterSize minimum length of cluster to retain (default = 10)
#' @param CCmethod Methods for correlation coefficient calculation (default = "pearson"). Users can also pick "spearman".
#' @param positiveCorrelation This determines if correlation matrix should convert to positive (with abs function) or not.
#' @param normalization Determine if normalization is needed on massive correlation coefficient matrix.
#' @return QCMObject - An S4 Class with lmQCM results
#'
#' @examples
#' library(lmQCM)
#' library(Biobase)
#' data(sample.ExpressionSet)
#' data = assayData(sample.ExpressionSet)$exprs
#' data = fastFilter(data, 0.2, 0.2)
#' lmQCM(data)
#'
#' @import genefilter
#' @import Biobase
#' @import stats
#' @import methods
#' @export
lmQCM <- function(data_in,gamma=0.55,t=1,lambda=1,beta=0.4,minClusterSize=10,CCmethod="pearson",positiveCorrelation=F,normalization=F) {
  message("Calculating massive correlation coefficient ...")
  cMatrix <- cor(t(data_in), method = CCmethod) # Correlation coefficient, with pearson, matrix (nxn, n : features ?)
  diag(cMatrix) <- 0 # each feature with itself has a variance of 0

  if (positiveCorrelation){
    cMatrix <- abs(cMatrix)
  }

  if(normalization){
    # Normalization
    D <- rowSums(cMatrix)
    D.half <- 1/sqrt(D)

    cMatrix <- apply(cMatrix, 2, function(x) x*D.half ) # apply() takes Data frame or matrix as an input and gives output in vector, 2 means performed on columns
    cMatrix <- t(apply(cMatrix, 1, function(x) x*D.half )) #1 means performed on rows

  }

  C <- localMaximumQCM(cMatrix, gamma, t, lambda)

  clusters <- merging_lmQCM(C, beta, minClusterSize)
  # map rownames to clusters
  clusters.names = list()
  for (i in 1:length(clusters)){
    mc = clusters[[i]]
    clusters.names[[i]] = rownames(data_in)[mc]
  }
  # calculate eigengene
  eigengene.matrix <- matrix(0, nrow = length(clusters), ncol = dim(data_in)[2]) # Clusters * Samples

  for (i in 1:(length(clusters.names))) {
    geneID <- as.matrix(clusters.names[[i]])
    X <- data_in[geneID,]
    mu <- rowMeans(X)
    stddev <- rowSds(as.matrix(X), na.rm=TRUE) # standard deviation with 1/(n-1)
    XNorm <- sweep(X,1,mu) # normalize X
    XNorm <- apply(XNorm, 2, function(x) x/stddev)
    SVD <- svd(XNorm)
    eigenvector.first = SVD$v[,1]


    # Compute the sign of the eigengene.
    # 1. Correlate the eigengene value with each of the gene's expression in that module across all samples used to generate the module.
    # 2. If >50% of the correlations is negative, then assign a – sign to the eigengene.
    # 3. If 50% or more correlation is positive, the eigengene remains positive.
    # 4. Output the eigene value table with the sign carried (if it is negative).

    negative_ratio = sum(cor(t(X), eigenvector.first) < 0)/dim(X)[1]
    if (negative_ratio > 0.5){
      eigenvector.first = -eigenvector.first
    }

    eigengene.matrix[i,] <- t(eigenvector.first)
  }
  eigengene.matrix = data.frame(eigengene.matrix)
  colnames(eigengene.matrix) = colnames(data_in)

  QCMObject <- methods::new("QCMObject", clusters.id = clusters, clusters.names = clusters.names,
                            eigengene.matrix = eigengene.matrix)

  message("Done.")
  return(eigengene.matrix)

}




#data <- data(sample.ExpressionSet)
#data = assayData(sample.ExpressionSet)$exprs
#lmQCM(data)
#print(data)

args <- commandArgs(trailingOnly = TRUE) # does not work for some reason




cancer_name <- paste(readLines("/Users/marlon/Desktop/Project/TCGAData/currentcancer.txt"), collapse="\n")
view_names <- paste(readLines("/Users/marlon/Desktop/Project/TCGAData/cancerviews.txt"), collapse="\n")
mode <- paste(readLines("/Users/marlon/Desktop/Project/TCGAData/eigengene_mode.txt"), collapse="\n")
view_names_list <- as.list(scan(text=view_names, what="\n"))

print(view_names_list)



path <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/RPPA_for_r.csv")

# RPPA
if (file.exists(path) & ('RPPA' %in% view_names_list))  {
  print("Creating RPPA eigengene matrix for training data ....")

  data_RPPA <- read.csv(path)
  data_RPPA = subset(data_RPPA, select= -c(X)) # remove added X column
  matrix_RPPA <- t(as.matrix(data_RPPA))
  class(matrix_RPPA) <- "numeric"
  eset_RPPA <- new("ExpressionSet", expr = matrix_RPPA)
  data_new_RPPA = assayData(eset_RPPA)$exprs

  eigengene_RPPA = lmQCM(data_new_RPPA)
  eigengene_RPPA = t(eigengene_RPPA)

  if ("all" == mode) {
    temp <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/RPPA_test_for_r.csv")
    print("Creating RPPA eigengene matrix for test data ....")
    data_test_RPPA <- read.csv(temp)
    data_test_RPPA = subset(data_test_RPPA, select= -c(X)) # remove added X column
    matrix_test_RPPA <- t(as.matrix(data_test_RPPA))
    class(matrix_test_RPPA) <- "numeric"
    eset_test_RPPA <- new("ExpressionSet", expr = matrix_test_RPPA)
    data_test_new_RPPA = assayData(eset_test_RPPA)$exprs

    eigengene_test_RPPA = lmQCM(data_test_new_RPPA)
    eigengene_test_RPPA = t(eigengene_test_RPPA)

  }


  temp4 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/RPPA_val_for_r.csv")
  print("Creating RPPA eigengene matrix for validation data ....")
  data_val_RPPA <- read.csv(temp4)
  data_val_RPPA = subset(data_val_RPPA, select= -c(X)) # remove added X column
  matrix_val_RPPA <- t(as.matrix(data_val_RPPA))
  class(matrix_val_RPPA) <- "numeric"
  eset_val_RPPA <- new("ExpressionSet", expr = matrix_val_RPPA)
  data_val_new_RPPA = assayData(eset_val_RPPA)$exprs

  eigengene_val_RPPA = lmQCM(data_val_new_RPPA)
  eigengene_val_RPPA = t(eigengene_val_RPPA)




  temp2 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/RPPA_eigengene_matrix.csv")
  temp5 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/RPPA_val_eigengene_matrix.csv")
  write.csv(eigengene_RPPA, temp2, row.names = TRUE)
  write.csv(eigengene_val_RPPA, temp5, row.names = TRUE)

  if ("all" == mode) {
    temp3 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/RPPA_test_eigengene_matrix.csv")
    write.csv(eigengene_test_RPPA, temp3, row.names = TRUE)

  }

}


path <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MICRORNA_for_r.csv")

if (file.exists(path) & ('microRNA' %in% view_names_list)) {
  print("Creating microRNA eigengene matrix for training data ....")
  data_microRNA <- read.csv(path)
  data_microRNA = subset(data_microRNA, select= -c(X)) # remove added X column
  matrix_microRNA <- t(as.matrix(data_microRNA))
  class(matrix_microRNA) <- "numeric"
  eset_microRNA <- new("ExpressionSet", expr = matrix_microRNA)
  data_new_microRNA = assayData(eset_microRNA)$exprs

  eigengene_microRNA = lmQCM(data_new_microRNA)
  eigengene_microRNA = t(eigengene_microRNA)

  if ("all" == mode) {
    print("Creating microRNA eigengene matrix for test data ....")
    temp <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MICRORNA_test_for_r.csv")
    data_test_microRNA <- read.csv(temp)
    data_test_microRNA = subset(data_test_microRNA, select= -c(X)) # remove added X column
    matrix_test_microRNA <- t(as.matrix(data_test_microRNA))
    class(matrix_test_microRNA) <- "numeric"
    eset_test_microRNA <- new("ExpressionSet", expr = matrix_test_microRNA)
    data_test_new_microRNA = assayData(eset_test_microRNA)$exprs

    eigengene_test_microRNA = lmQCM(data_test_new_microRNA)
    eigengene_test_microRNA = t(eigengene_test_microRNA)

  }




  temp4 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MICRORNA_val_for_r.csv")
  print("Creating microRNA eigengene matrix for validation data ....")
  data_val_microRNA <- read.csv(temp4)
  data_val_microRNA = subset(data_val_microRNA, select= -c(X)) # remove added X column
  matrix_val_microRNA <- t(as.matrix(data_val_microRNA))
  class(matrix_val_microRNA) <- "numeric"
  eset_val_microRNA <- new("ExpressionSet", expr = matrix_val_microRNA)
  data_val_new_microRNA = assayData(eset_val_microRNA)$exprs

  eigengene_val_microRNA = lmQCM(data_val_new_microRNA)
  eigengene_val_microRNA = t(eigengene_val_microRNA)


  temp2 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MICRORNA_eigengene_matrix.csv")
  temp5 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MICRORNA_val_eigengene_matrix.csv")
  write.csv(eigengene_microRNA, temp2, row.names = TRUE)
  write.csv(eigengene_val_microRNA, temp5, row.names = TRUE)

  if ("all" == mode) {
    temp3 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MICRORNA_test_eigengene_matrix.csv")
    write.csv(eigengene_test_microRNA, temp3, row.names = TRUE)

  }

}





#DNA
path <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/DNA_for_r.csv")

if (file.exists(path) & ('DNA' %in% view_names_list)) {
  print("Creating DNA eigengene matrix for training data ....")
  data_DNA <- read.csv(path)
  data_DNA = subset(data_DNA, select= -c(X)) # remove added X column
  matrix_DNA <- t(as.matrix(data_DNA))
  class(matrix_DNA) <- "numeric"
  eset_DNA <- new("ExpressionSet", expr = matrix_DNA)
  data_new_DNA = assayData(eset_DNA)$exprs

  eigengene_DNA = lmQCM(data_new_DNA)
  #print(eigengene_dna)
  eigengene_DNA = t(eigengene_DNA)

  if ("all" == mode) {
    temp <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/DNA_test_for_r.csv")
    print("Creating DNA eigengene matrix for test data ....")
    data_test_DNA <- read.csv(temp)
    data_test_DNA = subset(data_test_DNA, select= -c(X)) # remove added X column
    matrix_test_DNA <- t(as.matrix(data_test_DNA))
    class(matrix_test_DNA) <- "numeric"
    eset_test_DNA <- new("ExpressionSet", expr = matrix_test_DNA)
    data_test_new_DNA = assayData(eset_test_DNA)$exprs

    eigengene_test_DNA = lmQCM(data_test_new_DNA)
    eigengene_test_DNA = t(eigengene_test_DNA)

  }


  temp4 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/DNA_val_for_r.csv")
  print("Creating DNA eigengene matrix for validation data ....")

  data_val_DNA <- read.csv(temp4)
  data_val_DNA = subset(data_val_DNA, select= -c(X)) # remove added X column
  matrix_val_DNA <- t(as.matrix(data_val_DNA))
  class(matrix_val_DNA) <- "numeric"
  eset_val_DNA <- new("ExpressionSet", expr = matrix_val_DNA)
  data_val_new_DNA = assayData(eset_val_DNA)$exprs

  eigengene_val_DNA = lmQCM(data_val_new_DNA)
  eigengene_val_DNA = t(eigengene_val_DNA)

  temp2 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/DNA_eigengene_matrix.csv")
  temp5 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/DNA_val_eigengene_matrix.csv")
  write.csv(eigengene_DNA, temp2, row.names = TRUE)
  write.csv(eigengene_val_DNA, temp5, row.names = TRUE)

  if ("all" == mode) {
    temp3 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/DNA_test_eigengene_matrix.csv")
    write.csv(eigengene_test_DNA, temp3, row.names = TRUE)

  }


}
#mRNA
path <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MRNA_for_r.csv")
if (file.exists(path) & ('mRNA' %in% view_names_list)) {
  print("Creating mRNA eigengene matrix for training data ....")

  data_mRNA <- read.csv(path)
  data_mRNA = subset(data_mRNA, select= -c(X)) # remove added X column
  matrix_mRNA <- t(as.matrix(data_mRNA))
  class(matrix_mRNA) <- "numeric"
  eset_mRNA <- new("ExpressionSet", expr = matrix_mRNA)
  data_new_mRNA = assayData(eset_mRNA)$exprs

  eigengene_mRNA = lmQCM(data_new_mRNA)

  eigengene_mRNA = t(eigengene_mRNA)

  if ("all" == mode) {
    temp <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MRNA_test_for_r.csv")
    print("Creating mRNA eigengene matrix for test data ....")
    data_test_mRNA <- read.csv(temp)
    data_test_mRNA = subset(data_test_mRNA, select= -c(X)) # remove added X column
    matrix_test_mRNA <- t(as.matrix(data_test_mRNA))
    class(matrix_test_mRNA) <- "numeric"
    eset_test_mRNA <- new("ExpressionSet", expr = matrix_test_mRNA)
    data_test_new_mRNA = assayData(eset_test_mRNA)$exprs

    eigengene_test_mRNA = lmQCM(data_test_new_mRNA)

    eigengene_test_mRNA = t(eigengene_test_mRNA)

  }

  temp4 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MRNA_val_for_r.csv")

  print("Creating mRNA eigengene matrix for validation data ....")
  data_val_mRNA <- read.csv(temp4)
  data_val_mRNA = subset(data_val_mRNA, select= -c(X)) # remove added X column
  matrix_val_mRNA <- t(as.matrix(data_val_mRNA))
  class(matrix_val_mRNA) <- "numeric"
  eset_val_mRNA <- new("ExpressionSet", expr = matrix_val_mRNA)
  data_val_new_mRNA = assayData(eset_val_mRNA)$exprs

  eigengene_val_mRNA = lmQCM(data_val_new_mRNA)
  eigengene_val_mRNA = t(eigengene_val_mRNA)


  temp2 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MRNA_eigengene_matrix.csv")
  temp5 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MRNA_val_eigengene_matrix.csv")
  write.csv(eigengene_mRNA, temp2, row.names = TRUE)
  write.csv(eigengene_val_mRNA, temp5, row.names = TRUE)

  if ("all" == mode) {
    temp3 <- paste0("/Users/marlon/Desktop/Project/TCGAData/",cancer_name,"/MRNA_test_eigengene_matrix.csv")
    write.csv(eigengene_test_mRNA, temp3, row.names = TRUE)
  }



}





