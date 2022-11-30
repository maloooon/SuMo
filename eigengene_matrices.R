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
        # message(sprintf("The length of merged Cluster: %d", length(mergedCluster)))
      }
    }
    sizeMergedCluster <- matrix(0, nrow = 0, ncol = length(mergedCluster))
    for (i in 1 : length(mergedCluster)) {
      sizeMergedCluster[i] <- length(mergedCluster[[i]])
    }
    res <- sort.int(sizeMergedCluster, decreasing = TRUE, index.return=TRUE)
    sortSize <- res$x
    sortMergedInd <- res$ix
    mergedCluster <- mergedCluster[sortMergedInd]
    currentInd <- 0
  }
  for (i in 1:length(mergedCluster)){
    mergedCluster[[i]] <- unname(mergedCluster[[i]])
  }
  message(sprintf(" %d Modules remain after merging.", length(mergedCluster)))
  return(mergedCluster)
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
  maxV <- apply(cMatrix, 2, max)
  maxInd <- apply(cMatrix, 2, which.max) # several diferrences comparing with Matlab results
  
  
  # Step 1 - find the local maximal edges
  # maxEdges <- matrix(0, nrow = 0, ncol = 2)
  # maxW <- matrix(0, nrow = 0, ncol = 1)
  # for (i in 1:nRow){
  #   if (maxV[i] == max(cMatrix[maxInd[i], ])) {
  #     maxEdges <- rbind(maxEdges, c(maxInd[i], i))
  #     maxW <- rbind(maxW, maxV[i])
  #   }
  # }
  lm.ind <- which(maxV == sapply(maxInd, function(x) max(cMatrix[x,])))
  maxEdges <- cbind(maxInd[lm.ind], lm.ind)
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
  cMatrix <- cor(t(data_in), method = CCmethod)
  diag(cMatrix) <- 0
  
  if (positiveCorrelation){
    cMatrix <- abs(cMatrix)
  }
  
  if(normalization){
    # Normalization
    D <- rowSums(cMatrix)
    D.half <- 1/sqrt(D)
    
    cMatrix <- apply(cMatrix, 2, function(x) x*D.half )
    cMatrix <- t(apply(cMatrix, 1, function(x) x*D.half ))
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

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("Biobase")


library(Biobase)


install.packages("matrixStats")

library("matrixStats")


#data <- data(sample.ExpressionSet)
#data = assayData(sample.ExpressionSet)$exprs
#lmQCM(data)
#print(data)

data <- read.csv("/Users/marlon/DataspellProjects/MuVAEProject/MuVAE/TCGAData/RPPA_for_r.csv")
data = subset(data, select= -c(X)) # remove added X column 
matrix <- t(as.matrix(data))
class(matrix) <- "numeric"
eset <- new("ExpressionSet", expr = matrix)
data_new = assayData(eset)$exprs

print(data_new)
#lmQCM(data_new)


