
library("Hmisc")
library(corrplot)

# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients: r-values
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}



SIGT_fitted_data <- read.csv("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/1_SIGT_PP/Data/2_Fitted/SIGT_fitted_data.csv")
names(SIGT_fitted_data)


IGT__SIGT_v1 <- rcorr((as.matrix(SIGT_fitted_data[ ,c(2:11)])), type = c("pearson"))  ## rcorr will create both r-values & p-values
IGT__SIGT_v1 <- flattenCorrMatrix(IGT__SIGT_v1$r, IGT__SIGT_v1$P)





