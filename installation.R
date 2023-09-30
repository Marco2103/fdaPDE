system.file(package='Rcpp')
system.file(package='RcppEigen')

# load Rcpp library

library(Rcpp)

#setwd(paste0(getwd(), "/wrappers/R/"))

# update RCppExports.cpp
compileAttributes(".")

# install fdaPDE
install.packages(".", type="source", repos=NULL)
