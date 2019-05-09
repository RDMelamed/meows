.libPaths(c("/home/melamed/bin/R"))
source("coxeff.R")
args = commandArgs(trailingOnly=TRUE)    
xval(args[1],args[2])

