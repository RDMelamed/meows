.libPaths(c("/home/melamed/bin/R"))
source("/project2/melamed/wrk/iptw/code/matchweight/coxeff.R")
args = commandArgs(trailingOnly=TRUE)
if(args[2] == "False"){
    xval(args[1])
}else{
    singleoutcome(args[1])
    }

