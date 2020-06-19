.libPaths(c("/home/melamed/bin/R"))
source("/project2/melamed/wrk/iptw/code/matchweight/coxeff.R")
source("/project2/melamed/wrk/iptw/code/matchweight/msm_surv.R")
args = commandArgs(trailingOnly=TRUE)
if(args[2] == "multi_cross_section"){
    xval(args[1], args[3]=="True")
}else if(args[2] == "single_cross_section"){
    singleoutcome(args[1], args[3]=="True")
}else{
    cat("temporal:" ,args[1], ' ', args[4],' ', args[3],"\n")
    effcut(args[1],args[4], args[3], args[5])
    }

