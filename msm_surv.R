library(survey)
library(survival)


effcut <- function(pattern, treatment="treat", suff="X",weightprefix=""){
    tc = c(".ctl",".trt")
    x = paste("pattern='",pattern,"'; treatment= '",treatment, "'; suff='",suff,"'; weightprefix='", weightprefix,"'\n",sep="")
    cat(x)
    if (suff != "X"){
        tc = paste(tc, suff,sep="")
    }else{
        suff = ""
    }
    weightcol = "cum_wt"
    if(weightprefix != "X"){
        weightcol = paste(weightprefix,weightcol, sep=".")
        }
    cat("going:",pattern, suff,   weightcol)
    dwt = data.frame()
    for (i in 1:2){
        cat("reading ", paste(pattern,tc[i], '.censwt.bz2',sep=""),"\n")
        dat = read.table(paste(pattern,tc[i], '.censwt.bz2',sep=""),header=T,sep="\t",row.names = 1)
        dat$treat = i - 1
        dwt = rbind(dwt, dat)
    }
    dwt[dwt$treat==0,'as_treated'] = 0
    astreated = dwt[dwt$treat==1,'as_treated']
    top = quantile(astreated, .8)
    astreated = ifelse(astreated > top , top, astreated)
    astreated = (top- astreated)/(top-1) + 1  ## treated go from 1 to 2
    #astreated = (top-astreated)/(top-1)
    #astreated = astreated/mean(astreated)/2 + .5 ## treated have mean of 1
    dwt[dwt$treat==1,'as_treated'] =  astreated #ifelse(astreated > top , top, astreated)/top    
    if(treatment=="as_treated"){
        suff = paste(suff, "as_treated.06.11",sep=".")
        dwt$treat = dwt$as_treated
    }
    cat(" pattern:", pattern, "  top=",top,"\n")

    intervalcol = colnames(dwt)[1:9]
    intervalcol = c(intervalcol, "treat","as_treated")    
    cancers = setdiff(colnames(dwt), intervalcol) #colnames(dwt)[9:ncol(dwt)]
    cutz = c(1,3,5 ,10) 
                                        #gecol = c("survco","survse") #,"svy","svyse")
    resdfs = list()
    effsave = paste(pattern, suff, ".msm",sep="")
    for(stat in c("eff","mse")){
        fi =  paste(effsave,stat,".txt",sep="")
        if(file.exists(fi)){
            cat("reading:",fi)
            resdfs[[stat]] = read.table(fi) #paste(effsave,stat,".txt",sep=""))
        }else{
            cat("not  reading",fi)
                resdfs[[stat]] = data.frame(matrix(nrow=length(cutz),
                                           ncol = length(cancers)),
                                    row.names=as.character(cutz))
                colnames(resdfs[[stat]]) = cancers
               }
    }
    ##  dwt = dwt[dwt$censored==0,]
    #bv2 = data.frame(matrix(nrow=length(cutz), ncol = length(gecol)),row.names=as.character(cutz))
    #colnames(bv2) = gecol
    nc = colSums(dwt[,cancers]==1)
    nas = is.na(dwt['cum_wt'])
    cat("\ntotal na=",sum(nas)," ids = ",dat[nas,'ids']," wtcol=",weightcol,"\n")
    write.table(nc,paste(pattern,suff,".N.txt",sep=""))
    for(cancer in cancers){
                                        #cancer = "Breast_Cancer"
        #cat("\n",cancer)
        if(sum(dwt[,cancer]==1) < 20){
            next
        }
        
        cancdat = dwt[dwt[,cancer] >= 0, c(intervalcol, cancer)]

        cancdat$interval_start = cancdat$interval_start - 1
        colnames(cancdat)[colnames(cancdat)==cancer] = "cancer"

        for (cu in cutz){
            ins = as.character(cu)
            if(!is.na(resdfs[['eff']][ins,cancer])){
                #cat("skipping ", cancer, cu)
                next
            }
            
            cancdat$clipwt = ifelse(cancdat[,weightcol] > cu , cu,
                                    cancdat[,weightcol])
            #cat("\ntotal na=",sum(is.na(cancdat[,weightcol]))," clipped =",  sum(is.na(cancdat$clipwt))," ids = ",dat[is.na(cancdat$clipwt),'ids']," wtcol=",weightcol,"\n")
            fit = 0
            if (cu ==  1){
                fit  = coxph(Surv(interval_start, interval_end, cancer) ~ treat + cluster(ids),
                                     data = cancdat)
            }else{
                fit  = coxph(Surv(interval_start, interval_end, cancer) ~ treat + cluster(ids),
                             data = cancdat, weights = clipwt)
                }
        #binsvymodel=svyglm(cancer ~ treat, family=quasibinomial,design=svydesign(~1,weights= ~clipwt, data=cancdat))
        #toins = c(coef(fit), SE(fit),
        #          coef(binsvymodel)['treat'], SE(binsvymodel)['treat'])
        
        #cat(ins," ")
        resdfs[["eff"]][ins, cancer] = coef(fit)
        resdfs[["mse"]][ins, cancer] = SE(fit)
        #bv2[ins,] = toins
        #rr2[[ins]] = toins
    }
        write.table(resdfs[['eff']],paste(effsave, "eff.txt",sep=""),sep="\t")
        write.table(resdfs[['mse']],paste(effsave, "mse.txt",sep=""),sep="\t")        
        #write.table(bv2,paste(pattern, suff, ".msmeff.txt",sep=""),sep="\t")
        
    }
    cat("\nFINISHED",effsave,"\n")
}

#effcut("0.4-0.001.",suff="R")
bootit <- function(cancdat){
cu  = 20
cancdat$clipwt = ifelse(cancdat$cum_wt > cu , cu, cancdat$cum_wt)
nbootstrap = 100
xx = data.frame(matrix(nrow=3, ncol=nbootstrap),
                row.names = c("eff","se","ncancer"))
uids =  unique(cancdat$ids)
for (b in 1:nbootstrap){
    boot = sample(uids,length(uids),replace=T)
    bdat = cancdat[cancdat$ids %in% boot,]
    bcount = table(boot)
    for(i in 2:max(bcount)){
        newi  = cancdat[cancdat$ids %in% names(bcount[bcount >=  i]),]
        newi$ids = newi$ids  +  i/max(bcount)
        bdat = rbind(bdat, newi)
        }
    fit  = coxph(Surv(interval_start, interval_end, cancer) ~ treat + cluster(ids),
               data = bdat, weights = clipwt)
    xx[,b] = c(coef(fit), SE(fit), sum(bdat$cancer))
}
write.table(t(xx),"booteff.txt",sep="\t")
}
