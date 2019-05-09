library(survey)
library(survival)
weightedest <- function(weightdat){
        todo = data.frame(weightdat$lab, weightdat$id, weightdat$ipw, 
                          ifelse(weightdat$outcome > 0, weightdat$outcome, weightdat$deenroll),
                          weightdat$outcome > 0,
                          weightdat$deenroll)
        #cat("wtest-- ",dim(todo),"\n")
        colnames(todo) = c("lab","id","ipw","deathcens","death","deenroll")
        sfit3 <- coxph(Surv(deathcens, death) ~ lab + cluster(id), data=todo, weight=ipw)
        return( c(coef(sfit3), SE(sfit3)['lab']) )
}

xval <- function(pref,pattern){
    cat("coxeff ",pattern,"\n")
                                        #cat("making df:",ncol(outcAll)-3)
    todo = list.files(pref,pattern=paste(pattern,".*.iptw$",sep=""))
    outcomes = gsub(".iptw","",gsub(pattern,"",todo,fixed=T),fixed=T)
    est = data.frame(matrix(nrow=length(outcomes),ncol=4))
    dimnames(est) = list(outcomes, c("surv","survse","N","events"))
    outfile = paste(pref,pattern,"eff",sep="")
    cat(dim(est))
    outix = 1
    for (wtfile in todo){
        #cat("ONIT:",wtfile,"\n")
        x = read.table(paste(pref,wtfile,sep=""),sep="\t",header=T) #,row.names=0)
        est[outcomes[outix],] = c(weightedest(x),nrow(x), sum(x$outcome > 0))
        outix = outix + 1
        write.table(est, file=outfile)
    }
    write.table(est, file=outfile)
    #e = data.frame(e=exp(est$surv), l=exp(est$surv - 1.96*est$survse), u=exp(est$surv + 1.96*est$survse))
    }

bootstrapsurv <- function(outc, weightdat, savename, nbootstrap){
    startat = 2
    cat("making df:",ncol(outc)-(startat -1))
    cols = c("surv","survse")

    #xx = data.frame(matrix(nrow=ncol(outc)-(startat-1), ncol=length(cols)))
    outcnames = colnames(outc)[startat:ncol(outc)]
    colnames = 1:nbootstrap
    xx = list(surv=data.frame(matrix(nrow=length(outcnames), ncol=length(colnames)),
                                         row.names=outcnames),#,column.names=colnames),
                  se=data.frame(matrix(nrow=length(outcnames), ncol=length(colnames)),
                                         row.names=outcnames)) #,column.names=colnames))
    for (cn in colnames(outc)[startat:ncol(outc)]){
        cat(cn,"\n")
        todo = data.frame(weightdat$lab, weightdat$id, weightdat$ipw, 
                          ifelse(outc[,cn] > 0, outc[,cn], outc$deenroll),
                          outc[,cn] > 0,
                          outc$deenroll)

        colnames(todo) = c("lab","id","ipw","deathcens","death","deenroll")
        #if (cn=="Benign_Endocrine_Neoplasm"){
        #cat("tot outc = ",sum(todo$death))
        #            }
        
        for (b in colnames){
            boot = sample(nrow(todo),nrow(todo),replace=T)
            #if (cn=="Benign_Endocrine_Neoplasm"){
            #cat("n outc = ",sum(todo$death[boot]))
            #        }
            sfit3 <- coxph(Surv(deathcens, death) ~ lab + cluster(id),
                           data=todo[boot,], weight=ipw)

            xx$surv[cn,paste("X",b,sep="")] = coef(sfit3)
            xx$se[cn,paste("X",b,sep="")] = SE(sfit3)['lab']
           }
        write.table(xx$surv,file=paste(savename,"bootsurv.txt",sep="."))
        write.table(xx$se,file=paste(savename,"bootsurvse.txt",sep="."))
    }
}
    

oneboot <- function(wtfile){
    outcAll = read.table(gsub(".ipw",".outmat",wtfile),sep="\t",header=T,row.names="rowId")
    cat("GOTCHA:", wtfile, dim(outcAll),"\n")
    weightdat = read.table(wtfile,sep="\t",header = TRUE)
    bootstrapsurv(outcAll, weightdat,wtfile, 100)
    
    }
