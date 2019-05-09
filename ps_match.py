import tables
import sys
sys.path.append("../../code")
import bin_match
import pandas as pd
import numpy as np
import pickle
import pdb
import csv
import time
from collections import Counter, defaultdict
from scipy import sparse
import os
import tables
import glob
import subprocess
import his2ft #h2b_vi2clid as h2b
import caliper2 as caliper
import matching  #match_in_bin
import ps
import pd_helper
import regression_splines as rs
from sklearn.metrics import roc_auc_score
import weights_outcomes
### right now this is not doing Any vocabularies, just straight-up features
def get_ps(modname):
    psmod = pickle.load(open(modname,'rb'))
    settings = list(psmod['preds'].keys())
    modsetting = settings[0] if len(settings) == 1 else psmod['xval'].mean(axis=1).idxmax() 
    return pd.Series(psmod['preds'][modsetting],index=psmod['ids'])

def runctl_psmatch(hisdir, trtname, pairname, drugid,trt_to_exclude, psmatch, outcomes, alpha, l1, vocab2superft={}): ##comm_ids will be allready removed from ctl via first run
    #print("runctl:--",hisdir)

    
    pair_allid = "PSM" + pairname
    #ctlbins = get_binned_ids(hisdir + savename + "binembeds.pytab")
    ### FIrst step: overall PS matching
    ##  - across all bins, get PS
    ##  - then, after this, match PS within bins

    outcome_ord = sorted(outcomes.keys())

    #### this does the matching, given a PS (2 different PS constructions)
    #### then it gets the PS distr of the matched populations
    def match_eval(psmod, savename):
        osaves = [savename + "." + oname + ".ids" for oname in outcome_ord]
        ### match on PS
        did_match = True
        nmatch = {}                
        for ix, osave in enumerate(osaves):
            if not os.path.exists(hisdir + osave + ".trt"):
                did_match = False
                break
            else:
                nmatch[ix] = np.loadtxt(hisdir + osave + ".trt").shape[0]
        #pdb.set_trace()

        if not did_match:
            outc_match = match_ps(hisdir, pairname, drugid,trtname, psmatch, psmod,
                                    outcome_ord)
            for ix, osave in enumerate(osaves):
                nmatch[ix] = len(outc_match[ix][0])
                np.savetxt(hisdir + osave + ".trt", outc_match[ix][0])        
                np.savetxt(hisdir + osave + ".ctl", outc_match[ix][1])
            
            del outc_match

        ### now get PS -- for each outcome
        #pdb.set_trace()        
        ### for smallest outcome, do cross-validation:        
        smallest = pd.Series(nmatch).idxmin()
        f, xval = ps.ctl_propensity_score(hisdir,trtname, pair_allid,osaves[smallest],['ago'],transfunc=ps.agobins,alphas=alpha, l1s=l1)
        x = xval.mean(axis=1).idxmax().split("-")
        l1s_do = [float(x[0])]
        alph_do = [float(x[1])]

        for ix, osave in enumerate(osaves):
            xval = ps.ctl_propensity_score(hisdir,trtname, pair_allid,osave,['ago'],transfunc=ps.agobins,alphas=alph_do, l1s=l1s_do)
        weights_outcomes.outcome_info(hisdir, savename, trtname, drugid, pairname, outcome_ord, trt_to_exclude)            
    ## removing Superft, see ps_match2 for the code
        
    ### "sparsematched"
    match_eval(get_ps(hisdir + pair_allid + ".ids.psmod.pkl"),
               "PSM" + str(psmatch) + ".spm." + pairname)

    ### now, match on a PS of emb
    match_eval(runctl_psmatch_emb(hisdir,trtname,pairname, drugid,
                                  trt_to_exclude, pair_allid),
               "PSM" + str(psmatch) + ".ebm." + pairname)
'''    
    ### match on PS
    psmod = get_ps(hisdir + pair_allid + ".ids.psmod.pkl")
    savename =  "PSM" + str(psmatch) + ".spm." + pairname
    
    if not os.path.exists(hisdir + savename + ".ids.trt"):
        trtid, ctlid = match_ps(hisdir, pairname, drugid,trtname, psmatch, psmod)
        np.savetxt(hisdir + savename + ".ids.trt", trtid)        
        np.savetxt(hisdir + savename + ".ids.ctl", ctlid)
    ### evaluate using pre-matched feature files (trtname, pair_allid), filter with savename
    ps.ctl_propensity_score(hisdir,trtname, pair_allid,savename + '.ids',['ago'],transfunc=control_match3.agobins,alphas=alpha, l1s=l1)
    ## removing Superft, see ps_match2 for the code    
    psmod = runctl_psmatch_emb(hisdir,trtname,pairname, drugid,
                                trt_to_exclude, pair_allid)
    ### match on PS
    savename =  "PSM" + str(psmatch) + ".spm." + pairname            
    if not os.path.exists(hisdir + savename + ".ids.trt"):
        trtid, ctlid = match_ps(hisdir, pairname, drugid,trtname, psmatch, psmod)
        np.savetxt(hisdir + savename + ".ids.trt", trtid)        
        np.savetxt(hisdir + savename + ".ids.ctl", ctlid)
    ps.ctl_propensity_score(hisdir,trtname, pair_allid,savename + '.ids',['ago'],transfunc=control_match3.agobins,alphas=alpha, l1s=l1)        
'''
        
def match_ps(hisdir,ctlfile, trt, trtname,psmatch, psmod, ordered_outcomes):

    trtinfo = his2ft.get_trt_info(hisdir + trtname, trt)
    ## only need this to get the bins to patients
    ctlbins = tables.open_file(his2ft.gen_embname(hisdir, ctlfile) ,mode="r") #.replace("PSM" + str(psmatch),"")
    #ctloutc = tables.open_file(his2ft.gen_outcname(hisdir, ctlfile) ,mode="r") #.replace("PSM" + str(psmatch),"")    
    
    TRTID = []
    CTLID = []
    caliper = psmatch*psmod.var()
    ix = 0
    scored_ids = psmod.index
    print("DOING:",trtinfo['bindf'].shape[0])
    outcome_matches = {o:[[],[]] for o in range(len(ordered_outcomes))}
    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        if not "/" + binid in ctlbins:
            continue
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        #pdb.set_trace()
        #node = trtinfo['drugbins'].get_node("/" + binid)
        #trtids = set(node[:,0]) & scored_ids
        tsel = np.isin(trt_compare.index,scored_ids)
        trt_ids = trt_compare.index[tsel]
        trt_compare = psmod.loc[trt_ids]
        trtoutc = trtoutc[tsel]

        if trt_compare.shape[0] == 0:
            continue
        #trt_compare = psmod.loc[set(trt_compare.index) & scored_ids]
        if psmatch > 1:
            if len(trt_ids) > 1:
                other_treated = np.tile(trt_compare.transpose(),
                                    (trt_compare.shape[0],1)).transpose()
                caliper = np.percentile(np.abs(pd_helper.upper_tri(trt_compare.values - other_treated)),psmatch)
            else:
                caliper = psmod.var()

        ctldat = ctlbins.get_node("/" + binid)
        csel = np.isin(ctldat[:,0], scored_ids)
        ctlids = ctldat[:,0][csel]
        if len(ctlids) != ctldat.shape[0]:
            pdb.set_trace()
        ctldat = psmod.loc[ctlids]
        ctloutc_bin = ctldat.get_node("/outcomes/" + binid)[csel]
        to_exclude = matching.out_exclude(trt_compare, trtoutc, ctlids, ctloutc_bin)
        #pdb.set_trace()
        tall,call = matching.one_bin_PSM(trt_compare,
                                         ctldat,
                                     caliper,binid)
        
        for i in range(len(ordered_outcomes)):
            #if i == 7:
            #    pdb.set_trace()
            #for outcome, toexcl in to_exclude.items():
            if not i in to_exclude:
                outcome_matches[i][0] += tall
                outcome_matches[i][1] += call
            else:
                t,c = matching.one_bin_PSM(trt_compare.loc[~np.isin(trt_compare.index,to_exclude[i])],
                                         ctldat.loc[~np.isin(ctldat.index,to_exclude[i])],
                                         caliper,binid)
                if len(t) > 0:
                    outcome_matches[i][0] += t
                    outcome_matches[i][1] += c
            #if outcome==6: # and len(outcome_matches[outcome-1][0]) > 2: #min(trt_compare.shape[0],ctldat.shape[0]) > 5 and len(t) == 0:
            #    pdb.set_trace()
            #    print(outcome)
        #pdb.set_trace()
        ix += 1
        if ix % 50 == 0:
            print('{:d} have {:d}\n'.format(ix,len(outcome_matches[0][0])))

    #pdb.set_trace()
            
    return outcome_matches
    #ctl_propensity_score(hisdir,name, ctlfile,outname)


    featlab2modpred(dense, hisft, lab, alphas, l1s)
#def runctl_psmatch_emb(hisdir, trtname, savename, coxoutc, coxfilt, outc,emb50,ctlf,drugid,psmatch):
def runctl_psmatch_emb(hisdir, trtname, savename, trt, trt_to_exclude,idname, alphas=10**np.arange(-4.0,-1.0),l1s=[.2,.3,.4]): ##comm_ids = in trt & ctl but ctl will have these removed from binembeds
    print("runctl:--",hisdir)
    psmod_savefile = hisdir + idname + ".embedps.pkl"
    if os.path.exists(psmod_savefile):
        mod, psm = pickle.load(open(psmod_savefile,'rb'))
        return psm
    '''
    trtdense = pd.read_csv(hisdir + trtname + ".den",sep="\t",header=None)

    spline_info = rs.get_spline_info(trtdense[:,1:])
    splinify = rs.info2splines(spline_info)
    trtdense = pd.DataFrame(trtdense[:,1:],index=trtdense[0])
    ctldense = pd.read_csv(hisdir + savename + ".den",sep="\t",header=None,index_col=0)
    '''
    ## fit PS models on everyone in the same bins
    ctlbins = tables.open_file(his2ft.gen_embname(hisdir, savename) ,mode="r")
    #trtbins = tables.open_file(hisdir + trtname +"binembeds.pytab" ,mode="r")
    trtinfo = his2ft.get_trt_info(hisdir + trtname, trt)
    mods = rs.make_mods(3, alphas,l1s,class_weight=None)
    #ntrt = np.loadtxt(hisdir + "PSM" + str(psmatch) + savename + ".ids.trt").shape[0]

    ### making weights: because we are doing partial_fit
    ntrt = np.loadtxt(hisdir + idname + ".ids.trt").shape[0]    
    nctl = np.loadtxt(hisdir + idname + ".ids.ctl").shape[0]
    # class_weight = {int(i):tot_dat/(len(class_counts)*class_counts[i]) for i in class_counts}
    # sklearn doc: n_samples / (n_classes * np.bincount(y))
    class_weight = {i:(nctl + ntrt)/(2*ct) for i,ct in enumerate([nctl, ntrt])}

    def get_x(binid):
        if not "/" + binid in ctlbins:
            return None
        #trt_compare, ctldat, cutoff = match_in_bin.binfo(trtinfo, binid)
        node = trtinfo['drugbins'].get_node("/" + binid)
        ids = node[:,0]
        sel = ~np.isin(ids,trt_to_exclude)
        ids = ids[sel]
        if ids.shape[0] == 0:
            return None
        #trt_compare = node[sel,:]
        #trtdem = splinify(trt_compare[:,:6])
        trt_compare = trtinfo['scaler'].transform(node[:,6:][sel,:])
        #trt_compare, trtoutc = matching.binfo(trtinfo, binid)

        ctldat = ctlbins.get_node("/" + binid)
        #ctldem = splinify([:,:6])

        ### compile together to return
        lab = np.append(np.ones(trt_compare.shape[0]),np.zeros(ctldat.shape[0]))
        ids = np.hstack([ids, ctldat[:,0]])
        return np.vstack((trt_compare, trtinfo['scaler'].transform(ctldat[:,6:]))), lab, ids
         
    

    ### fit models
    desired_iter = 5000000    
    iter = 0
    numfold = 5    
    while iter < desired_iter:
        #print("DOING:",trtinfo['bindf'].shape[0])
        for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
            X = get_x(binid)
            if not X:
                continue
            X, lab, ids = X
            sel = np.ones(lab.shape[0])==1
            if len(mods) > 1:
                splits = rs.get_splitvec(lab.shape[0], nf = numfold)
                sel = splits != numfold-1
            sample_weight = lab*class_weight[1] + (1-lab)*class_weight[0]
            for m in mods:
                mods[m].partial_fit(X[sel], lab[sel],classes=[0,1],
                                    sample_weight=sample_weight[sel])
            iter += (splits != numfold-1).sum()

    ### get preds to select model
    opt = list(mods.keys())[0]
    if len(mods) > 1:
        pred_test = {m:[] for m in mods}
        lab_test = []

        for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
            X = get_x(binid)
            if not X:
                continue
            X, lab, ids = X

            splits = rs.get_splitvec(lab.shape[0], nf = numfold)
            sel = splits == numfold-1

            if sel.sum() == 0:
                continue
            lab_test.append(lab[sel])        
            for m in mods:
                pred_test[m].append(mods[m].predict_proba(X[sel])[:,1])
        lab_test = np.hstack(lab_test)
        rocs = {m:roc_auc_score(lab_test, np.hstack(pred_test[m])) for m in mods}
        #if True:
        #    return mods, pred_test, lab_test, rocs

        opt = pd.Series(rocs).idxmax()

    ### make PS from my model
    ids = []
    pscore = []
    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        X = get_x(binid)
        if not X:
            continue
        X, lab, binids = X
        pscore.append(mods[opt].predict_proba(X)[:,1]     )
        ids.append(binids)
    pscore = pd.Series(np.hstack(pscore),index=np.hstack(ids))
    f = open(psmod_savefile,'wb')
    pickle.dump((mods, pscore),f)
    f.close()
    return pscore 
