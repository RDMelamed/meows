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

def runctl_psmatch(hisdir, trtname, pairname, drugid,trt_to_exclude, psmatch, outcomes, alpha, l1, vocab2superft={},single_outcomes=True,pair_allid_prefix="PSM", do_ebm=True): ##comm_ids will be allready removed from ctl via first run
    #print("runctl:--",hisdir)

    
    pair_allid = pair_allid_prefix + pairname
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
        did_match = os.path.exists(hisdir + savename + "BLAH.ids.trt")
        nmatch = {}
        if not single_outcomes:
            did_match = True
            for ix, osave in enumerate(osaves):
                if not os.path.exists(hisdir + osave + ".trt"):
                    did_match = False
                    break
                else:
                    nmatch[ix] = np.loadtxt(hisdir + osave + ".trt").shape[0]


        if not did_match:
            matches = match_ps(hisdir, pairname, drugid,trtname, psmatch, psmod,
                                    outcome_ord, single_outcomes)
            pdb.set_trace()                    
            if single_outcomes:
                np.savetxt(hisdir + savename + ".ids.trt", matches[0])        
                np.savetxt(hisdir + savename + ".ids.ctl", matches[1])
            else:
                for ix, osave in enumerate(osaves):
                    nmatch[ix] = len(matches[ix][0])
                    np.savetxt(hisdir + osave + ".trt", matches[ix][0])        
                    np.savetxt(hisdir + osave + ".ctl", matches[ix][1])

                del matches


        if single_outcomes:
            f, xval = ps.ctl_propensity_score(hisdir,trtname, pairname,savename + ".ids",alphas=alpha, l1s=l1)
        else:
            ### now get PS -- for each outcome            
            ### for smallest outcome, do cross-validation:
            smallest = pd.Series(nmatch).idxmin()
            f, xval = ps.ctl_propensity_score(hisdir,trtname, pairname,osaves[smallest],alphas=alpha, l1s=l1)
            x = xval.mean(axis=1).idxmax().split("-")
            l1s_do = [float(x[0])]
            alph_do = [float(x[1])]
            for ix, osave in enumerate(osaves):
                xval = ps.ctl_propensity_score(hisdir,trtname, pairname,osave,['ago'],transfunc=ps.agobins,alphas=alph_do, l1s=l1s_do)
        weights_outcomes.outcome_info(hisdir, savename, trtname, drugid, pairname, outcome_ord, trt_to_exclude, single_outcomes=single_outcomes)            
    ## removing Superft, see ps_match2 for the code
        
    ### "sparsematched"
    match_eval(get_ps(hisdir + pair_allid + ".ids.psmod.pkl"),
               pair_allid_prefix + str(psmatch) + ".spm." + pairname)

    if do_ebm:
        ### now, match on a PS of emb
        match_eval(runctl_psmatch_emb(hisdir,trtname,pairname, drugid,
                                      trt_to_exclude, pair_allid),
                   pair_allid_prefix + str(psmatch) + ".ebm." + pairname)
        
def match_ps(hisdir,ctlfile, trt, trtname,psmatch, psmod, ordered_outcomes, single_outcomes):

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
    tc_matches = [[],[]]
    #pdb.set_trace()
    if not single_outcomes:
        tc_matches = {o:[[],[]] for o in range(len(ordered_outcomes))}
    tbase = time.time()
    BTS = {}
    accBTS = {}    
    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        if not "/" + binid in ctlbins:
            continue
        ta = time.time()
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        #if binid == 't412':
        #    pdb.set_trace()
        #node = trtinfo['drugbins'].get_node("/" + binid)
        #trtids = set(node[:,0]) & scored_ids
        #if binid == "t60":
        #    pdb.set_trace()
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
        #if len(ctlids) != ctldat.shape[0]:
        #    pdb.set_trace()
        ctldat = psmod.loc[ctlids]
        accBTS[binid] = time.time() - ta
        if (time.time() - ta) > 30:
            pdb.set_trace()

        t0 = time.time()
        tall,call = matching.one_bin_PSM(trt_compare,
                                         ctldat,
                                     caliper,binid)
        
        t1 = time.time()
        BTS[binid] = t1 - t0
        if (t1 - t0) > 30:
            pdb.set_trace()
        #if len(BTS) > 100:
        #    break
        if single_outcomes:
            tc_matches[0] += tall
            tc_matches[1] += call            
        else:
            #pdb.set_trace()
            ctloutc_bin = ctlbins.get_node("/outcomes/" + binid)[csel]
            to_exclude = matching.out_exclude(trt_compare, trtoutc, ctlids, ctloutc_bin)
            #pdb.set_trace()

            for i in range(len(ordered_outcomes)):
                #if i == 7:
                #    pdb.set_trace()
                #for outcome, toexcl in to_exclude.items():
                if not i in to_exclude:
                    tc_matches[i][0] += tall
                    tc_matches[i][1] += call
                else:
                    t,c = matching.one_bin_PSM(trt_compare.loc[~np.isin(trt_compare.index,to_exclude[i])],
                                             ctldat.loc[~np.isin(ctldat.index,to_exclude[i])],
                                            caliper,binid)
                    if len(t) > 0:
                        tc_matches[i][0] += t
                        tc_matches[i][1] += c
        ix += 1
        if ix % 50 == 0:
            print('{:d} have {:d}\n'.format(ix,
                                            len(tc_matches[0]) if single_outcomes else len(tc_matches[0][0])))
    BTS['tot'] = time.time() - tbase
    #pdb.set_trace()
    f = open("BTS.pkl",'wb')
    pickle.dump(BTS, f)
    f.close()
    return tc_matches
    #ctl_propensity_score(hisdir,name, ctlfile,outname)



def runctl_psmatch_emb_nonpartial(hisdir, trtname, savename, trt, trt_to_exclude,idname, alphas=10**np.arange(-4.0,-1.0),l1s=[.2,.3,.4]): ##comm_ids = in trt & ctl but ctl will have these removed from binembeds
    print("runctl:--",hisdir)

    psmod_savefile = hisdir + idname + ".embedps.pkl"
    if os.path.exists(psmod_savefile):
        mod, psm, xval = pickle.load(open(psmod_savefile,'rb'))
        return psm
    ## fit PS models on everyone in the same bins
    ctlbins = tables.open_file(his2ft.gen_embname(hisdir, savename) ,mode="r")
    #trtbins = tables.open_file(hisdir + trtname +"binembeds.pytab" ,mode="r")
    trtinfo = his2ft.get_trt_info(hisdir + trtname, trt)
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
        demo = np.vstack((node[:,2:9][sel,:], ctldat[:,2:9]))
        return np.vstack((trt_compare,
                          trtinfo['scaler'].transform(ctldat[:,6:]))), lab, ids, demo

    Xs = []
    labs = []
    ids = []; demos = []
    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        X = get_x(binid)
        if not X:
            continue
        X, lab, xids, demo = X
        Xs.append(X); labs.append(lab); ids.append(xids); demos.append(demo)
    demos = np.vstack(demos); labs = np.hstack(labs)
    spline_info = rs.get_spline_info(demos[labs==1,:])    
    splinify = rs.info2splines(spline_info)
    demos = splinify(demos)
    Xs = np.hstack((demos, np.vstack(Xs)))
    del demos
    xval = rs.cross_val(Xs, labs, 5,iter=5000000, alphas=alphas, l1s=l1s)
    optmod = xval[4].mean(axis=1).idxmax()
    pscore = xval[3][optmod]
    pscore = pd.Series(xval[3][optmod],index=np.hstack(ids))
    #pscore = pd.Series(np.hstack(pscore),index=np.hstack(ids))
    #pdb.set_trace()    
    f = open(psmod_savefile,'wb')
    pickle.dump((xval[0], pscore, xval[4]),f)
    f.close()
    return pscore 



def runctl_psmatch_emb(hisdir, trtname, savename, trt, trt_to_exclude,idname, alphas=10**np.arange(-4.0,-1.0),l1s=[.2,.3,.4]): ##comm_ids = in trt & ctl but ctl will have these removed from binembeds
    print("runctl:--",hisdir)
    psmod_savefile = hisdir + idname + ".embedps.pkl"
    if os.path.exists(psmod_savefile):
        mod, psm, xval = pickle.load(open(psmod_savefile,'rb'))
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
    trtget = np.loadtxt(hisdir + idname + ".ids.trt") ### this would already have trt_to_exclude excluded
    ntrt = trtget.shape[0]
    ctlget = np.loadtxt(hisdir + idname + ".ids.ctl")
    nctl = ctlget.shape[0]
    # class_weight = {int(i):tot_dat/(len(class_counts)*class_counts[i]) for i in class_counts}
    # sklearn doc: n_samples / (n_classes * np.bincount(y))
    class_weight = {i:(nctl + ntrt)/(2*ct) for i,ct in enumerate([nctl, ntrt])}

    def get_x(binid):
        if not "/" + binid in ctlbins:
            return None
        #trt_compare, ctldat, cutoff = match_in_bin.binfo(trtinfo, binid)
        node = trtinfo['drugbins'].get_node("/" + binid)
        ids = node[:,0]
        #sel = ~np.isin(ids,trt_to_exclude)
        sel = np.isin(ids,trtget)        
        ids = ids[sel]
        if ids.shape[0] == 0:
            return None
        #trt_compare = node[sel,:]
        #trtdem = splinify(trt_compare[:,:6])
        trt_compare = trtinfo['scaler'].transform(node[:,6:][sel,:])
        #trt_compare, trtoutc = matching.binfo(trtinfo, binid)

        ctldat = ctlbins.get_node("/" + binid)
        ctldat = ctldat[:][np.isin(ctldat[:,0], ctlget),:]
        #ctldem = splinify([:,:6])

        ### compile together to return
        lab = np.append(np.ones(trt_compare.shape[0]),np.zeros(ctldat.shape[0]))
        ids = np.hstack([ids, ctldat[:,0]])
        return np.vstack((trt_compare, trtinfo['scaler'].transform(ctldat[:,6:]))), lab, ids
         
    

    ### fit models
    desired_iter = 15000000    
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
    rocs = {}
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
    pickle.dump((mods, pscore,rocs),f)
    f.close()
    return pscore 
