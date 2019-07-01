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
import file_names
### right now this is not doing Any vocabularies, just straight-up features
def get_ps(modname):
    psmod = pickle.load(open(modname,'rb'))
    settings = list(psmod['preds'].keys())
    modsetting = settings[0] if len(settings) == 1 else psmod['xval'].mean(axis=1).idxmax() 
    return pd.Series(psmod['preds'][modsetting],index=psmod['ids'])

def runctl_psmatch(hisdir, ctl, drugid,trt_to_exclude, calipers_psmatch, outcome_ord, alpha, l1, vocab2superft={},single_outcomes=True,idfile_name="PSM", do_ebm=True): ##comm_ids will be allready removed from ctl via first run
    #print("runctl:--",hisdir)

    
    #pair_allid = pair_allid_prefix + pairname
    #pairname = file_names.get_savename(drugid, ctl)
    runname, trtname = file_names.get_trt_names(hisdir, drugid)
    pairname = runname + str(ctl)
    #pair_allid = pairname + "." + idfile_name #pair_allid_prefix + pairname

    #ctlbins = get_binned_ids(hisdir + savename + "binembeds.pytab")
    ### FIrst step: overall PS matching
    ##  - across all bins, get PS
    ##  - then, after this, match PS within bins

    #outcome_ord = sorted(outcomes.keys())

    #### this does the matching, given a PS (2 different PS constructions)
    #### then it gets the PS distr of the matched populations

    def match_eval(psmod, save_prefix):
        nmatch = match_ps(hisdir, ctl, drugid, calipers_psmatch, psmod,
                               outcome_ord, single_outcomes, save_prefix)

        for psk in calipers_psmatch:

            saven = ".".join([pairname, save_prefix, str(psk)])
            #saven = file_names.get_savename_prefix(pairname, save_prefix + str(psk))            
            if single_outcomes:
                f, xval = ps.ctl_propensity_score(hisdir,drugid, ctl,saven + ".ids",alphas=alpha, l1s=l1)
            else:
                ### now get PS -- for each outcome            
                ### for smallest outcome, do cross-validation:
                smallest = pd.Series(nmatch).idxmin()
                osaves = [".".join([saven, oname, "ids"])
                          for oname in outcome_ord]
                f, xval = ps.ctl_propensity_score(hisdir,drugid,ctl,osaves[smallest],alphas=alpha, l1s=l1)
                x = xval.mean(axis=1).idxmax().split("-")
                l1s_do = [float(x[0])]
                alph_do = [float(x[1])]
                for ix, osave in enumerate(osaves):
                    xval = ps.ctl_propensity_score(hisdir,drugid, ctl,osave,alphas=alph_do, l1s=l1s_do)
            weights_outcomes.outcome_info(hisdir, saven, drugid, ctl, outcome_ord, trt_to_exclude, single_outcomes=single_outcomes)            
    ## removing Superft, see ps_match2 for the code
        
    ### "sparsematched"
    match_eval(get_ps(file_names.get_ps_file(pairname, idfile_name)),
               idfile_name + "." + "spm") #+ str(psmatch) + ".spm." + pairname)

    if do_ebm:
        ### now, match on a PS of emb
        match_eval(runctl_psmatch_emb(hisdir,trtname,pairname, drugid,
                                      trt_to_exclude, pair_allid),
                   pair_allid_prefix + str(psmatch) + ".ebm." + pairname)
        
def match_ps(hisdir,ctl, trt, calipers_psmatch, psmod, ordered_outcomes, single_outcomes,save_prefix):
    
    ### match on PS
    runname, trtname = file_names.get_trt_names(hisdir, trt)
    pairname = runname + str(ctl) #mixed_histories/Target.4904/4410
    #pairname + save_prefix + 
    did_match = os.path.exists(".".join([pairname, save_prefix, str(calipers_psmatch[0]), "ids.trt"]))
    nmatch = {}
    osaves = [".".join([pairname, save_prefix, str(calipers_psmatch[0]), oname, "ids"]) for oname in ordered_outcomes]
    if not single_outcomes:
        did_match = True
        for ix, osave in enumerate(osaves):
            if not os.path.exists(osave + ".trt"):
                did_match = False
                break
            else:
                nmatch[ix] = np.loadtxt(hisdir + osave + ".trt").shape[0]
    if did_match:
        return nmatch
    #matches = match_ps(hisdir, pairname, drugid,trtname, psmatch, psmod,
    #                       outcome_ord, single_outcomes)
    
    trtinfo = his2ft.get_trt_info(trtname, trt)
    ## only need this to get the bins to patients
    ctlbins = tables.open_file(his2ft.gen_embname(pairname) ,mode="r") #.replace("PSM" + str(psmatch),"")
    #ctloutc = tables.open_file(his2ft.gen_outcname(hisdir, ctlfile) ,mode="r") #.replace("PSM" + str(psmatch),"")    
    
    bin_caliper = {psk:psk*psmod.var() for psk in calipers_psmatch}
    ix = 0
    scored_ids = psmod.index
    print("DOING:",trtinfo['bindf'].shape[0])
    tc_matches = {k:[[],[]] for k in calipers_psmatch}

    if not single_outcomes:
        tc_matches = {k:{o:[[],[]] for o in range(len(ordered_outcomes))}
                      for k in calipers_psmatch}
    tbase = time.time()
    BTS = {}
    accBTS = {}    
    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        if not "/" + binid in ctlbins:
            continue
        ta = time.time()
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        tsel = np.isin(trt_compare.index,scored_ids)
        trt_ids = trt_compare.index[tsel]
        trt_compare = psmod.loc[trt_ids]

        if trt_compare.shape[0] == 0:
            continue
        #trt_compare = psmod.loc[set(trt_compare.index) & scored_ids]


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
        match_by_cal = {}
        for psk in calipers_psmatch:
            if psk > 1:
                if len(trt_ids) > 1:
                    #pdb.set_trace()
                    other_treated = np.tile(trt_compare.transpose(),
                                        (trt_compare.shape[0],1)).transpose()
                    bin_caliper[psk] = np.percentile(np.abs(pd_helper.upper_tri(trt_compare.values - other_treated)),psk)
                else:
                    bin_caliper[psk] = psmod.var()
            
            match_by_cal[psk] = matching.one_bin_PSM(trt_compare, ctldat, bin_caliper[psk],binid)
        #pdb.set_trace() #     x = [len(m[0]) for m in match_by_cal.values()]       
        if single_outcomes:
            for psk in bin_caliper:
                tc_matches[psk][0] += match_by_cal[psk][0]
                tc_matches[psk][1] += match_by_cal[psk][1]
        else:
            #pdb.set_trace()
            trtoutc = trtoutc[tsel]            
            ctloutc_bin = ctlbins.get_node("/outcomes/" + binid)[csel]
            to_exclude = matching.out_exclude(trt_compare, trtoutc, ctlids, ctloutc_bin)
            #pdb.set_trace()

            for i in range(len(ordered_outcomes)):
                #if i == 7:
                #    pdb.set_trace()
                #for outcome, toexcl in to_exclude.items():
                if not i in to_exclude:
                    for psk in bin_caliper:
                        tc_matches[psk][i][0] += match_by_cal[psk][0]
                        tc_matches[psk][i][1] += match_by_cal[psk][1]
                else:
                    for psk in bin_caliper:
                        t,c = matching.one_bin_PSM(trt_compare.loc[~np.isin(trt_compare.index,to_exclude[i])],
                                             ctldat.loc[~np.isin(ctldat.index,to_exclude[i])],
                                            bin_caliper[psk],binid)
                        if len(t) > 0:
                            tc_matches[psk][i][0] += t
                            tc_matches[psk][i][1] += c
        ix += 1
        if ix % 50 == 0:
            print('{:d} have {:d}\n'.format(ix,
                                            len(list(tc_matches.values())[0][0]) if single_outcomes else len(list(tc_matches.values())[0][0][0]))) ## zeroth ps, zeroth outcome, t
    BTS['tot'] = time.time() - tbase
    #pdb.set_trace()
    f = open("BTS.pkl",'wb')
    pickle.dump(BTS, f)
    f.close()
    #pdb.set_trace()
    if single_outcomes:
        for psk, matches in tc_matches.items():
            saveto = ".".join([pairname, save_prefix, str(psk), "ids"])
            np.savetxt(saveto + ".trt", matches[0])        
            np.savetxt(saveto + ".ctl", matches[1])
    else:
        for psk, matches in tc_matches.items():
            osaves = [".".join([pairname, save_prefix, str(psk), oname, "ids"]) for oname in ordered_outcomes]        
            for ix, osave in enumerate(osaves):
                nmatch[ix] = len(matches[ix][0])
                np.savetxt(osave + ".trt", matches[ix][0])        
                np.savetxt(osave + ".ctl", matches[ix][1])

        del matches


    return nmatch
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
