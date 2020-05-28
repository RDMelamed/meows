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

def runctl_psmatch(hisdir, ctl, drugid,trt_to_exclude, calipers_psmatch, outcome_ord, alpha, l1, single_outcomes=True,idfile_name="PSM", ft_exclude=[], time_chunk=0): ##comm_ids will be allready removed from ctl via first run
    #print("runctl:--",hisdir)

    print("runctl_psmatch time_chunk=",time_chunk)    
    #pair_allid = pair_allid_prefix + pairname
    #pairname = file_names.get_savename(drugid, ctl)
    runname, trtname = file_names.get_trt_names(hisdir, drugid)
    pairname = runname + str(ctl)
    pair_allid = pairname + "." + idfile_name #pair_allid_prefix + pairname

    #ctlbins = get_binned_ids(hisdir + savename + "binembeds.pytab")
    ### FIrst step: overall PS matching
    ##  - across all bins, get PS
    ##  - then, after this, match PS within bins

    #### this does the matching, given a PS 
    #### then it gets the PS distr of the matched populations
    def match_eval(psmod, save_prefix):
        nmatch = match_ps(hisdir, ctl, drugid, calipers_psmatch, psmod,
                               outcome_ord, single_outcomes, save_prefix)

        for psk in calipers_psmatch:

            saven = ".".join([pairname, save_prefix, str(psk)])
            weights_outcomes.outcome_info(hisdir, saven, drugid, ctl, outcome_ord, trt_to_exclude, single_outcomes=single_outcomes, weighting = False,time_chunk=time_chunk)            

    match_eval(file_names.get_ps_file(pairname, idfile_name),
               idfile_name + "." + "spm") #+ str(psmatch) + ".spm." + pairname)

        
def match_ps_big(hisdir,ctl, trt, calipers_psmatch, psmod, ordered_outcomes,save_prefix):

    ### match on PS
    runname, trtname = file_names.get_trt_names(hisdir, trt)
    pairname = runname + str(ctl) #mixed_histories/Target.4904/4410
    #pairname + save_prefix + 
    did_match = os.path.exists(".".join([pairname, save_prefix, str(calipers_psmatch[0]), "ids.trt"]))
    nmatch = {}

    if did_match:
        return nmatch
    
    trtinfo = his2ft.get_trt_info(trtname, trt)
    ## only need this to get the bins to patients
    ctlbins = tables.open_file(his2ft.gen_embname(pairname) ,mode="r") #.replace("PSM" + str(psmatch),"")

    psmod = pickle.load(open(psmod ,'rb'))
    lab = psmod['lab']
    hyperparam = ps.get_optimal_settingn(psmod)
    optmod  = psmod['mods'][hyperparam]
    #psmod = ps.get_ps(psmod)
    bin_caliper = {psk:psk*psmod['preds'][hyperparam].var() for psk in calipers_psmatch}
    #mod_pred = 
    ix = 0
    #scored_ids = psmod.index
    scored_trt = psmod.index[lab==1]
    scored_ctl = psmod.index[lab==0]    
    print("DOING:",trtinfo['bindf'].shape[0])
    tc_matches = {k:[[],[]] for k in calipers_psmatch}

    tbase = time.time()
    trt_h5 = tables.open_file(file_names.sparseh5_names(hisdir, trt),'r')
    ctl_h5 = tables.open_file(file_names.sparseh5_names(hisdir, ctl),'r')
    sparse_index = ps.get_sparseindex(hisdir, trt)
    BTS = {}
    accBTS = {}
    curbins = []

    it = iter(trtinfo['bindf']['binid'])
    BIGNESS = 300000
    stopped = False
    while not stopped:
        tnum = 0; cnum = 0
        trtagg = [];  ctlagg = []
        
        while tnum < BIGNESS and cnum <  BIGNESS:
            try:
                binid = next(it)
                trt_compare, trtoutc = matching.binfo(trtinfo, binid)
                if trt_compareshape[0] == 0:
                    continue
                trtagg.append(trt_compare.index)
                ctldat = ctlbins.get_node("/" + binid)

                ctlagg.append(ctldat[:,0])
            except StopIteration:
                stopped = True
                break
            dense,  hisft, lab, ids = ps.load_stack(trt_h5, ctl_h5, sparse_index,
                                                list(chain.from_iterable(trtagg)), list(chain.from_iterable(ctlagg)))
            hisft = psmod['scaler'].transform(sparse.hstack((dense, hisft),format='csr'))
            del dense
            for bin_ids in zip(*tuple((trtagg, ctlagg))):
                
                hisft[np.isin(ids, binids[0]),:]

    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        if not "/" + binid in ctlbins:
            continue
        ta = time.time()
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        tsel = np.isin(trt_compare.index,scored_trt)
        trt_ids = trt_compare.index[tsel]
        trt_compare = psmod.loc[trt_ids]
        #if len(set(trt_ids) & set([  3780886,   1784601,   8838199,   1123636, 117297190])) > 0:
        #    pdb.set_trace()
        if trt_compare.shape[0] == 0:
            continue
        #trt_compare = psmod.loc[set(trt_compare.index) & scored_ids]


        ctldat = ctlbins.get_node("/" + binid)
        csel = np.isin(ctldat[:,0], scored_ctl)
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
        for psk in bin_caliper:
            tc_matches[psk][0] += match_by_cal[psk][0]
            tc_matches[psk][1] += match_by_cal[psk][1]
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
    for psk, matches in tc_matches.items():
        saveto = ".".join([pairname, save_prefix, str(psk), "ids"])
        np.savetxt(saveto + ".trt", matches[0])        
        np.savetxt(saveto + ".ctl", matches[1])

    return nmatch
    #ctl_propensity_score(hisdir,name, ctlfile,outname)


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
    psmod = pickle.load(open(psmod ,'rb'))
    lab = psmod['lab']
    psmod = ps.get_ps(psmod)
    bin_caliper = {psk:psk*psmod.var() for psk in calipers_psmatch}
    ix = 0
    #scored_ids = psmod.index
    scored_trt = psmod.index[lab==1]
    scored_ctl = psmod.index[lab==0]    
    print("DOING:",trtinfo['bindf'].shape[0])
    tc_matches = {k:[[],[]] for k in calipers_psmatch}

    if not single_outcomes:
        tc_matches = {k:{o:[[],[]] for o in range(len(ordered_outcomes))}
                      for k in calipers_psmatch}
    tbase = time.time()
    trt_h5 = tables.open_file(file_names.sparseh5_names(hisdir, trt),'r')
    ctl_h5 = tables.open_file(file_names.sparseh5_names(hisdir, ctl),'r')
    sparse_index = ps.get_sparseindex(hisdir, trt)
    BTS = {}
    accBTS = {}
    curbins = []

    for binid in trtinfo['bindf']['binid']: # drugbins.walk_nodes("/","EArray"):
        if not "/" + binid in ctlbins:
            continue
        ta = time.time()
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        tsel = np.isin(trt_compare.index,scored_trt)
        trt_ids = trt_compare.index[tsel]
        trt_compare = psmod.loc[trt_ids]
        #if len(set(trt_ids) & set([  3780886,   1784601,   8838199,   1123636, 117297190])) > 0:
        #    pdb.set_trace()
        if trt_compare.shape[0] == 0:
            continue
        #trt_compare = psmod.loc[set(trt_compare.index) & scored_ids]


        ctldat = ctlbins.get_node("/" + binid)
        csel = np.isin(ctldat[:,0], scored_ctl)
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


