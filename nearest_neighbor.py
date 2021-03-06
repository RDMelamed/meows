import pandas as pd
import numpy as np
import pickle
import pdb
from scipy import sparse
import scipy
import caliper2 as caliper
from collections import defaultdict
import os
import his2ft
import tables
import ps_match
import weights_outcomes
import ps
from matching import *
import file_names
#trtname, savename
def runctl(hisdir, ctl, drugid, filt_trt, calipername, outcomes, idfile_name, vocab2superft={}, psmatch_caliper=95, hideous='',single_outcomes=True): #psmatcher='',
    #his2ft.bin_pats(hisdir,emb50,coxfilt,ctlf, savename, filtid = comm_trt_ctl)
    #tables.file._open_files.close_all()

    #outcomes = sorted(outcomes.keys())
    ## match based on embed vectors in the bins

    ## matchname will be same as savename UNLESS we are doing PSmatch then savename.PSfilt

    matchname, matchcounts = matchctl(hisdir, ctl,drugid,idfile_name,calipername, outcomes, pscaliper=psmatch_caliper, hideous=hideous, single_outcomes=single_outcomes) #ps_matcher=psmatcher,
    print(matchname, " did matching...")    
    '''    
    tids = set([])
    cids = set([])    
    for o in outcomes:
        tids |= set(np.loadtxt(hisdir + matchname + "." + o + ".ids.trt"))
        cids |= set(np.loadtxt(hisdir + matchname + "." + o + ".ids.ctl"))
    np.savetxt(hisdir + matchname + ".union.ids.trt",np.array(list(tids)))
    np.savetxt(hisdir + matchname + ".union.ids.ctl",np.array(list(cids)))
    ### after matching,outputs IDs, then we go back to the files & make sparsemat
    ### - trtname = for opening sparse-index only
    ### - savename = the sparse npz files
    ### - coxfilt = the same but should have been filtered in the bin_pats step
    ### - idfile=savename = the file where the IDs (rows) to do are stored
    ### - vi2groupi = everything will be in raw/full voc -> translate (optionally)
    print(savename, " making sparsemat...")
    his2ft.prepare_sparsemat2(hisdir,trtname, matchname,coxfilt,ctlf,
                           idfile=matchname + ".union", vocab2superft=vocab2superft)
    '''
    alpha=[.0001,.001,.01]; l1=[.2,.3]
    #pdb.set_trace()
    if single_outcomes:
        f, xval = ps.ctl_propensity_score(hisdir,drugid, ctl, 
                                      matchname + ".ids",alphas=alpha, l1s=l1)
    else:
        smallest = pd.Series(matchcounts).idxmin()
        #pdb.set_trace()    
        f, xval = ps.ctl_propensity_score(hisdir,trtname,savename,
                                          matchname + "." + outcomes[smallest] + ".ids",
                                          ['ago'],transfunc=ps.agobins,alphas=alpha, l1s=l1)
        x = xval.mean(axis=1).idxmax().split("-")
        print(f, xval.mean(axis=1).idxmax())
        ls_do = [float(x[0])]
        alph_do = [float(x[1])]
        #("" if not transfunc else "." + transfunc.__name__)
        for o in outcomes:
            r = ps.ctl_propensity_score(hisdir,trtname, savename, matchname + "." + o + '.ids',alphas=alph_do, l1s=ls_do)
    weights_outcomes.outcome_info(hisdir, matchname, drugid, ctl, outcomes, filt_trt, single_outcomes=single_outcomes)

#def get_binned_ids(tabfile):
#    return ids

import time
# filt_trt, removing "filt_trt" because we are now only doing selected ids...
def matchctl(hisdir,ctl, drugid, idfile_name, calipername,ordered_outcomes,
             pscaliper=95,hideous='', single_outcomes=False): #ps_matcher='',

    runname, trtname = file_names.get_trt_names(hisdir, drugid)
    pairname = runname + str(ctl)
    savename = ".".join([pairname, idfile_name, "NN"+ hideous, calipername, str(pscaliper)])    
    #savename = ('NNPSfilt' + str(pscaliper) if ps_matcher else 'NN')+ hideous + "." + ctlfile
    did_match = True
    counts = {}
    #pdb.set_trace()
    if single_outcomes:
        did_match = os.path.exists(savename + ".ids.trt")
    else:
        for ix, osave in enumerate(ordered_outcomes):
            if not os.path.exists(savename + "." + osave + ".ids.trt"):
                did_match = False
                break
            else:
                counts[ix] = np.loadtxt(savename + "." + osave + ".ids.trt").shape[0]
    if did_match:
        return savename, counts
    trtinfo = his2ft.get_trt_info(trtname, drugid,hideous)
    ctlbins = tables.open_file(his2ft.gen_embname(pairname,hideous=hideous) ,mode="r")
    print("using caliper:",trtname + hideous +".caliper." + calipername + ".pkl")
    embcaliper = pickle.load(open(trtname + hideous +".caliper." + calipername + ".pkl",'rb'))
    #psmod = [] if not ps_matcher else ps_match.get_ps(ps_matcher)
    ## if adding a PS caliper on top of NN matching... currently commented out
    psmod = ps.get_ps(pickle.load(open(file_names.get_ps_file(pairname, idfile_name),'rb')))
    ps_var = psmod.var()    
    bin_caliper = pscaliper*ps_var
    scored_ids = psmod.index        
    looking_for = 30
    TRTID = []
    CTLID = []
    tc_matches = [[], []]
    if not single_outcomes:
        tc_matches = {o:[[],[]] for o in range(len(ordered_outcomes))}
    ix = 0
    print("DOING:",trtinfo['bindf'].shape[0])
    t0 = time.time()
    for binid in trtinfo['bindf']['binid']: 
        if not "/" + binid in ctlbins:
            continue
        trt_compare, trtoutc = binfo(trtinfo, binid)
        #PS caliper!
        tsel = np.isin(trt_compare.index,scored_ids)
        trtoutc = trtoutc[tsel] #~trt_compare.index.isin(filt_trt)]
        trt_compare = trt_compare.loc[tsel,:] #~trt_compare.index.isin(filt_trt),:]
        if trt_compare.shape[0] == 0:
            continue
        
        ctldat = ctlbins.get_node("/" + binid)
        ctlids = ctldat[:,0]
        csel = np.isin(ctldat[:,0], scored_ids)
        ctlids = ctldat[:,0][csel]
        ctldat = ctldat[:,6:][csel,:]
        if ctldat.shape[0] == 0:
            continue
        ctldat = pd.DataFrame(trtinfo['scaler'].transform(ctldat),index=ctlids)
        
        #ctloutcom = ctloutcomes.get_node("/" + binid)
        #to_exclude = out_exclude(trt_compare, trtoutc, ctlids, ctloutcomes, binid)
        #ctloutc_bin = 

        '''
        to_exclude = defaultdict(list) # {o:[[],[]] for o in outcome}
        poo = list(zip(*tuple((ctlids, list(ctloutcomes.get_node("/" + binid)))))) + list(zip(*tuple((trt_compare.index, trtoutc))))
        for pid, ctlpat in poo:
            for ou in [ctlpat[i] for i in range(1,len(ctlpat), 2)
                       if ctlpat[i+1] <= 0 ]:
                to_exclude[ou].append(pid)
        '''
        embcutoff = embcaliper[binid] #if len(caliper) > 0 else 0
        if trt_compare.shape[0] < 2:
            print("TINY BIN:"+ savename +" "+binid)
        pscutoff = caliper.ps_caliper(trt_compare, trtinfo, binid,True,pscaliper,psmod) if trt_compare.shape[0] > 1 else 1
        #pscutoff = 0
        #pscutoff = caliper.percentile_pscaliper(trt_compare,pscaliper, 0) if trt_compare.shape[0] > 1 else 1
        #pdb.set_trace()
        tmatch_all, cmatch_all = one_bin_NN_pscaliper(trt_compare, ctldat,
                                            embcutoff, pscutoff,
                                            trtinfo['prec'],psmod)
        ix += 1        
        if ix % 50 == 0:
            print('{:d} {:s} have {:d} in {:s} in {:2.1f} minutes\n'.format(ix,binid,len(tc_matches[0]) if single_outcomes else len(tc_matches[0][0]),savename, (time.time()-t0)/60))
        if single_outcomes:
            tc_matches[0] += tmatch_all
            tc_matches[1] += cmatch_all
        else:
            to_exclude = out_exclude(trt_compare, trtoutc, ctlids, ctlbins.get_node("/outcomes/" + binid))                
            for outcome in range(len(ordered_outcomes)):
                #if outcome==11:
                #    pdb.set_trace()
                if outcome not in to_exclude:
                    tc_matches[outcome][0] += tmatch_all
                    tc_matches[outcome][1] += cmatch_all
                else:
                    #pdb.set_trace()
                    to_excl = to_exclude[outcome]
                    t = trt_compare.loc[~np.isin(trt_compare.index, to_excl),:]
                    c = ctldat.loc[~np.isin(ctldat.index, to_excl),:]
                    if t.shape[0] == 0 or c.shape[0]==0:
                        continue
                    tmatch, cmatch = one_bin_NN_pscaliper(t, c,
                                                    embcutoff, pscutoff,
                                                    trtinfo['prec'],psmod)
                    tc_matches[outcome][0] += tmatch
                    tc_matches[outcome][1] += cmatch
        #if ix > 5:
        #    break
        #print('{:d} {:s} have {:d}\n'.format(ix,binid,len(outcome_matches[0][0])))        
    #pdb.set_trace()
    counts = {}
    if single_outcomes:
        np.savetxt(savename + ".ids.trt", tc_matches[0])
        np.savetxt(savename + ".ids.ctl", tc_matches[1])
    else:
        for ix, oname in enumerate(ordered_outcomes):
            np.savetxt(savename + "." + oname + ".ids.trt", tc_matches[ix][0])
            np.savetxt(savename + "." + oname + ".ids.ctl", tc_matches[ix][1])
            counts[ix] = len(tc_matches[ix][1])

    return savename, counts
