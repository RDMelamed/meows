import tables
import sys
sys.path.append("../../code")
import pandas as pd
import numpy as np
import pickle
import pdb
import datetime
import time
from collections import Counter, defaultdict
from scipy import sparse
import scipy
import os
import tables
import glob
import subprocess
import his2ft
from itertools import chain
#sys.path.append("../11.16_ctl_match/")
import ps_match
#sys.path.append("../02.27_match2nc/")
import caliper2 as caliper
import weights_outcomes
import matching
import ps
import nearest_neighbor
import multiprocessing as mp
#import match_ps_caliper
### now you have 2 files
###   -- the trt dat in "hisdir"
import json

def get_trt_names(drugid):
    runname = 'Target.' + str(drugid) +"."
    return runname, runname + 'trt'            
def load_emb(hideous=''):
    femb, fdrugemb, fpred = pickle.load(open("../05.02_bigdrugs/emb50-500.pkl" if not hideous else hideous,'rb'))
    hname = os.path.basename(hideous).split(".")[0]    
    return femb.values, hname

def load_ugly(doid,hideous=''):
    if len(doid)==0:
        doid = [int(i) for i in open('big_neuropsycho').read().strip("\n").split("\n")]
    newvoc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")
    dodict = newvoc.loc[newvoc['type']=='rx',:].set_index('id').loc[doid,['vi']].to_dict()['vi']
    #dodict = newvoc.loc[newvoc['type']=='rx',:].set_index('id').loc[doid,['clid']].to_dict()['clid']
    femb, hname = load_emb(hideous)
    return dodict, femb, hname

def drug_group2(drugid, doid, hisdir,NN_pscaliper,outcomes, vocab2superft={},
                psmatch=[],
                caliper_percentile=90, caliper_median_method=False,hideous='',
                multi=True, single_outcomes=False):
    if not drugid in doid:
        doid.append(drugid)
    if not hisdir.endswith("/"): hisdir += "/"

    runname, trtname = get_trt_names(drugid)
    print(runname)
    trtfiles = hisdir + "*." + str(drugid)
    trtfiles = glob.glob(trtfiles)

    ### for the TREATED: get all pat; get bins & sparse-index, make sparse mat
    ###   -- then get calipers per bin
    trtid = hisdir + "trtid" + str(drugid)
    if not os.path.exists(trtid):
        subprocess.call("cut -f 1 " + " ".join(trtfiles) + " | sort > " + trtid,shell=True)


    dodict, emb50, hideous_name = load_ugly(doid)
    print("dodict:", dodict)    
    his2ft.bin_pats(hisdir,emb50,{},trtfiles,trtname, is_trt=True)
    tables.file._open_files.close_all()    
    if hideous: ### nearest neighbor will use this one 
       emb, hideous_name = load_emb(hideous)
       his2ft.bin_pats(hisdir,emb,{},trtfiles,trtname, is_trt=True,hideous=hideous_name)
       tables.file._open_files.close_all()    
       caliper.caliperdict(hisdir + trtname, drugid,
                        median=caliper_median_method,percentile=caliper_percentile, hideous=hideous_name)
    else:
        caliper.caliperdict(hisdir + trtname, drugid,
                        median=caliper_median_method,percentile=caliper_percentile)
    calipername = ('perc' if caliper_median_method==False else 'med') + str(caliper_percentile)   



    ### create sparsemat ONCE rather than doing it for every pair...
    his2ft.prepare_sparsemat2(hisdir,trtname, trtname,{},trtfiles)
    tables.file._open_files.close_all()    
    procs = []
    nproc = 3
    Q = mp.Queue(maxsize=500000)
    plock = mp.Lock()
    if not multi:

        ### test!
        Q.put(doid[0])
        Q.put(None)
        run_ctl_proc(hisdir,runname, trtname,trtfiles,drugid, doid, calipername, Q,
                     outcomes, psmatch,NN_pscaliper,plock, hideous, single_outcomes)
    else:
        for i in range(nproc):
            p = mp.Process(target = run_ctl_proc,
                       args = (hisdir, runname, trtname,trtfiles,drugid,doid, calipername, Q,
                               outcomes, psmatch,NN_pscaliper,plock,hideous, single_outcomes))
            p.start()
            procs.append(p)
        for ctl in list(set(dodict.keys()) - set([drugid])):
            Q.put(ctl)
        for i in range(nproc):
            Q.put(None)
        for p in procs:
            p.join()
            print("got one!")


    print("GOT ALL!")
def run_ctl_proc(hisdir, runname, trtname,trtfiles,drugid, doid, calipername, Q,
                 outcomes, psmatch,NN_pscaliper,plock, hideous='', single_outcomes =False):

    dodict, emb50, hideous_name = load_ugly(doid)
    #hideous = '' 
    times_record = {}
    for ctl in iter(Q.get, None):
        coxfilt = {'exclude':set([dodict[drugid], dodict[ctl]])}
        savename = runname + str(ctl)
        times_record[ctl] = onectl(hisdir, trtname, trtfiles, drugid, calipername,
               savename, coxfilt, emb50,ctl, outcomes,
                                   psmatch=psmatch,NN_pscaliper=NN_pscaliper,hideous=hideous, single_outcomes=single_outcomes)
        plock.acquire()        
        with open("times_record_tmp",'a') as f:
            f.write(json.dumps(times_record))
        plock.release()
        
    plock.acquire()
    with open("times_record",'a') as f:
        pd.DataFrame(times_record).to_csv(f)
    plock.release()        
    print("QUITTING run_ctl_proc")
    
def onectl(hisdir, trtname,trtfiles,drugid, calipername,
           savename, coxfilt, emb50,ctl, outcomes,
           psmatch=[],NN_pscaliper=[], vocab2superft={},hideous='', single_outcomes=False):    
    ## those in both TRT, CTL  will be removed:
    ##  - trt_to_exclude: treated have already been chosen before this control is Thus.
    #     * considered removing any patient with incident-use of both drugs ("comm_ids"),
    #     * must include in this list any treated with history of ctl drug (from "get_his_ids")
    ##  - comm_ids: also to removed from CTL bin embed files (via filtid argument + coxfilt), thus not in Ctl either.  filtid = will remove those with past or future of the other drug;
    import time
    t0 = time.time()
    times_record = {}
    ctlf = glob.glob(hisdir + "valid*." + str(ctl))
    commidfile = hisdir + savename + ".comm_id"
    comm_ids = []
    if not os.path.exists(commidfile):
        comm_ids = [float(i) for i in
            subprocess.check_output("cut -f 1 " + " ".join(ctlf) +
                                " | sort | comm -1 -2  - " + hisdir + "trtid" +
                                str(drugid),shell=True).decode("utf-8").split("\n")[:-1]]
        np.savetxt(commidfile, comm_ids)
    else:
        comm_ids = np.loadtxt(commidfile)
    trt_to_exclude = []
    trtexcl_file = hisdir + savename + ".trt_excl"
    if not os.path.exists(trtexcl_file):        
        trt_to_exclude = sorted(set(his2ft.get_his_ids(trtfiles, coxfilt))
                            | set(comm_ids))
        np.savetxt(trtexcl_file, trt_to_exclude,fmt="%d")
    else:
        trt_to_exclude = np.loadtxt(trtexcl_file,dtype=int)
    print(savename, " make trt remove:",len(trt_to_exclude),
          ' @{:2.2f}'.format((time.time() - t0)/60))            
    np.random.shuffle(ctlf)
    
    ### bin the ctl people as they fall in same bin as trt
    tx = time.time()
    print(savename, "  binning...")

    his2ft.bin_pats(hisdir,emb50,coxfilt,ctlf, savename, filtid = comm_ids)
    if hideous:
        emb50, hideous = load_emb(hideous)
        his2ft.bin_pats(hisdir,emb50,coxfilt,ctlf, savename, filtid = comm_ids,hideous=hideous)
    tables.file._open_files.close_all()
    print("Finish bdinning... @{:2.2f}".format((time.time() - tx)/60))            
    times_record['setup'] = (time.time() - t0)/60
    ## get all binned IDs: treated and ctl that are in the bins
    ## NOTE: in the "binning" step, we already remove the CTL with TRT-events
    pair_allid = "PSM" + savename    
    if not os.path.exists(hisdir + pair_allid + ".ids.ctl"):
        print(pair_allid, "preparing ids...")
        ctlbins = tables.open_file(his2ft.gen_embname(hisdir, savename) ,mode="r")
        ids = []
        for bindat in ctlbins.walk_nodes("/","EArray"):
            ids.append(bindat[:,0])
        ## save IDs so that we can work with only patients in same bins
        ## NOTE: for TRT, need to filter out ppl with CTL-events
        
        np.savetxt(hisdir + pair_allid + ".ids.ctl", np.hstack(ids)) 
        np.savetxt(hisdir + pair_allid + ".ids.trt",np.array(list(set(np.loadtxt(hisdir + 'trtid' + str(drugid),dtype=int))-set(trt_to_exclude))))

    ## now prepare all data for PS:
    tx = time.time()
    his2ft.prepare_sparsemat2(hisdir,trtname, savename ,coxfilt,ctlf,
                           idfile=pair_allid)
    print("Finish sparsemat... @{:2.2f}".format((time.time() - tx)/60))            

    ### make propensity score of all & get PS scores
    tx = time.time()
    #pair_allid

    BIGNESS = 300000
    NSAMP = 5
    trtid = np.loadtxt(hisdir + pair_allid + ".ids.trt")
    ctlid = np.loadtxt(hisdir + pair_allid + ".ids.ctl")
    #pdb.set_trace()
    if len(trtid) > 2*BIGNESS or len(ctlid) > 2*BIGNESS:
        for i in range(NSAMP):
            pair_allid_prefix = "PSM" + str(i)
            pair_allid_samp = pair_allid_prefix + savename
            if not os.path.exists(hisdir + pair_allid_samp + ".ids.trt"):
                print('creating sample:' + hisdir + pair_allid_samp + ".ids.*")
                if len(trtid) <= 2*BIGNESS:
                    np.savetxt(hisdir + pair_allid_samp + ".ids.trt", trtid)
                else:
                    np.savetxt(hisdir + pair_allid_samp + ".ids.trt", np.random.choice(trtid, BIGNESS,False))
                if len(ctlid) <= 2*BIGNESS:
                    np.savetxt(hisdir + pair_allid_samp + ".ids.ctl", ctlid)
                else:
                    np.savetxt(hisdir + pair_allid_samp + ".ids.ctl", np.random.choice(ctlid, BIGNESS,False))
            #pdb.set_trace()
            tr = ps2match2ipw(hisdir, trtname,drugid, calipername,
                              savename,  outcomes,trt_to_exclude,
                              pair_allid_prefix=pair_allid_prefix,
                        psmatch=psmatch,NN_pscaliper=NN_pscaliper, hideous=hideous, single_outcomes=single_outcomes)
            times_record.update(tr)
    else:
        tr = ps2match2ipw(hisdir, trtname,drugid, calipername,
                     savename,  outcomes,trt_to_exclude,
                     psmatch=psmatch,NN_pscaliper=NN_pscaliper, hideous=hideous, single_outcomes=single_outcomes)
        times_record.update(tr)        
    return times_record
def ps2match2ipw(hisdir, trtname,drugid, calipername,
                 savename,  outcomes,trt_to_exclude,pair_allid_prefix="PSM",
           psmatch=[],NN_pscaliper=[], vocab2superft={},hideous='', single_outcomes=False):
    alpha=[.0001,.001,.01]; l1=[.2,.3]
    tx = time.time()
    ps.ctl_propensity_score(hisdir,trtname,savename,pair_allid_prefix + savename + '.ids', alphas=alpha, l1s=l1)
    print("Finish big ps... @{:2.2f}".format((time.time() - tx)/60))
    times_record = {}
    times_record[pair_allid_prefix + 'bigps'] = (time.time() - tx)/60
    #oorder = sorted(outcomes.keys())
    #outcomes = [outcomes[k] for k in oorder]
    #for psperc in psmatch:
    tx = time.time()
    ps_match.runctl_psmatch(hisdir, trtname, savename, drugid,trt_to_exclude, psmatch, outcomes,alpha, l1, vocab2superft=vocab2superft, single_outcomes=single_outcomes,do_ebm = False, pair_allid_prefix=pair_allid_prefix) 
    print("Finish psmatch... @{:2.2f}".format((time.time() - tx)/60))
    times_record[ pair_allid_prefix + 'psm'] = (time.time() - tx)/60
    #times_record[ pair_allid_prefix + 'psm' + str(psperc)] = (time.time() - tx)/60
    '''
    psmod = hisdir + "PSM" + savename + ".ids.psmod.pkl"
    for perc in NN_pscaliper:
        tx = time.time()
        nearest_neighbor.runctl(hisdir,trtname, savename, drugid, trt_to_exclude, calipername, outcomes, vocab2superft, psmatcher = psmod, psmatch_caliper=perc,hideous=hideous, single_outcomes = single_outcomes)
        print("Finish NNmatch... @{:2.2f}".format((time.time() - tx)/60))
        times_record['NN' + str(perc)] = (time.time() - tx)/60
    '''
    return times_record


