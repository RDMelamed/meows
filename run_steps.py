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
import regression_splines
#import match_ps_caliper
### now you have 2 files
###   -- the trt dat in "hisdir"
import json
from file_names import *
def load_emb(hideous=''):
    femb, fdrugemb, fpred = pickle.load(open("../05.02_bigdrugs/emb50-500.pkl" if not hideous else hideous,'rb'))
    hname = os.path.basename(hideous).split(".")[0]    
    return femb.values, hname

def load_ugly(doid,noemb, hideous=''):
    if len(doid)==0:
        doid = [int(i) for i in open('big_neuropsycho').read().strip("\n").split("\n")]
    newvoc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")
    dodict = newvoc.loc[newvoc['type']=='rx',:].set_index('id').loc[doid,['vi']].to_dict()['vi']
    #dodict = newvoc.loc[newvoc['type']=='rx',:].set_index('id').loc[doid,['clid']].to_dict()['clid']
    femb, hname = load_emb(hideous)
    if noemb:
        femb = np.zeros(0)
    return dodict, femb, hname



    
def drug_group2(drugid, doid, hisdir,outcomes, 
                psmatch=[],NN_pscaliper=[],vocab2superft={},
                caliper_percentile=90, caliper_median_method=False,hideous='',
                multi=True, single_outcomes=False, ft_exclude=[],featweights='',
                weight_method=''):
    
    if not drugid in doid:
        doid.append(drugid)
    if not hisdir.endswith("/"): hisdir += "/"
    outcomes = sorted(pickle.load(open(outcomes,'rb')).keys())
    runname, trtname = get_trt_names(hisdir, drugid)
    print(runname)
    trtfiles = todo_files(hisdir, drugid)
    #trtfiles = hisdir + "*." + str(drugid)
    #trtfiles = glob.glob(trtfiles)

    ### for the TREATED: get all pat; get bins & sparse-index, make sparse mat
    ###   -- then get calipers per bin
    trtid = runname + "trtid"
    #pdb.set_trace()
    if not os.path.exists(trtid):
        subprocess.call("cut -f 1 " + " ".join(trtfiles) + " | sort > " + trtid,shell=True)

    noemb = caliper_percentile == 0
    dodict, emb50, hideous_name = load_ugly(doid, noemb)
    #print("dodict:", dodict)
    if os.path.exists(runname + "complete_log"):
        finished = open(runname+"complete_log").read().strip().split("\n")
        remaining = set(dodict.keys()) - set([drugid])  - set([int(i.replace("complete",""))
                                                               for i in finished if i != "complete"])
        if len(remaining)== 0:
            print("COMPLETED-SKIPPING:" + runname )
            return
        dodict = {k:dodict[k] for k in list(remaining) + [drugid]}
    
    
    his2ft.bin_pats(runname,emb50,{},trtfiles,0, is_trt=True,featweights=featweights, weightmode=weight_method)
    tables.file._open_files.close_all()
    if hideous: ### nearest neighbor will use this one 
       emb, hideous_name = load_emb(hideous)
       his2ft.bin_pats(runname,emb,{},trtfiles, 0, is_trt=True,hideous=hideous_name)
       tables.file._open_files.close_all()    
       caliper.caliperdict(trtname, drugid,
                        median=caliper_median_method,percentile=caliper_percentile, hideous=hideous_name)
    else:
        #pdb.set_trace()
        caliper.caliperdict(trtname, drugid,
                            median=caliper_median_method,percentile=caliper_percentile,
                            hideous=featweights + weight_method)
    calipername = ('perc' if caliper_median_method==False else 'med') + str(caliper_percentile)   


    
    ### create sparsemat ONCE rather than doing it for every pair...
    #his2ft.prepare_sparsemat2(hisdir,trtname, trtname,{},trtfiles)
    tables.file._open_files.close_all()    
    procs = []
    nproc = 3
    Q = mp.Queue(maxsize=500000)
    plock = mp.Lock()
    if not multi:

        ### test!
        for ctl in list(set(dodict.keys()) - set([drugid])):
            Q.put(ctl)
        Q.put(None)

        run_ctl_proc(hisdir, trtfiles,drugid, doid, calipername, Q,
                               outcomes,psmatch,NN_pscaliper, plock, hideous,noemb,featweights,weight_method,
                               single_outcomes, ft_exclude)
        
    else:
        for ctl in list(set(dodict.keys()) - set([drugid])):
            Q.put(ctl)
        for i in range(nproc):
            Q.put(None)
        for i in range(nproc):
            p = mp.Process(target = run_ctl_proc,
                       args = (hisdir, trtfiles,drugid, doid, calipername, Q,
                               outcomes,psmatch,NN_pscaliper, plock, hideous,noemb,featweights,weight_method,
                               single_outcomes, ft_exclude))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            print("got one!")
        '''
        run_preps(hisdir, trtfiles,drugid, doid,  Q,
                  outcomes,plock, hideous,False,featweights,weight_method)

        #filter_multi(hisdir, drugid, doid)
        Q.put(doid[0])
        Q.put(None)
        for i in range(nproc):
            p = mp.Process(target = run_ctl_proc,
                       args =(hisdir,trtfiles,drugid, doid, calipername, Q,
                     outcomes, psmatch,NN_pscaliper,plock, hideous+ featweights + weight_method,
                     single_outcomes, ft_exclude))
            
            p.start()
            procs.append(p)
        for ctl in list(set(dodict.keys()) - set([drugid])):
            Q.put(ctl)
        for i in range(nproc):
            Q.put(None)
        for p in procs:
            p.join()
            print("got one!")
        ''' 
    finished = open(runname+"complete_log").read().strip().split("\n")
    did_finish = False
    if len(set(dodict.keys()) - set([drugid])  - set([int(i) for i in finished if not "complete" in i]))== 0:
        did_finish = True
        with open(runname+ "complete_log",'a') as f:
            for pytab in glob.glob(runname + "*pytab"):
                os.remove(pytab)
            subprocess.call("cp --parents " + runname + "*eff " + hisdir +"/effs",shell=True)
            #subprocess.call("tar -czf " + runname.strip("/") + ".tgz" + " " + runname,shell=True)        
            f.write("complete\n")
    print("GOT ALL!")

    return did_finish


        
'''
def run_preps(hisdir, trtfiles,drugid, doid,  Q,
                 outcomes,plock, hideous='',noemb=False,featweights='',weight_method=''):

    dodict, emb50, hideous_name = load_ugly(doid, noemb)
    #hideous = '' 
    times_record = {}
    for ctl in iter(Q.get, None):
        coxfilt = {'exclude':set([dodict[drugid], dodict[ctl]])}
        #savename = runname + str(ctl)
        times_record[str(ctl)] = prep_match(hisdir, trtfiles,drugid,
           coxfilt, emb50,ctl, outcomes,hideous=hideous,featweights=featweights,weight_method=weight_method)
    print("QUITTING run_preps")
def run_ctl_proc(hisdir, trtfiles,drugid, doid, calipername, Q,
                 outcomes, psmatch,NN_pscaliper,plock, hideous='', single_outcomes =False, ft_exclude=[]):
    times_record = {}
    for ctl in iter(Q.get, None):
        times_record[str(ctl)] = onectl(hisdir, drugid, calipername,
               ctl, outcomes, psmatch=psmatch,NN_pscaliper=NN_pscaliper,hideous=hideous, single_outcomes=single_outcomes, ft_exclude=ft_exclude)
        plock.acquire()        
        with open("times_record_tmp",'a') as f:
            f.write(json.dumps(times_record))
        plock.release()
        
    plock.acquire()
    with open("times_record",'a') as f:
        pd.DataFrame(times_record,index=['time']).to_csv(f)
    plock.release()        
    print("QUITTING run_ctl_proc")
'''
        
def run_ctl_proc(hisdir, trtfiles,drugid, doid, calipername,  Q,
                 outcomes,psmatch,NN_pscaliper, plock, hideous='',noemb=False,featweights='',weight_method='',
              single_outcomes =False, ft_exclude=[]):
    dodict, emb50, hideous_name = load_ugly(doid, noemb)
    #hideous = '' 
    times_record = {}
    for ctl in iter(Q.get, None):
        coxfilt = {'exclude':set([dodict[drugid], dodict[ctl]])}
        #savename = runname + str(ctl)
        t0 = time.time()
        x = prep_match(hisdir, trtfiles,drugid,
           coxfilt, emb50,ctl, outcomes,hideous=hideous,featweights=featweights,weight_method=weight_method)
        t1 = time.time()
        times_record[str(ctl)] = onectl(hisdir, drugid, calipername,
               ctl, outcomes, psmatch=psmatch,NN_pscaliper=NN_pscaliper,hideous=hideous+ featweights + weight_method,
                     single_outcomes=single_outcomes, ft_exclude=ft_exclude)
        plock.acquire()        
        with open("times_record_tmp",'a') as f:
            f.write(json.dumps(times_record))
        plock.release()
        
    plock.acquire()
    with open("times_record",'a') as f:
        pd.DataFrame(times_record,index=['time']).to_csv(f)
    plock.release()        
    print("QUITTING run_ctl_proc")
    
def prep_match(hisdir, trtfiles,drugid,
           coxfilt, emb50,ctl, outcomes,
           hideous='', featweights='',weight_method=''):    
    import time
    t0 = time.time()

    #ctlf = glob.glob(hisdir + "valid*." + str(ctl))
    ctlf = todo_files(hisdir, ctl)
    runname, trtname = get_trt_names(hisdir, drugid)
    #savename = get_savename(drugid, ctl)
    pair_prefix = runname + str(ctl)

    ## those in both TRT, CTL  will be removed:
    ##  - trt_to_exclude: treated have already been chosen before this control is Thus.
    #     * considered removing any patient with incident-use of both drugs ("comm_ids"),
    #     * must include in this list any treated with history of ctl drug (from "get_his_ids")
    ##  - comm_ids: also to removed from CTL bin embed files (via filtid argument + coxfilt), thus not in Ctl either.  filtid = will remove those with past or future of the other drug;
    
    commidfile = pair_prefix + ".comm_id"
    comm_ids = []
    if not os.path.exists(commidfile):
        comm_ids = [float(i) for i in
            subprocess.check_output("cut -f 1 " + " ".join(ctlf) +
                                " | sort | comm -1 -2  - " + runname + "trtid",
                                    shell=True).decode("utf-8").split("\n")[:-1]]
        np.savetxt(commidfile, comm_ids)
    else:
        comm_ids = np.loadtxt(commidfile)
    trt_to_exclude = []
    trtexcl_file = pair_prefix + ".trt_excl"
    if not os.path.exists(trtexcl_file):        
        trt_to_exclude = sorted(set(his2ft.get_his_ids(trtfiles, coxfilt))
                           | set(comm_ids))
        np.savetxt(trtexcl_file, trt_to_exclude,fmt="%d")
    else:
        trt_to_exclude = np.loadtxt(trtexcl_file,dtype=int)
    print(pair_prefix, " make trt remove:",len(trt_to_exclude),
          ' @{:2.2f}'.format((time.time() - t0)/60))            
    np.random.shuffle(ctlf)
    
    ### bin the ctl people as they fall in same bin as trt
    tx = time.time()
    print(pair_prefix, "  binning...")
    #his2ft.bin_pats(runname,emb50,{},trtfiles,0, is_trt=True)
    #pdb.set_trace()
    his2ft.bin_pats(runname,emb50,coxfilt,ctlf, ctl, filtid = comm_ids, featweights=featweights, weightmode=weight_method)
    if hideous:
        emb50, hideous = load_emb(hideous)
        #his2ft.bin_pats(hisdir,emb50,coxfilt,ctlf, savename, filtid = comm_ids,hideous=hideous)
        his2ft.bin_pats(runname,emb50,coxfilt,ctlf, ctl, filtid = comm_ids,hideous=hideous)
    tables.file._open_files.close_all()
    print("Finish bdinning... @{:2.2f}".format((time.time() - tx)/60))            
    #times_record['setup'] = (time.time() - t0)/60
    ## get all binned IDs: treated and ctl that are in the bins
    ## NOTE: in the "binning" step, we already remove the CTL with TRT-events
    pair_allid_name, pair_allid = pair_names(pair_prefix)
    if not os.path.exists(pair_allid + ".ids.ctl"):
        print(pair_allid, "preparing ids...")
        ctlbins = tables.open_file(his2ft.gen_embname(pair_prefix,hideous+featweights+weight_method) ,mode="r")
        ids = []
        for bindat in ctlbins.walk_nodes("/","EArray"):
            ids.append(bindat[:,0])
        
        ## save IDs so that we can work with only patients in same bins
        ## NOTE: for TRT, need to filter out ppl with CTL-events
        np.savetxt(pair_allid + ".ids.trt",
                   np.array(list(set(np.loadtxt(runname + 'trtid',dtype=int))-set(trt_to_exclude))))
        
        np.savetxt(pair_allid + ".ids.ctl", np.hstack(ids) if len(ids) else np.zeros(0)) 

def pair_names(pair_prefix):
    pair_allid_name = "PSM"
    pair_allid = pair_idfile(pair_prefix, pair_allid_name)
    return pair_allid_name, pair_allid

def filter_multi(hisdir, drugid, doid):
    runname, trtname = get_trt_names(hisdir, drugid)
    if os.path.exists(runname + "removed_ids"):
        print('skipping filter_multi '  + runname + "removed_ids" + " exists")
        return
    idgot = set()
    multi_id = defaultdict(set)
    def get_name(ctl_i):
        pair_prefix = runname + str(ctl_i) 
        return pair_names(pair_prefix)[1]

    for ctl_i in range(len(doid)):
        pair_allid_i = get_name(doid[ctl_i])
        id_i = np.loadtxt(pair_allid_i + ".ids.ctl")
        for ctl_j in range(ctl_i + 1, len(doid)):        
            pair_allid_j = get_name(doid[ctl_j])

            id_j = np.loadtxt(pair_allid_j + ".ids.ctl")
            for comm in set(id_i) & set(id_j):
                multi_id[comm] |= set([doid[ctl_i], doid[ctl_j]])
        np.savetxt(pair_allid_i + ".ids.ctl",
                   np.array(list(set(id_i) - set(multi_id.keys()))))
    #pdb.set_trace()
    towrite = defaultdict(list)
    dup = 0
    f = open(runname + "removed_ids",'w')
    for pat, drugs in multi_id.items():
        #pdb.set_trace()
        chosen = np.random.choice(list(drugs),1)[0]
        towrite[chosen].append(pat)
        dup += len(drugs)
        f.write(str(int(pat)) + "\t" + ",".join([str(d) for d in set(drugs) - set([chosen])])+"\n")
    f.close()
    print('dup',dup, len(multi_id))

    for fn, pats in towrite.items():
        
        with open(get_name(fn) + ".ids.ctl", 'ab') as f:
            np.savetxt(f, np.array(pats))
            
def onectl(hisdir,drugid, calipername,
           ctl, outcomes,
           psmatch=[],NN_pscaliper=[], vocab2superft={},
           hideous='', single_outcomes=False, ft_exclude=[]):    

    ## now prepare all data for PS:
    #tx = time.time()
    #his2ft.prepare_sparsemat2(hisdir, '', savename ,coxfilt,ctlf, idfile=pair_allid)
    #print("Finish sparsemat... @{:2.2f}".format((time.time() - tx)/60))            

    ### make propensity score of all & get PS scores
    tx = time.time()
    #pair_allid
    times_record = {}
    BIGNESS = 300000
    NSAMP = 5
    runname, trtname = get_trt_names(hisdir, drugid)
    pair_prefix = runname + str(ctl)
    pair_allid_name, pair_allid = pair_names(pair_prefix)    
    trtid = np.loadtxt(pair_allid + ".ids.trt")
    ctlid = np.loadtxt(pair_allid + ".ids.ctl")
    #print(pair_allid)
    #pdb.set_trace()
    #idfile_name = "PSM"
    #pair_idname =  pair_prefix + "." + idfile_name
    if len(np.atleast_1d(ctlid)) < 200  or len(np.atleast_1d(trtid)) < 200:
        print("SKIPPING ", pair_allid, " too small!")
        return
    BFRAC = 1.75
    if max(len(trtid), len(ctlid)) > BFRAC*BIGNESS:
        trt_splits = 1 if len(trtid) <= BFRAC*BIGNESS else min(int(np.ceil(len(trtid)/BIGNESS)), 10)
        ctl_splits = 1 if len(ctlid) <= BFRAC*BIGNESS else min(int(np.ceil(len(ctlid)/BIGNESS)), 10)
        nsplits = min(trt_splits, ctl_splits) ## if they are both big guys then split both but limits # splits
        if nsplits == 1:  ## if only 1 is big then do that one
            nsplits = max(trt_splits, ctl_splits)
        trt_split_index = regression_splines.get_splitvec(len(trtid), trt_splits)
        ctl_split_index = regression_splines.get_splitvec(len(ctlid), ctl_splits)        

        for i in range(nsplits):
            #idfile_name_samp = idfile_name + str(i)
            pair_allid_samp_name = pair_allid_name + str(i)
            pair_allid_samp =  pair_idfile(pair_prefix, pair_allid_samp_name) + ".ids"    
            if not os.path.exists(pair_allid_samp + ".trt"):
                print('creating sample:' + pair_allid_samp + ".*")
                if len(trtid) <= BFRAC*BIGNESS:
                    np.savetxt(pair_allid_samp + ".trt", trtid)
                else:
                    np.savetxt(pair_allid_samp + ".trt", trtid[trt_split_index==(i % trt_splits)]) #np.random.choice(trtid, BIGNESS,False))
                    #if i==(trt_splits - 1) and nsplits > trt_splits:
                    #    np.random.shuffle(trt_split_index)
                if len(ctlid) <= BFRAC*BIGNESS:
                    np.savetxt(pair_allid_samp + ".ctl", ctlid)
                else:
                    np.savetxt(pair_allid_samp + ".ctl", ctlid[ctl_split_index==(i % ctl_splits)])  #np.random.choice(ctlid, BIGNESS,False))
                    #if i==(ctl_splits - 1) and nsplits > ctl_splits:
                    #    np.random.shuffle(ctl_split_index)
                    
            #pdb.set_trace()
            tr = ps2match2ipw(hisdir, drugid, calipername,
                              ctl,  outcomes,[],
                              idfile_name=pair_allid_samp_name, #idfile_name,
                        psmatch=psmatch,NN_pscaliper=NN_pscaliper, hideous=hideous, single_outcomes=single_outcomes, ft_exclude = ft_exclude)
            times_record.update(tr)
    else:
        #idfile_name = ".PSM" + str(i)
        #pair_allid_samp =  pair_prefix + idfile_name        
        tr = ps2match2ipw(hisdir, drugid, calipername,
                          ctl,  outcomes,[], idfile_name = pair_allid_name,
                          psmatch=psmatch,NN_pscaliper=NN_pscaliper, hideous=hideous, single_outcomes=single_outcomes, ft_exclude = ft_exclude)
        times_record.update(tr)
    with open(runname + "complete_log", 'a') as f:
        f.write(str(ctl) + "\n")
    return times_record
def ps2match2ipw(hisdir, drugid, calipername,
                 ctl,  outcomes,trt_to_exclude, ## trt_to_exclude -- don't really need this b/c using files of ids to match
                 idfile_name="PSM",
                 psmatch=[],NN_pscaliper=[], vocab2superft={},hideous='', single_outcomes=False, ft_exclude = []):
    alpha=[.0001,.001,.01]; l1=[.2,.3]
    tx = time.time()
    runname, trtname = get_trt_names(hisdir, drugid)
    #id_savename = runname + str(ctl) + idfile_name #get_savename_prefix(drugid, ctl, pair_allid_prefix)
    #ps.ctl_propensity_score(hisdir,trtname,savename,pair_allid_prefix + savename + '.ids', alphas=alpha, l1s=l1)
    #pairname = file_names.get_pair_name(hisdir, trt_drugid, ctl_drugid)
    #id_list  =

    ps.ctl_propensity_score(hisdir,drugid,ctl, runname + str(ctl) +"."+ idfile_name + ".ids",ft_exclude = ft_exclude,  alphas=alpha, l1s=l1)
    print("Finish big ps... @{:2.2f}".format((time.time() - tx)/60))
    times_record = {}
    times_record[idfile_name + 'bigps'] = (time.time() - tx)/60
    #oorder = sorted(outcomes.keys())
    #outcomes = [outcomes[k] for k in oorder]
    #for psperc in psmatch:
    tx = time.time()
    ps_match.runctl_psmatch(hisdir, ctl, drugid,trt_to_exclude, psmatch, outcomes,
                            alpha, l1, vocab2superft=vocab2superft, single_outcomes=single_outcomes,do_ebm = False, idfile_name=idfile_name, ft_exclude = ft_exclude) 
    print("Finish psmatch... @{:2.2f}".format((time.time() - tx)/60))
    times_record[ idfile_name + 'psm'] = (time.time() - tx)/60
    #times_record[ pair_allid_prefix + 'psm' + str(psperc)] = (time.time() - tx)/60

    #psmod = hisdir + "PSM" + savename + ".ids.psmod.pkl"
    for perc in NN_pscaliper:
        tx = time.time()
        nearest_neighbor.runctl(hisdir,ctl, drugid, trt_to_exclude, calipername, outcomes, idfile_name, vocab2superft, psmatch_caliper=perc,hideous=hideous, single_outcomes = single_outcomes)  #psmatcher = psmod, 
        print("Finish NNmatch... @{:2.2f}".format((time.time() - tx)/60))
        times_record['NN' + str(perc)] = (time.time() - tx)/60

    return times_record

def writelog(strw, writeinfo):
    writeinfo[0].acquire()
    with open(writeinfo[1],'a') as f:
        f.write(strw + "\n")
    writeinfo[0].release()

def run_comparators(trtname, hisdir,multiproc=True):
    if not os.path.exists("tmp/"):
        mkdir("tmp/")
    comparators = pickle.load(open("comparison_drug_mech.pkl",'rb'))        
    outcome = "../06.11_all_neighbors/outcomes_no_nonmel.pkl"
    voc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")
    drugid = voc.loc[voc['name']==trtname,'id'].values[0]
    compid = get_compid(trtname, comparators, voc)
    #comp  = comparators[trtname]['red-umls']
    #comp = comp[:10]
    #compid = list(voc.loc[voc['name'].isin(comp),'id'].values)
    #print('''
    
    drug_group2(drugid, compid, hisdir,outcome, psmatch=[.25],
                      NN_pscaliper=[], multi=multiproc, single_outcomes=True,caliper_percentile=0) #''')
    
    #drug_group2(drugid, compid, hisdir,outcome, psmatch=[.25],
    #                  NN_pscaliper=[50], multi=True, single_outcomes=True,caliper_percentile=70) #''')
    #run_steps.drug_group2(2305, [2293], hisdir, outcome, psmatch=[.25],NN_pscaliper =[82],
     #                 multi=True, single_outcomes=True,caliper_percentile=82,featweights='canc_nonzero',weight_method='P')

def get_compid(trtname, comparators, voc):
    comp = voc.set_index('name').loc[comparators[trtname]['red-umls'],:] #.sort_values('ct')
    if comp.shape[0] > 20:
        comp = comp.loc[comp['ct'] > 100000,:].iloc[:20,:]

    #pdb.set_trace()
    compid = list(comp['id'].values)  # list(voc.loc[voc['name'].isin(comp),'id'].values)
    #print(trtname,len(compid))        
    comp = voc.set_index('name').loc[comparators[trtname]['othermech'],:] #.sort_values('ct')
    if comp.shape[0] > 20:
        comp = comp.loc[comp['ct'] > 100000,:].iloc[:20,:]
    compid.extend(list(comp['id'].values))   #voc.set_index('name').loc[comparators[trtname]['othermech'],'id'].values))
    compid = list(set(compid))
    return compid

def run_Q(Q, plock, hisdir,multi):
    if not os.path.exists("tmp/"):
        mkdir("tmp/")
    #comparators = pickle.load(open("all_comparisons.pkl",'rb'))
    comparators = pickle.load(open("comparison_drug_mech.pkl",'rb'))        
    outcome = "../06.11_all_neighbors/outcomes_no_nonmel.pkl"
    voc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")
    for trtname in iter(Q.get, None):
        #
        print(">",trtname,"<")
        drugid = voc.loc[voc['name']==trtname,'id'].values[0]


        runname, _ = get_trt_names(hisdir, drugid)
        trtid = runname + "trtid"
        if os.path.exists(runname + "complete_log"):
            finished = [i for i in open(runname+"complete_log").read().strip().split("\n") if "complete" in i]
            if len(finished) > 0:
                with open(hisdir + "todo.finished",'a') as f:
                    f.write(trtname + "\n")
                continue
        #return
        #comp  = comparators[trtname]
        #comp  = comparators[trtname]['red-umls'][:15]
        comp = voc.set_index('name').loc[comparators[trtname]['red-umls'],:] #.sort_values('ct')
        if comp.shape[0] > 20:
            comp = comp.loc[comp['ct'] > 100000,:].iloc[:20,:]

        #pdb.set_trace()
        compid = list(comp['id'].values)  # list(voc.loc[voc['name'].isin(comp),'id'].values)
        #print(trtname,len(compid))        
        comp = voc.set_index('name').loc[comparators[trtname]['othermech'],:] #.sort_values('ct')
        if comp.shape[0] > 20:
            comp = comp.loc[comp['ct'] > 100000,:].iloc[:20,:]
        compid.extend(list(comp['id'].values))   #voc.set_index('name').loc[comparators[trtname]['othermech'],'id'].values))
        compid = list(set(compid))
        if len(compid) == 0:
            writelog("skip-No comp!\t" + trtname, plock)
            with open(hisdir + "todo.finished",'a') as f:
                f.write(trtname + "\n")
            continue

        #print(trtname,len(compid))

        writelog("starting\t" + trtname + "\tdoing "+str(len(compid)), plock)

        finished = drug_group2(drugid, compid, hisdir,outcome, psmatch=[.25],
                          NN_pscaliper=[], multi=multi, single_outcomes=True,caliper_percentile=0)

        writelog("finished\t" + trtname, plock)

        with open(hisdir + "todo.finished",'a') as f:
            f.write(trtname + "\n")


def run_many(hisdir, subdo):
    print("START: " + subdo)
    Q = mp.Queue(maxsize=500000)
    #open('vallogf','w').close()
    plock = mp.Lock()
    #subdo = open(subdo).read().split("\n")
    #comparators = pickle.load(open("all_comparisons.pkl",'rb'))
    def readtodo(todo):
        return open(todo).read().strip().split("\n")
    todo = readtodo(hisdir + "todo")
    finished = readtodo(hisdir + "todo.finished")
    todo_updated = set(todo) - set(finished)
    if not len(todo_updated) == len(todo):
        with open(hisdir + "todo",'w') as f:
            f.write("\n".join(todo_updated) + "\n")
    for currtodo in glob.glob(hisdir + "todo.curr*"):
        todo_updated = todo_updated - set(readtodo(currtodo))
    thisdo = list(todo_updated)[:min(len(todo_updated), 30)]
    with open(hisdir + "todo.curr." + subdo,'w') as f:
        f.write("\n".join(thisdo) + "\n")

    for c in thisdo: #open(subdo).read().strip().split("\n"):
        Q.put(c)
    nproc = 6
    ps = []
    for p in range(nproc):
        ps.append(mp.Process(target = run_Q,
                         args = (Q, [plock, "log."+subdo],hisdir, False)))
        ps[-1].start()
        Q.put(None)
    for p in ps:
        p.join()
        print("got one in run_many!")
        
if __name__ == "__main__":
    #if len(sys.argv) > 2:
    #    run_comparators(sys.argv[1], sys.argv[2])
    #else:
    run_many(sys.argv[1], sys.argv[2])
#sbatch ../../code/matchweight/run_onedrug.sh bupropion_hydrochloride mixed_histories/    
