import pandas as pd
import numpy as np
import pickle
import pdb
import csv
from sklearn.metrics import roc_auc_score
from scipy import sparse
import scipy
from sklearn.preprocessing import MaxAbsScaler
import sys
import os
sys.path.append("../12.21_allcode_ctlmatch")
import his2ft
import tables
import matching
import subprocess
def get_mod(mod, whichdo=''):
    return whichdo if whichdo else (list(mod['preds'].keys())[0]
                                        if len(mod['preds'])==1
                                        else mod['xval'].mean(axis=1).idxmax())

def ipw(mod, whichdo=''):
    whichmod = get_mod(mod, whichdo)
    ps = mod['preds'][whichmod]
    
    return pd.Series(np.clip((mod['lab']*(mod['lab'].mean()/np.clip(ps,0,1)) +
                      (1-mod['lab'])*((1-mod['lab'].mean())/np.clip(1-ps,10**-6,1))),
                             0,10000),
                     index = mod['ids'])

def omat(nodeinfo,omax):
    pdat = []
    for pato in list(nodeinfo):
        pato = list(pato)
        ixy = 1
        outs = [pato[0]]
        ogot = 0
        while ixy < len(pato):
            while pato[ixy] > ogot:
                outs += [0]
                #print("no {:d}, putting 0".format(ogot))                
                ogot += 1
            outs.append(pato[ixy + 1]) #if pato[ixy + 1] > 0 else 0
            #print(ogot, outs[-1], pato[ixy + 1], ixy)
            ogot += 1
            ixy += 2
        #outs = outs + [0]*(omax - ogot)
        pdat.append(outs + [0]*(omax - ogot))
    return np.array(pdat) #pdat 

def outcome_info(hisdir, pspref,trtname,drugid, ctlfile,outcomes_sorted, trt_to_exclude=[],whichdo=''):
    outc_fnames = {outc:hisdir + pspref  + "." + outc
                   for outc in outcomes_sorted}
    savepref = 'min-' if whichdo else ''
    if os.path.exists(savepref + list(outc_fnames.values())[0] + ".iptw"):
        print("Exists, returning: " + savepref + list(outc_fnames.values())[0] + ".iptw")
        return

    trtinfo = his2ft.get_trt_info(hisdir + trtname, drugid)
    #trtbins = tables.open_file(his2ft.gen_embname(hisdir, trtname) ,mode="r")
    #trtoutcomes = tables.open_file(his2ft.gen_outcname(hisdir, trtname) ,mode="r")
    
    ctlbins = tables.open_file(his2ft.gen_embname(hisdir, ctlfile) ,mode="r")
    #ctloutcomes = tables.open_file(his2ft.gen_outcname(hisdir, ctlfile) ,mode="r")
    num_out = len(outcomes_sorted)
    #outcomes_sorted = outcomes_sorted[:6]
    #print("Not outcoming all outcomes!")
    ipwm = {outc:ipw(pickle.load(open(outc_fnames[outc] + ".ids.psmod.pkl",'rb')),whichdo)
             for outc in outcomes_sorted}
    
    towrite = {outc:pd.DataFrame() for outc in outcomes_sorted}

    def write_outcome(outcome):
        towrite[outcome].index.name = 'id'
        write_name = savepref + outc_fnames[outcome]+".iptw"
        do_header = not os.path.exists(write_name)
        with open(write_name,'a') as f:
            towrite[outcome].to_csv(f,sep="\t",header=do_header)
    
    for binid in trtinfo['bindf']['binid']:
        if not "/" + binid in ctlbins:
            continue
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        trt_ids = trt_compare.index
        ctldat = ctlbins.get_node("/" + binid)
        ctlids = ctldat[:,0]
        trtoutc = omat(trtoutc,num_out )
        ctloutc = omat(ctlbins.get_node("/outcomes/" + binid),num_out )
        #pdb.set_trace()            
        for ix, (outcome,ipwt) in enumerate(ipwm.items()):
            #psmodmatchwt in psmod:
            ctlk = np.isin(ctlids, ipwt.index)
            trtk = np.isin(trt_ids, ipwt.index)
            co = ctloutc[ctlk,:]
            c = pd.DataFrame({"deenroll":co[:,0],
                          "outcome":co[:,ix + 1],
                          "ipw":ipwt.loc[ctlids[ctlk]]},index=ctlids[ctlk])
            c['label'] = 0
            co = trtoutc[trtk,:]            
            t = pd.DataFrame({"deenroll":co[:,0],
                          "outcome":co[:,ix + 1],
                          "ipw":ipwt.loc[trt_ids[trtk]]},index=trt_ids[trtk])
            t['label'] = 1
            
            if (c['outcome'] < 0).sum() > 0 or (t['outcome'] < 0).sum() > 0 or np.isin(trt_ids[trtk],trt_to_exclude).sum()>0:
                print("not excluded...",binid)
                pdb.set_trace()
            towrite[outcome] = pd.concat((towrite[outcome],c,t),axis=0)
            if towrite[outcome].shape[0] > 10000:
                write_outcome(outcome)
                towrite[outcome] = pd.DataFrame()
    for outcome in outc_fnames:
        write_outcome(outcome)
    subprocess.call("Rscript --vanilla run_eff.R " + savepref + hisdir + " " +
                    pspref + ".",shell=True)
