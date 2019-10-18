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
import file_names
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
    pdat = np.array(pdat) #pdat
    anyo = np.where(pdat[:,1:] > 0, pdat[:,1:], 5000).min(axis=1).reshape(-1,1)
    anyo = np.where(anyo==5000,0,anyo)
    return np.hstack((pdat,anyo))

def outcome_info(hisdir, outc_fnames, drugid, ctl, outcomes_sorted, trt_to_exclude=[],whichdo='', single_outcomes=True, weighting=True):

    def run_R_est(outc_fnames):
        print("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/run_eff.R " + outc_fnames + " " + " " + str(single_outcomes) + " " + str(weighting) )
        subprocess.call("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/run_eff.R " + outc_fnames + " " + " " + str(single_outcomes) + " " + str(weighting) ,shell=True)

    #outc_fnames = hisdir + pspref
    savepref = 'min-' if whichdo else ''                
    if single_outcomes:
        if os.path.exists(outc_fnames + ".iptw"): #("" if not weighting else ".unwt") +
            print("Exists, returning: " + savepref + outc_fnames + ".iptw")
            if not os.path.exists(outc_fnames + ("" if not weighting else ".unwt") + ".eff"):
                run_R_est(outc_fnames)
            return
    else:

        outc_fnames = {outc:outc_fnames + savepref  + "." + outc
                   for outc in outcomes_sorted}
        if os.path.exists(list(outc_fnames.values())[0] +savepref +  ".iptw"):
            print("Exists, returning: " + list(outc_fnames.values())[0] +savepref +  ".iptw")
            return

    runname, trtname = file_names.get_trt_names(hisdir, drugid)
    #ctlfile = file_names.get_savename(drugid, ctl)
    pairname = runname + str(ctl)
    trtinfo = his2ft.get_trt_info(trtname, drugid)
    ctlbins = tables.open_file(his2ft.gen_embname(pairname) ,mode="r")
    num_out = len(outcomes_sorted)
    ipwm = {}
    towrite = {}
    print("making iptw file: " + outc_fnames + ".iptw")
    if single_outcomes:
        if weighting:
            psname = outc_fnames + ".ids.psmod.pkl"
            if not os.path.exists(psname):
                print("No ps {:s}, returning".format(psname))
                return
            ipwm = ipw(pickle.load(open(outc_fnames + ".ids.psmod.pkl",'rb')),whichdo)
        else:
            matchid = np.append(np.loadtxt(outc_fnames + ".ids.trt"),
                                np.loadtxt(outc_fnames + ".ids.ctl"))
            if matchid.shape[0] == 0:
                print("no ids!", outc_fnames, " returning")
                return
            ipwm = pd.Series(np.ones(matchid.shape[0]), index=matchid)

        towrite = pd.DataFrame()
    else:
        ipwm = {outc:ipw(pickle.load(open(outc_fnames[outc] + ".ids.psmod.pkl",'rb')),whichdo)
             for outc in outcomes_sorted}
        towrite = {outc:pd.DataFrame() for outc in outcomes_sorted}
    def write_outcome(df, fname):
        df.index.name = 'id'
        write_name = savepref + fname +  ".iptw"
        do_header = not os.path.exists(write_name)
        with open(write_name,'a') as f:
            df.to_csv(f,sep="\t",header=do_header)
            
    def select_in(ctlids, trt_ids, ctloutc, trtoutc, ipwti):
        ctlk = np.isin(ctlids, ipwti)
        trtk = np.isin(trt_ids, ipwti)
        trtids_o = trt_ids[trtk]            
        co = ctloutc[ctlk,:]
        tco = trtoutc[trtk,:]
        ctlids_o = ctlids[ctlk]
        return tco, co, trtids_o, ctlids_o
    columns = ['deenroll'] + outcomes_sorted + ['any_outcome']
    for binid in trtinfo['bindf']['binid']:
        if not "/" + binid in ctlbins:
            continue
        trt_compare, trtoutc = matching.binfo(trtinfo, binid)
        trt_ids = trt_compare.index
        ctldat = ctlbins.get_node("/" + binid)
        ctlids = ctldat[:,0]
        trtoutc = omat(trtoutc,num_out )
        ctloutc = omat(ctlbins.get_node("/outcomes/" + binid),num_out )
        if single_outcomes:
            tco, co, trtids_o, ctlids_o = select_in(ctlids, trt_ids, ctloutc, trtoutc, ipwm.index)
            
            co = pd.DataFrame(co, index=ctlids_o, columns = columns)
            co['ipw'] = ipwm.loc[ctlids_o]
            co['label'] = 0
            tco = pd.DataFrame(tco, index=trtids_o, columns = columns)
            tco['ipw'] = ipwm.loc[trtids_o]
            tco['label'] = 1
            towrite = pd.concat((towrite, co, tco),axis=0)
            if towrite.shape[0] > 10000:
                #pdb.set_trace()
                write_outcome(pd.concat((towrite.loc[:,['ipw','label']],
                                         towrite.drop(['ipw','label'],axis=1)),axis=1),
                                        outc_fnames)
                towrite = pd.DataFrame()
        else:
            for ix, (outcome, ipwt) in enumerate(ipwm.items()):
                tco, co, trtids_o, ctlids_o = select_in(ctlids, trt_ids, ctloutc, trtoutc, ipwt.index) 
                #psmodmatchwt in psmod:
                '''
                ctlk = np.isin(ctlids, ipwt.index)
                trtk = np.isin(trt_ids, ipwt.index)
                trtids_o = trt_ids[trtk]            
                co = ctloutc[ctlk,:]
                tco = trtoutc[trtk,:]
                ctlids_o = ctlids[ctlk]
                '''
                c = pd.DataFrame({"deenroll":co[:,0],
                              "outcome":co[:,ix + 1],
                                  "ipw":ipwt.loc[ctlids_o]},index=ctlids_o)
                c['label'] = 0

                t = pd.DataFrame({"deenroll":co[:,0],
                              "outcome":co[:,ix + 1],
                              "ipw":ipwt.loc[trtids_o]},index=trtids_o)
                t['label'] = 1

                if (c['outcome'] < 0).sum() > 0 or (t['outcome'] < 0).sum() > 0 or np.isin(trtids_o,trt_to_exclude).sum()>0:
                    print("not excluded...",binid)
                    pdb.set_trace()
                towrite[outcome] = pd.concat((towrite[outcome],c,t),axis=0)
                if towrite[outcome].shape[0] > 10000:
                    write_outcome(towrite[outcome], outc_fnames[outcome])
                    towrite[outcome] = pd.DataFrame()
    if single_outcomes:
        if towrite.shape[0] > 0:
            write_outcome(pd.concat((towrite.loc[:,['ipw','label']],
                         towrite.drop(['ipw','label'],axis=1)),axis=1),
                        outc_fnames)
    else:
        for outcome in outc_fnames:
            
            write_outcome(towrite[outcome], outc_fnames[outcome])
    #pdb.set_trace()
    run_R_est(outc_fnames)
    #savepref + hisdir + " " + pspref + "." +

'''    
for i in glob.glob("*iptw"):
    if not os.path.exists(i.replace("iptw","eff")):
        print("doing :",i)
        subprocess.call("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/run_eff.R " + i.replace(".iptw","") + " " + " " + str(True) + " " + str(True) ,shell=True)    
'''

def boot_one(outc_fnames, namereplace = 'PSM',
             single_outcomes=True, weighting=True):
    if single_outcomes == False:
        raise Exception("multi outcome bootstrap NOT IMPLEMENTED")
    outc = pd.read_csv(outc_fnames + ".iptw",sep="\t")
    t = outc.loc[outc['label']==1,:]
    c = outc.loc[outc['label']==0,:]    
    for i in range(10):
        sel = np.random.choice(t.shape[0], t.shape[0])
        fn = outc_fnames.replace(namereplace, namereplace + "boot" + str(i)) 
        pd.concat((t.iloc[sel,:], c.iloc[sel,:]),axis=0).to_csv(fn+ ".iptw",sep="\t",index=False)
        print("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/run_eff.R " + fn  + " " + " " + str(single_outcomes) + " " + str(weighting) )
        subprocess.call("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/run_eff.R " + fn + " " + " " + str(single_outcomes) + " " + str(weighting) ,shell=True)
