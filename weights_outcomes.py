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
#sys.path.append("../12.21_allcode_ctlmatch")
import his2ft
import tables
import matching
import subprocess
import file_names
import regression_splines as rs
import ps
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


def run_temporal(hisdir, outc_fnames, drugid, ctl, outcomes_sorted,time_chunk,agg=1):

    #if not os.path.exists(outc_fnames + ".trt.censwt.bz2") and
    if time_chunk0:        
        censoring_weights(hisdir, outc_fnames, drugid, ctl, outcomes_sorted,drugid, time_chunk, agg=agg)
        #if not os.path.exists(outc_fnames + ".ctl.censwt.bz2") and time_chunk>0:                    
        censoring_weights(hisdir, outc_fnames, ctl, drugid, outcomes_sorted,drugid, time_chunk,agg=agg)

    #if not os.path.exists(outc_fnames + ".msmeff.txt") and time_chunk>0:                           
    subprocess.call("Rscript --vanilla run_eff.R " + outc_fnames + "  temporal" +
                    (" agg-" + str(agg) if agg > 1 else " X"),shell=True)

## single_outcomes = if TRUE then you have ALL outcomes must not have happened at time of drug prescription. if FALSE then you are considering outcomes independently, in which case some of the people in the data might have a history of the outcome and you would need to exclude those people from the matching, do a separate matching  for each outcome, more or less a pain in ass
def outcome_info(hisdir, outc_fnames, drugid, ctl, outcomes_sorted, trt_to_exclude=[],whichdo='', single_outcomes=True, weighting=False,time_chunk=0,agg=1):
    print("outcome_info time_chunk=",time_chunk)    
    def run_R_est(outc_fnames):
        print("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/run_eff.R " + outc_fnames + " " + " " + str(single_outcomes) + " " + str(weighting) )
        subprocess.call("Rscript --vanilla run_eff.R " + outc_fnames + " " + 
                        ("single_cross_section"  if single_outcomes else "multi_cross_section")
                        + " " + str(weighting) ,shell=True)
    time_chunk = int(time_chunk)
    def run_temporal():
        if time_chunk==0:
            return
        #if not os.path.exists(outc_fnames + ".trt.censwt.bz2") and
        
        censoring_weights(hisdir, outc_fnames, drugid, ctl, outcomes_sorted,drugid, time_chunk, agg=agg)
            #if not os.path.exists(outc_fnames + ".ctl.censwt.bz2") and time_chunk>0:                    
        censoring_weights(hisdir, outc_fnames, ctl, drugid, outcomes_sorted,drugid, time_chunk,agg=agg)
            
        #if not os.path.exists(outc_fnames + ".msmeff.txt") and time_chunk>0:                           
        subprocess.call("Rscript --vanilla run_eff.R " + outc_fnames + "  temporal" +
                        (" agg-" + str(agg) if agg > 1 else " X"),shell=True)
        
    #outc_fnames = hisdir + pspref
    savepref = 'min-' if whichdo else ''
    #pdb.set_trace()
    if single_outcomes:
        if os.path.exists(outc_fnames + ".iptw"): #("" if not weighting else ".unwt") +
            print("Exists, returning: " + savepref + outc_fnames + ".iptw")
            if not os.path.exists(outc_fnames + ("" if weighting else ".unwt") + ".eff"):
                run_R_est(outc_fnames)
            print("... for" + savepref + outc_fnames + ".iptw time_chunk=",time_chunk)
            run_temporal()
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
    if time_chunk > 0:
        run_temporal()
    #savepref + hisdir + " " + pspref + "." +

def get_weights(XS, lab, full_interval, savename):
    den = {}
    if not os.path.exists(savename):
        den = rs.cross_val(XS, lab, 5,iter=15000000,alphas=10**np.arange(-4.0,-2.0),
                           l1s=[.3,.2], fit_sel=full_interval==1)
        den = {'xval':den[4],'mods':den[0],
                                'scaler':den[1],'preds':den[3]}
        
        f = open(savename,'wb')
        pickle.dump(den,f)
        f.close()
    else:
        den = pickle.load(open(savename,'rb'))
    setting = den['xval'].mean(axis=1).idxmax()
    preds = den['preds'][setting]
    #pdb.set_trace()
    preds = full_interval*preds #np.where(full_interval==0, TIME_CHUNK, full_interval)/TIME_CHUNK*preds
    return preds



def censoring_weights(hisdir, outname, drug1, drug2, outcomes_sorted,trt, TIME_CHUNK,
                      agg=1, FILT_ERA=0, drug1_washout=np.inf):
    #dense, spmat, futmat = [np.array(), sparse.csr_matrix(), sparse.csr_matrix()]
    #if not os.path.exists(outname + ".den.trt.pkl") or os.path.exists(outname + ".den.ctl.pkl"):
    save_suff = (".trt" if drug1==trt else ".ctl")
    suff2 =  (".agg-" + str(agg) if agg > 1 else "") +  (".filt-" +  str(FILT_ERA) if FILT_ERA > 0 else '')
    if drug1_washout < np.inf:
        suff2 +=  ".drug1-" + str(drug1_washout)
    print("Starting: ", outname + save_suff + suff2+  ".censwt.bz2")    
    if os.path.exists(outname + save_suff + suff2 +  ".censwt.bz2"):
        print("exists: ", outname + save_suff + suff2+  ".censwt.bz2")
        return
    drug_ids = np.loadtxt(outname + ".ids" + save_suff)
    past_sparse_index = ps.get_sparseindex(hisdir, trt)
    myh5 = tables.open_file(file_names.sparseh5_names(hisdir, drug1),'r')
    _, my_sparse = ps.load_selected(myh5, drug_ids, past_sparse_index)
    my_sparse = sparse.vstack(my_sparse, format='csr')
    keep = np.array((my_sparse > 0).sum(axis=0))[0,:]
    keep = (keep > 50) & (keep < .7*my_sparse.shape[0])
    del _, my_sparse
    #pdb.set_trace()
    past_sparse_index = past_sparse_index[keep]
    
    dense, spmat, futmat, lab, fut_sparse_ix, cens_info = his2ft.censored_sparsemat(hisdir,  past_sparse_index,
                                                                         np.loadtxt(outname + ".ids" + save_suff), drug1, drug2, TIME_CHUNK, agg=agg, washout=drug1_washout)
    #pdb.set_trace()
    if agg > 1:
        TIME_CHUNK = TIME_CHUNK*agg
    lab = np.array(lab)

    '''
    cens_info["as_treated"] = cens_info['interval_end'] if drug1==trt else np.zeros(cens_info.shape[0])
    cens_info = pd.DataFrame({'ids':dense[:,0],

                              #"week":dense[:,2],
                              "censored":lab})
    ## transform week to week SINCE drug
    def offs(w): return list(w - w.min())
    woffs  = pd.DataFrame(dense[:,[0,2]],columns=['ids','week']).groupby("ids")['week'].agg(offs)

    cens_info["interval_start"] = np.hstack([woffs[k] for k in
                                   cens_info['ids'].drop_duplicates()])
    interval_length = np.where(dense[:,1]==0,TIME_CHUNK, dense[:,1])
    cens_info["interval_end"] = cens_info['interval_start'] + interval_length
    '''
    cens_info["as_treated"] = cens_info['interval_end'] if drug1==trt else np.zeros(cens_info.shape[0])
    #pdb.set_trace()    
    if FILT_ERA >  0:
        sel = np.where(cens_info['interval_start']  < FILT_ERA)[0]
        cens_info = cens_info.iloc[sel,:]
        dense = dense[sel,:]
        spmat = spmat[sel,:]
        futmat = futmat[sel,:]
        lab = lab[sel]
    spline_info = rs.get_spline_info(dense[:,2:])
    splinify = rs.info2splines(spline_info)
    formula = rs.get_formula(spline_info)

    ### adding in time since index date 
    dense = np.hstack((splinify(dense[:,2:]), cens_info['interval_start'].values.reshape(-1,1), cens_info['interval_start'].values.reshape(-1,1)**2))
    #full_interval = dense[:,1]
    spmat = ps.agoexp(spmat)
    #xx = sparse.hstack((dense,spmat, futmat),format='csr')
    #xx = np.savetxt("48856160.txt",np.array(xx[np.where(cens_info['ids']==48856160.0)[0],:].todense()))
    #pdb.set_trace()
    keep = np.array((spmat > 0).sum(axis=0))[0,:]
    keep = (keep > 100) & (keep < .7*spmat.shape[0])
    if (~keep).sum() > 0:
        spmat = spmat[:,keep]
        print("FILTERING ultrasparse MSM data:",(~keep).sum(), "->",spmat.shape, ' for ', outname + save_suff)
    cens_info['den'] = get_weights(sparse.hstack((dense,spmat,futmat), format='csr'), lab, interval_length/TIME_CHUNK,
                                   outname + save_suff + suff2 + ".den.pkl")
    cens_info['num'] = get_weights(sparse.hstack((dense,spmat), format='csr'), lab, interval_length/TIME_CHUNK, outname + save_suff +suff2+  ".num.pkl")  

    cens_info['single_wt'] = (1-lab)*(1-cens_info['num'])/(1-cens_info['den']) + \
                             lab*cens_info['num']/(cens_info['den'])
    cens_info['cum_wt'] = cens_info.groupby('ids')['single_wt'].agg('cumprod')

    outcomes = pd.read_csv(outname + ".iptw",sep="\t")
    outcomes = outcomes.loc[outcomes['label']==int(drug1==trt),:].set_index("id")
    def cmat(week):
        patid = week.name
        #print("got patid", patid, week)
        #cancs = np.tile(outc.loc[patid,:][4:],(1,len(week)))
        cancs = outcomes.loc[patid,:][4:].values
        #print(cancs.shape)
        week = np.tile(week, (cancs.shape[0],1)).transpose()
        #print(week.shape)
        inchunk = np.array((cancs >= week) & (cancs < week + TIME_CHUNK) & (cancs > 0), dtype=int)
        befchunk = np.array((cancs < week) & (cancs > 0), dtype=int)
        #print(inchunk.shape)
        return inchunk - befchunk

    gb = cens_info.groupby('ids')['interval_start'].apply(cmat)    
    gb = np.vstack([gb[k] for k in cens_info['ids'].drop_duplicates()])
    cens_info = pd.concat((cens_info,
                           pd.DataFrame(gb,columns=outcomes.columns[4:])),axis=1)
    cens_info = cens_info.loc[cens_info['censored']==0,:] #.drop('lab',axis=1)
    print("SAVING: ", outname + save_suff + suff2+  ".censwt.bz2",  cens_info.shape)
    #pdb.set_trace()
    #with open("LOGGY",'a') as f:
    #    f.write("SAVING: "+ outname + save_suff + suff2+  ".censwt.bz2")
    #cens_info.to_csv(outname + save_suff +suff2+ ".censwt",sep="\t",header=True)        
    cens_info.to_csv(outname + save_suff +suff2+ ".censwt.bz2",sep="\t",header=True,compression='bz2')
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
