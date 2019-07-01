import pandas as pd
import numpy as np
import pickle
import pdb
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import regression_splines as rs
from scipy import sparse
import scipy
from sklearn.preprocessing import MaxAbsScaler
import caliper2 as caliper
from collections import defaultdict
import os
import his2ft
import tables
import ps_match
'''
def gen_embname(hisdir, savename):
    return hisdir + savename + "binembeds.pytab"
def gen_outcname(hisdir, savename):
    return hisdir + savename + "outcomes.pytab"
'''

    #ctl_propensity_score(hisdir,name, ctlfile,outname)
def out_exclude(trtcompare,trtoutc, ctlids, ctloutcomes):
    to_exclude = defaultdict(list) # {o:[[],[]] for o in outcome}
    poo = list(zip(*tuple((ctlids, list(ctloutcomes))))) + list(zip(*tuple((trtcompare.index, trtoutc))))
    for pid, ctlpat in poo:
        for ou in [ctlpat[i] for i in range(1,len(ctlpat), 2)
                   if ctlpat[i+1] <= 0 ]:
            to_exclude[ou].append(pid)
    return to_exclude


def binfo(trtinfo, binid):
    node = trtinfo['drugbins'].get_node("/" + binid)
    ids = node[:,0]
    bindat = node[:,1]
    if node.shape[1] > 6:
        bindat = trtinfo['scaler'].transform(node[:,6:])
    trt_compare = pd.DataFrame(bindat,index=ids)      
    #outc = trtinfo['outcomes'].get_node("/" + binid)
    outc = trtinfo['drugbins'].get_node("/outcomes/" + binid)
    return trt_compare, outc

def one_bin_NN_pscaliper(trt_compare, ctldat, dcutoff, cutoff, prec, psmod):
    trtps = psmod.loc[trt_compare.index]
    ctlids = ctldat.index
    x = np.tile(psmod.loc[ctlids],(trt_compare.shape[0],1))
    smaller = ctldat; bigger = trt_compare

    ## this one exists to check if there are any possible matches in caliper
    pairrectangle = pd.DataFrame((trtps.values - x.transpose()).transpose(),index=trtps.index, columns=ctlids).abs()
    ## this one just exists to look up if particular pairs are in caliper
    pairstack = pairrectangle.transpose().stack()
    smaller_is_ctl = True
    if ctldat.shape[0] > trt_compare.shape[0]:
        smaller = trt_compare; bigger = ctldat
        pairstack = pairrectangle.stack()        
        pairrectangle = pairrectangle.transpose()
        smaller_is_ctl = False

    SMALL_OUT = []
    BIG_OUT = []    
    #pdb.set_trace()
    i = 0
    while smaller.shape[0] > 0 and bigger.shape[0] > 0:
        #pdb.set_trace()
        ### filter out = only try to match if there's a PS match in caliper
        '''
        rowsel = pairrectangle.min(axis=1) < cutoff
        colsel = pairrectangle.min(axis=0) < cutoff
        bigger = bigger.loc[rowsel,:]
        smaller = smaller.loc[colsel,:]
        pairrectangle = pairrectangle.loc[rowsel,colsel]
        if bigger.shape[0] == 0:
            break
        '''
        #idout_len = len(IDOUT)
        #t0 = time.time()
        #print("NN")
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric="mahalanobis",metric_params={'VI':prec}).fit(bigger)
        distances, indices = nbrs.kneighbors(smaller)
        #pdb.set_trace()
        nns = pd.DataFrame({'dist':distances[:,0],
                            'ind':bigger.index[indices[:,0]]},index=smaller.index)
        ### remove with PS match > caliper
        nns = nns.loc[np.array(pairstack.loc[list(zip(*tuple((nns.index, nns['ind']))))] < cutoff),:].sort_values('dist')
        '''
        nncaliper = np.array(nns['dist'] < dcutoff)
        remove = nns.loc[~nncaliper,:]
        #pairrectangle.loc[nncaliper.index, nncaliper['ind']]
        pr = pairrectangle.transpose().stack()
        pr.loc[list(zip(*tuple((remove.index, remove['ind']))))] = cutoff
        pairrectangle = pr.unstack().transpose()
        '''
        nnb = nns.loc[np.array(nns['dist'] < dcutoff),:].drop_duplicates('ind')
        i += 1
        #if i > 10:
        #    pdb.set_trace()
        if nnb.shape[0] == 0:
            break
        BIG_OUT += list(nnb['ind']) ### ind --> bigger
        SMALL_OUT += list(nnb.index) ### index --> smaller            
        smaller = smaller.drop(nnb.index,axis=0)
        bigger = bigger.drop(nnb['ind'],axis=0)
        #pdb.set_trace()

        #pairrectangle = pairrectangle.drop(nnb['ind'], axis=0).drop(nnb.index,axis=1)

    return (BIG_OUT, SMALL_OUT) if smaller_is_ctl else (SMALL_OUT, BIG_OUT) 


def one_bin_PSM(trt_compare, ctldat, cutoff, binid):
    IDOUT = []
    
    smaller = ctldat; bigger = trt_compare
    smaller_is_ctl = True    
    if ctldat.shape[0] > trt_compare.shape[0]:
        smaller = trt_compare; bigger = ctldat
        smaller_is_ctl = False
    #pdb.set_trace()
    i = 0
    SMALL_OUT = []
    BIG_OUT = []    
    while smaller.shape[0] > 0 and bigger.shape[0] > 0:
        #idout_len = len(IDOUT)
        i += 1
        #if i > 500:
        #    pdb.set_trace()
        trtadd = []
        ctladd = []
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric="manhattan").fit(bigger.values.reshape(-1,1))
        distances, indices = nbrs.kneighbors(smaller.values.reshape(-1,1))
        nns = pd.DataFrame({'dist':distances[:,0],
                            'ind':bigger.index[indices[:,0]]},index=smaller.index)
        nnb = nns.sort_values('dist').drop_duplicates('ind')
        #pdb.set_trace()

        nnb = nnb.loc[nnb['dist'] < cutoff,:]
        if nnb.shape[0] == 0:
            #pdb.set_trace()
            #print("ran out for " + binid)
            break
        #ctladd = list(ctlids[nnb['ind']])
        BIG_OUT += list(nnb['ind']) ### ind --> bigger
        SMALL_OUT += list(nnb.index) ### index --> smaller            
        smaller = smaller.drop(nnb.index,axis=0)
        bigger = bigger.drop(nnb['ind'],axis=0)
    return (BIG_OUT, SMALL_OUT) if smaller_is_ctl else (SMALL_OUT, BIG_OUT) 
'''
def featlab2modpred(dense, hisft, lab, alphas, l1s):
    spline_info = rs.get_spline_info(dense[lab==1,:])
    splinify = rs.info2splines(spline_info)
    dense = splinify(dense)
    XS = sparse.hstack((dense,hisft), format='csr')
    scaler = MaxAbsScaler()    
    XS = scaler.fit_transform(XS)

    #splits = rs.get_splitvec(lab.shape[0], nf = numfold)
    #pens = [.0001]
    #pens = 10**np.arange(-6.0,-2.0)            #.001, .0005
    #pdb.set_trace()


    #### get cross validation
    #xval = rxsd.run_regrs(XS, lab, splits, numfold,iter=5000000)

    
    xval = np.zeros(5)
    #pdb.set_trace()
    #if len(alphas) > 1:
    xval = rs.cross_val(XS, lab, 5,iter=5000000, alphas=alphas, l1s=l1s)
    mods, preds, roc = mod_pred(XS, lab, alphas, l1s)    
    modpred = {'mod':mods, 'lab':lab,'scaler':scaler,
                 'preds':preds,'roc':roc,'xval':xval[4]}
    return modpred

def mod_pred(XS, lab, alphas, l1_rs):    
    iter = int(5000000/XS.shape[0]) #500 00000


    mods = rs.make_mods(iter, alphas, l1_rs)
    preds = {}
    roc = {}
    for k in mods:
        mods[k].fit(XS, lab)
        mods[k].intercept_ = mods[k].intercept_ + np.log(lab.mean()/(1-lab.mean()))
        #pdb.set_trace()
        preds[k] = mods[k].predict_proba(XS)[:,1]
        roc[k] = roc_auc_score(lab, preds[k])
    return mods, preds, roc
'''
    
def one_bin_NN(trt_compare, ctldat, cutoff, binid, prec,nlimit=0):

    IDOUT = []
    if nlimit > 0 and trt_compare.shape[0] > nlimit:
        sel = np.random.choice(np.arange(trt_compare.shape[0]), nlimit,replace=False)
        trt_compare = trt_compare.iloc[sel,:]

    
    smaller = ctldat; bigger = trt_compare
    if ctldat.shape[0] > trt_compare.shape[0]:
        smaller = trt_compare; bigger = ctldat
    #pdb.set_trace()
    i = 0
    while smaller.shape[0] > 0 and bigger.shape[0] > 0:
        #idout_len = len(IDOUT)
        i += 1
        #if i > 500:
        #    pdb.set_trace()
        trtadd = []
        ctladd = []
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',metric="mahalanobis",metric_params={'VI':prec}).fit(bigger)
        distances, indices = nbrs.kneighbors(smaller)
        nns = pd.DataFrame({'dist':distances[:,0],
                            'ind':bigger.index[indices[:,0]]},index=smaller.index)
        nnb = nns.sort_values('dist').drop_duplicates('ind')
        #pdb.set_trace()
        nnb = nnb.loc[nnb['dist'] < cutoff,:]
        if nnb.shape[0] == 0:
            #pdb.set_trace()
            #print("ran out for " + binid)
            break
        #ctladd = list(ctlids[nnb['ind']])
        IDOUT.append(list(nnb['ind']))
        IDOUT.append(list(nnb.index))
        #if len(IDOUT) == idout_len:
        #    break
        smaller = smaller.drop(nnb.index,axis=0)
        bigger = bigger.drop(nnb['ind'],axis=0)
    return IDOUT

