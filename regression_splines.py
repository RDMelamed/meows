from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from collections import Counter
import numpy as np
import pandas as pd
import sys
import numpy as np
import pickle
import pdb
import tables
import subprocess
import glob
import resource
import psutil
import os
from itertools import chain
from scipy import sparse
from patsy import dmatrix

os.environ['MKL_THREADING_LAYER'] = 'GNU'
from sklearn.preprocessing import MaxAbsScaler
sys.path.append("../../code")

def getpercentiles(urx,perc=[.2,.4,.6,.8,1]):
    urx = np.cumsum(urx.loc[urx['ct'] > 0,:])
    return [ urx.loc[urx['ct'] <=p*urx['ct'].max(),'ct'].idxmax() for p in perc]

def get_spline_info(trt):
    #mins = np.array([10000]*2)
#maxs = np.array([-10000]*2)
    urx = pd.DataFrame(np.zeros(100),columns=['ct'])
    trx = pd.DataFrame(np.zeros(1000),columns=['ct'])
    tdx = pd.DataFrame(np.zeros(1000),columns=['ct'])
    tpx = pd.DataFrame(np.zeros(1000),columns=['ct'])    
    uy = set([])
    donzo = 0
    #for it in iter(trainbatch_q.get, None):
    #    for dat in it:
    nt = trt.shape[0]
    tstart = 0
    treatper = 10000
    trt = trt[np.random.permutation(trt.shape[0]),:]
    while tstart < nt and donzo < 50000:
        endt =min(tstart + treatper, nt)        
        dft = trt[tstart:endt,:] ### dense = 0->7 (excl)
        donzo += dft.shape[0]
        #mat = dft[:,[0,1]]
        #mins = np.vstack([mins, mat.min(axis=0)]).min(axis=0)
        #maxs = np.vstack([mins, mat.max(axis=0)]).max(axis=0)
        uy |= set(np.array(np.array(dft[:,0]/52.0,dtype=int)))
        urx = urx.add(pd.DataFrame(Counter(dft[:,3]),index=['ct']).transpose(),fill_value=0)
        trx = trx.add(pd.DataFrame(Counter(dft[:,4]),index=['ct']).transpose(),fill_value=0)
        tdx = tdx.add(pd.DataFrame(Counter(dft[:,5]),index=['ct']).transpose(),fill_value=0)
        tpx = tpx.add(pd.DataFrame(Counter(dft[:,6]),index=['ct']).transpose(),fill_value=0)
        #print("its," + str(donzo))
        if donzo > 50000:
            break
            #weekrange = [min(weekrange[0], dft[sel,0].min()),
            #            max(weekrange[1], dft[sel,0].max())]
            #agerange = [min(agerange[0], dft[sel,1].min()),
            #            max(agerange[1], dft[sel,1].max())]
    xtreme = [.05,.95]
    for_formula = {'age':[2, [0,5,10,15,25,40,55,70]  + [90], 1], 
                       'week':[2, list(np.array(sorted(uy))*52) + [12*52],0],
                       'urx':[3,getpercentiles(urx), 3], 
                       'trx':[3,getpercentiles(trx),4],
                       'tdx':[3,getpercentiles(tdx),5],
                       'tpx':[3,getpercentiles(tpx),6]} 

    return for_formula


def info2splines(for_formula):
    '''
    sp = ["bs("+k + ",knots="+str(quant[:-1])+ 
            ",degree="+str(deg) + ",lower_bound=0" +
          ",upper_bound=" + str(quant[-1]) + ",include_intercept=True)-1" 
          for k, (deg, quant, col) in for_formula.items()]
    sp.append('cc(month,knots=[1, 4, 7,10])-1')
    formula = "+".join(sp)
    '''
    formula = get_formula(for_formula)
    def makeX(Xin):
        #d2 = clipper(denseft)
        pt = dmatrix(formula, get_patsydict(for_formula, Xin))
        ###  (sex, splines of other demo pt) ###, sparseft g)
        
        return np.hstack((Xin[:,2].reshape(-1,1), pt, Xin[:,6:])) #, pt.design_info.column_names
    return makeX

def get_patsydict(for_formula, Xin):
    for c, percentiles, col in for_formula.values():
        Xin[:,col] = np.clip(Xin[:,col],1,percentiles[-1])
    dictin = {k:Xin[:,for_formula[k][2]] for k in for_formula}
    dictin['month'] = np.array(Xin[:,for_formula['week'][2]]/4,dtype=int) % 12
    return dictin

def info2dmatrix(for_formula, dns):
    #for c, percentiles, col in for_formula.values():
    #    dns[:,col] = np.clip(dns[:,col],1,percentiles[-1])

    return dmatrix(get_formula(for_formula),get_patsydict(for_formula, dns))
                   
    #               {k:dns[:,for_formula[k][2]] for k in for_formula})

def single_formula(k, deg, quant, col):
    #+  quant[0] + #lower_bound=0
    sp = ("bs("+k + ",knots="+str(quant[:-1])+ 
            ",degree="+str(deg) + ",lower_bound=0" +  
          ",upper_bound=" + str(quant[-1]) + ",include_intercept=True)-1")

    return sp # "+".join(sp)
    
def get_formula(for_formula):
    sp = [single_formula(k, deg, quant,col)
          for k, (deg, quant, col) in for_formula.items()]
    sp.append('cc(month,knots=[1, 4, 7,10])-1')    
    formula = "+".join(sp)
    return formula

def make_mods(iter, alphas, l1s, class_weight='balanced'):
    mods = {}
    for l1 in l1s:
        for p in alphas:
            mods[str(l1) + "-" + str(p)] = SGDClassifier(loss="log", penalty="elasticnet",
                                                alpha=p, l1_ratio=l1,max_iter=iter,
                                                         class_weight= class_weight)
    return mods

def cross_val(XS, lab, numfold, iter=5000000, alphas=[.001,.0005,.0001],l1s=[.3,.2]):
    iter = int(iter/XS.shape[0]) #500 00000
    sgdmods = make_mods(iter, alphas, l1s)
    scaler = MaxAbsScaler()    
    '''
    if iter > 3:
        iter = int(iter/XS.shape[0]) #500 00000

    sgdmods = {str(alph):SGDClassifier(loss="log", penalty="elasticnet",
                                       alpha=alph, l1_ratio=.3,max_iter=iter,
                                        class_weight= 'balanced')
               for alph in alphas}

    for p in alphas:
        sgdmods['r5.' + str(p)] = SGDClassifier(loss="log", penalty="elasticnet",
                                                alpha=p, l1_ratio=.2,max_iter=iter,
                                                class_weight= 'balanced')
    '''
    splits = get_splitvec(lab.shape[0], nf = numfold)        
    rocs = pd.DataFrame(index=sgdmods.keys(), columns=np.arange(numfold))

    #pdb.set_trace()
    if len(l1s) > 1 or len(alphas) > 1:
        #print("hiii")
        for f in range(numfold):
            print("starting fold:")
            scaler.fit(XS[splits!=f,:])
            #pdb.set_trace()        
            Xval = scaler.transform(XS[splits==f,:])
            X = scaler.transform(XS[splits!=f,:])
            print("about to fit fold:")
            for k in sgdmods:
                sgdmods[k].fit(X, lab[splits!=f])
            labval = lab[splits == f]
            print("about to eval fold:")            
            preds = {k:sgdmods[k].predict_proba(Xval)[:,1]
                     for k in sgdmods}
            for k in preds:
                roc = roc_auc_score(labval, preds[k])
                #if roc < .5:
                #    pdb.set_trace()
                print(k,roc)
                rocs.loc[k,f] = roc
            
            print("FOLD ",f)
    preds = {}
    xtrans = scaler.fit_transform(XS)    
    for k in sgdmods:
        sgdmods[k].fit(xtrans, lab)
        sgdmods[k].intercept_ = sgdmods[k].intercept_ + np.log(lab.mean()/(1-lab.mean()))
        preds[k] = sgdmods[k].predict_proba(xtrans)[:,1]
    return sgdmods, scaler, lab, preds, rocs

def get_splitvec(shape, nf=5):
    return np.array(list(np.arange(nf))*int(1+shape/nf))[:shape]

def get_names(pref, dictinfo, suffix="",concat=[]):
    (sparse_index,ct) = pickle.load(open(pref + "trtindex.pkl",'rb'))
    x = pd.read_table(pref + "ctl.den" + suffix,sep="\t",header=None,nrows=1000)
    splinify = info2dmatrix(dictinfo, x.values)
    splinnames = splinify.design_info.column_names
    vocab = pd.read_pickle("../../data/vocab_mat.pkl")
    sparse_names = vocab.set_index('eltid').loc[sparse_index,'name']
    if len(concat) > 1:
        sparse_names = list(chain.from_iterable([list(c + '.' + sparse_names) for c in concat]))
    else:
        sparse_names = list(sparse_names)
    return ['sex'] + splinnames + sparse_names


def get_roc_stats(res):
    return pd.DataFrame({k:res[k][4].mean(axis=1) for k in res})


