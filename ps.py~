import time
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
import regression_splines as rs
import scipy
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score
import file_names

def get_ps(psmod):
    settings = list(psmod['preds'].keys())
    modsetting = settings[0] if len(settings) == 1 else psmod['xval'].mean(axis=1).idxmax() 
    return pd.Series(psmod['preds'][modsetting],index=psmod['ids'])


def get_sparse(pref, ftsuffix):
    sp = [sparse.load_npz(pref + ("" if ft =="ago" else "." + ft) + ".npz" )
          for ft in ftsuffix]
    if len(sp)==1:
        return sp[0]
    else:
        return sparse.hstack(sp,format='csr')
def agobins(hisft):
    hisft.data = 800 - hisft.data
    sta = sparse.hstack([hisft >= (800-4), hisft >= (800 - 52), hisft > 0],format='csr')
    return sta

def load_sparse_mat(name, filename='store.h5'):
    """
    Load a csr matrix from HDF5

    Parameters
    ----------
    name: str
        node prefix in HDF5 hierarchy

    filename: str
        HDF5 filename

    Returns
    ----------
    M : scipy.sparse.csr.csr_matrix
        loaded sparse matrix
    """
    with tables.open_file(filename) as f:

        # get nodes
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(getattr(f.root, "{:s}_{:s}".format(name,attribute)).read())

    # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M

def load_chunk(trt_h5, tn, trt_useids, sparse_index):
    #node = trt_h5.get_node("/" + tn + "_den")
    node = trt_h5.get_node("/" + tn )
    #den = node.den
    ids = node.den[:,0] #ids = node[:,0]
    sel = np.isin(ids, trt_useids)
    if sel.sum() == 0:
        return None
    #trt_dense = node[:][sel,:]
    trt_dense = node.den[:][sel,:]    
    attributes = []
    for attribute in ('data', 'indices', 'indptr', 'shape'):
        #attributes.append(getattr(node, "{:s}_{:s}".format(tn,attribute)).read())
        attributes.append(getattr(node, attribute).read())
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    #sm = np.array((M > 0).sum(axis=0))[0,:][sparse_index-1]
    M = M[sel,:][:,sparse_index - 1] ## subtract 1 because took out "zero", the column labels if we had them would start from 1
    #M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3]).sum(axis=0)[sparse_index]
    return trt_dense, M #, sm

def chunk_list(trt_h5):
    return [i._v_name for i in trt_h5.walk_groups()][1:]
    #return [n.name.split("_")[0] for n in trt_h5.list_nodes(trt_h5.root)
    #           if n.name.endswith("_data")]

def load_selected(trt_h5, trt_useids, sparse_index):
    trt_sparse = []
    trt_dense = []
    #ftsum = np.zeros(30285) #sparse_index.shape[0])
    for tn in chunk_list(trt_h5):
        d = load_chunk(trt_h5, tn, trt_useids, sparse_index)
        if not d:
            continue
        d, s = d
        trt_dense.append(d)
        trt_sparse.append(s)
        #ftsum += sm
    #pdb.set_trace()
    return trt_dense, trt_sparse
                
## hisdir = where all files are
## name = name of treated -> look for name+".den", name +".npz"
## ctlname = same as name but for ctl
## idfile = if you need to filter the IDs.  in some cases, the ids will exactly match what's in the feature files
def ctl_propensity_score(hisdir, trt_drugid, ctl_drugid,idfile, ftsuffix=['ago'], transfunc=agobins,
                         save_prefix = '',ft_exclude=[],
                         alphas=10**np.arange(-6.0,-2.0),l1s=[.3,.2],batch_test=False,index_in = ''): #, eltfilter={}, demfilter={}):
    if len(ftsuffix) > 1:
        save_prefix += "-".join(ftsuffix)


    fsave = idfile + ".psmod.pkl" + ("BT" if batch_test else "") # hisdir + save_prefix
    if index_in:
        fsave = fsave.replace("ids.psmod",index_in + ".ids.psmod")

    if os.path.exists(fsave):
        modpred = pickle.load(open(fsave,'rb'))
        return [fsave,modpred['xval']]

    #runname, trtname = file_names.get_trt_names(hisdir, trt_drugid)

    trt_useids = np.loadtxt(idfile + ".trt")
    ctl_useids = np.loadtxt(idfile + ".ctl")
    if len(trt_useids) + len(ctl_useids) < 100:
        print("Aborting ps... < 100 ids",idfile)
        return "", 0
    print("reading file:",  idfile)
    #pdb.set_trace()
    trt_h5 = tables.open_file(file_names.sparseh5_names(hisdir, trt_drugid),'r')
    ctl_h5 = tables.open_file(file_names.sparseh5_names(hisdir, ctl_drugid),'r')
    modpred = {}
    #sparse_index = file_names.get_sparse_index(hisdir, trtid)
    elct = pickle.load(open(file_names.sparse_index_name(hisdir, trt_drugid),'rb'))
    SPFT_CUT = 100
    sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > SPFT_CUT,:].index)),
                           dtype = int)
    if sparse_index.shape[0] == 0:
        sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > 10,:].index)),
                           dtype = int)
        
    sparse_index = np.delete(sparse_index,0)
    if index_in:
        pdb.set_trace()
        for i in ['.trt','.ctl']:
            os.symlink(os.path.basename(idfile + i), idfile.replace(".ids","."+ index_in +".ids" )+i)
        sparse_index = np.loadtxt(index_in)
    sparse_index = sparse_index[~np.isin(sparse_index, ft_exclude)]

    #print("ALWAYS BATCH")

    trt_dense, trt_sparse = load_selected(trt_h5, trt_useids, sparse_index)
    ctl_dense, ctl_sparse = load_selected(ctl_h5, ctl_useids, sparse_index)
    trt_dense = np.vstack(trt_dense)
    ctl_dense = np.vstack(ctl_dense)
    dense = np.vstack((trt_dense, ctl_dense))
    lab = np.hstack((np.ones(trt_dense.shape[0]),np.zeros(ctl_dense.shape[0])))
    del trt_dense, ctl_dense
    ids = dense[:,0]
    dense = dense[:,1:]
    hisft = sparse.vstack(trt_sparse+ ctl_sparse,format='csr')
    del trt_sparse, ctl_sparse
    if transfunc:
        hisft = transfunc(hisft)
    keep = np.array((hisft > 0).sum(axis=0))[0,:]
    keep = (keep > 100) & (keep < .7*hisft.shape[0])

    if (~keep).sum() > 0:
        hisft = hisft[:,keep]
        print("FILTERING ultrasparse:",(~keep).sum(), "->",hisft.shape, ' for ', fsave )
    modpred = featlab2modpred(dense, hisft, lab, alphas, l1s)
    modpred['ids'] = ids
    #if batch_test or len(trt_useids) + len(ctl_useids) < 1500000: #200000000: #150: #        
    #else:
    #    modpred = batch_ps(trt_h5, trt_useids, ctl_h5, ctl_useids,transfunc,alphas,l1s)
    f = open(fsave,'wb')
    pickle.dump(modpred,f)
    f.close()
    return fsave, modpred['xval']

            

def agoexp(hisft):
    xx = np.array(hisft.data)
    hisft.data = np.exp(-1*(xx)**2/360)
    return hisft

    
def featlab2modpred(dense, hisft, lab, alphas, l1s):
    spline_info = rs.get_spline_info(dense[lab==1,:])
    splinify = rs.info2splines(spline_info)
    dense = splinify(dense)
    
    XS = sparse.hstack((dense,hisft), format='csr')
    #print("soaving!")
    #f = open("test5358.pkl",'wb')
    #pickle.dump((XS, lab), f)
    #f.close()


    modpred =  {'lab':lab} 
    if len(l1s) > 1 or len(alphas) > 1:
        xval = rs.cross_val(XS, lab, 5,iter=15000000, alphas=alphas, l1s=l1s)
        #xval = xval[4]
        #pdb.set_trace()
        modpred.update({'xval':xval[4],'mods':xval[0],
                        'scaler':xval[1],'preds':xval[3]})
    else:
        ## only after cross-val should you do this...
        scaler = MaxAbsScaler()    
        XS = scaler.fit_transform(XS)
        mods, preds, roc = mod_pred(XS, lab, alphas, l1s)
        modpred.update({'xval':np.zeros(5), 'mods':mods,
                        'scaler':scaler,'preds':preds})
    #modpred = {'mod':mods, 'lab':lab,'scaler':scaler,
    #             'preds':preds,'roc':roc,'xval':xval}
    return modpred

def mod_pred(XS, lab, alphas, l1_rs):    
    max_iter = int(15000000/XS.shape[0]) #500 00000
    mods = rs.make_mods(max_iter, alphas, l1_rs)
    preds = {}
    roc = {}
    for k in mods:
        mods[k].fit(XS, lab)
        mods[k].intercept_ = mods[k].intercept_ + np.log(lab.mean()/(1-lab.mean()))
        #pdb.set_trace()
        preds[k] = mods[k].predict_proba(XS)[:,1]
        roc[k] = roc_auc_score(lab, preds[k])
    return mods, preds, roc

def batch_ps(trt_h5, trt_useids, ctl_h5, ctl_useids, transfunc, alpha, l1):
    tot_people = len(trt_useids) + len(ctl_useids)    
    class_weight = {i:tot_people/(2*ct) for i,ct in
                    enumerate([len(ctl_useids), len(trt_useids)])}

    ### 1) go through and get feat counts, and get spline info
    trt_sparse = []
    trt_dense = []
    splinify = None
    ftarr = None
    tf = time.time()    
    t0 = time.time()
    for tn in chunk_list(trt_h5):
        d, s = load_chunk(trt_h5, tn, trt_useids)
        if not splinify:
            spline_info = rs.get_spline_info(d[:,1:])
            splinify = rs.info2splines(spline_info)
        s = transfunc(s).sum(axis=0)[0,:]
        if not ftarr:
            ftarr = s
        else:
            ftarr += s
    for tn in chunk_list(ctl_h5):
        d, s = load_chunk(ctl_h5, tn, ctl_useids)
        ftarr += transfunc(s).sum(axis=0)[0,:]

    keep = np.array((ftarr > 100 ) & (ftarr < .7*(tot_people)))[0,:]
    numfold = 5
    max_iter = int(np.ceil(15000000/(tot_people*(numfold - 1)/numfold)))

    ### 2) fit scalers
    fscaler = [MaxAbsScaler() for _ in range(numfold)]
    tot_scaler = MaxAbsScaler() 
    def fit_scaler_xval(XS, lab):
        splits = rs.get_splitvec(XS.shape[0], nf = len(fscaler))
        for i, scaler in enumerate(fscaler):
            scaler.partial_fit(XS[splits != i])
        tot_scaler.partial_fit(XS)
    batch_do(trt_h5, trt_useids, ctl_h5, ctl_useids, splinify, transfunc, keep, fit_scaler_xval,
             chunk=100000)
    
    print("scaler in:",(time.time()-t0)/60)
    t0 = time.time()
    alph_do = [alpha[0]]; l1s_do = l1[0];
    if len(alpha) > 1 or len(l1) > 1:
        ### 3) fit cross-val models
        sgdmods, xval_pred, xval_lab = pickle.load(open("blah.pkl",'rb'))
        '''
        sgdmods = [rs.make_mods(42, alpha, l1,class_weight=None)
                   for _ in range(numfold)]
        def fit_mod_xval(XS, lab):
            sample_weight = lab*class_weight[1] + (1-lab)*class_weight[0]
            splits = rs.get_splitvec(XS.shape[0], nf = numfold)
            for i, mods in enumerate(sgdmods):
                for m in sgdmods[i]:
                    sgdmods[i][m].partial_fit(fscaler[i].transform(XS[splits != i,]), lab[splits!=i], sample_weight=sample_weight[splits!=i],classes=[0,1])
        for it in range(max_iter):
            batch_do(trt_h5, trt_useids, ctl_h5, ctl_useids, splinify, transfunc, keep, fit_mod_xval,
                 chunk=100000)
        print("xval in:",(time.time()-t0)/60)
        t0 = time.time()

        ### 4) predict from cross -val models
        xval_pred = [{k:[] for k in sgdmods[0]} for _ in range(numfold)]
        xval_lab = [[] for _ in range(numfold)]           
        def eval_mod_xval(XS, lab):
            #sample_weight = lab*class_weight[1] + (1-lab)*class_weight[0]
            splits = rs.get_splitvec(XS.shape[0], nf = numfold)
            for i, mods in enumerate(sgdmods):
                xval_lab[i].append(lab[splits==i])
                for m in mods:
                    xval_pred[i][m].append(sgdmods[i][m].predict_proba(fscaler[i].transform(XS[splits == i,]))[:,1])
        batch_do(trt_h5, trt_useids, ctl_h5, ctl_useids, splinify, transfunc, keep, eval_mod_xval,
                 chunk=100000)
        print("xval-eval in:",(time.time()-t0)/60)
        t0 = time.time()
        '''
        ### get cross-val ROC
        xval = pd.DataFrame(index = list(sgdmods[0].keys()), columns = np.arange(numfold))
        for f in range(numfold):
            labs = np.hstack(xval_lab[f])
            for m in sgdmods[f]:
                xval.loc[m,f] = roc_auc_score(labs, np.hstack(xval_pred[f][m]))

        ### 5) just do the one best model: without cross-validation
        modsel = xval.mean(axis=1).idxmax().split("-")
        l1s_do = [float(modsel[0])]
        alph_do = [float(modsel[1])]

    optmod = list(rs.make_mods(42, alph_do, l1s_do, class_weight=None).values())[0]
    def final_fit(XS, lab):
        sample_weight = lab*class_weight[1] + (1-lab)*class_weight[0]
        optmod.partial_fit(tot_scaler.transform(XS), lab, sample_weight=sample_weight,classes=[0,1])
    for it in range(max_iter):
        batch_do(trt_h5, trt_useids, ctl_h5, ctl_useids, splinify, transfunc, keep,final_fit,
             chunk=100000)
    print("opt-mod in:",(time.time()-t0)/60)
    t0 = time.time()

    ### 6) after full iters, get final predictions
    frac_treated = trt_useids/tot_people
    optmod.intercept_ = optmod.intercept_ + np.log(frac_treated/(1-frac_treated))    
    preds = []
    labs = []
    def final_pred(XS, lab):
        labs.append(lab)
        preds.append(optmod.predict_proba(tot_scaler.transform(XS))[:,1])
    batch_do(trt_h5, trt_useids, ctl_h5, ctl_useids, splinify, transfunc, keep,final_pred,
             chunk=100000)
    print("final pred in:",(time.time()-t0)/60)
    print("TOT in:",(time.time()-tf)/60)

    return {'xval':xval, 'mod':optmod, 'scaler':scaler,
            'preds':np.hstack(preds), 'lab':np.hstack(labs)}
        
def batch_do(trt_h5, trt_useids, ctl_h5, ctl_useids, splinify, transfunc, keep, do_thing,
             chunk=100000):
    tot_people = len(trt_useids) + len(ctl_useids)
    num_chunks = int(tot_people/chunk)
    chunk = int(tot_people/num_chunks)
    trt_desired = int(len(trt_useids)/tot_people*chunk)
    ctl_desired = int(len(ctl_useids)/tot_people*chunk)
    ctl_chunks = chunk_list(ctl_h5)
    trt_chunks = chunk_list(trt_h5)

    tsparse = []; csparse = [];
    tdense = []; cdense = [];
    ttot = 0; ctot = 0;            
    trt_chix = 0; ctl_chix = 0;
    #pdb.set_trace()
    tdone = 0; cdone = 0
    chunk_number = 0
    #while ctl_chix < len(ctl_chunks) and trt_chix < len(trt_chunks):

    while tdone < len(trt_useids) and cdone < len(ctl_useids):
        ### get this chunk of treated
        chunk_number += 1
        while ttot < trt_desired:
            d, s = load_chunk(trt_h5, trt_chunks[trt_chix], trt_useids)
            ttot += d.shape[0]
            trt_chix += 1
            tdense.append(d)
            tsparse.append(s)
        tsparse_do = sparse.vstack(tsparse)
        tdense_do = np.vstack(tdense)
        if chunk_number < num_chunks:
            tsparse = [tsparse_do[trt_desired:,:]]
            tdense = [tdense_do[trt_desired:,:]]
            tsparse_do = tsparse_do[:trt_desired,:]
            tdense_do = tdense_do[:trt_desired,:]

        ### get this chunk of ctl
        while ctot < ctl_desired:
            d, s = load_chunk(ctl_h5, ctl_chunks[ctl_chix], ctl_useids)
            ctot += d.shape[0]
            ctl_chix += 1
            cdense.append(d)
            csparse.append(s)
        csparse_do = sparse.vstack(csparse)
        cdense_do = np.vstack(cdense)
        if chunk_number < num_chunks:
            csparse = [csparse_do[ctl_desired:,:]]
            cdense = [cdense_do[ctl_desired:,:]]
            csparse_do = csparse_do[:ctl_desired,:]
            cdense_do = cdense_do[:ctl_desired,:]

        ###
        ttot = tdense[0].shape[0]
        ctot = cdense[0].shape[0]
        tdone += tsparse_do.shape[0]; cdone += cdense_do.shape[0]
        #### now, fit
        dense_do = np.vstack((tdense_do, cdense_do))
        ids = dense_do[:,0]
        dense_do = splinify(dense_do[:,1:])
        sparse_do = transfunc(sparse.vstack((tsparse_do, csparse_do),format='csr'))[:,keep]
        lab = np.hstack((np.ones(tdense_do.shape[0]),np.zeros(cdense_do.shape[0])))
        #pdb.set_trace()

        XS = sparse.hstack((dense_do,sparse_do), format='csr')
        do_thing(XS, lab)
        print("batch do:",tdone, cdone)
    print("of wanted:", len(trt_useids),  len(ctl_useids))

'''        
            for mod in mods:
                mod.partial_fit(XS,lab,classes=[0,1], sample_weight=sample_weight)
    #iter = int(5000000/(splits != f).sum())
    sgdmods = rs.make_mods(iter, alpha, l1,class_weight=None)
    for r in range(int(iter)*4):
        for i in range(0,XS.shape[0],chunk):
            chunkix = np.arange(i,min(i+chunk,XS.shape[0]))
            chunkix = chunkix[splits[chunkix]!=f]
            sample_weight = lab[chunkix]*class_weight[1] + (1-lab[chunkix])*class_weight[0]
            sgdmods['0.2-0.0001'].partial_fit(XS[chunkix,:],lab[chunkix],classes=[0,1],
                               sample_weight=sample_weight)
        #print("finished iter",r)
    t1 = time.time()
    xvroc[f] = roc_auc_score(lab[splits==f], sgdmods['0.2-0.0001'].predict_proba(XS[splits==f])[:,1])
    print(f, t1-t0, xvroc[f])    
    t0 = time.time()
    dense = pd.read_csv(hisdir + name + ".den",sep="\t",header=None)
    hisft = get_sparse(hisdir + name, ftsuffix)
    ids = dense[0]    
    dense = dense.values[:,1:] ## first col is ID
    
    ### for treated, you need to get the IDs that matched to this CTL
    ### because treated history files are not specific to this CTL

    sel = np.isin(ids,useids)
    if (~np.isin(useids,ids)).sum() > 0:
        pdb.set_trace()
    ids = ids[sel]
    hisft = hisft[sel,:]
    dense = dense[sel,:]

    dense2 = pd.read_csv(hisdir + ctlname + ".den",sep="\t",header=None)
    hisft2 = get_sparse(hisdir + ctlname, ftsuffix)
    ctlids = dense2[0]
    ### For NN matching: for CTL, we already filtered (dense, ft) so that
    ### they are the matched only in the prepare_sparsemat step
    ###   -- BUT if ps match it hasn't been filtered yet
    #pdb.set_trace()

    if (~np.isin(filt_cid,ctlids)).sum() > 0:
        pdb.set_trace()
    
    if len(filt_cid) != dense2.shape[0]:
        sel = np.isin(ctlids,filt_cid)
        ctlids = ctlids[sel]
        dense2 = dense2.loc[sel,:]
        hisft2 = hisft2[sel,:]


    ids = np.hstack((ids, ctlids))

    lab = np.hstack((np.ones(dense.shape[0]),np.zeros(dense2.shape[0])))
    dense = np.vstack((dense, dense2.values[:,1:])) #.values[:,2:]))
    hisft = sparse.vstack((hisft, hisft2),format='csr')
    if transfunc:
        hisft = transfunc(hisft)

    t_load = time.time()

    keep = np.array((hisft > 0).sum(axis=0))[0,:]
    #print("removing high: ", (keep >= .7*hisft.shape[0]).sum())
    #print("removing low: ", (keep <= 100).sum())    
    keep = (keep > 100) & (keep < .7*hisft.shape[0])

    if (~keep).sum() > 0:
        hisft = hisft[:,keep]
        print("FILTERING ultrasparse:",(~keep).sum(), "->",hisft.shape, ' for ', fsave )

    t_colfilt = time.time()                
    modpred = featlab2modpred(dense, hisft, lab, alphas, l1s)
    t_regr = time.time()                    
    modpred['ids'] = ids #p = list(modpred['preds'].values())[0]
    print("time:" + ("{:1.2f}\t"*3).format(t_load - t0, t_colfilt - t_load, t_regr - t_colfilt))
'''
