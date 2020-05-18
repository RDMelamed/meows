from collections import defaultdict
import tables
import sys
sys.path.append("../../code")
import bin_match
import pandas as pd
import numpy as np
import pickle
import pdb
import csv
import datetime
import time
from scipy.spatial.distance import mahalanobis
import his2ft
import os
import ps_match
import file_names
def caliperdict(trtname, trt,median=True, percentile=90,redo=None,hideous=''):
    if percentile == 0:
        return None
    #savedir, trtname = file_names.get_trt_names(hisdir, trt)
    fname = trtname + hideous + ".caliper." + \
             ("median" if median else "perc") + str(percentile) + ".pkl"
    if os.path.exists(fname): return
    print("making caliper:", fname)
    trtinfo = his2ft.get_trt_info(trtname, trt,hideous)
    
    cuts = defaultdict(float) #pd.DataFrame(np.zeros(bindf.shape[0]),index=bindf['binid'])
    if redo:
        cuts = redo
    for (i,binid) in enumerate(trtinfo['bindf']['binid']):
        #if binid=='t15':
        #    pdb.set_trace()
        if redo and binid in cuts:
            continue
        node = trtinfo['drugbins'].get_node("/" + binid)
        ids = node[:,0]
        lab = node[:,1]==trt
        bindat = trtinfo['scaler'].transform(node[:,6:])
        trt_compare = pd.DataFrame(bindat[lab,:],index=ids[lab])

        #cuts[binid] = get_caliper(trt_compare, trtinfo['drugbins'], binid, trtinfo['scaler'], trtinfo['bindf'], trtinfo['levels'], trt, trtinfo['prec'], median,percentile)

        cuts[binid] = get_caliper(trt_compare, trtinfo, binid, median,percentile)        
        if i % 20 == 0:
            print("caliper:",i,binid, cuts[binid])
    f = open(fname,'wb')
    pickle.dump(cuts, f)
    f.close()
    return f

#def get_caliper(trt_compare, drugbins, binid, scaler, bindf, levels, trt,
#                prec,median,percentile):
def get_caliper(trt_compare, trtinfo, binid, median, percentile):
    looking_for = 30
    trt_more = [trt_compare.values]        
    if trt_compare.shape[0] < looking_for:
        for neighbor in get_neighbors(trtinfo['bindf'], binid, his2ft.binners,
                                      trtinfo['levels'], looking_for):
            node = trtinfo['drugbins'].get_node("/" + neighbor)
            nodelab = node[:,1]==trtinfo['trt']
            
            trt_more.append( trtinfo['scaler'].transform(node[:,6:][nodelab,:]))

    trt_more = pd.DataFrame(np.vstack(trt_more))
    trt_more.index = list(trt_compare.index) + list(set(np.arange(2*trt_more.shape[0]))-
                                   set(trt_compare.index))[:(trt_more.shape[0]
                                                  - trt_compare.shape[0])]
    trtdist = pd.DataFrame()
    cutoff = 10

    if trt_compare.shape[0] > 10000:
        tshape = trt_compare.shape[0]
        print("bigboy! ",tshape)
        dists = []
        for k in range(int(trt_compare.shape[0]/1000)):
             x = trt_compare.iloc[k:min(trt_compare.shape[0],k+1000),:].apply(lambda x: trt_more.drop(x.name,axis=0).iloc[np.random.choice(tshape-1,500,replace=False),:].apply(lambda y:  mahalanobis(x,y, trtinfo['prec']), axis=1),axis=1)
             dists.append(x.apply(lambda q: np.percentile(q[~pd.isnull(q)],percentile), axis=1))
        return np.median(np.hstack(dists))
    
    elif trt_compare.shape[0] > 1000:
        #pdb.set_trace()
        tshape = trt_compare.shape[0]
        trtdist = trt_compare.apply(lambda x: 
                      trt_more.drop(x.name,axis=0).iloc[np.random.choice(tshape-1,100,replace=False),:].apply(lambda y: 
                                                        mahalanobis(x,y, trtinfo['prec']),
                                                         axis=1),axis=1)

    elif trt_more.shape[0] >= 30:
        trtdist = trt_compare.apply(lambda x: 
                  trt_more.drop(x.name,axis=0).apply(lambda y: 
                                                    mahalanobis(x,y, trtinfo['prec']),
                                                     axis=1),axis=1)
    #pdb.set_trace()
    if median:
        return trtdist.apply(lambda q: np.percentile(q[~pd.isnull(q)],percentile), axis=1).median()
    else:
        return np.percentile(trtdist.stack(), percentile)
def percentile_pscaliper(trt_compare, psk, psvar):
    if trt_compare.shape[0] > 1:
        #pdb.set_trace()
        other_treated = np.tile(trt_compare.transpose(),
                            (trt_compare.shape[0],1)).transpose()
        bin_caliper[psk] = np.percentile(np.abs(pd_helper.upper_tri(trt_compare.values - other_treated)),psk)
    else:
        bin_caliper[psk] = psmod.var()
    
def ps_caliper(trt_compare, trtinfo, binid, median, percentile, psmod):
    looking_for = 30
    trtps = psmod.loc[trt_compare.index]
    trt_more = [trtps]
    if trt_compare.shape[0] < looking_for:
        for neighbor in get_neighbors(trtinfo['bindf'], binid, his2ft.binners,
                                      trtinfo['levels'], looking_for):
            node = trtinfo['drugbins'].get_node("/" + neighbor)
            nodelab = (node[:,1]==trtinfo['trt']) & np.isin(node[:,0], psmod.index)
            trt_more.append(psmod.loc[node[:,0][nodelab]])
            #pdb.set_trace()
            if pd.isnull(trt_more[-1]).sum() > 0:
                pdb.set_trace()

    trt_more = pd.Series(np.hstack(trt_more))
    #trt_more.index = list(trt_compare.index) + list(set(np.arange(2*trt_more.shape[0]))-
    # #                              set(trt_compare.index))[:(trt_more.shape[0]
    #                                              - trt_compare.shape[0])]
    other_treated = np.tile(trt_more,(trt_compare.shape[0],1)).transpose()
    #pdb.set_trace()
    dists = np.abs(trtps.values - other_treated)
    #pdb.set_trace()    
    if median and trt_compare.shape[0]>2:
        #pdb.set_trace()
        #return dists.apply(lambda q: np.percentile(q[q!=0], percentile)).median()
        return np.median(np.apply_along_axis(lambda q: np.percentile(q[q!=0], percentile), 0, dists))
    else:
        dists2 = np.where(dists==0, np.float('NaN'),dists).reshape(-1,1)
        #pdb.set_trace()
        return np.percentile(dists2[~np.isnan(dists2)], percentile)
                 


def get_neighbors(bindf,binid,binners,levels, lookingfor):
    #t_compare = np.array(hasdrug.copy())
    seed = list(bindf.index[bindf['binid']==binid])    
    cur_bins = set(seed)  #list(bindf.index[bindf['binid']==binid]))
    all_bins = set(bindf.index)
    searched_bins = set(seed)
    #cur_bins = set([bindo])
    #searched_bins = set([])
    
    lohis = []    
    for i,f in enumerate(binners): #range(len(binners)):
        lohis.append([np.where(levels[f]==seed[0][i])[0][0]]*2)
        
    it = 0
    tot_trt = int(bindf.loc[bindf['binid']==binid,'binnedct'])
    #pdb.set_trace()
    while tot_trt < lookingfor:
        ## find new bins
        for i, dim in enumerate(binners):
            values = levels[dim]            
            lo, hi = lohis[i]
            #pdb.set_trace()
            #addto = list(cur_bins)
            addto = list(searched_bins)
            for bini in addto:
                if lo > 0 and bini[i] == values[lo]:
                    #print '...' + str(values[lo - 1])
                    toadd = list(bini)
                    toadd[i] = values[lo - 1]
                    toadd = tuple(toadd)
                    #print(toadd)
                    #print('trying ' + dim + ' @ ' + str(values[lo-1]))
                    if not toadd in searched_bins:
                        searched_bins.add(toadd)
                        if toadd in all_bins:

                            #pdb.set_trace()
                            cur_bins.add(toadd)
                            tot_trt += int(bindf.loc[toadd,'binnedct'])
                            #print(toadd , str(int(bindf.loc[toadd,'binnedct'])))
                            #print('   got ' + str(int(bindf.loc[toadd,'ct'])))

                            if tot_trt > lookingfor:
                                break
                if hi < len(values)-1 and bini[i] == values[hi]:
                    #print '...' + str(values[hi + 1])
                    toadd = list(bini)
                    toadd[i] = values[hi + 1]
                    toadd = tuple(toadd)
                    #print('trying ' + dim + ' @ ' + str(values[hi+1]))
                    if not toadd in searched_bins:
                        searched_bins.add(toadd)
                        if toadd in all_bins :
                            #pdb.set_trace()                        
                            cur_bins.add(toadd)
                            tot_trt += int(bindf.loc[toadd,'binnedct'])
                            #print(toadd , str(int(bindf.loc[toadd,'binnedct'])))
                            #print('   got ' + str(int(bindf.loc[toadd,'ct'])))
                            if tot_trt > lookingfor: break
            lohis[i] = [max(lo-1,0), min(hi + 1,len(values))]
        #print(str(it) + 'got:' + str(len(cur_bins)) + " ->" + str(tot_trt))
        #done_bins |= set(cur_bins)
        #cur_bins.extend(neighbors)
        it += 1
        if it > 10:
            print("Trouble for ",binid,seed)
            break
    return [bindf.loc[c,'binid'] for c in cur_bins if not c in seed]
    
