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
from collections import Counter, defaultdict
from scipy import sparse
import scipy
from itertools import chain
from sklearn.preprocessing import StandardScaler
import subprocess
import os
import json
import glob
sys.path.append("../../code/matchweight")

dem_length = 7

import file_names
def gen_embname(savename,hideous=''):
    return savename + hideous + ".binembeds.pytab"

### Create bins of patients
# - bin contents = pat ID & dem info; if doing embed matching, put an embed vector in each bin (prob could be more efficient)
# -filters = to filter out individual patient histories by: presence of element in history or demographics
## arguments:
# embmat = data frame of embeddings
# filters = {'exclude':[trt, ctl]} ie, remove anyone with history of the other drug
# todo_files = list of files to parse
# drugid = only used for generating the name to save
def bin_pats(savedir, embmat, filters, todo_files, drugid, #savename,
             is_trt=False, filtid=[], vi2groupi={},hideous='',featweights='', weightmode=''):

    savename = savedir + (str(drugid) if not is_trt else "trt")

    namesuff = hideous + featweights + weightmode
    tmp = "tmp/" + savename.replace("/",".") + namesuff
    if not os.path.exists("tmp/"):
        os.mkdir("tmp/")
    if len(glob.glob(tmp + "*")) > 0:
        for f in glob.glob(tmp + "*"):
            os.remove(f)
        tabrem = gen_embname(savename,namesuff)
        os.remove(tabrem)
        print("removed ",tabrem)
    
    if os.path.exists(gen_embname(savename,namesuff)):
        return
    print('making: ',gen_embname(savename,namesuff))
    ##################
    ### first, go through the target "treated" (trt)
    ### - get the bins using the predetermined dims
    ### - set up a file to save based on treated
    ##################
    #trtfile = hisdir + "trt"
    chunksize = 100000            
    def get_binner_bindf():
        if is_trt:
            header = ['week','age','gender','urx','rx','dx','px']
            demo = []        
            for trtfile in todo_files:
                for row in csv.reader(open(trtfile), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC):
                    patid, outcome, drug, bindemoi, nonbin_demoi, x, future = parse_row(row)
                    ### 
                    demo.append(bindemoi + nonbin_demoi) # + [bindemoi[3]])
            demo = pd.DataFrame(demo, columns=header)

            bindf, binner, dims = bin_match.get_coarsened_bins(demo, chunksize=chunksize)
            return bindf, binner, dims
        else:
            trtfile = savedir + 'trt' #hisdir + ".".join(savename.split(".")[:-1]) + ".trt"
            bindf = pd.read_pickle(trtfile + "bindf.pkl")
            dims = pd.read_pickle(trtfile + "bindims.pkl")            
            binner = bin_match.get_binner_func(dims, chunksize)
            #binner = pd.read_pickle(trtfile + "binner.pkl")
            return bindf, binner, dims
    bindf, binner, bindims = get_binner_bindf()
    binsseen = set([])    
    ### prepare to bin

    relevant_bins = set(bindf.index)
    bid2i = dict(zip(*tuple((bindf.index, bindf['binid']))))
    
    drugbins = tables.open_file(gen_embname(savename,namesuff) ,mode="a")
    #outcome_tab = tables.open_file(hisdir + savename+ "outcomes.pytab" ,mode="a")    
    tabcache = defaultdict(list)
    outcomecache = defaultdict(list)    
    tabcacheCt = defaultdict(int)
    scaler = StandardScaler()
    #print("capping bins at 50k!!!!")
    overflowlist = []

    
    featweights_mat = np.zeros(0)
    if featweights:
        featweights_mat = pd.read_csv(featweights,header=None,index_col=0,names=['id','coef'])
        if weightmode == 'P':
            featweights_mat = featweights_mat[featweights_mat['coef'] > 0]
        featweights_mat = featweights_mat.abs()/featweights_mat.abs().sum()
    def bin_write(onebin):
        binid = bid2i[onebin]
        #pdb.set_trace()        
        if not "/" + binid in drugbins:
            ncol = tabcache[onebin][0].shape[1]
            drugbins.create_earray(drugbins.root, binid,
                           tables.FloatAtom(), 
                             shape=(0, ncol),
                         chunkshape=(50, ncol))
            #outcome_tab.create_vlarray(outcome_tab.root, binid, tables.Int32Atom())
        #pdb.set_trace()
        
        if True: #drugbins.get_node("/" + binid).shape[0] < 50000:
            to_write = np.vstack(tabcache[onebin])
            drugbins.get_node("/" + binid).append(to_write)
            '''
            with open("tmp/e" + savename + binid,'ab') as f:
                np.savetxt(f, to_write)
            for i in outcomecache[onebin]:
                outcome_tab.get_node("/" + binid).append(list(i))
            '''
            #if to_write.shape[1] > 6:
            #    pdb.set_trace()
            if to_write.shape[1] > 6:

                scaler.partial_fit(to_write[:,6:])  ###patid, drug, [4-d bin] = 6
            with open(tmp + binid,'a') as f:
                f.write("\n".join([json.dumps(list(i)) for i in outcomecache[onebin]])+'\n')
        #elif binid not in overflowlist:
        #    print("exceeded 50000 for " + binid)
        #    overflowlist.append(binid)
        tabcacheCt[onebin] = 0
        tabcache[onebin] = []
        outcomecache[onebin] = []        
    #pbig = set(np.array(np.loadtxt("pbig"),dtype=int))
    #pdb.set_trace()
    def bin_store(bindemo, bincontents,binsseen, binoutcomes):
        bindemo = pd.DataFrame(bindemo,columns=['week','age','gender','urx'])
        bincontents = np.array(bincontents)
        #if len(pbig & set(bincontents[:,0])):
        #    pdb.set_trace()
        binoutcomes = np.array(binoutcomes)
        sel = ~np.isin(bincontents[:,0], filtid)
        bindemo = bindemo.loc[sel,:].reset_index().drop("index",axis=1);
        bincontents = bincontents[sel,:]; binoutcomes =  binoutcomes[sel]
        #if not isinstance(binoutcomes[0], list):
        #    binoutcomes[0][0] = 
        binchunk = binner(bindemo)
        groups = binchunk.groupby(bindf.index.names)
        chunk_d0_bins = relevant_bins & set(groups.groups.keys())
        binsseen |= chunk_d0_bins
        #pdb.set_trace()        
        for onebin in chunk_d0_bins:
            ixes = groups.groups[onebin]
            ## for each bin, get relevant rows, and "savecols" skips binning vars
            tabcache[onebin].append(bincontents[ixes, :])
            outcomecache[onebin] += list(binoutcomes[ixes])
            tabcacheCt[onebin] += len(ixes)
            if tabcacheCt[onebin] > 50:
                bin_write(onebin)
                
    dofilts = make_dofilts(filters)                
    sparse_elements = defaultdict(int) ## only needed for the "treated" 
    trtid = 0
    pdone = 0
    ##################
    ### now go through treated and control treat both the same, mostly
    ### - parse rows
    ### - store in cache & write to pytables file
    ##################    ctloutcome_length=0,

    for fdo in todo_files:
        bindemo = []; bincontents = []; binoutcomes = []
        print(fdo)
        for row in csv.reader(open(fdo), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC): #open(hisdir + "/trt"):
            patid, outcome, drug, bindemoi, nonbin_demoi, x, future = parse_row(row)

            ### study design filters
            if dofilts(x, bindemoi): # or patid in filtid:
                continue
            if is_trt: ### only for treated, for future reference save list of elt
                trtid = drug
                for el in x[0,:]:
                    sparse_elements[el] += 1
            #if patid in pbig:
            #    pdb.set_trace()
            bindemo.append(bindemoi) ##                    
            ## save: <bin label info>: week, age, gender, urx
            ##       <bin contents>: id, drug, bin info, non-bin demo, emb
            x = x[:,x[0,:]!=0]
            wid = 50
            times = x[1,:]
            weights = np.exp(-1*(times)**2/wid)

                
            #weights = timefunc(x[1,:]) #np.exp(-1*(x[6,:])**2/360)
            x = np.vstack((x[0,:], weights))
            if featweights:
                ftsel = np.isin(x[0,:],featweights_mat.index)
                x = x[:,ftsel]
                canccoef = featweights_mat.loc[x[0,:],'coef'].values
                if weightmode=='WA':
                    x[1,:] = np.multiply(x[1,:], canccoef)
                if weightmode=='WT':
                    times = times[ftsel]
                    for tu in set(times):
                        timeco = canccoef[times==tu]
                        x[1,times==tu] = np.multiply(x[1,times==tu], timeco/timeco.max())
            savelist = [patid, drug]
            #if patid==3358822:
            #    pdb.set_trace()
            if embmat.shape[0] > 0:
                savelist += bindemoi + nonbin_demoi + \
                            list((embmat[x[0,:].astype(int),:].transpose()*x[1,:]/(x[1,:].sum()+10**-6)).sum(axis=1))
            bincontents.append(savelist)
            #pdb.set_trace()
            binoutcomes.append(outcome)
            
            ## above: embmat weighted by time (exponential decay)
            pdone += 1

            ## every chunk-size rows, save it!
            if len(bindemo) == chunksize:
                bin_store(bindemo, bincontents,binsseen, binoutcomes)
                bindemo = []; bincontents = []; binoutcomes = [];
                #print("@chunk",len(binsseen))
        #pdb.set_trace()
        bin_store(bindemo, bincontents,binsseen,binoutcomes)
        #if pdone > 3000000:
        #    print("breaking at 3M")
        #    break
    for b, v in tabcache.items():
        if len(v) > 0:
            bin_write(b)
    #outcome_tab.close()
    #outcome_tab = None
    #if not os.path.exists(gen_outcname(hisdir, savename)):
    #outcome_tab = tables.open_file(gen_outcname(hisdir, savename, hideous) ,mode="a")
    #b2_tab = tables.open_file(hisdir + savename+ "binembeds2.pytab" ,mode="a")    
    #pdb.set_trace()
    group = drugbins.create_group("/", 'outcomes', 'outcome matched to')
    for b in tabcache:
        binid = bid2i[b]
        nrow = drugbins.get_node("/" + binid).shape[0]
        #vl = outcome_tab.create_vlarray(outcome_tab.root, binid,
        #                           tables.Int32Atom(),expectedrows=nrow,chunkshape=nrow)
        vl = drugbins.create_vlarray(group, binid,
                                   tables.Int32Atom(),expectedrows=nrow,chunkshape=nrow)
        for i in open(tmp + binid):
            vl.append(json.loads(i))
        os.remove(tmp + binid)
    #outcome_tab.close()
    if not is_trt:
        print("Finishing control data!!! did:" + str(pdone))
        return

    bnames = []
    ctbinned = []
    for b in drugbins.walk_nodes("/","EArray"): #.root.walk_nodes()
        bnames.append(b.name)
        ctbinned.append((b[:,1]==trtid).sum())
    ctbinned = dict(zip(*tuple((bnames,ctbinned))))
    ctbinned = [0 if k not in ctbinned else ctbinned[k] for k in bindf['binid']]
    bindf['binnedct'] = ctbinned
    bindf.to_pickle(savename+ "bindf.pkl")    
    drugbins.close()
    ct = pd.DataFrame(sparse_elements,index=['ct']).transpose()
    index =list(ct.loc[ct['ct'] > 100,:].index)
    #index = list((ct['ct'] > 100).index)
    sparse_index = index
    f = open(savename +"sparse_index.pkl",'wb')
    pickle.dump((ct), f)
    f.close()
    f = open(savename+ namesuff + "standardscaler.pkl",'wb')
    pickle.dump(scaler,f)
    f.close()
    f = open(savename +"bindims.pkl",'wb')
    pickle.dump(bindims, f)
    f.close()

nft = 1

def parse_row(row, nft=1):

    patid = row[0]
    outcome_length = np.where(np.array(row)==-1)[0].max()
    #deenroll = row[1]
    outcome = row[1:(outcome_length)] ## num outcomes + pat id + deenroll

    drug = row[outcome_length + 1]

    row = row[(outcome_length + 1 + 1):] ## skip past num outcome + 1 for the "-1" + 1 for drug
    demo = row[:dem_length]
    ## week, age, sex, urx
    bindemo = [demo[0], demo[1], demo[2],demo[3]]
    nonbin_demo = [demo[4],demo[5],demo[6]]

    row = row[dem_length:]
    future_start = np.where(np.array(row)==-2)[0]
    future = []
    if future_start.shape[0] > 0:
        future_start = future_start.min()
        future = row[future_start:]
        row = row[:future_start]
    if len(row) % 2 != 0:
        print("OH NO-- incorrect format")
        pdb.set_trace()
    nsparse = int(len(row)/(nft +  1))

    x = np.vstack((row[:nsparse],
                   np.array(row[nsparse:]).reshape(-1,nft).transpose()))
    ret = [patid, outcome, drug, bindemo, nonbin_demo, x, future]
    return ret


def get_his_ids(files, filters):
    dofilts = make_dofilts(filters)
    filtid = []
    for fdo in files:
        for row in csv.reader(open(fdo), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC):
            patid, outcome, drug, bindemoi, nonbin_demoi, x, future = parse_row(row)
            if dofilts(x, bindemoi):
               filtid.append(patid)
    return filtid
    
def align_to_sparseindex(x, sparse_index):
    x = x[:,np.isin(x[0,:], sparse_index)]
    x = x[:,np.argsort(x[0,:])]
    mycol = np.where(np.isin(sparse_index, x[0,:]))[0]
    return x, mycol


def make_dofilts(filters): #exclude=set([]), include=set([]), filt):    
    def dofilts(x, demo):
        if 'exclude' in filters and len(filters['exclude'] & set(x[0,:])) > 0:
            #if x[1,np.isin(x[0,:],list(filters['exclude']))].min() == 1:
            #    pdb.set_trace()                    
            return True
        if 'include' in filters and len(filters['include'] & set(x[0,:])) == 0: 
            #if x[1,np.isin(x[0,:],list(exclude))].min() < 1:
                #pdb.set_trace()                    
            return True
        if 'dem' in filters:
            #pdb.set_trace()
            for k,v in filters['dem'].items():
                if len(v)==1 and not demo[k]==v[0]:
                    #pdb.set_trace()        
                    return True
                elif len(v)>1 and not (demo[k]>v[0] and demo[k] < v[1]):
                    #pdb.set_trace()                            
                    return True
        return False
    return dofilts
    
def timefunc(b):
    return np.exp(-1*(b)**2/360)

##############################
## sparseindex_name = saved list of columns to use, made in the bin step
## outname = save teh results
## filters = elements in the vocab of the data that cause remove a patient (see make_dofilts)
## todo_files = files to become the rows of our matrices
## idfile = another filter, only keep IDs in these files, as when you are doing sparsemat to evaluate a matching
def prepare_sparsemat2(hisdir, sparseindex_name, drugdo, filters, idfile = ''):
    
    savename = file_names.sparseh5_names(hisdir, drugdo) #hisdir + outname 
    todo_files = file_names.todo_files(hisdir, drugdo)
    if os.path.exists(savename):
        if TESTING:
            subprocess.call("rm " + savename,shell=True)
        else:
            return
    save_size = 300000
    useids = []
    if idfile:
        useids = set(np.loadtxt(hisdir + idfile+'.ids.trt')) | set(np.loadtxt(hisdir + idfile+'.ids.ctl'))
    VMAX = 30285
    sparse_index = np.arange(1,VMAX+1)    
    if sparseindex_name:
        sparse_index = get_sparse_index(hisdir + sparseindex_name, cut=100)
        #pdb.set_trace()
        print("removing ZERO (invalid) from my feature set")

        sparse_index = np.delete(sparse_index,0)
    #else:
    #    sparse_index = np.arange(1,vocab['vi'].max()+1)
    ncol = len(sparse_index)
    dofilts = make_dofilts(filters)
    dense = []

    h5tab = tables.open_file(savename, 'a') 
    #densefile = savename +".den"
    rowlist = []
    collist = []
    #freqdat = []
    timedat = []
    r = 0
    node_name = 0
    def store_to(node_name):
        #if node_name < 2:
        #    pdb.set_trace()
        name = 'c' + str(node_name)        
        group = h5tab.create_group("/", 'c' + str(node_name), str(node_name))
        sp_store = sparse.csr_matrix((timedat,(rowlist,collist)),
                              shape=(r,len(sparse_index)))

        for attribute in ('data', 'indices', 'indptr', 'shape'):
            #full_name = f'{name}_{attribute}'

            # add nodes
            arr = np.array(getattr(sp_store, attribute))
            atom = tables.Atom.from_dtype(arr.dtype)
            ds = h5tab.create_carray(group, attribute, atom, arr.shape,
                                     chunkshape = arr.shape)
            ds[:] = arr
        arr = np.array(dense)
        atom = tables.Atom.from_dtype(arr.dtype)
        #full_name = f'{name}_den'
        ds = h5tab.create_carray(group, "den", atom, arr.shape)
        ds[:] = arr

    for fdo in todo_files:
        for row in csv.reader(open(fdo), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC):
            #if row[0] == 1845:
            #    pdb.set_trace()
            patid, outcome, drug, bindemoi, nonbin_demoi, x, future = parse_row(row)
            #pdb.set_trace()
            if useids and not patid in useids:
                continue
            if dofilts(x, bindemoi):
                continue

            ### then get indexed version (some super rare wouldn't be included but maybe yes if are part of a group...)
            x2, mycol = align_to_sparseindex(x, sparse_index)
            timedat += list(x2[1,:] + .5) ### NOTE adding .5 so sparse zero (==NEVER) != zero days before!

            rowlist += [r]*len(mycol)
            collist += list(mycol) #list(x + len(den))
            dense.append([patid] + bindemoi + nonbin_demoi)

            r += 1
            if len(dense) > save_size:
                store_to(node_name)
                dense = []
                rowlist = []
                collist = []
                timedat = []
                r = 0
                node_name += 1
    store_to(node_name)
    h5tab.close()

#TIME_CHUNK = 25
def censored_sparsemat(hisdir, past_sparse_index, use_ids, this_drug, other_drug,TIME_CHUNK, agg=1,  washout=np.inf):
    t0 = time.time()
    '''
    elct = pickle.load(open(file_names.sparse_index_name(hisdir, trt_drugid),'rb'))
    SPFT_CUT = 100
    past_sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > SPFT_CUT,:].index)),
                           dtype = int)
    if past_sparse_index.shape[0] == 0:
        past_sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > 10,:].index)),
                           dtype = int)
    past_sparse_index = np.delete(past_sparse_index,0)
    '''
    voc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")        
    #fut_sparse_index = np.arange(1,voc['vi'].max()+1)
    #fut_made_index = False
    other_vi = voc.loc[(voc['type']=="rx") & (voc['id']==other_drug),"vi"].values[0]
    this_vi = voc.loc[(voc['type']=="rx") & (voc['id']==this_drug),"vi"].values[0]
    
    ncol = len(past_sparse_index)
    
    dense = []
    spmat = {'rows':[], 'cols':[], 'dat':[], 'ncol':len(past_sparse_index)}

    lab = []
    r = 0
    fut_sparse_elements = defaultdict(int)
    import psutil
    process = psutil.Process(os.getpid())

    for fdo in file_names.todo_files(hisdir, this_drug):
        for row in csv.reader(open(fdo), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC):
            patid, outcome, drug, bindemoi, nonbin_demoi, x, future = parse_row(row)
            if not patid in use_ids:
                continue
            future_periods = np.where(np.array(future)==-2)[0]
            for el in set(chain.from_iterable([future[(future_periods[fp]+2):future_periods[fp+1]]
                                               for fp in range(len(future_periods)-1)])):
                fut_sparse_elements[el] += 1
    print("t1: make future index {:1.2f}".format(time.time() - t0))
    fut_sparse_elements.pop(other_vi)
    fut_sparse_elements.pop(0)    
    ct = pd.DataFrame(fut_sparse_elements,index=['ct']).transpose()
    fut_sparse_index = np.array(ct.loc[ct['ct'] > 100,:].index)
    #pdb.set_trace()
    #sel = np.where(np.isin(fut_elts, fut_sparse_index))[0]
    futmat = {'rows':[], 'cols':[], 'dat':[], 'ncol':len(fut_sparse_index)}            
    print("SPECIFY CHUNK LENGTH {:d} past elt & {:d} fut elt".format(past_sparse_index.shape[0], fut_sparse_index.shape[0]))
    for fdo in file_names.todo_files(hisdir, this_drug):
        for row in csv.reader(open(fdo), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC):
            patid, outcome, drug, bindemoi, nonbin_demoi, x, future = parse_row(row)
            if not patid in use_ids:
                continue
            ### then get indexed version (some super rare wouldn't be included but maybe yes if are part of a group...)
            x2, mycol = align_to_sparseindex(x, past_sparse_index)
            future_periods = np.where(np.array(future)==-2)[0]
            if agg > 1:
                #future_periods_agg = future_periods[np.arange(0, len(future_periods), agg)]
                future_agg = []
                f_ind = 0
                #pdb.set_trace()
                while f_ind < len(future_periods):
                    chunk_len  = 0
                    chunk_contents = []
                    #future_periods_agg += future_periods[f_ind]
                    
                    for toagg in range(agg):
                        if f_ind == len(future_periods):
                            break
                        fstart = future_periods[f_ind]
                        last_chunk = f_ind == len(future_periods)-1
                        chunk_len += future[fstart + 1] if last_chunk else TIME_CHUNK
                        chunk_contents += future[fstart + 1:(len(future) if last_chunk else future_periods[f_ind+1])]
                        f_ind +=  1
                        
                    future_agg += [-2, chunk_len] + sorted(set(chunk_contents))
                    #pdb.set_trace()
                future = future_agg
                future_periods = np.where(np.array(future)==-2)[0]            
                                      
            #pdb.set_trace()
            last_drug1_ago = 0
            for fut in range(len(future_periods)):
                spmat['dat'] += list(x2[1,:] + .5) ### NOTE adding .5 so sparse zero (==NEVER) != zero days before!
                spmat['rows'] += [r]*len(mycol)
                spmat['cols'] += list(mycol) #list(x + len(den))
                last_chunk = fut >= len(future_periods)-1

                #pdb.set_trace()
                dense.append([patid, 
                              0 if not last_chunk else future[future_periods[fut]+1],##chunk len
                              bindemoi[0]+(TIME_CHUNK*agg)*fut, ## study week
                              bindemoi[1] + TIME_CHUNK*agg/52*fut] ##age
                             + bindemoi[2:] + nonbin_demoi)

                ### get columns of the future periods co
                fut_elt = future[future_periods[fut]+1:
                                 (len(future) if last_chunk else future_periods[fut+1])]
                fut_mycol = np.where(np.isin(fut_sparse_index, fut_elt))[0]
                #fut_x2, fut_mycol = align_to_sparseindex(fut_elt, fut_sparse_index)
                futmat['rows'] += [r]*len(fut_mycol)
                futmat['cols'] += list(fut_mycol)
                futmat['dat'] += [1]*len(fut_mycol) #fut_mycol)
                r += 1
                if this_vi not in fut_elt:
                    last_drug1_ago += TIME_CHUNK*agg
                else:
                    last_drug1_ago = 0
                if other_vi in fut_elt or last_drug1_ago > washout:
                    #pdb.set_trace()
                    lab += [1]
                    break
                else:
                    lab += [0]


            if len(dense) % 500000 < 20: 
                print("BIG nrow={:d} {:2.2f}".format(len(dense), process.memory_info().rss/10**9))
    dense = np.array(dense)
    spmat = sparse.csr_matrix((spmat['dat'],(spmat['rows'],spmat['cols'])),
                              shape=(r,spmat['ncol']))

    #pdb.set_trace()
    futmat = sparse.csr_matrix((futmat['dat'],(futmat['rows'],futmat['cols'])),
                              shape=(r,futmat['ncol']))

    t1  = time.time()
    print("END nrow={:d} {:2.2f}, time = {:1.2f} min".format(dense.shape[0], process.memory_info().rss/10**9,(t1 - t0)/60))

    cens_info = pd.DataFrame({'ids':dense[:,0],
                              "censored":lab})
    ## transform week to week SINCE drug
    def offs(w): return list(w - w.min())
    woffs  = pd.DataFrame(dense[:,[0,2]],columns=['ids','week']).groupby("ids")['week'].agg(offs)

    cens_info["interval_start"] = np.hstack([woffs[k] for k in
                                   cens_info['ids'].drop_duplicates()])
    interval_length = np.where(dense[:,1]==0,TIME_CHUNK, dense[:,1])
    cens_info["interval_end"] = cens_info['interval_start'] + interval_length

    
    return dense, spmat, futmat, lab, fut_sparse_index, cens_info
            
    #store_to(node_name)
    #h5tab.close()
    
    
def get_covmat(hisdir, trt,hideous='',noscale=False):
    scaler =pickle.load(open(hisdir +hideous+ "standardscaler.pkl",'rb'))
    #bindf = tables.open_file(hisdir + "binembeds.pytab" ,mode="r")
    #pdb.set_trace()
    drugbins = tables.open_file(hisdir +hideous+ ".binembeds.pytab" ,mode="r")
    if not hasattr(scaler,"mean_"):
        return None
    nvar = scaler.mean_.shape[0]
    prodsums = np.zeros((nvar, nvar))
    varsums = np.zeros(nvar)
    meanest = np.zeros(0)
    sums = np.zeros(nvar)
    
    #meanest = np.zeros(0)    
    #for bin in drugbins:
    nrow = 0
    #pdb.set_trace()
    print("noscale",noscale)
    for bindat in drugbins.walk_nodes("/","EArray"):
        sel = bindat[:,1]==trt
        if sel.sum()==0:
            continue
        if ~sel.sum() > 0:
            pdb.set_trace()
        forcov = bindat[:,6:][sel,:]
        if not noscale:
            forcov = scaler.transform(forcov)
        # should be centered after transform - meanest
        prodsums += forcov.transpose().dot(forcov)
        varsums += forcov.sum(axis=0)
        sums += forcov.sum(axis=0)
        nrow += sel.sum()
    di = np.tile(varsums,(nvar,1)) #np.diag(prodsums)

    covest = (prodsums - (di * di.transpose())/nrow)/nrow
    prec = np.linalg.inv(covest)    
    return prec,covest
binners=['urx','year','age']

def get_trt_info(hisdir, trt, hideous=''):
    prec = []; cov = []
    pc = get_covmat(hisdir, trt,hideous)
    if pc:
        prec, cov = pc
    scaler = pd.read_pickle(hisdir +hideous + "standardscaler.pkl")    
    bindf = pd.read_pickle(hisdir + "bindf.pkl")
    bindf = bindf.loc[bindf['binnedct'] > 0,:]

    levels = {i[0]:np.array(i[1]) for i in zip(*tuple((bindf.index.names,
                                                  bindf.index.levels)))}
    bindf = bindf.reorder_levels(binners + list(levels.keys()-set(binners)),axis=0).sort_index()
    #bindf = bindf.reset_index().set_index('binid')
    #for binid in bindf.index:
    #    if bindf.loc[binid,'ct'] < 30:
    #caliper = pickle.load(open("calipermedcoag.pkl",'rb'))
    drugbins = tables.open_file(hisdir +hideous +  ".binembeds.pytab" ,mode="r")
    if hideous:
        hideous = "." + hideous 
    #outcomes = tables.open_file(hisdir +  ".outcomes.pytab" ,mode="r")    
    return {'prec':prec, 'scaler':scaler, 'bindf':bindf,
            'levels':levels,'drugbins':drugbins, 'trt':trt} #,'outcomes':outcomes} #


TESTING = False
#sbatch ../../code/matchweight/run_sparsmat.sh mixed_histories/ drug_neighbor_counts.pkl
import multiprocessing as mp
if __name__ == "__main__":
    dirn = sys.argv[1]
    druginfo = sys.argv[2] if len(sys.argv) > 2 else dirn + "/drug_neighbor_counts.pkl"
    todo = pd.read_pickle(druginfo)

    #if TESTING:
    #    todo = [3512, 4747]
    
    if not os.path.exists(dirn + "/sparsemat"):
        os.mkdir(dirn + "/sparsemat")
        
    if TESTING:
        subprocess.call("rm " + dirn + "/sparsemat/*",shell=True)
        prepare_sparsemat2(dirn,"",todo.index[0], {})
        exit
    todo = list(todo.loc[todo[0] > 20000, :].index)        
    pool = mp.Pool(processes=11 if not TESTING else 2)        
    res = [pool.apply_async(prepare_sparsemat2,
                            args=(dirn, "", drug, {}))
           for drug in todo]
    results = [p.get() for p in res] ## do i need this?
    
'''
def gen_outcname(savename,hideous=''):
    return savename + hideous + ".outcomes.pytab"

def outcome_info(hisdir, outname, noutcomes,
                       todo_files, idfile = ''):

    savename = outname +  ".outcomes"
    #pdb.set_trace()
    if os.path.exists(savename):
        return savename
    useids = []
    if idfile:
        useids = set(np.loadtxt(hisdir + idfile+'.ids.trt')) | set(np.loadtxt(hisdir + idfile+'.ids.ctl')) #"coag_timeonly/duperfiltbinonly")
    
    #dofilts = make_dofilts(filters)
    outcomelist = []
    open(savename,'w').close()

    for fdo in todo_files:
        for row in csv.reader(open(fdo), delimiter='\t',quoting = csv.QUOTE_NONNUMERIC):
            patid, outcome, drug, bindemoi, nonbin_demoi, x, deenroll = parse_row(row)
            #pdb.set_trace()
            if useids and not patid in useids:
                continue
            #if dofilts(x, bindemoi):
            #    continue
            outcome = np.array(outcome,dtype=int).reshape(-1,1).reshape((int(len(outcome)/2),2)).transpose()
            sv = np.zeros(noutcomes,dtype=int)
            sv[outcome[0,:]] = outcome[1,:]
            outcomelist.append([patid, deenroll] + list(sv))
            if len(outcomelist) > 10000:
                with open(savename,'ab') as f:
                    np.savetxt(f, np.array(outcomelist), delimiter="\t",fmt="%d")
                outcomelist = []
    with open(savename,'ab') as f:
        np.savetxt(f, np.array(outcomelist), delimiter="\t")                
    return savename
'''
