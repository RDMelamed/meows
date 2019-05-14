from collections import defaultdict
import tables
import subprocess
import glob
import time
import multiprocessing as mp
import os
import pandas as pd
import sys
import numpy as np
import pickle
import pdb
import datetime
import resource
#import apsw
import sqlite3
import json
from itertools import chain
import skips
from collections import Counter
import shutil

os.system('taskset -p 0xffffffff %d' % os.getpid())
cache_size = 50
TESTING = False #False

if TESTING:
    cache_size = 1
drugs = []
#indexdir = "/project2/arzhetsky/MSdb/kvdb2/MG.rxindices/"
indexdir = "/project2/arzhetsky/MSdb/kvdb2/MG.rxindices/"
datdir = "/project2/melamed/wrk/iptw/data/"
import enQ
from skips import urx_rx_dx, rxc, demc, dxc

def history_parse(label, visits, dat, decode, outcomes, codesuffix, TEST=False):
    urx, rx, dx, px = visits  ### Rx and Dx are now in vocab-ids.  urx is not.
    rx = rx[rx[:,0] != decode['rxi2' + codesuffix][label],:] ## remove the precise week of incident Rx
    entry = urx[urx[:,rxc['generic']]==label,:][0]
    week = entry[rxc['week']]
    ret = [label, week,entry[rxc['age']], dat['dem'][demc['female']],urx.shape[0]]
    history_id = []
    history_feat = []
    #pdb.set_trace()
    for feat in [rx, dx, px]:
        fthistory  = get_elements_weights(feat, week)
        history_id += fthistory[0] 
        history_feat += fthistory[1]
        #pdb.set_trace()
        ret += [int(fthistory[2])]
    
    ret += list(history_id) + list(history_feat)
    
    #### get outcomes
    dxout = np.array(dat['dx'])
    olist = [max(dat['enroll'][1]-week,0)]    
    if dxout.shape[0] > 0:
        olist = [max(olist[0], dxout[:,dxc['week']].max()-week)]
        for outi, codes in enumerate(outcomes):

            #oweeks = dx[dx[:,2 if outcomeicd else 0]==oc,1] ### column 2 = icd codes rather than phe!
            #pdb.set_trace()
            oweeks = dxout[np.isin(dxout[:,dxc['icd']], codes),dxc['week']] ### column 2 = icd codes rather than phe!
            if len(oweeks) > 0: # and oweeks.min() > week:
                oweeks = oweeks - week
                #pdb.set_trace()                
                olist.append(outi)                                
                if oweeks.min() > 0:
                    olist.append(oweeks.min())
                else:
                    ## get the last week before trt... up to ZERO
                    #return None ##  if any of htese outcomes happens BEFORE drug, we exclude this person
                    olist.append(oweeks[oweeks <= 0].max())
            '''
            if len(oweeks) > 0 and oweeks.min() > week:
                #pdb.set_trace()                
                oweeks = oweeks - week
                olist.append(outi)
                if oweeks.max() <= 0:  ## this seems left over & obsolete
                    olist.append(oweeks.max())
                else:
                    olist.append(oweeks[oweeks > 0].min())
            '''
    ret = (ret, olist) #datentry
    return ret #"\t".join([str(i) for i in datentry]) + '\n' #, list(history_id)


def get_history(elwk, wmax):
    hist = elwk[(elwk[:,1]<=wmax),:]
    hist[:,1] = wmax - hist[:,1] # - rxhistory[:,rxc['week']] ## time backwards
    hist = hist[::-1,:] ## reverse
    return hist

def get_elements_weights(elwk, wmax):
    hist = get_history(elwk, wmax)
    numelt = float(hist.shape[0])
    if numelt == 0:
        return [[], [], 0]
    histfre = hist[ hist[:,1]<=52,:]
    uhist = hist[np.unique(hist[:,0],return_index=True)[1],:]
    return [list(uhist[:,0]), list(uhist[:,1]), numelt]
    
    '''
    cv = {}
    if histfre.shape[0] > 0:
        cuts = np.arange(0,52,4)
        v = np.digitize(histfre[:,1],cuts)
        v = np.unique(np.vstack((histfre[:,0],v)).transpose(),axis=0)
        cv = Counter(v[:,0])
    '''
'''
    cv = Counter(histfre[:,0])
    cuts = [0,4,12,28,52,1000]
    v = np.digitize(hist[:,1],cuts)
    v = np.unique(np.vstack((hist[:,0],v)).transpose(),axis=0)
    g = pd.DataFrame(np.hstack((v,np.ones(v.shape[0]).reshape(-1,1))),
                     columns=['id','cut','1']).groupby(['id','cut']
                     ).sum().unstack('cut',fill_value=0)['1']
    if g.shape[1] < len(cuts)-1:
        for i in set(np.arange(1,len(cuts))) - set(g.columns):
            g[i] = 0
        g = g.sort_index(axis=1)
    #pdb.set_trace()    
    g['time'] = hist[np.unique(hist[:,0],return_index=True)[1],1]
    g['freq'] = [cv[i] if i in cv else 0 for i in g.index]

    return [list(g.index.astype(int)), list(g.astype(int).values.reshape(-1)), numelt]
'''

def stringify(res):
    return "\t".join([str(i) for i in res]) + '\n'

def historyloop(Q, doid, name, prefix, outcomes, mix=False):
    decode = pickle.load(open(datdir +"decode.12.18.clid.allvocab.pkl",'rb'))

    #elt_freqs = pickle.load(open("elt_freqs.pkl"))
    #(eltfreq_new,elt2freq) = pickle.load(open(datdir +vocabfile,'rb'))
    #idfs = {k:-np.log(v) for k,v in elt2freq.items()}
    conn, pcurs = skips.get_conn_curs()    


    econn = sqlite3.Connection("/project2/melamed/db/enr2.db")
    ecurs = econn.cursor()
    ecurs.execute("PRAGMA cache_size=2000;")

    print("With " + str(len(outcomes)) + " outcomes")
    #pref = 'ost'
    def getfname(it, drug):
        fname = prefix + '/valid.' + str(name)+ '.'+ str(it) + ("." + str(drug) if drug > 0 else "")
            
        open(fname,'w').close()
        return fname

    it = 0
    #fname = getfname(it)
    logf = prefix + '/vallogf'
    with open(logf,'a') as f:
        f.write('STart!' + name + ' ' + str(it) +'\n')
        

    obs = {0:''} if mix else defaultdict(str)
    count = {0:0} if mix else defaultdict(int)
    fname = {0:getfname(it,0)} if mix else defaultdict(str)
    codesuffix = 'vi'

    tried = 0
    gots = 0      
    for (pdone, person) in iter(Q.get, None):
        #print("P:" , person)
        tried += 1
        pcurs.execute('select * from kv where person = ?',(person,))

        dat = json.loads(pcurs.fetchone()[1])
        ecurs.execute('select * from kv where person = ?',(person,))
        dat['enroll'] = json.loads(ecurs.fetchone()[1])
        '''
        try:
            res = history_parse(dat, decode, doid, outcomes, codesuffix)
        except ValueError:
            print("VALUEERROR", person)
            break
        '''
        #if TESTING:        
        #    print person, res
        alldo, visits = urx_rx_dx(dat, doid, decode, codesuffix)

        for label in alldo:
            try:
                res = history_parse(label, visits,dat, decode, outcomes, codesuffix)
            except ValueError:
                print("VALUEERROR", person)
                break
            if not res:
                continue
            gots += 1
            drug = res[0][0]
            res = [person] + res[1] + [-1] + res[0]
            obs[drug] += stringify(res)
            count[drug] += 1
        #else:
        #    with open(logf,'a') as f:
        #        f.write('no: '+str(person))
        #if person == 2:
        #    print 'person = 2' + str(len(res))
        if pdone % 100000 == 0:
            with open(logf,'a') as f:
                f.write('done ' + str(pdone) + '\n') # + ' ' + list(fname.values())[0] +'\n')
        #if len(obs) > 50000:
        for k in count:
            if not k in fname:
                fname[k] = getfname(it, k)
            if count[k] > 5000:        
                with open(fname[k] ,'a') as f:
                    f.write(obs[k])
                obs[k] = ''
            if count[k] >= 100000:
                it += 1
                fname[k] = getfname(it, k)
                count[k] = 0

    print("tried:", tried, " got: " , gots)       
    with open(logf,'a') as f:
        f.write('Finished!' + list(fname.values())[0] + " tried:" + str(it)+'\n')
    for k in count:
        if count[k] == 0: continue
        if not k in fname:
            fname[k] = getfname(it, k)
        
        with open(fname[k] ,'a') as f:
            f.write(obs[k])

def enQminus(Q, doid, nprocs,ndo=0, samp=.1):
    i = 0
    d2f = {i:open(indexdir + str(d)) for i,d in enumerate(doid)}
    curp = np.array([float(d2f[i].readline().strip()) for i in d2f])
    for person in range(1, 151104811):
        ### if person took drugs in doid then SKIP
        if person in curp:
            for fix in np.where(curp==person)[0]:
                nextpat = d2f[fix].readline()
                if nextpat:
                    curp[fix] = float(nextpat.strip())
                else:
                    curp[fix] = np.float('inf')
                    with open('logf','a') as f:
                        f.write('enQ finished ' + str(fix) +" " + str(i) +  "\n")
            continue
        if np.random.rand(1)[0] < samp:
            Q.put((i,int(person)), block=True)
            i += 1
        if ndo >0 and i > ndo:
            break
    for i in range(nprocs):
        Q.put(None, block=True)

def main(doid,prefix, outcomes, ndo=0,neg_enQ=False):
    if os.path.exists(prefix):
        print(prefix + " EXISTS--QUITTING!")
        return
        #shutil.rmtree(prefix)
    os.mkdir(prefix)
    ndo = 1000 if TESTING else 0
    nprocs = 2 if TESTING else 14
    loadprocs = []
    Q = mp.Queue(maxsize=500000)
    #open('vallogf','w').close()
    plock = mp.Lock()
    enQproc = mp.Process(target = enQ.enQ if not neg_enQ else enQminus,
                         args = (Q, doid,nprocs,ndo))
    enQproc.start()
    for i in range(nprocs):
        loadprocs.append(mp.Process(target = historyloop,
                                    args = (Q, doid if not neg_enQ else set([]),
                                            str(i), prefix,outcomes)))
        loadprocs[-1].start()
    for p in loadprocs:
        p.join()
            
if __name__=="__main__":
    print(sys.argv[1])
    ## <which drugs>, 
    doid = [int(i) for i in open(sys.argv[1]).read().strip("\n").split("\n")]
    print(sys.argv[2])    
    prefix = sys.argv[2]
    #codesuffix = sys.argv[3]
    outcomes = pickle.load(open(sys.argv[3],'rb'))
    oorder = sorted(outcomes.keys())
    outcomes = [outcomes[k] for k in oorder]
        #outcomes = [([int(j) for j in i.split("=")[0].split(",")], i.split("=")[1]) for i in sys.argv[5].split("*")] #.strip(")").strip(",")]
    #print(outcomes)
    #doid = [4047, 5610] ## warfarin, dabig
    main(doid, prefix, outcomes)

