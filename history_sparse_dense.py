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

from collections import Counter
import shutil
import psutil

#os.system('taskset -p 0xffffffff %d' % os.getpid())
cache_size = 50
TESTING =  False

if TESTING:
    cache_size = 1
drugs = []
#indexdir = "/project2/arzhetsky/MSdb/kvdb2/MG.rxindices/"
indexdir = "/project2/arzhetsky/MSdb/kvdb2/MG.rxindices/"
datdir = "/project2/melamed/wrk/iptw/data/"
sys.path.append("/project2/melamed/wrk/iptw/code")
import enQ
import skips

def history_parse(label, visits, dat, decode, outcomes, codesuffix, time_chunk_size, TEST=False):
    urx, rx, dx, px = visits  ### Rx and Dx are now in vocab-ids.  urx is not.
    entry = urx[urx[:,skips.rxc['generic']]==label,:][0]
    week = entry[skips.rxc['week']]
    rx = rx[! ((rx[:,0] == decode['rxi2' + codesuffix][label]) &
               (rx[:,1] == week)) ,:] ## remove the precise incident Rx
    
    urxwk = urx[:,skips.rxc['week']] ## changing to number unique drugs in past year
    ret = [label, week,entry[skips.rxc['age']], dat['dem'][skips.demc['female']],
           sum((urxwk < week) & (urxwk >= week - 52))]
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
    olist = [max(max([dat['enroll'][1]] + [i[:,1].max() if i.shape[0] > 0 else 0
                                           for i in [rx, dx,px]])
                 -week,0)]    
    if dxout.shape[0] > 0:
        olist = [max(olist[0], dxout[:,skips.dxc['week']].max()-week)]
        for outi, codes in enumerate(outcomes):
            oweeks = dxout[np.isin(dxout[:,skips.dxc['icd']], codes),skips.dxc['week']] ### column 2 = icd codes rather than phe!
            if len(oweeks) > 0: # and oweeks.min() > week:
                oweeks = oweeks - week
                #pdb.set_trace()                
                olist.append(outi)                                
                if oweeks.min() > 0:
                    olist.append(oweeks.min())
                else:
                    ## get the last week before trt... up to ZERO
                    ##  if any of htese outcomes happens BEFORE drug, we exclude this person
                    if TEST:
                        return [-10] + list(set(dxout[np.isin(dxout[:,skips.dxc['icd']], codes) &
                                                  (dxout[:,skips.dxc['week']] < week),skips.dxc['icd']]))
                    else:
                        return None
                                       
                    #olist.append(oweeks[oweeks <= 0].max())

    #### get post-treatment health info
    future_chunks = []
    #pdb.set_trace()    
    for time_chunk in range(week, week + olist[0], time_chunk_size):
        
        #x = feat[(feat[:,1] >= time_chunk)& (feat[:,1] < time_chunk + time_chunk_size),0]
        #pdb.set_trace()
        chunkadd = [sorted(set(feat[(feat[:,1] >= time_chunk) &
                                    (feat[:,1] < time_chunk + time_chunk_size) ,0]))
                    for feat in [rx, dx, px]]
        #if 0 in chunkadd:
        #    pdb.set_trace()
        pref = [-2, min(time_chunk + time_chunk_size, week + olist[0])- time_chunk] if time_chunk + time_chunk_size >= week + olist[0] else [-2]
        future_chunks += pref + list(chain.from_iterable(chunkadd))
            
    ret = (ret, olist, future_chunks) #datentry
    return ret #"\t".join([str(i) for i in datentry]) + '\n' #, list(history_id)


def get_history(elwk, wmax):
    hist = elwk[(elwk[:,1]<=wmax),:]
    hist[:,1] = wmax - hist[:,1] # - rxhistory[:,skips.rxc['week']] ## time backwards
    hist = hist[::-1,:] ## reverse
    return hist

def get_elements_weights(elwk, wmax):
    hist = get_history(elwk, wmax)

    if hist.shape[0] == 0:
        return [[], [], 0]
    uhist = hist[np.unique(hist[:,0],return_index=True)[1],:]
    return [list(uhist[:,0]), list(uhist[:,1]), float(sum(hist[:,1]<=52))]


def stringify(res):
    return "\t".join([str(i) for i in res]) + '\n'


    
def historyloop(Q, doid, name, prefix, outcomes, time_chunk_size):
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
        fname = prefix + '/hisinfo.' + str(name)+ '.'+ str(it) 
        open(fname,'w').close()
        return fname

    it = 0
    #fname = getfname(it)
    logf = prefix + '/vallogf'
    with open(logf,'a') as f:
        f.write('STart!' + name + ' ' + str(it) +'\n')
        
    obs = ''
    count = 0
    '''
    obs = {0:''} if mix else defaultdict(str)
    count = {0:0} if mix else defaultdict(int)
    fname = {0:getfname(it,0)} if mix else defaultdict(str)
    '''
    codesuffix = 'vi'

    tried = 0
    gots = 0
    t0 = time.time()
    for (pdone, person) in iter(Q.get, None):
        #print("P:" , person)

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
        alldo, visits = skips.urx_rx_dx(dat, doid, decode)
        if alldo:
            tried += 1                        
        for label in alldo:
            try:
                res = history_parse(label, visits,dat, decode, outcomes, codesuffix,
                                    time_chunk_size)
            except ValueError:
                print("VALUEERROR", person)
                pdb.set_trace()
                break
            if not res:
                continue
            gots += 1
            drug = res[0][0]
            res = [drug, person] + res[1] + [-1] + res[0] + res[2]
            obs  += stringify(res)
            count += 1
        if pdone % 100000 == 0:
            with open(logf,'a') as f:
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
                #print('memory use:', memoryUse)                
                f.write('done ' + str(pdone) +
                        "time={:1.2f}min, mem={:1.2f}Gb".format((time.time()-t0)/60, memoryUse) + '\n') # + ' ' + list(fname.values())[0] +'\n')
        #if len(obs) > 50000:
        if count > 100000:
            with open(getfname(it,0), 'a') as f:
                f.write(obs)
            obs = ''
            count = 0
            it += 1
        '''
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
        '''
    #pdb.set_trace()
    with open(getfname(it,0),'a') as f:
        f.write(obs)
        
    print("tried:", tried, " got: " , gots)

    with open(logf,'a') as f:
        f.write('Finished!' +str(name)  + " tried:" + str(it)+'\n')
    '''        
    with open(logf,'a') as f:
        f.write('Finished!' + list(fname.values())[0] + " tried:" + str(it)+'\n')

    for k in count:
        if count[k] == 0: continue
        if not k in fname:
            fname[k] = getfname(it, k)
        
        with open(fname[k] ,'a') as f:
            f.write(obs[k])
    '''
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

def main(doid,prefix, outcomes, time_chunk_size, ndo=0,neg_enQ=False, history=True):
    doid = [int(i) for i in open(doid).read().strip("\n").split("\n")]
    outcomes = pickle.load(open(outcomes,'rb'))
    oorder = sorted(outcomes.keys())
    outcomes = [outcomes[k] for k in oorder]
    prefix += "-" + str(time_chunk_size)
    if os.path.exists(prefix):
        if TESTING:
            shutil.rmtree(prefix)
        else:
            print(prefix + " EXISTS--QUITTING!")
            return

    os.mkdir(prefix)
    ndo = ndo if TESTING else 0
    nprocs = 3 if TESTING else 14

    Q = mp.Queue(maxsize=500000)
    #open('vallogf','w').close()
    plock = mp.Lock()
    enQproc = mp.Process(target = enQ.enQ if not neg_enQ else enQminus,
                         args = (Q, doid,nprocs,ndo))
    enQproc.start()
    if TESTING:
        target = historyloop if history else mysteryloop
        target(Q, doid if not neg_enQ else set([]),
                                            str(0), prefix,outcomes, time_chunk_size)
    else:
        loadprocs = []        
        for i in range(nprocs):
            loadprocs.append(mp.Process(target = historyloop if history else mysteryloop,
                                        args = (Q, doid if not neg_enQ else set([]),
                                                str(i), prefix,outcomes, time_chunk_size)))
            loadprocs[-1].start()
        for p in loadprocs:
            p.join()
    postparse(prefix)

def splitcat(fdir, dname):
    tosave = defaultdict(str)
    ct = defaultdict(int)
    ix = 0
    t0 = time.time()
    fname = fdir + "hisinfo_cat." + str(dname)
    if not os.path.exists(fname):
        print("NO " + fname)
        return
    for line in open(fname):
        line = line.strip().split("\t")
        drug = line[0]
        tosave[drug] += "\t".join(line[1:]) + "\n"
        ct[drug] += 1

        if ct[drug] > 2000:
            with open(fdir + "/split." + dname + "." + drug,'a') as f:
                f.write(tosave[drug])
            ct[drug] = 0
            tosave[drug] = ''
            #if TESTING:
            #    break
        ix += 1            
        if ix % 1000000 == 0:
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
            #print('memory use:', memoryUse)
            print(str(dname) + ' done ' + str(ix) +
                    "time={:1.2f}min, mem={:1.2f}Gb".format((time.time()-t0)/60, memoryUse) + '\n') # + ' ' + list(fname.values())[0] +'\n')
        
    for drug, drugct in ct.items():
        if drugct > 0:
            with open(fdir + "/split." + dname + "." + drug,'a') as f:
                f.write(tosave[drug])
            tosave[drug] = ''
    print("finished ", dname)
    
def postparse(fdir):

    if not fdir.endswith("/"):
        fdir += "/"
    ndo = 1000 if TESTING else 0
    nprocs = 2 if TESTING else 14

    import subprocess
    for dname in range(nprocs):
        #x = glob.glob("mixed_histories/hisinfo." + str(dname) + ".*")
        fis = {int(os.path.basename(i).split(".")[2]):i for i in glob.glob(fdir + "/hisinfo." + str(dname) + ".*")}
        for ix in range(len(fis)):
            subprocess.call("cat " + fis[ix] + " >> " + fdir + "/hisinfo_cat." + str(dname),shell=True)
    
    loadprocs = []
    if TESTING:
        splitcat(fdir, str(0))
    else:
        for i in range(nprocs):
            loadprocs.append(mp.Process(target = splitcat,
                                        args = (fdir, str(i))))
            loadprocs[-1].start()
        for p in loadprocs:
            p.join()

    #subprocess.call("wc -l " + fdir +  " split* | awk '{print $1"\t"$2" + "skipct",shell=True)
    #subprocess.call("wc -l " + fdir + " split* | awk '{print $1" + '"\t"' + "$2}' > " + fdir + "skipct",shell=True)
    subprocess.call("wc -l " + fdir + "split* | awk '{print $1" + '"\t"' + "$2}' > " + fdir + "skipct",shell=True)
    gotct = pd.read_table(fdir + "/skipct",header=None)
    gotct = gotct.loc[gotct[1].str.contains('split'),:]
    gotct['dr'] = [i.split(".")[2] for i in gotct[1]]
    gotct['dr'] = gotct['dr'].map(int)
    gotct[0] =  gotct[0].map(int)
    decode = pickle.load(open("../../data/decode.12.18.clid.allvocab.pkl",'rb'))
    z= gotct.groupby('dr').sum()
    z['name'] = [decode['i2gennme'][str(i)] for i in z.index]
    #
    z.to_pickle(fdir + "/drug_neighbor_counts.pkl")
    ##subprocess.call("bash run_sparsemat.sh " + fdir + " " + fdir + "/drug_neighbor_counts.pkl", shell=True)
            
if __name__=="__main__":
    '''
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
    #main(doid, prefix, outcomes)
    '''
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
        
def mysteryloop(Q, doid, name, prefix, outcomes, mix=False):
    decode = pickle.load(open(datdir +"decode.12.18.clid.allvocab.pkl",'rb'))
    conn, pcurs = skips.get_conn_curs()    
    econn = sqlite3.Connection("/project2/melamed/db/enr2.db")
    ecurs = econn.cursor()
    ecurs.execute("PRAGMA cache_size=2000;")

    codesuffix = 'vi'
    tried = 0
    gots = 0
    excludereasons = defaultdict(int)
    for (pdone, person) in iter(Q.get, None):
        #print("P:" , person)
        tried += 1
        pcurs.execute('select * from kv where person = ?',(person,))

        dat = json.loads(pcurs.fetchone()[1])
        ecurs.execute('select * from kv where person = ?',(person,))
        dat['enroll'] = json.loads(ecurs.fetchone()[1])

        alldo, visits = urx_rx_dx(dat, doid, decode, codesuffix)

        for label in alldo:
            res = history_parse(label, visits,dat, decode, outcomes, codesuffix, TEST=True)
            if res[0]==-10:
                for r in res[1:]:
                    excludereasons[r] += 1
        if pdone % 100000 == 0:
            with open('logf','a') as f:
                f.write('done ' + str(pdone) + '\n') # + ' ' + list(fname.values())[0] +'\n')

    print("tried:", tried, " got: " , gots)       
    f = open("excludereasons" + str(name) + ".pkl",'wb')
    pickle.dump(excludereasons, f)
    f.close()
