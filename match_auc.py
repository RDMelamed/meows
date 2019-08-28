import numpy as np
import pandas as pd
import sys
import numpy as np
import pickle
import pdb
import subprocess
import glob
import os

def gensuff(trt,ctl,suff):
    return "Target." + str(trt) + '.' + str(ctl) + suff 


def ren(df, dovoc=np.array([])):
    df = df.rename({"":"NN"},axis=1)
    if dovoc.shape[0] > 0:
        df = df.rename({i:str(i) +'-'+ dovoc.loc[i,'name'] for i in df.index})
    return df

def get_entries(entries, trt, ctls, pref, suff, dovoc):
    prefdo = "multidrug_neuropsycho/" + pref
    #do = list(dovoc.index)
    df = pd.DataFrame(np.zeros((len(ctls), len(entries))),index=ctls,columns=entries)
    for s in entries:
        for d in ctls:
            #print(d)
            #print(prefdo + s + gensuff(trt, d, suff))
            #print(prefdo + s + gensuff(trt, d, suff))
            mod = pickle.load(open(prefdo + s + gensuff(trt, d, suff),'rb'))
            whichmod = list(mod['preds'].keys())[0]
            if mod['xval'].shape[0] > 1:
                whichmod = mod['xval'].mean(axis=1).idxmax()
            df.loc[d,s] = mod['xval'].loc[whichmod,:].mean() #list(mod['roc'].values())[0]
    
    return ren(df, dovoc)

def get_roc_suff(doid, drug, pref, suff, dovoc,nops=True):
    do = list(set(doid) - set([drug]))
    ctl = do[0]
    suffdo = gensuff(drug, ctl,suff)
    prefdo = "multidrug_neuropsycho/" + pref 
    entries = glob.glob(prefdo + "*" + suffdo )
    if nops:
        entries = [g for g in entries if not "PSM" in g]
    entries = [i for i in entries if "Xfer" not in i] + [i for i in entries if "Xfer" in i]

    entries = [i.replace(prefdo,"").replace(suffdo,"") for i in entries]
    #print("hm:",entries[0])
    #entries = [i.replace(prefdo,"").replace(".idsago.agobins.psmod.pkl","") for i in entries]

    rvocabs = get_entries(entries, drug,do,pref,suff, dovoc)
    #print(entries)
    return rvocabs


def gen_mod_name(hisdir, m, trt, ctl):
    return hisdir + m[0] + "Target." + str(trt) + "." + str(ctl) + m[1]

def rendict(dovoc):
    return {d:str(d) + "-" + dovoc.loc[d,'name'] for d in dovoc.index}


def moddo_auc(moddo, trt, ctls, hisdir,ft='ago'):
    levct = mod_count(trt, moddo, ctls)    
    postname = ".ids" + ft + ".agobins.psmod.pkl"
    #do = list(dovoc.index)
    modnames = [getnm(m) for m in moddo]
    #df = pd.DataFrame(np.zeros((len(ctls), len(moddo))),index=ctls,columns=modnames)
    comp = {}
    rawauc = {}
    for d in ctls:
        if d == trt:
            continue
        mod = pickle.load(open(hisdir + 'PSMTarget.' + str(trt) +"." + str(d) + '.idsago.agobins.psmod.pkl','rb'))
        rawauc[d] = {'auc':mod['xval'].mean(axis=1)[0],
                        'ct':mod['ids'].shape[0],
                        'tct':mod['lab'].sum(),'cct':(1-mod['lab']).sum()}        
        aucd = {}
        for i,m in enumerate(moddo):
            mod = pickle.load(open(gen_mod_name(hisdir, m,trt,d) + postname,'rb'))
            whichmod = list(mod['preds'].keys())[0]
            if mod['xval'].shape[0] > 1:
                whichmod = mod['xval'].mean(axis=1).idxmax()
            aucd[modnames[i]] = mod['xval'].loc[whichmod,:].mean() #list(mod['roc'].values())[0]

        levphen = loadncs(trt,int(d), moddo, hisdir)
        modsnc = levphen['ci'].columns.levels[0]
        ui = levphen['ci'].xs("surv.UI",level=1,axis=1)
        li = levphen['ci'].xs("surv.LI",level=1,axis=1)
        pairinfo = {
            'auc':pd.Series(aucd).loc[modsnc],
            'ct':levct.loc[d,:].loc[modsnc],
            'abone':(1- levphen['ci'].xs("surv",level=1,axis=1)).abs().mean(axis=0),
            'var':levphen['ci'].xs("surv",level=1,axis=1).var(axis=0),
            'wid':(ui-li).mean(axis=0), 'TN':((ui > 1) & (li < 1)).mean() }
        comp[d] = pd.DataFrame(pairinfo)
    return pd.concat(comp,axis=0), pd.DataFrame(rawauc).transpose() #            


def wcdf(fn, drug, dovoc):
    wc = pd.read_table(fn,sep="\t",header=None)    
    wc = wc.loc[wc[1]!='total',:]
    #d = 5387
    d = drug
    comp = [int(s[(s.find(str(d))+len(str(d))+1):].split(".")[0]) for s in wc[1]]
    wc['comp'] = comp
    meth = []
    for i in wc.index:
        s = wc.loc[i,1]
        ct = wc.loc[i,'comp']
        meth.append(s[:s.find('Target')] + ('.embedmatched' if 'embedmatched' in s else '') + 
                    ('.sparsematched' if 'sparsematched' in s else ''))
    wc['meth'] = meth
    #return wc
    #print("dropping psfilt!")
    #wc = wc.loc[~wc[1].str.contains("PSfilt"),:]
    wc.loc[wc[1].str.contains("PSfilt"),'meth'] = 'PSfilt'
    a = wc.set_index(['comp','meth']).drop(1,axis=1).transpose().stack('meth').transpose()
    a.columns = a.columns.droplevel(0)
    todrop = ['PSM0.1','PSM0.2','PSM0.2.sparsematched']

    todrop = set(todrop) & set(a.columns)
    if todrop:
        a = a.drop(todrop,axis=1)
    return ren(a, dovoc)

def get_count(drug,dovoc=np.array([]),ctl=''):
    #wc  -l *Target.5387*trt | awk '{print $1"\t"$2}' > nmatch5387
    fn = "multidrug_neuropsycho/nmatch" + ctl + str(drug)
    print("yo!",fn)
    if not os.path.exists(fn):
        print("make!",fn)
        subprocess.call("cd multidrug_neuropsycho/; wc  -l *Target." + str(drug) +  
                        '*' + ('trt' if not ctl else 'ctl') +' | awk \'{print $1"\t"$2}\' > nmatch' + ctl + str(drug),
                       shell=True)

    mname = [getnm(m) for m in modperc]
    df = pd.DataFrame(index=set(doid)-set([trt]), columns = mname)
    for d in doid:
        if d == trt: continue
        for m in modperc:
            df.loc[d,match_auc.getnm(m)] = wc.loc[wc[1]==match_auc.gen_mod_name('', m, trt, d) + ".ids.trt",0].values[0]    

    return wcdf(fn, drug, dovoc)


def mod_count(drug, modperc,doid, ctl=''):
    #wc  -l *Target.5387*trt | awk '{print $1"\t"$2}' > nmatch5387
    fn = "multidrug_neuropsycho/nmatch" + ctl + str(drug)
    if not os.path.exists(fn):
        print("make!",fn)
        subprocess.call("cd multidrug_neuropsycho/; wc  -l *Target." + str(drug) +  
                        '*' + ('trt' if not ctl else 'ctl') +' | awk \'{print $1"\t"$2}\' > nmatch' + ctl + str(drug),
                       shell=True)

    mname = [getnm(m) for m in modperc]
    wc = pd.read_table(fn,sep="\t",header=None)    
    df = pd.DataFrame(index=set(doid)-set([drug]), columns = mname)
    for d in doid:
        if d == drug: continue
        for m in modperc:
            #print(gen_mod_name('', m, drug, d) + ".ids." + ('trt' if not ctl else 'ctl'))
            df.loc[d,getnm(m)] = wc.loc[wc[1]==gen_mod_name('', m, drug, d) + ".ids." + ('trt' if not ctl else 'ctl'),0].values[0]    
    return df

sys.path.append("../../code")
def load_coxeff(f):
    wtres = pd.read_table(f,sep=" ")
    #wtres = get_r_out(f)
    cid = {}
    i = 'surv'
    cid[i] = np.exp(wtres[i]) 
    cid[i + ".LI"] = np.exp(wtres[i] - 1.96*wtres[i + "se"]) 
    cid[i + ".UI"] = np.exp(wtres[i] + 1.96*wtres[i + "se"])
    cid["event"] = wtres['events']
    
    return pd.DataFrame(cid)
    
def load_regr(regrres):
    tocat = []
    expci = {}
    for m,f in regrres.items():
        #print(f)
        
        expci[m] = load_coxeff(f)
        #wtres = wtres.rename(columns={k:m + "."+k for k in wtres.columns})        
        #tocat.append(wtres)
    return  pd.concat(expci,axis=1) #pd.concat(tocat,axis=1),

def getnm(m): return (m[0] if m[0] != "" else "NN") + m[1]

def loadncs(trt, ctl,moddo, sdir,ft='ago'):
    sdir = sdir.strip("/")  + ".ests/"
    #print(sdir + m[0] + "Target." + str(trt) + "." + str(ctl) + m[1] + ".ipw.effects.txt")
    #dd = {getnm(m):sdir + m[0] + "Target." + str(trt) + "." + str(ctl) + m[1] + ft + ".ipw.effects.txt"
    #                                             for m in moddo}
    #print(dd)

    coxse, coxci = load_regr({getnm(m):sdir + m[0] + "Target." + str(trt) + "." + str(ctl) + m[1] + ft + ".ipw.effects.txt"
                                                 for m in moddo})
    
    outcs = {}
    matchct = {}
    for m in moddo:
        #print(m[0] + 'Target.'+ str(trt) + "." + str(ctl) + m[1] )
        o = pd.read_table(sdir + m[0] + 'Target.'+ str(trt) + "." + str(ctl) + m[1] + ft + 
                                                  '.outmat',dtype=int)
        nm = getnm(m)
        outcs[nm] = (o > 0).sum(axis=0)[2:]
        matchct[nm] = o.shape[0]
    outcs = pd.DataFrame(outcs)
    outcsel = list(outcs.loc[outcs.min(axis=1) > 100].index)
    #coxci = coxci.loc[outcsel,:]
    coxci = coxci.loc[outcsel,:]
    outcs = outcs.loc[outcsel,:]
    return {'ci':coxci, 'oct':outcs,'mct':matchct}

def load_eff(hisdir, ctl,trt, prefs, addpath=[]):
    maincomp = {k:hisdir + k+".Target." + str(trt) + "." + str(ctl) + ".eff" for k in prefs}
    if addpath:
        for p in addpath:
            maincomp['min-' + p] = 'min-' + hisdir+ p +".Target." + str(trt) + "." + str(ctl) + ".eff"
    eff2634 = load_regr(maincomp)
    return eff2634.loc[eff2634.xs('event',level=1,axis=1).min(axis=1) > 200,:]

def get_comp(drug, mod49,sdir):

    levauc =match_auc.get_roc_suff(doid, drug, "",".idsago.agobins.psmod.pkl", dovoc)
    levsp = match_auc.get_roc_suff(doid, drug, "",".sparsematched.idsago.agobins.psmod.pkl",dovoc,nops=False)

    levemb = match_auc.get_roc_suff(doid, drug, "",".embedmatched.idsago.agobins.psmod.pkl",dovoc, nops=False)
    levauc = pd.concat((levauc['full'], levsp.rename({i:i+ ".sparsematched" for i in levsp.columns},axis=1), 
                        levemb.rename({i:i + ".embedmatched" for i in levsp.columns},axis=1)),axis=1)

    levct = match_auc.get_count(drug,dovoc)
    #levauc = levauc.
    levauc = levauc.rename({'full':'NN'})
    levauc = levauc.rename({a:a.split("-")[0] for a in levauc.index},axis=0)   
    todrop = set(['vi2bigi','vi2clid']) & set(levct.columns)
    if len(todrop):
        levct = levct.drop(todrop,axis=1)
    levct = levct.rename({a:a.split("-")[0] for a in levct.index},axis=0)    
    comp ={}
    rawauc = {}
    for pair in levauc.index:
        levphen = match_auc.loadncs(drug,int(pair), mod49, sdir)
        mod = pickle.load(open(sdir + 'PSMTarget.' + str(drug) +"." + pair + '.idsago.agobins.psmod.pkl','rb'))
        rawauc[pair] = {'auc':mod['xval'].mean(axis=1)[0],'tct':levct.loc[str(pair),'PSM']}
        levphen['ci'] = levphen['ci'].rename(columns={i:i.replace("blah","") for i in levphen['ci'].columns.levels[0]},level=0)
        pairinfo = {'auc':levauc.loc[str(pair),:].rename({'full':'NN'}).loc[levphen['ci'].columns.levels[0]],
                'ct':levct.loc[str(pair),:].rename({'full':'NN'}).loc[levphen['ci'].columns.levels[0]],
                'nct':levct.loc[str(pair),:].rename({'full':'NN'}).loc[levphen['ci'].columns.levels[0]]/levct.loc[str(pair),'PSM'],
                 #'abone':(1- levphen['ci'].xs("surv",level=1,axis=1).mean(axis=0)).abs(),
                'abone':(1- levphen['ci'].xs("surv",level=1,axis=1)).abs().mean(axis=0),
            'var':levphen['ci'].xs("surv",level=1,axis=1).var(axis=0),
        'wid':(levphen['ci'].xs("surv.UI",level=1,axis=1)-levphen['ci'].xs("surv.LI",level=1,axis=1)).mean(axis=0)}
        comp[pair] = pd.DataFrame(pairinfo)
    comp  = pd.concat(comp,axis=0)
    
    carbrawauc = pd.DataFrame(rawauc).transpose().rename(rendict).sort_values('auc')
    carbrawauc['P95m.auc'] = comp.xs('PSM95.sparsematched',axis=0,level=1)['auc'].rename(rendict)
    ctlct = match_auc.get_count(drug,dovoc,ctl='ctl')
    carbrawauc['ctlct'] = ctlct['PSM'] #carbcomp.xs('PSM95.sparsematched',axis=0,level=1)['auc'].rename(rendict)

    carbrawauc['mct'] = comp.xs('PSM95.sparsematched',axis=0,level=1)['ct'].rename(rendict)
    carbrawauc['NNauc'] = comp.xs('NN',axis=0,level=1)['auc'].rename(rendict)

    carbrawauc['mctNN'] = comp.xs('NN',axis=0,level=1)['ct'].rename(rendict)    
    return comp,rawauc, carbrawauc
