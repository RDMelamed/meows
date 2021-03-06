
import numpy as np
import pandas as pd
import sys
import numpy as np
import pickle
import pdb
import subprocess
import glob


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
sys.path.append('/project2/melamed/wrk/iptw/code')
import plot_helper    
def plotci(est, coxci, matcher=['wt','match','strat'], tots=[],tot_not_crude=True,sord_in = []):
    sord = coxci.xs(est,level=1,axis=1).mean(axis=1).sort_values()
    if len(sord_in) == 0:
        coxci = coxci.loc[sord.index,:]
    else:
        coxci = coxci.loc[sord_in.index,:]
        sord = sord_in
    f, axen = plt.subplots(1,figsize=(11,3))


    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=len(matcher))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    inc = 1/(len(matcher) + 2)
    ym = np.percentile(coxci.xs('surv.UI',axis=1,level=1).stack(),95)
    ypos = np.percentile(coxci.xs('surv.UI',axis=1,level=1).stack(),90)    
    yl = np.percentile(coxci.xs('surv.LI',axis=1,level=1).stack(),2)    
    for (i, ro) in enumerate(coxci.index):
        for (j, meh) in enumerate(matcher):
            axen.plot([i+j*inc,i+j*inc], [coxci.loc[ro,(meh,est + ".LI")],coxci.loc[ro,(meh,est + ".UI")]],c=scalarMap.to_rgba(j) if len(matcher) > 1 else "y")
            if coxci.loc[ro,(meh,est + ".LI")]>1 or coxci.loc[ro,(meh,est + ".UI")]<1:
                axen.plot(i+j*inc,ypos,"+",c='w',label="_nolegend_",markersize=10)
                axen.plot(i+j*inc,ypos,"+",c=scalarMap.to_rgba(j),label="_nolegend_")
    axen.set_xticks(np.arange(coxci.shape[0]))
    axen.set_xticklabels(list(coxci.index),rotation=90)
    leglab = ['{:s} CI0={:1.2f}, CIwid={:1.2f}'.format(meh, 
        ((coxci.loc[:,(meh,est + '.UI')] > 1) & (coxci.loc[:,(meh,est + '.LI')] < 1)).mean(),
        (coxci.loc[:,(meh,est + '.UI')] - coxci.loc[:,(meh,est + '.LI')]).mean())
              for meh in matcher]
    print(leglab)
    axen.plot([0,coxci.shape[0]],[1,1],':',color='gray')
    plot_helper.trleg(axen,leglab)
    if len(tots) > 0:
        
        ax = axen.twinx()
        if tot_not_crude:
            ax.plot(np.arange(sord.shape[0]), tots.loc[sord.index,'tot'],':')
            ax.set_yscale("log")
            ax.set_ylabel("# events")
        else:
            ax.plot(np.arange(sord.shape[0]), tots.loc[sord.index,'crudeOR'],':',color='blue')
            ax.set_ylabel("crude OR")
            t = tots.loc[sord.index,'tot']
            t = (t - t.min())/t.max()*tots['crudeOR'].max()
            ax.plot(np.arange(sord.shape[0]), t,':',color='gray')            
    return f, axen,[yl, ym]
from IPython.display import display, HTML
import file_names
def pairci(dird,trt,ctl,newvoc,sord_in=[],outc_sel = [],filt='',tots=True):
    runname, trtname = file_names.get_trt_names(dird, trt)
    pairname = runname + str(ctl)
    prefs = {i.replace(pairname +".","").replace(".eff","").replace(".unwt",""):i
             for i in glob.glob(pairname + "*.eff")}
    if filt:
        prefs = {k:v for k,v in prefs.items() if filt in k}
    eff325 = match_auc.load_regr(prefs)
    eff325 = eff325.loc[eff325.xs('event',level=1,axis=1).min(axis=1) > 100,:]
    if outc_sel:
        eff325 = eff325.loc[outc_sel,:]
    #eff325 = match_auc.load_eff(dird,ctl, trt, prefs)
    f, ax ,yl= plotci('surv',eff325, matcher=prefs, 
                      tots=[] if not tots else pd.DataFrame(eff325.xs('event',axis=1,level=1).min(axis=1),
                                                         columns=['tot']),sord_in = sord_in) #outcs.min(axis=1)[outcsel])
    #
    tname = newvoc.loc[(newvoc['id']==trt) &(newvoc['type']=='rx'),'name'].values[0]
    cname = newvoc.loc[(newvoc['id']==ctl) &(newvoc['type']=='rx'),'name'].values[0]
    ax.set_title(tname + " vs CTL=" + cname)
    return eff325, f, ax


def readmse(est):
    eff =  pd.read_table(est + ".msmeff.txt",sep="\t")
    se =  pd.read_table(est + ".msmmse.txt",sep="\t")
    N = pd.DataFrame(index=eff.columns,  columns = ['x'])
    N.iloc[:] = 200
    if os.path.exists(est + ".N.txt"):
        N =   pd.read_table(est + '.N.txt',sep=" ")
    ret = {}
    for cancer in eff.columns:
        reg = pd.DataFrame({'survco':eff[cancer],
                       'survse':se[cancer]})
        reg['mse']= reg['survco']**2 + reg['survse']**2
        ret[cancer]=  reg
    return N, pd.concat(ret,axis=1)

def msmpairci(dird,trt,ctl,newvoc,cuts, suff  =  '', antisuff='', sord_in=[],outc_sel = [],tots=True):
    runname, trtname = file_names.get_trt_names(dird, trt)
    pairname = runname + str(ctl)
    tname = newvoc.loc[(newvoc['id']==trt) &(newvoc['type']=='rx'),'name'].values[0]
    cname = newvoc.loc[(newvoc['id']==ctl) &(newvoc['type']=='rx'),'name'].values[0]
    title = tname + " vs CTL=" + cname
    return msmeff(pairname, title,cuts, suff = suff, antisuff=antisuff, sord_in=sord_in,outc_sel = outc_sel,tots=tots)

def comboci(dird,trt, drug2,prefix,newvoc,cuts, suff  =  '', antisuff='', sord_in=[],outc_sel = [],tots=True):
    runname = file_names.combo_names(dird, trt)
    pairname = runname + prefix + str(drug2)
    tname = newvoc.loc[(newvoc['id']==trt) &(newvoc['type']=='rx'),'name'].values[0]
    cname = newvoc.loc[(newvoc['vi']==drug2) &(newvoc['type']=='rx'),'name'].values[0]
    title = tname + " + drug2=" + cname
    return msmeff(pairname, title,cuts, suff = suff, antisuff=antisuff, sord_in=sord_in,outc_sel = outc_sel,tots=tots)

    
def msmeff(pairname, title, cuts, suff  =  '', antisuff='', sord_in=[],outc_sel = [],tots=True):
    print("pn",pairname, " anti=",antisuff)
    prefs= [i.replace(".msmeff.txt","") for i in glob.glob(pairname + "*" + suff + ".msmeff.txt")]
    if antisuff:
        prefs= [i for i in prefs if not antisuff in i]

        
    #print("\n".join(prefs))
    prefs = {i.replace(pairname +".",""):i for i in prefs}
    pnames = []
    agg = {}
    Ns = {}
    for p, fname in prefs.items():
        N,  estf = readmse(fname)
        for cut in cuts:
            est = pd.DataFrame(estf.loc[cut,:]).transpose().stack().transpose()
            est.columns= est.columns.droplevel(0)
            pn  = p + "-" + str(cut)
            pnames.append(pn)
            agg[pn] = pd.DataFrame({"surv":np.exp(est['survco']),
                                   "surv.LI":np.exp(est['survco']  - 1.96*est['survse']),
                                   "surv.UI":np.exp(est['survco']  + 1.96*est['survse']),
                                   "event":N['x']})
        #Ns[p]= N['x']
        #Ndo = N.drop("treat").loc[N['x']> 100].index

    #Ns = pd.DataFrame(Ns)
    eff325 = pd.concat(agg, axis=1)
    eff325 = eff325.loc[eff325.xs("event",axis=1,level=1).min(axis=1) > 80,:]        
    if outc_sel:
        eff325 = eff325.loc[outc_sel,:]
    #eff325 = match_auc.load_eff(dird,ctl, trt, prefs)
    f, ax ,yl= plotci('surv',eff325, matcher=pnames, 
                      tots=[] if not tots else pd.DataFrame(eff325.xs('event',axis=1,level=1).min(axis=1),
                                                         columns=['tot']),sord_in = sord_in) #outcs.min(axis=1)[outcsel])
    #
    ax.set_title(title)
    return eff325, f, ax


def save(dird,trt,ctl,newvoc,cuts, suff  =  '', sord_in=[],outc_sel = [],tots=True):
    runname, trtname = file_names.get_trt_names(dird, trt)
    pairname = runname + str(ctl)
    #print("pn",pairname)
    prefs= [i.replace(".msmeff.txt","") for i in glob.glob(pairname + "*" + suff + ".msmeff.txt")]
    #print("\n".join(prefs))
    prefs = {i.replace(pairname +".",""):i for i in prefs}
    agg = {}
    Ns = {}
    for p, fname in prefs.items():
        for cut in cuts:
            N,  est = readmse(fname)
            est = pd.DataFrame(est.loc[cut,:]).transpose().stack().transpose()
            est.columns= est.columns.droplevel(0)
            agg[p + "-" + str(cut)] = pd.DataFrame({"surv":np.exp(est['survco']),
                                   "surv.LI":np.exp(est['survco']  - 1.96*est['survse']),
                                   "surv.UI":np.exp(est['survco']  + 1.96*est['survse']),
                                   "event":N['x']})
        #Ns[p]= N['x']
        #Ndo = N.drop("treat").loc[N['x']> 100].index

    #Ns = pd.DataFrame(Ns)
    eff325 = pd.concat(agg, axis=1)
    eff325 = eff325.loc[eff325.xs("event",axis=1,level=1).min(axis=1) > 80,:]        
    if outc_sel:
        eff325 = eff325.loc[outc_sel,:]
    #eff325 = match_auc.load_eff(dird,ctl, trt, prefs)
    f, ax ,yl= plotci('surv',eff325, matcher=prefs, 
                      tots=[] if not tots else pd.DataFrame(eff325.xs('event',axis=1,level=1).min(axis=1),
                                                         columns=['tot']),sord_in = sord_in) #outcs.min(axis=1)[outcsel])
    #
    ax.set_title(tname + " vs CTL=" + cname)
    return eff325, f, ax

def single_outcome(dird,trt, ctllist, outc,voc):
    runname, trtname = file_names.get_trt_names(dird, trt)
    ctldat = dict()
    for ctl in ctllist:
        pairname = runname + str(ctl)
        expci = {}
        todof = glob.glob(pairname + ".PSM*.eff")
        if len(todof) == 0:
            continue
        for i in todof:
            nicename = i.replace(pairname +".","").replace(".eff","").replace(".unwt","")
            expci[nicename] = match_auc.load_coxeff(i)
        x = pd.DataFrame(pd.concat(expci,axis=1).loc[outc,:]).transpose().stack().transpose()
        x.columns = x.columns.droplevel(level=0)
        ctldat[ctl] = x
        
    ctllist = list(ctldat.keys())
    print(ctllist)
    f, axen = plt.subplots(1,figsize=(11,3))
    inc = 1/(len(ctllist) + 2)
    coxci = pd.concat(ctldat,axis=0)
    ym = np.percentile(coxci['surv.UI'],95)
    ym = np.percentile(coxci['surv.UI'],2)    
    ypos = np.percentile(coxci['surv.UI'],90)

    for (i, ro) in enumerate(ctllist):
        toplot = ctldat[ro]
        print(i,ro)
        for (j, meh) in enumerate(toplot.index):
            #print(ro,meh)
            lty = '-'
            if toplot.loc[meh,'event'] < 200:
                lty = ':'
            axen.plot([i+j*inc,i+j*inc], [toplot.loc[meh,"surv.LI"],toplot.loc[meh,"surv.UI"]],lty)
            if toplot.loc[meh,"surv.LI"] > 1 or toplot.loc[meh,"surv.LI"] > 1 :
                axen.plot(i+j*inc,ypos,"+",c='w',label="_nolegend_",markersize=10)
                axen.plot(i+j*inc,ypos,"+",c='k',label="_nolegend_")
    axen.set_xticks(np.arange(len(ctllist)))
    names = list(voc.loc[voc['type']=='rx',:].set_index('id').loc[ctllist,'name'].values)
    axen.set_xticklabels(names,rotation=90)
    axen.plot([0,len(ctllist)],[1,1],':',color='gray')
    return f, axen,pd.concat(ctldat, axis=0)




# Assuming that dataframes df1 and df2 are already defined:
def distFromZero(coagregci):
    f, ax = plt.subplots(1,3,figsize=(14,4))
    abone = (1- coagregci.xs('surv',level=1,axis=1)).abs().mean(axis=0) #).abs()
    var = coagregci.xs('surv',level=1,axis=1).var(axis=0)
    dod = pd.DataFrame({'abone':abone,'var':var})
    display(dod)
    
    dod = dod.loc[dod['abone'] < .15,:]
    ax[0].plot(dod['abone'],dod['var'],'.')
    for i in dod.index:
        ax[0].text(dod.loc[i,'abone'], dod.loc[i,'var'],i,rotation=45,ha='left',va='bottom')
    #ax[0].set_xlim(.05,.15)
    #ax[0].set_ylim(0,.1)
    ax[0].set_xlabel("est distance from 1")
    ax[0].set_ylabel("est variance")
    cidf = pd.DataFrame({'n0':((coagregci.xs('surv.UI',level=1,axis=1) > 1) &(coagregci.xs('surv.LI',level=1,axis=1) < 1)).sum(axis=0)/coagregci.shape[0],
                        'wid':(coagregci.xs('surv.UI',level=1,axis=1)-coagregci.xs('surv.LI',level=1,axis=1)).mean(axis=0)})
    ax[1].plot(cidf['n0'],cidf['wid'],'.')
    for i in cidf.index:
        ax[1].text(cidf.loc[i,'n0'], cidf.loc[i,'wid'],i,rotation=45,ha='left',va='bottom')
    ax[1].set_xlabel("fraction CI covering 0")
    ax[1].set_ylabel("CI widthse")
    #return dod, cidf
    points = "-"
    try:
        dod['oldind'] = list(dod.index)
        dod.index = dod.index.map(float)
        cidf.index = cidf.index.map(float)
        dod = dod.sort_index()
    except ValueError:
        points = "."
        print("unsorted")
    ax[2].plot(dod['abone'],cidf.loc[dod.index,'n0'],points)
    for i in dod.index:
        ax[2].text(dod.loc[i,'abone'], cidf.loc[i,'n0'],i,rotation=45,ha='left',va='bottom')
    ax[2].set_xlabel("mean distance from 0")
    ax[2].set_ylabel("fraction CI covering 0")
    
    return dod, cidf    

def cipair2(levphen,notot=False,sord_in = []):
    dod, cidf = distFromZero(levphen)
    f, ax ,yl= plotci('surv',levphen, matcher=cidf.index, tots=[] if notot else pd.DataFrame(levphen.xs('event',axis=1,level=1).min(axis=1),columns=['tot']),sord_in = sord_in) #outcs.min(axis=1)[outcsel])
    ax.set_ylim(yl)
    return f, ax,yl


import glob

def ps2pref(ps,lab):
    v = np.log(ps/(1-ps)) - np.log(lab.mean()/(1-lab.mean()))
    return np.exp(v)/(1+np.exp(v))

import os
#sys.path.append("/project2/melamed/wrk/iptw/work/02.27_match2nc")


def plot_ps(modfn, ids, ax):
    fullps = pickle.load(open(modfn,'rb'))
    modsetting = fullps['xval'].mean(axis=1).idxmax() 
    fullps = pd.Series(ps2pref(fullps['preds'][modsetting], fullps['lab']), index=fullps['ids'])
    #plt.plot(xval, density(xval))
    #ax.fill(x, zer, x,a,'b', alpha=.2,label='unexposed')
    ids = fullps['ids'][fullps['lab']==1]
    if len(set(ids) - set(fullps.index)) > 0:
        print("fucked at:", fn,len(set(ids) - set(fullps.index)))
        ids = ids[np.isin(ids,fullps.index)]
    dens = gaussian_kde(fullps.loc[ids])
    yv = dens(xval) 
    colorVal = scalarMap.to_rgba(i)
    i += 1
    nm = mod[0] + mod[1]
    nm = nm if nm else "NN"
    #print("@",fn, "PSMTarget" in fn)
    ax[i][0].fill_between(xval, zer,yv,facecolor=colorVal,alpha=.3,label=nm)  
    ax[i][0].set_title(nm)  
    dens = gaussian_kde(fullps.loc[m['ids'][m['lab']==0]])
    yv = dens(xval)  
    ax[i][0].fill_between(xval, zer,yv,facecolor='gray',alpha=.3,label=nm)
    import weights_outcomes
    ipw = weights_outcomes.ipw(m)
    dens = gaussian_kde(ipw[m['lab']==0])
    yv = dens(ipwx)  
    ax[i][1].fill_between(ipwx, zer,yv,facecolor='blue',alpha=.3,label=nm)
    dens = gaussian_kde(ipw[m['lab']==1])
    yv = dens(ipwx)  
    ax[i][1].fill_between(ipwx, zer,yv,facecolor='red',alpha=.3,label=nm)

def prefpair(pair, modperc):
    modperc = modperc
    fullps = pickle.load(open('multidrug_neuropsycho/PSMTarget.' + pair + '.idsago.agobins.psmod.pkl','rb'))
    settings = list(fullps['preds'].keys())
    modsetting = settings[0] if len(settings) == 1 else fullps['xval'].mean(axis=1).idxmax() 
    fullps = pd.Series(ps2pref(fullps['preds'][modsetting], fullps['lab']), index=fullps['ids'])
    
    #fullps = ps_match2.get_ps('multidrug_neuropsycho/PSMTarget.' + pair + '.idsago.agobins.psmod.pkl')
    i = 0
    from scipy.stats import gaussian_kde
    import matplotlib.cm as cmx
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    '''
    modperc = [m for m in modperc if not 'embed' in m[1]]
    if not os.path.exists(genfn(pair, ['','.PSfilt'])):
        print("removing psfilt")
        modperc = [m for m in modperc if not 'PSfilt' in m[1]]
    '''
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=len(modperc))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    f, ax = plt.subplots(len(modperc) + 1,2,figsize=(5,(len(modperc) + 1)*2),tight_layout=True)
    xval = np.linspace(0,1,30)
    ipwx = np.linspace(0,2,30)
    zer = np.zeros(xval.shape)    
    for mod in modperc:
        fn = genfn(pair, mod)
        #print(fn)
        m = pickle.load(open(fn,'rb'))
        
        whichmod = m['xval'].mean(axis=1).idxmax()
        

        #plt.plot(xval, density(xval))
        #ax.fill(x, zer, x,a,'b', alpha=.2,label='unexposed')
        ids = m['ids'][m['lab']==1]
        if len(set(ids) - set(fullps.index)) > 0:
            print("fucked at:", fn,len(set(ids) - set(fullps.index)))
            ids = ids[np.isin(ids,fullps.index)]
        dens = gaussian_kde(fullps.loc[ids])
        yv = dens(xval) 
        colorVal = scalarMap.to_rgba(i)
        i += 1
        nm = mod[0] + mod[1]
        nm = nm if nm else "NN"
        #print("@",fn, "PSMTarget" in fn)
        if not "PSMTarget" in fn:
            #print(nm)
            ax[0][0].fill_between(xval, zer,yv,facecolor=colorVal,alpha=.3,label=nm)  
        ax[i][0].fill_between(xval, zer,yv,facecolor=colorVal,alpha=.3,label=nm)  
        ax[i][0].set_title(nm)  
        dens = gaussian_kde(fullps.loc[m['ids'][m['lab']==0]])
        yv = dens(xval)  
        ax[i][0].fill_between(xval, zer,yv,facecolor='gray',alpha=.3,label=nm)
        
        ipw = weights_outcomes.ipw(m)
        dens = gaussian_kde(ipw[m['lab']==0])
        yv = dens(ipwx)  
        ax[i][1].fill_between(ipwx, zer,yv,facecolor='blue',alpha=.3,label=nm)
        dens = gaussian_kde(ipw[m['lab']==1])
        yv = dens(ipwx)  
        ax[i][1].fill_between(ipwx, zer,yv,facecolor='red',alpha=.3,label=nm)
    ax[i][0].set_xlabel('ps (before matching)')
    ax[i][1].set_xlabel('IPW')    
    ax[0][1].remove()
    ax[0][0].legend(loc=2,bbox_to_anchor=(1.05,1))

    import matplotlib.cm as cmx
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

import match_auc
def bounds_plot(sorted_ctl, method_prefixes, trt, auc, dovoc):
    bounds = {}
    for i in sorted_ctl:
        levi = match_auc.load_eff(i,trt,method_prefixes)
        ui = levi.xs('surv.UI',axis=1,level=1)
        li = levi.xs('surv.LI',axis=1,level=1)
        val = ui.where(ui < 1, other=li.where(li > 1,other=1))
        #val = val.loc[set(val.index) & bounds.columns,:].transpose()
        bounds[i] = val.transpose()
    b2 = pd.concat(bounds,axis=0)
    matchers = b2.loc[sorted_ctl[0],:].index
    b3 = b2.loc[pd.MultiIndex.from_product([sorted_ctl, matchers]),:] #sord,axis=0,level=0)
    b4=  b3.loc[:,b3.loc[(sorted_ctl[0],method_prefixes[0]),:].sort_values().index]
    b4l = np.log(b4)

    a = b4l
    sz = 5
    imgheight = .6
    imgwidth = .6
    imgbottom = .1    
    imgleft = .2        
    fig = plt.figure(figsize=(sz*3,sz*2))

    axmatrix = fig.add_axes([0.1,imgbottom,imgwidth,imgheight])
    cmin= -.2
    cmax = .2
    '''
    norm = colors.Normalize(vmin=-.3, vmax=.3)
    cmap = cmx.bwr
    m = cmx.ScalarMappable(norm=norm, cmap=cmap)
    '''
    masked_array = np.ma.array(a, mask=pd.isnull(a))
    cmap = cmx.get_cmap('bwr')
    cmap.set_bad('gray',1.)
    cax = axmatrix.imshow(masked_array,  cmap=cmap,vmin=cmin, vmax = cmax, #-1*absmax, vmax=absmax,
                          interpolation='nearest', aspect = 'auto')
    axmatrix.set_xticks(range(a.shape[1]))
    axmatrix.set_yticks(range(a.shape[0]))
    axmatrix.set_xticklabels(a.columns,rotation=90,fontsize=12)
    axcbar = fig.add_axes([ .65 ,0.05,0.1,0.6])
    axcbar.set_xticks([])
    axcbar.set_yticks([])
    #axcbar.set_axes([])
    axcbar.set_axis_off()
    cbar = fig.colorbar(cax, ax=axcbar,ticks=np.linspace(cmin,cmax,7)) #rientation='horizontal',
    cbar.set_ticklabels([round(i,2) for i in np.exp(cbar.get_ticks())])
    axmatrix.set_yticks([])
    axauc = fig.add_axes([.08,imgbottom,.02,imgheight])
    axauc.imshow(pd.DataFrame(auc.loc[b4l.index],dtype=float),cmap=cmx.get_cmap('bone'),vmin=.5,vmax=.8,interpolation='nearest',aspect='auto')
    axauc.set_xticks([0])
    axauc.set_yticks([])
    axauc.set_xticklabels(['auc'])
    jet = plt.get_cmap('gist_ncar')
    cNorm = colors.Normalize(vmin=0, vmax=len(sorted_ctl))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    print('yl',axauc.get_ylim())
    for ix,i in enumerate(sorted_ctl):
        c = scalarMap.to_rgba(ix)
        axauc.text(-1,5*(ix + 1)-3,dovoc.loc[i,'name'],ha='right',color=c,fontweight='bold')
        axauc.plot([.4,.4],[ix*5 - .5, ix*5 + 4.5],color=c,linewidth=5)
