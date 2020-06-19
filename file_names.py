import glob
import os

def get_trt_names(hisdir, drugid):
    runname = hisdir + 'TargetON.' + str(drugid) +"/"
    if not os.path.exists(runname):
        os.mkdir(runname)
    return runname, runname + 'trt'


def combo_names(hisdir, drugid):
    runname = hisdir + 'Combo.' + str(drugid) +"/"
    if not os.path.exists(runname):
        os.mkdir(runname)
    return runname, runname + 'trt'

def get_pair_name(hisdir, drugid,ctl):
    runname, trtname = get_trt_names(hisdir,drugid)
    return runname + str(ctl) #"Target.{:d}/{:d}".format(drugid, ctl)    

#def get_savename_prefix(drugid,ctl, prefix):
#    return get_savename(drugid, ctl) + "." + prefix

def sparseh5_names(hisdir, drugid):
    spdir = hisdir + "/sparsemat/"
    return spdir + str(drugid) + ".h5"

def todo_files(hisdir, drugid):
    return glob.glob(hisdir + "split.*." + str(drugid))

#def trt_id_list(hisdir, drugid):
#    hisdir + "trtid" + str(drugid)    
def sparse_index_name(hisdir, trtid):
    tr, trtname = get_trt_names(hisdir, trtid)
    return trtname + "sparse_index.pkl"

def get_sparse_index(hisdir, trtid, cut = 100):
    elct = pickle.load(open(sparse_index_name(hisdir, trtid),'rb'))
    sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > cut,:].index)))
    return sparse_index

def pair_idfile(pair, idfile):
    return pair + "." + idfile

def get_ps_file(pair, idfile):
    return pair_idfile(pair,idfile) + ".ids.psmod.pkl"    

def idfile(hisdir, drugid):
    return hisdir + "drug_ids/" + str(drugid)

def commid(pair_prefix):
    return pair_prefix + ".comm_id", pair_prefix + ".trt_excl", pair_prefix + ".removed_history"

def selection_model(idfile):
    return idfile + ".selmod.pkl"

def id_name_path(pair_prefix, idfile_name):
    return pair_prefix +"."+ idfile_name + ".ids"
