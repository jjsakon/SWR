import numpy as np
from copy import copy
import itertools

def get_recall_clustering(positions, recalls):
    from scipy.stats import percentileofscore
    from scipy.spatial.distance import euclidean
    #Get temporal/semantic clustering scores. 

    #Positions: array of semantic/temporal values
    #Recalls: array of indices for true recall sequence (zero indexed), e.g. [0, 2, 3, 5, 9, 6]

    positions = copy(np.array(positions).astype(float))
    all_pcts = []
    all_possible_trans = list(itertools.combinations(range(len(positions)), 2))
    for ridx in np.arange(len(recalls)-1):  #Loops through each recall event, except last one
        possible_trans = [comb for comb in all_possible_trans if (recalls[ridx] in comb)]
        dists = []
        for c in possible_trans:
            try:
                dists.append(euclidean(positions[c[0], :], positions[c[1], :]))
            except:
                #If we did this transition, then it's a NaN, so append a NaN
                dists.append(np.nan)
        dists = np.array(dists)
        dists = dists[np.isfinite(dists)]

        true_trans = euclidean(positions[recalls[ridx], :], positions[recalls[ridx+1], :])
        pctrank = 1.-percentileofscore(dists, true_trans, kind='strict')/100.
        all_pcts.append(pctrank)

        positions[recalls[ridx]] = np.nan

    return all_pcts

def remove_repeats(recalls):
    #Takes array of serial positions and remove second instance of a repeated word
    items_to_keep = np.ones(len(recalls)).astype(bool)
    items_seen = []
    idx_removed = []
    for idx in range(len(recalls)):
        if recalls[idx] in items_seen:
            items_to_keep[idx] = False
            idx_removed.append(idx)
        items_seen.append(recalls[idx])

    final_vec = np.array(recalls)[items_to_keep]
    return final_vec, idx_removed

def find_temporal_runs(serial_pos):
    if (0 in serial_pos[:2]) or (1 in serial_pos[:2]):  #see if the first or second word are recalled first or second
        #See if there are 3 or more successive recalls 
        diff = np.abs(np.diff(np.concatenate(([0], np.diff(serial_pos)==1, [0]))))   #this identifies the indices where runs of 1 in the differential change
        ranges = np.where(diff==1)[0]
        if len(ranges)>0:   #make sure there was any consecutive run at all
            pass
        else:
            return np.ones(len(serial_pos)).astype(bool)
        mask = np.ones(len(serial_pos)).astype(bool)  #mask to remove recalls from serial_pos
        mask[ranges[0]:ranges[1]+1] = False  #want the range, inclusive of both ends!
        if (np.sum(mask==False)>=2) & ((ranges[0]==0) | (ranges[0]==1)):   #we'll only do this if there are 3 or more successive recalls that begin with the first or second word
            return mask
        else:
            return np.ones(len(serial_pos)).astype(bool)
    else:
        return np.ones(len(serial_pos)).astype(bool)