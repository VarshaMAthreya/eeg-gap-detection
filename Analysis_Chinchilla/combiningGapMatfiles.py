# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:20:05 2023

@author: vmysorea
"""
# Merge the matfiles for the three gap durations to one for each subject

from scipy import io
from scipy.io import savemat

data_loc = 'D:/PhD/Data/Chin_Data/AnalyzedGDT_matfiles/'

subjlist =  [ 'Q363', 'Q364', 'Q368',
            'Q402',  'Q406', 'Q407', 
             'Q424', 'Q426'] 

for subj in range(len(subjlist)):
    sub = subjlist [subj]
    dat1 = io.loadmat(data_loc + sub + '_16ms_2-20Hz.mat', squeeze_me=True)
    dat2 = io.loadmat(data_loc + sub + '_32ms_2-20Hz.mat', squeeze_me=True)
    dat3 = io.loadmat(data_loc + sub + '_64ms_2-20Hz.mat', squeeze_me=True)
    dat4 = io.loadmat(data_loc + sub + '_LightSedation_NoGap_2-20Hz.mat', squeeze_me=True)
    
    dat1.keys()
    dat2.keys()
    dat3.keys()
    dat4.keys()
    
    mastoid16 = dat1['ep_mastoid']
    vertex16 = dat1['ep_vertex']
    ground16= dat1['ep_ground']
    cap16 = dat1['ep_all']
    gap_mastoid16 = dat1['gap_mastoid']
    gap_vertex16 = dat1['gap_vertex']
    gap_ground16= dat1['gap_ground']
    gap_cap16 = dat1['gap_cap']
    picks= dat1['picks']
    t=dat1['t']
    t_full=dat1['t_full']
    
    mastoid32 = dat2['ep_mastoid']
    vertex32 = dat2['ep_vertex']
    ground32= dat2['ep_ground']
    cap32 = dat2['ep_all']
    gap_mastoid32 = dat2['gap_mastoid']
    gap_vertex32 = dat2['gap_vertex']
    gap_ground32= dat2['gap_ground']
    gap_cap32 = dat2['gap_cap']
    
    mastoid64 = dat3['ep_mastoid']
    vertex64 = dat3['ep_vertex']
    ground64= dat3['ep_ground']
    cap64 = dat3['ep_all']
    gap_mastoid64 = dat3['gap_mastoid']
    gap_vertex64 = dat3['gap_vertex']
    gap_ground64= dat3['gap_ground']
    gap_cap64 = dat3['gap_cap']
    
    mastoid = dat4['ep_mastoid']
    vertex = dat4['ep_vertex']
    ground= dat4['ep_ground']
    cap = dat4['ep_all']
    gap_mastoid = dat4['gap_mastoid']
    gap_vertex = dat4['gap_vertex']
    gap_ground= dat4['gap_ground']
    gap_cap = dat4['gap_cap']

    mat_ids = dict(mastoid16 = mastoid16,  vertex16 = vertex16, ground16= ground16, cap16 = cap16, 
               gap_mastoid16 = gap_mastoid16, gap_vertex16 = gap_vertex16, gap_ground16= gap_ground16, gap_cap16 = gap_cap16, 
               mastoid32 = mastoid32,  vertex32 = vertex32, ground32= ground32, cap32 = cap32, 
               gap_mastoid32 = gap_mastoid32, gap_vertex32 = gap_vertex32, gap_ground32= gap_ground32, gap_cap32 = gap_cap32, 
               mastoid64 = mastoid64,  vertex64 = vertex64, ground64= ground64, cap64 = cap64, 
               gap_mastoid64 = gap_mastoid64, gap_vertex64 = gap_vertex64, gap_ground64= gap_ground64, gap_cap64 = gap_cap64, 
               mastoid = mastoid,  vertex = vertex, ground= ground, cap = cap, 
               gap_mastoid = gap_mastoid, gap_vertex = gap_vertex, gap_ground= gap_ground, gap_cap = gap_cap,
               t_full = t_full, t = t, picks = picks)
               
    savemat(data_loc + sub + '_WithNoGap_2-20Hz.mat', mat_ids)