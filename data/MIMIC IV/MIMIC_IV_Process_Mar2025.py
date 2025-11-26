# -*- coding: utf-8 -*-
"""
Modified 03/2025: now saves out 1 .hdf5 file with the ECGs, and a .csv to use as labels
"""

import os
import numpy as np
import scipy.signal as sig
import numpy as np
import pandas as pd

import h5py
import torch # to set manual seed
from torch.utils.data import random_split
import wfdb # wfdb?

from datetime import datetime


# %% move to correct folder
fold_path = os.path.join(os.path.dirname(os.getcwd()), 'RAW_DATA', 'Cardio_Clinic_ECGs')
os.chdir(fold_path)


# %%
# Adjust ECGs to match Code-15 standard
def adjust_sig(signal):
    # MIMIC IV signals are 10s at 500Hz, 5k total samples
    # we seek 10s @ 400Hz, centered, padded to 4096.
    # so resample to 4k samples
    # then pad with 48 0's on both sides of axis=0
    new_sig = sig.resample(signal, 4000, axis=0)
    new_sig = np.pad(new_sig, ((48,48),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig

# file_path = os.path.join(os.getcwd(), 'MIMIC IV', 'machine_measurements_survival.csv')
file_path = '//lab-share//CHIP-Lacava-e2//Public//physionet.org//files//mimic-iv-ecg//1.0//machine_measurements_survival_01152025.csv'

record_list = pd.read_csv(file_path, low_memory=False)

# ecg_dir = os.path.join(os.getcwd(), 'MIMIC IV', 'ECG') # where ECG lives
ecg_dir = '//lab-share//CHIP-Lacava-e2//Public//physionet.org//files//mimic-iv-ecg//1.0//files'

import time

# Parse all files, assemble them into dictionaries
Dat_Dict = {}
PID_List = []
# a = time.time()

for entry_ind in range(len(record_list['subject_id'])):
    
    # get everything from the labels.csv you care about
    
        # Basics for survival analysis
    PID   = record_list['subject_id'][entry_ind]
    SID   = record_list['study_id'][entry_ind]
    TTE   = record_list['time_to_event'][entry_ind]
    Event = record_list['event'][entry_ind]
    
        # If admitted (for multimodal)
    #When_Admitted = record_list['admittime_before_ecg'][entry_ind]
    
    # Demographics
    Gender = record_list['gender'][entry_ind]
    
    # Age updated 01/24/25; 'Age' instead of 'Anchor Age'
    Age    = record_list['Age'][entry_ind]
    
    # ECG order
    ECG_Order = record_list['ECG_Order'][entry_ind]
    
        # Machine Measurements raw
    RR          = record_list['rr_interval'][entry_ind]        
    P_Start     = record_list['p_onset'][entry_ind]        
    P_End       = record_list['p_end'][entry_ind]
    QRS_Start   = record_list['qrs_onset'][entry_ind]
    QRS_End     = record_list['qrs_end'][entry_ind]
    T_End       = record_list['t_end'][entry_ind]
    P_Axis      = record_list['p_axis'][entry_ind]
    QRS_Axis    = record_list['qrs_axis'][entry_ind]
    T_Axis      = record_list['t_axis'][entry_ind]
        
        # From those, pick what we want to use and mark if it is valid
    P_Axis     = P_Axis
    QRS_Axis = QRS_Axis
    T_Axis     = T_Axis
    RR             = RR
    P_Dur       = P_End - P_Start
    PQ_Dur     = QRS_Start - P_End 
    QRS_Dur   = QRS_End - QRS_Start 
    # ST is unavailable
    QT_Dur     = T_End - QRS_Start 

    # grab the ECG
    P4 = 'p' + str(PID)[:4] # used in file name
    PID = 'p' + str(PID)
    SID2 = str(SID)
    SID = 's' + str(SID)
    rec_path = f'{ecg_dir}/{P4}/{PID}/{SID}/{SID2}' 
    rd_record = wfdb.rdrecord(rec_path)
    
    signal = rd_record.p_signal # 5k x 12 signal
    
    # filter out nans
    if (np.isnan(signal).any()):
        continue
    
    # ready to store
    if PID not in Dat_Dict.keys():
        Dat_Dict[PID] = {}
        PID_List.append(PID)
        
    if SID not in Dat_Dict[PID].keys():
        Dat_Dict[PID][SID] = {}
        Dat_Dict[PID][SID]['Signal'] = adjust_sig(signal) # store adjusted signal
        
        # time to event is in days, convert to years
        TTE = float(TTE) /365 # adjust TTE. sometimes get '0' for same-day, so place a lower value of 0.5 /365.
        if (TTE < 1e-4):
            TTE = 0.5 / 365
            
        Dat_Dict[PID][SID]['PID']             = int(PID[1:])            # int
        Dat_Dict[PID][SID]['SID']             = int(SID[1:])            # int
        Dat_Dict[PID][SID]['TTE_Mortality']             = TTE            # float
        Dat_Dict[PID][SID]['E_Mortality']           = Event          # bool
        Dat_Dict[PID][SID]['Is_Male']          = (Gender=='M')  # bool
        Dat_Dict[PID][SID]['Age']             = Age            # int
        Dat_Dict[PID][SID]['P_Axis']          = P_Axis         # int
        Dat_Dict[PID][SID]['QRS_Axis']        = QRS_Axis       # int
        Dat_Dict[PID][SID]['T_Axis']          = T_Axis         # int
        Dat_Dict[PID][SID]['RR']              = RR             # int
        Dat_Dict[PID][SID]['P_Dur']           = P_Dur          # int
        Dat_Dict[PID][SID]['PQ_Dur']          = PQ_Dur         # int
        Dat_Dict[PID][SID]['QRS_Dur']         = QRS_Dur        # int
        Dat_Dict[PID][SID]['QT_Dur']          = QT_Dur         # int
        Dat_Dict[PID][SID]['ECG_Order']       = ECG_Order # int

    # rd_record = wfdb.rdrecord(rec_path) 
    # wfdb.plot_wfdb(record=rd_record, figsize=(124,18), title='Study 41420867 example', ecg_grids='all')

print('debug. # entires: ' + str(len(PID_List)))

# %% go from dictionary with info per PID/SID to lists for saveout

# first, split the PID list randomly. 20% test, 80% train.
Te_Count = int(len(PID_List) * 0.2)
Tr_Count = len(PID_List) - Te_Count

torch.manual_seed(12345)
PID_Index_Train, PID_Index_Test = random_split(range(len(PID_List)), [Tr_Count, Te_Count])   

# per https://stackoverflow.com/questions/58089499/how-to-combine-many-numpy-arrays-efficiently
# add everything to list, then concatenate at end
# now we assemble. lists, then numpy, then hdf5.

# what order we want the labels to be in
ordered_key_list =  ['PID', 'Age', 'Is_Male', 'TTE_Mortality', 'E_Mortality'] # what order we want to save out data as
ordered_key_list += ['SID']
ordered_key_list += ['P_Axis', 'QRS_Axis']
ordered_key_list += ['T_Axis', 'RR', 'P_Dur']
ordered_key_list += ['PQ_Dur', 'QRS_Dur', 'QT_Dur']
ordered_key_list += ['ECG_Order']


# Now build ECG arrays and label lists exactly as before (to match previous order), but convert the labels to pandas lists
ALL_ECG = []
Train_Y = []
for i in PID_Index_Train:
    PID = PID_List[i]
    for SID in Dat_Dict[PID].keys():
        ALL_ECG.append(Dat_Dict[PID][SID]['Signal'])
        
        Y_as_List = []
        for key in ordered_key_list:
            Y_as_List.append( Dat_Dict[PID][SID][key] )
        Train_Y.append( Y_as_List )    

# Train_X = np.stack(Train_X, axis=0).astype(np.float32)
# Train_Y = np.stack(Train_Y, axis=0).astype(np.float64) # AUG CHANGE
Train_Y_pd = pd.Dataframe(data = Train_Y, columns = ordered_key_list)
Train_Y_pd['train'] = [True for k in range(Train_Y_pd.shape[0])]
Train_Y_pd['test'] = [False for k in range(Train_Y_pd.shape[0])]

# Test_X = []
Test_Y = []
for i in PID_Index_Test:
    PID = PID_List[i]
    for SID in Dat_Dict[PID].keys():
        ALL_ECG.append(Dat_Dict[PID][SID]['Signal'])
        
        Y_as_List = []
        for key in ordered_key_list:
            Y_as_List.append( Dat_Dict[PID][SID][key] )
        Test_Y.append( Y_as_List ) 
        
# Test_X = np.stack(Test_X, axis=0).astype(np.float32)
Test_Y_pd = pd.Dataframe(data = Test_Y, columns = ordered_key_list)
Test_Y_pd['train'] = [False for k in range(Test_Y_pd.shape[0])]
Test_Y_pd['train'] = [True for k in range(Test_Y_pd.shape[0])]
# Test_Y = np.stack(Test_Y, axis=0).astype(np.float64) # AUG CHANGE


# convert to arrays
Labels_pd = pd.concat(Train_Y_pd, Test_Y_pd)
ECGs = np.stack(ALL_ECG, axis=0).astype(np.float32)

# %% Save out
pd_path = os.path.join(os.getcwd(), 'Labels_MIMICIV_mort_032025_pd_8020.csv')
Labels_pd.to_csv (pd_path)

with h5py.File(os.path.join(os.getcwd(),'MIMIC_ECG.hdf5'), "w") as f:
    f.create_dataset('ECG',     data = ECGs, chunks = (1,4096,12),  compression="gzip")
    f.create_dataset('SID',     data = Labels_pd['SID'])
    f.create_dataset('PID',     data = Labels_pd['PID'])




