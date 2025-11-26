# -*- coding: utf-8 -*-
"""
This file reads in raw Code-15 data from ./RAW_DATA/Cdoe15/TRAIN
And saves out processed data (Numpy Arrays) for downstream work to ./RAW_DATA/Code15_Processed/TRAIN_DATA and ./RAW_DATA/Code15_Processed/TEST_DATA


It also works for data subsets (if you only include several "exams_part{N}.hdf5" of code15)
[Likely has pieces borrowed from stackoverflow or online tutorials]
"""
import numpy as np
import h5py
import os
import csv
import time
from torch import manual_seed
from torch.utils.data import random_split
import pandas as pd

manual_seed(12345)

# move to correct folder
fold_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'code15')
os.chdir(fold_path)

Code15_ECG_Directory = os.path.join(os.getcwd(),'ecg')
Code15_exams_loc = os.path.join(os.getcwd(),'exams.csv')  
pd_path = os.path.join(os.getcwd(), 'Labels_Code15_mort_032025_pd_8020.csv') 

# %% Adjust labels
labels = pd.read_csv(Code15_exams_loc)
labels = labels.loc[labels["trace_file"] == "exams_part0.hdf5"].copy().reset_index()

# rename columns
labels.rename(columns={'exam_id': 'SID'}, inplace=True)
labels.rename(columns={'is_male': 'Is_Male'}, inplace=True)
labels.rename(columns={'age': 'Age'}, inplace=True)
labels.rename(columns={'death': 'Mort_Event'}, inplace=True)
labels.rename(columns={'timey': 'Mort_TTE'}, inplace=True)
labels.rename(columns={'patient_id': 'PID'}, inplace=True)

# adjust tte_mortality to min of 1/365
labels.loc[labels['Mort_TTE']<1/365,'Mort_TTE'] = 1/365
legit_PID = labels.loc[np.isnan(labels['Mort_TTE'])==False]['PID'].to_numpy() # this matches np.unique(PIDs)

Te_Count = int(len(legit_PID) * 0.2)
Tr_Count = len(legit_PID) - Te_Count
PID_Index_Train, PID_Index_Test = random_split(range(len(legit_PID)), [Tr_Count, Te_Count])   

PID_labels = {}
for i in PID_Index_Train:
    temp_PID = legit_PID[i]
    PID_labels[temp_PID] = 'tr'
    
for i in PID_Index_Test:
    temp_PID = legit_PID[i]
    PID_labels[temp_PID] = 'te'
    
Test_Train_split_12345 = []
for PID,TTE in zip(labels['PID'],labels['Mort_TTE']):
    if (np.isnan(TTE)):
        Test_Train_split_12345.append('NegTTE')
    else:
        Test_Train_split_12345.append(PID_labels[PID])
        
labels['Test_Train_split_12345'] = Test_Train_split_12345

EIDs = []
ECGs = []
for f in os.listdir(Code15_ECG_Directory): 
    print(f)
    with h5py.File(os.path.join(Code15_ECG_Directory,f), "r") as h:
        print("Keys: %s" % h.keys())
        EIDs.append(h['exam_id'][:-1])
        ECGs.append(h['tracings'][:-1,648:-648,:].astype(np.float32)) # keep the middle 7 seconds @ 400Hz = 2800 samples
        
EIDs = np.hstack([k for k in EIDs])
ECGs = np.vstack([k for k in ECGs])

ECG_SID_label_dict = {}
for i,SID in enumerate(EIDs):
    ECG_SID_label_dict[SID] = i
    
reorder_arr = []
for SID in labels["SID"]:
    if (SID in ECG_SID_label_dict.keys()): 
        reorder_arr.append(ECG_SID_label_dict[SID])
    
EIDs = EIDs[reorder_arr]
ECGs = ECGs[reorder_arr]

with h5py.File('Code15_ECG.hdf5', "w") as f:
    f.create_dataset('ECG',       data = ECGs, chunks = (1,2800,12),  compression="gzip")
    f.create_dataset('SID',       data = EIDs)
    f.create_dataset('PID',       data = labels['PID'])
    
labels.to_csv (pd_path)
