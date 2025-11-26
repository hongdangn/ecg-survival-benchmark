# 1. Pull data, split, continue as usual until train


# %% Imports; Support functions before main functions...

# handle pycox folder requirement FIRST
import os 
os.environ['PYCOX_DATA_DIR'] = os.path.join(os.getcwd(),'Mandatory_PyCox_Dir') # next line requires this. it's a bad init call.



# from Model_Runner_Support import get_covariates

from Model_Runner_Support import Load_Labels
# from Model_Runner_Support import Clean_Data
from Model_Runner_Support import Apply_Horizon
from Model_Runner_Support import Split_Data
# from Model_Runner_Support import DebugSubset_Data
from Model_Runner_Support import set_up_train_folders
from Model_Runner_Support import set_up_test_folders

# from Model_Runner_Support import provide_data_details

# evaluation wrappers
from Model_Runner_Support import Gen_KM_Bootstraps
from Model_Runner_Support import Gen_Concordance_Brier_No_Bootstrap
# from Model_Runner_Support import Gen_Concordance_Brier_PID_Bootstrap
from Model_Runner_Support import Gen_AUROC_AUPRC
from Model_Runner_Support import print_classifier_ROC
from Model_Runner_Support import save_histogram


from MODELS.Support_Functions import Save_to_hdf5


import pandas as pd
import matplotlib.pyplot as plt

from sksurv.linear_model import CoxPHSurvivalAnalysis
from scipy.special import softmax 

import numpy as np
import torch
import time
import json

import collections
collections.Callable = collections.abc.Callable

import argparse 

# to see how the classifier did at its job
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# %% Compatability - bring back older version of scipy simpson function
import scipy
from MODELS.Support_Functions import simps
scipy.integrate.simps = simps

# %% 
def main(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args()
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
def Run_Model_via_String_Arr(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args(args[0])
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
    # 
def Run_Model(args):
    # input is paired dict of strings named args
    start_time = time.time()
    
    # %% 1. CUDA check
    # CUDA
    # for i in range(torch.cuda.device_count()):
    #    print(torch.cuda.get_device_properties(i).name)
       
    # if (torch.cuda.is_available() == False):
    #     print('No CUDA. Exiting.')
    #     exit()
       
    # %% 2. Arg Processing
    # Grab model name. No point in proceeding without it.
    if ('Model_Name' not in args.keys()):
        print('Model_Name not specified - cant train or pull models')
        exit()

    Model_Type = args['Model_Name'].split('_')[0]
    args['Model_Type'] = Model_Type
    
    if ('Train_Folder' not in args.keys()):
        print('Train_Folder not specified - cant train or pull models')
        exit()
    
    # 3. Random Seeds should really be from args. Note: "load"ed models overwrite these!
    if ('Rand_Seed' in args.keys()):
        args['Rand_Seed'] = int(args['Rand_Seed'])
        
    if ('Rand_Seed' not in args.keys()):
        np.random.seed()
        args['Rand_Seed'] = np.random.randint(70000,80000)
        print('Rand Seed Not Set. Picking a random number 70,000 - 80,000... ' + str(args['Rand_Seed']))    
    
    
    np.random.seed(args['Rand_Seed'])
    torch.manual_seed(args['Rand_Seed'])
    torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
    
    # Y covariate indices get passed here
    # val_covariate_col_list, test_covariate_col_list = get_covariates(args)
    
    
    
    # %% Process data: Load, Clean, Split
    train_df, test_df = Load_Labels(args)       # Data is a dict, is passed by reference
    # Clean_Data(Data, args)       # remove TTE<0 and NaN ECG
    Apply_Horizon(train_df, test_df, args)    # classifiers need to compact TTE and E into a single value, E*. Augments Data['y_'] for model train/runs without overwriting loaded information.
    train_df, valid_df = Split_Data(train_df)             # splits 'train' data 80/20 into train/val by PID
    
    # Can skip pulling ECG and re-ordering dataframe
    
    # Data, train_df, valid_df, test_df = Load_ECG_and_Cov(train_df, valid_df, test_df, args)
    # DebugSubset_Data(Data, train_df, test_df, args) # If args['debug'] == True, limits Data[...] to 1k samples each of tr/val/test.
    
    # if ('provide_data_details' in args.keys()):
    #     if (args['provide_data_details'] == 'True'):
    #         provide_data_details(args, Data, Train_Col_Names, Test_Col_Names)
    
    if ('covariates' in args.keys()):
        cov_list = args['covariates'][1:-1].split(',')
        Data = {}
        Data['Cov_train'] = train_df[cov_list].to_numpy().astype(np.float32)
        Data['Cov_valid'] = valid_df[cov_list].to_numpy().astype(np.float32)
        Data['Cov_test']  =  test_df[cov_list].to_numpy().astype(np.float32)
    else:
        breakpoint()
    
            
    # %% 9. set up trained model folders if they  don't exist
    set_up_train_folders(args)

    # %% 10. Select model, (maybe) load an existing model. ask for training (runs eval after training reqs met)
    
    # from https://xgboost.readthedocs.io/en/stable/get_started.html
    asdf = XGBClassifier(n_estimators=100, learning_rate=0.1, objective='binary:logistic')
    
    # do we need to normalize data first? yes
    # from https://xgboosting.com/xgboost-min-max-scaling-numerical-input-features/
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    Data['Cov_train'] = scaler.fit_transform(Data['Cov_train'])
    Data['Cov_valid'] = scaler.transform(Data['Cov_valid'])
    Data['Cov_test'] = scaler.transform(Data['Cov_test'])
    
    asdf.fit(Data['Cov_train'], train_df['E*'])

    # %% 11. Generate and save out results   
    print('got to eval. Total Time elapsed: ' ,'{:.2f}'.format(time.time()-start_time))
    if ('Test_Folder' in args.keys()):
    
        # get model outputs for test, validation sets (unshuffled)
        if ('Eval_Dataloader' not in args.keys()): # This lets you evaluate the model on its validation set instead of test set
            args['Eval_Dataloader'] = 'Test'
            
        if args['Eval_Dataloader'] == 'Test':
            test_outputs =  asdf.predict_proba(Data['Cov_test'])[:,1]
            test_correct_outputs = test_df['E*'] 
            
        if args['Eval_Dataloader'] == 'Train':
            test_outputs =  asdf.predict_proba(Data['Cov_train'])[:,1]
            test_correct_outputs = train_df['E*'] 
            
        if args['Eval_Dataloader'] == 'Valid':
            test_outputs =  asdf.predict_proba(Data['Cov_valid'])[:,1]
            test_correct_outputs = valid_df['E*'] 
            
        # get validation outputs anyway for Cox model fit
        val_outputs =  asdf.predict_proba(Data['Cov_valid'])[:,1]
        val_correct_outputs = valid_df['E*'] 
        
        # adjust output formats
        val_outputs  = np.squeeze(val_outputs)
        test_outputs = np.squeeze(test_outputs)
        
        # softmax the outputs (no need in XGB)
        # predict_proba is already 0-1, so don't need to softmax
        # val_outputs = np.array([softmax(k)[1] for k in val_outputs])
        # test_outputs = np.array([softmax(k)[1] for k in test_outputs])
        
        # --- From here, everything should go as before
        
        # Set up Folders
        set_up_test_folders(args)
        
        # Save out smx val/test model outputs + the labels ( [PID, TTE*, E*] are last three cols, model was trained on TTE*,E*)
        # From this we can recreate Cox models later
        tmp = os.path.join(args['Model_Eval_Path'], 'Classif_Outputs_and_Labels.hdf5')
        Save_to_hdf5(tmp, val_outputs, 'val_outputs')
        Save_to_hdf5(tmp, test_outputs, 'test_outputs')
        Save_to_hdf5(tmp, valid_df['E*'], 'valid_E*')
        Save_to_hdf5(tmp, valid_df['TTE*'], 'valid_TTE*')
        Save_to_hdf5(tmp, test_df['E*'], 'test_E*')
        Save_to_hdf5(tmp, test_df['TTE*'], 'test_TTE*')
        # Save_to_hdf5(tmp, Data['y_test'], 'y_test')
        
        # %% 13. Run Cox models
        # fit a Cox model on the VALIDATION set, evaluate on TEST set
        # 1. convert risk prediction (0-1)  to survival curves per subject
        # 2. measure concordance, brier, AUPRC, AUROC, etc.
        # NoteL Cox models are built on un-horizoned labels, even if the classifiers are trained on horizoned labels
        # (so this task uses all the time data, but the classifiers can only handle the one target)

        # build CoxPH curves on validation data
        zxcv = CoxPHSurvivalAnalysis() 
        a = valid_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
        b = valid_df['Mort_TTE'].to_numpy()
              
        tmp = np.array([ (a[k],b[k]) for k in range(a.shape[0]) ], dtype = [('event',bool),('time',float)] )
        zxcv.fit(np.expand_dims(val_outputs,-1), tmp   )
        
        # prep evaluation data - be sure to use actual time/event, not horizon
        if (args['Eval_Dataloader'] == 'Train'):
            disc_y_e = train_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
            disc_y_t = train_df['Mort_TTE'].to_numpy()
        elif (args['Eval_Dataloader'] == 'Validation'):
            disc_y_e = valid_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
            disc_y_t = valid_df['Mort_TTE'].to_numpy()
        else:
            disc_y_e = test_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
            disc_y_t = test_df['Mort_TTE'].to_numpy()
    
        # %% 14. Prep everything to match PyCox analysis
        # sample survival functions at a set of times (to compare to direct survival moels)
        upper_time_lim = max( b ) # the fit is limited to validation end times, so do 100 bins of tha
        sample_time_points = np.linspace(0, upper_time_lim, 100).squeeze()
        
        surv_funcs = zxcv.predict_survival_function(np.expand_dims(test_outputs,-1))
        surv = np.squeeze(  np.array([k(sample_time_points) for k in surv_funcs]))
        
        disc_y_e = disc_y_e.astype(int).squeeze()
        disc_y_t = np.array([np.argmax(sample_time_points>=k) if k<=upper_time_lim else len(sample_time_points)-1 for k in disc_y_t]) # bin times. none should appear in bin 0
        
        surv_df = pd.DataFrame(np.transpose(surv)) 
        
        # %% 15. Save out everything we need to recreate evaluation off-cluster: 
        hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
        Save_to_hdf5(hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(hdf5_path, disc_y_e, 'disc_y_e') # what really happened
        Save_to_hdf5(hdf5_path, disc_y_t, 'disc_y_t') # when it really happened, discretized
        Save_to_hdf5(hdf5_path, test_df['E*'].to_numpy(), 'Test E*')
        Save_to_hdf5(hdf5_path, surv, 'surv')

        print('Model_Runner: Generated survival curves. Total time elapsed: ' + str(time.time()-start_time) )
        
        # %% 16. evlauations
        
        # Save out KM. Add bootstrapping (20x). Saves KM values out separately in case you want to recreate that.
        Gen_KM_Bootstraps(surv, disc_y_e, disc_y_t, sample_time_points, args)

        # Concordance and Brier Score 
        time_points = [1,2,5,10,999]
        # across all ECG
        Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e, time_points, sample_time_points, args)
        # bootstrap: 1 ECG per patient x 20
        # Gen_Concordance_Brier_PID_Bootstrap(Data, args, disc_y_t, disc_y_e, surv_df, sample_time_points, time_points)
        
        # AUROC and AUPRC
        time_points = [1,2,5,10] # 999 doesn't work for AUROC
        Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points, args)
        print_classifier_ROC(test_correct_outputs, test_outputs)
        
        # histogram
        save_histogram(sample_time_points, disc_y_t, surv, args)
        
        print('Model_Runner: Finished evaluation. Total time elapsed: ' + str(time.time()-start_time) )
        
        
    #%% Test?
if __name__ == '__main__':
   main()