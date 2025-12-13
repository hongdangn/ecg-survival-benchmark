"""
MODEL_RUNNER_PyCox 

Overall:
    Process arguments
    Load training data
    Initialize a model
    (optional) load model parameters
    Train a model (can be skipped with --epoch_end -1)
    Evaluate a model
    Save model performance
    
Essential Arguments (no defaults):
    --Model_Name [ModelType + '_' + Name]       ex: --Model_Name RibeiroClass_Bob
    --Train_Folder [FolderName]                 ex: --Train_Folder Code15
    --Test_Folder [FolderName]                  ex: --Test_Folder Code15
    --Eval_Dataloader ['Validation', 'Train', or 'Test']   ex: --Eval_Dataloader Test

Optional Arguments:
    --Load ['Best' or 'Last'] - loads a model before training   ex: --Load Best
    --batch_size [int]                                     ex: --batch_size 128
    --validate_every [int]                                 ex: --validate_every 10
    --epoch_end [int]                                      ex: --epoch_end 25
       note: epoch_end -1 (default) skips training
    
---
Model_Runner:
    - Model_Runner Interprets passed arguments
        - Either from Command Line (main)
        - Or from another script (Run_Model_via_String_Arr)
            ex: args = '--Train_Folder '+folder+' --Model_Name Resnet18Class_asfd1 --Train True --Test_Folder '+folder+' --batch_size 128 --epoch_end 10 --validate_every 10'

    - Quick note on Models:
        - Model Type is set by the first half (up to an underscore) of their name: --Model_Name Resnet18Class_[asfd1]
        - Trained models live in Trained_Models/[their training data folder], as passed in: --Train_Folder [folder]
        - Model_Runner MUST get the Train_Folder and Model_Name arguments to know which model to build or use


main() parses command line inputs, then feeds a dict to run_model()
Run_Model_via_String_Arr() parses arguments from an array of strings, then feeds a dict to run_model()

DATA is assumed to be in N-H or N-H-W or N-C-H-W format. 
For time series, this would be N-Length or N-Length-Chan or N-[Color]-Length-Chan
Currently, all models reshape it into N-C-H-W

"""

import os

from src.zoo.utils import Load_Labels
from src.zoo.utils import Load_ECG_and_Cov
from src.zoo.utils import Split_Data
from src.zoo.utils import set_up_train_folders
from src.zoo.utils import set_up_test_folders

# evaluation wrappers
from src.zoo.utils import Gen_KM_Bootstraps
from src.zoo.utils import Gen_Concordance_Brier_No_Bootstrap
from src.zoo.utils import Gen_AUROC_AUPRC
from src.zoo.utils import save_histogram

from src.zoo.utils import Save_to_hdf5
from src.zoo.GenericModelDeepSurvival import GenericModelDeepSurvival
from src.zoo.utils import simps

import numpy as np
import torch
import matplotlib.pyplot as plt

import time

import collections
collections.Callable = collections.abc.Callable

import argparse
import scipy
scipy.integrate.simps = simps

    
def train(args):
    start_time = time.time()
    
    for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)
       
    if not torch.cuda.is_available():
        print('No CUDA. Exiting.')
        exit()
       
    # Grab model name. No point in proceeding without it.
    if ('Model_Name' not in args.keys()):
        print('Model_Name is not specified - cant train or pull models')
        exit()
    Model_Type = args['Model_Name'].split('_')[0]
    args['Model_Type'] = Model_Type

    if ('Train_Folder' not in args.keys()):
        print('Train_Folder not specified - cant train or pull models')
        exit()
    
    if ('Rand_Seed' in args.keys()):
        args['Rand_Seed'] = int(args['Rand_Seed'])
        
    if ('Rand_Seed' not in args.keys()):
        np.random.seed()
        args['Rand_Seed'] = np.random.randint(70000,80000)
        print('Rand_Seed Not Set. Picking a random number 70,000 - 80,000... ' + str(args['Rand_Seed']))    
    
    np.random.seed(args['Rand_Seed'])
    torch.manual_seed(args['Rand_Seed'])
    torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
    
    train_df, test_df = Load_Labels(args)       # Data is a dict, is passed by reference 
    train_df, valid_df = Split_Data(train_df)             # splits 'train' data 80/20 into train/val by PID
    
    print('Tr/Te contaminants:',np.intersect1d(train_df.PID.values,test_df.PID.values))
    print('Tr/Va contaminants:',np.intersect1d(train_df.PID.values,valid_df.PID.values))
    print('Va/Te contaminants:',np.intersect1d(test_df.PID.values,valid_df.PID.values))

    Data, train_df, valid_df, test_df = Load_ECG_and_Cov(train_df, valid_df, test_df, args)
    
    set_up_train_folders(args)
    
    print('Model_Runner:  Got to model init. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    asdf = GenericModelDeepSurvival(args, Data, train_df, valid_df, test_df)
      
    print('Model_Runner:  Got to Train. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    if( ('Load' in args.keys())):
        asdf.Load(args['Load'])
    asdf.train()
        
    
    if ('Test_Folder' in args.keys()):
        
        print('Model_Runner:  got to model eval. Total Time elapsed: ' ,'{:.2f}'.format(time.time()-start_time))
        
        if ('num_durations' not in args.keys()):
            args['num_durations'] = '100'
            print('By default, using 100 time intervals')
            num_durations = 100
        else:
            num_durations = int(args['num_durations'])
        
        if ('Eval_Dataloader' in args.keys()):
            cuts, disc_y_t, disc_y_e, surv, surv_df = asdf.Test(Which_Dataloader = args['Eval_Dataloader']) # This lets you evaluate the model on its validation set instead of test set
        else:
            cuts, disc_y_t, disc_y_e, surv, surv_df = asdf.Test() 
            
        sample_time_points = cuts # where are we sampling the survival functions?
        surv_df.index = np.arange(num_durations)
        
        set_up_test_folders(args)
             
        outputs_hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
        Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(outputs_hdf5_path, disc_y_t, 'disc_y_t') # when it really happened
        Save_to_hdf5(outputs_hdf5_path, disc_y_e, 'disc_y_e') # what really happened
        Save_to_hdf5(outputs_hdf5_path, test_df['Age'], 'Age')
        Save_to_hdf5(outputs_hdf5_path, test_df['Is_Male'], 'Is_Male')        
        Save_to_hdf5(outputs_hdf5_path, surv, 'surv')
        Save_to_hdf5(outputs_hdf5_path, test_df['SID'], 'SID')        
        
        
        Gen_KM_Bootstraps(surv, disc_y_e, disc_y_t, sample_time_points, args)

        time_points = [1,2,5,10,999]

        Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e, time_points, sample_time_points, args)

        time_points = [1,2,5,10] # 999 doesn't work for AUROC
        Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points, args)
        
        # histogram
        save_histogram(sample_time_points, disc_y_t, surv, args)
        
        print('Model_Runner: Finished evaluation. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
        
        if ('Multimodal_Out' in args.keys()):
            if (args['Multimodal_Out'] == 'True'):
                
                # Per Train, Validation, Test
                for eval_key in ['Train', 'Validation', 'Test']:
                    
                    if (eval_key == 'Train'):
                        y_key = 'y_train'
                    elif (eval_key == 'Validation'):
                        y_key = 'y_valid'
                    elif (eval_key == 'Test'):
                        y_key = 'y_test'
                        
                    # save the model's predictions at the 1 month mark
                    
                    multim_path = os.path.join(args['Model_Eval_Path'], 'Multimodal_Prediction_Out_'+eval_key+'.csv')
                    
                    # PID, TTE*, E* @ column inds [-3,-2,-1], respectively
                    surv_target_index = np.argmin( abs(sample_time_points - 30/365)) # mark the 1 month surv time point
                    
                    # regenerate surv
                    cuts, disc_y_t, disc_y_e, surv, surv_df = asdf.Test(Which_Dataloader = eval_key)
                    
                    # next line differs from _Survclass: Data['y_test'][:,-2], which is usually TTE*, gets discretized. 
                    temp = np.vstack( (Data[y_key][:,-3],1-surv[:,surv_target_index],disc_y_e,Data[y_key][:,int(args['y_col_test_time'])], Data[y_key][:,-1], Data[y_key][:,5], Data[y_key][:,1], Data[y_key][:,2]  ))
                    temp = np.transpose(temp)
                    headers = 'PID, surv_1mo_outputs, disc_y_e, TTE, E, SID, Age, Is_Male'
                    np.savetxt(multim_path, temp, header=headers, delimiter = ',')
                    
                
                    if ('Neg_One_Out' in args.keys()):
                        # also save out the features
                        neg_one_path = os.path.join(args['Model_Eval_Path'], 'Model_Features_Out_'+eval_key+'.hdf5')
                        neg_one_out = asdf.Get_Features_Out(Which_Dataloader = eval_key)
                        Save_to_hdf5(neg_one_path, headers, 'MM_headers')
                        Save_to_hdf5(neg_one_path, temp, 'MM_Out')
                        Save_to_hdf5(neg_one_path, neg_one_out, 'neg_one_out')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _, unknown_args = parser.parse_known_args()
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2]))

    args = argparse.Namespace(
        Train_Folder='Code15',
        Test_Folder='Code15',
        Model_Eval_Path='/train/code15',

        Model_Name='ECGTransForm',
        pycox_mdl='MTLR',
        batch_size=1,
        GPU_minibatch_limit=2,
        optimizer='Adam',
        
        epoch_end=0,
        validate_every=1,
        early_stop=100,
        
        Norm_Func='nchW',
        covariates='[Age,Is_Male]', 
        fusion_layers=3,
        fusion_dim=128,
        cov_layers=3,
        cov_dim=32,
        
        Scheduler=True,        
        Eval_Dataloader='Test',
        Rand_Seed=13,
        Multimodal_Out=False,  
        Neg_One_Out=False,     
    )
    args = vars(args)

    print(args)
    train(args)