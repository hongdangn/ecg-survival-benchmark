import numpy as np
import wfdb
import os
import tqdm
import h5py

FS = 100 
data_dir = "./HDF5_DATA/Code15/"
ecg_path = os.path.join(data_dir, "Code15_ECG.hdf5")
df_path = os.path.join(data_dir, 'Labels_Code15_mort_032025_pd_8020.csv') 
out_header_dir= os.path.join(data_dir, "headers/")

os.makedirs(out_header_dir, exist_ok=True)

with h5py.File(ecg_path, "r") as h:
    print("Keys: %s" % h.keys())
    ecgs = h['ECG'][()] 
    sids = h['SID'][()]

num_patients = len(sids)
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

for i in tqdm.tqdm(range(num_patients), total=num_patients):
    
    wfdb.wrsamp(
        record_name=f"sid_{sids[i]}",
        fs=FS,
        units=['mV'] * 12,
        sig_name=lead_names,
        p_signal=ecgs[i], 
        fmt=['16'] * 12,
        write_dir=out_header_dir
    )

print("Dump complete.")

