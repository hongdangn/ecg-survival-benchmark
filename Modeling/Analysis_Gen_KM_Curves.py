# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:44:10 2024

"""

import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py 

# C:\Users\ch242985\Desktop\Local ECG work\Trained_Models\MIMICIV\RibeiroReg_Best1YrMIMIC\EVAL

# something about fonts?
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 8

# %% Set up subplots

f = plt.figure(figsize=(3.35, 2.5))
grid = (6,9)


# ax1,ax2,ax3,ax4 = plt.subplot(2,2)
ax1 = plt.subplot2grid(grid, (0,0), rowspan=3,colspan=4)
ax3 = plt.subplot2grid(grid, (0,5), rowspan=3,colspan=4)
ax2 = plt.subplot2grid(grid, (4,0), rowspan=2,colspan=4)
ax4 = plt.subplot2grid(grid, (4,5), rowspan=2,colspan=4)

# ax = plt.subplot()

# ... 
# 1) break up into subplot regions (google))
# 2) remove unnecessary labels (google))
# 3) add new tick marks with N on separate line
# 4) size everything correctly with correct font size
# 5) axis tight


# %% Model 1
Model_One_Path = os.path.join(os.getcwd(), 'Trained_Models', 'MIMICIV', 'Ribeiro_042525_Cov_MIMICIV_8602958', 'EVAL', 'MIMICIV Test_Folder')
KM_Data_path = os.path.join(Model_One_Path, 'KM_Outputs.hdf5')
Hist_Path = os.path.join(Model_One_Path, 'Histogram.csv')

with h5py.File(KM_Data_path,'r') as f:
    mdl_median  = f['SF_mdl_median'][()]
    mdl_int_low = f['SF_int_low'][()]
    mdl_int_high= f['SF_mdl_int_high'][()]
    km_median   = f['SF_km_median'][()]
    km_int_low  = f['SF_km_int_low'][()]
    km_int_high = f['SF_km_int_high'][()]
    
hist_Data = np.genfromtxt(Hist_Path, dtype= float, skip_header=1, delimiter=',')

cuts = hist_Data[:,0] # end of the time bins
hist_coutns = hist_Data[:,1]
num_risk = hist_Data[:,2]


# fig1, ax = plt.savefig(args, kwargs)ubplots()
ax1.plot(cuts, km_median)
ax1.fill_between(cuts, km_int_low, km_int_high, color='b', alpha=.1)
ax1.plot(cuts,mdl_median, color='r')
ax1.fill_between(cuts, mdl_int_low, mdl_int_high, color='r', alpha=.1)
# ax1.legend(('KM Median','KM 5th-95th%','Model Median','Model 5th-95th%'))
# plt.xlabel('Years')
# plt.ylabel('Survival')
ax1.set(ylabel = 'Survival' ,title = 'MIMIC-IV')
ax1.set_xticks([])
# plt.text(-1, 400, 'Enrolled',color='r')

start = -1
end = 12
interval = (end - start) / 4

# plt.text(start, 500, str(int(num_risk[0])),color='r')
# plt.text(start + 1*interval, 500, str(int(num_risk[24])),color='r')
# plt.text(start + 2*interval, 500, str(int(num_risk[49])),color='r')
# plt.text(start + 3*interval, 500, str(int(num_risk[74])),color='r')
# plt.text(start + 4*interval, 500, str(int(num_risk[99])),color='r')

# quant, bin_loc = np.histogram(cuts[disc_y_t],bins=surv.shape[1])
ax2.bar(cuts,hist_coutns,width= (max(cuts)-min(cuts))/len(cuts))
# ax2.set(xlabel = 'Time to event or censor (years)' , ylabel = 'Count' )
ax2.set(ylabel = 'Count' )
ax2.set_yticklabels(['0','20k'])

# %% Model 2
Model_One_Path = os.path.join(os.getcwd(), 'Trained_Models', 'Code15', 'Ribeiro_042525_Cov_Code15_2211427', 'EVAL', 'Code15 Test_Folder')
KM_Data_path = os.path.join(Model_One_Path, 'KM_Outputs.hdf5')
Hist_Path = os.path.join(Model_One_Path, 'Histogram.csv')

with h5py.File(KM_Data_path,'r') as f:
    mdl_median  = f['SF_mdl_median'][()]
    mdl_int_low = f['SF_int_low'][()]
    mdl_int_high= f['SF_mdl_int_high'][()]
    km_median   = f['SF_km_median'][()]
    km_int_low  = f['SF_km_int_low'][()]
    km_int_high = f['SF_km_int_high'][()]
    
hist_Data = np.genfromtxt(Hist_Path, dtype= float, skip_header=1, delimiter=',')

cuts = hist_Data[:,0] # end of the time bins
hist_coutns = hist_Data[:,1]
num_risk = hist_Data[:,2]

# fig1, ax = plt.savefig(args, kwargs)ubplots()
ax3.plot(cuts, km_median)
ax3.fill_between(cuts, km_int_low, km_int_high, color='b', alpha=.1)
ax3.plot(cuts,mdl_median, color='r')
ax3.fill_between(cuts, mdl_int_low, mdl_int_high, color='r', alpha=.1)
ax3.legend(('KM Median','KM 5th-95th%','Model Median','Model 5th-95th%'), bbox_to_anchor=(1.1,0.05),ncol=2)
ax3.set_yticks([])
ax3.set_xticks([])
# plt.xlabel('Years')
# plt.ylabel('Survival')
ax3.set(title = 'Code-15')


start = -1
end = 12
interval = (end - start) / 4

# plt.text(-1, 400, 'Enrolled',color='r')
# plt.text(start, 500, str(int(num_risk[0])),color='r')
# plt.text(start + 1*interval, 500, str(int(num_risk[24])),color='r')
# plt.text(start + 2*interval, 500, str(int(num_risk[49])),color='r')
# plt.text(start + 3*interval, 500, str(int(num_risk[74])),color='r')
# plt.text(start + 4*interval, 500, str(int(num_risk[99])),color='r')

# quant, bin_loc = np.histogram(cuts[disc_y_t],bins=surv.shape[1])
ax4.bar(cuts,hist_coutns,width= (max(cuts)-min(cuts))/len(cuts))
ax4.set(xlabel = 'Time to event or censor (years)')
ax4.xaxis.set_label_coords(-0.1,-0.35)

# %% more changes



plt.tight_layout()

# %% Save out

os.makedirs(os.path.join(os.getcwd(),'Out_Figures'),exist_ok=True)
plot_file_path = os.path.join(os.getcwd(), 'Out_Figures', 'KM paper figure.pdf')
plt.savefig(plot_file_path,bbox_inches='tight')