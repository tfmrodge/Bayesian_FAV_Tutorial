# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:40:53 2022

@author: trodge01
"""

import uncertainties
import pandas as pd
import numpy as np
import pymc3 as pm
import pdb
import os
from IPython.core.interactiveshell import InteractiveShell
#This contains the FAV model set-up and display functions. For more details or for help with variable names/inputs
#etc, look at the function definitions and in-line comments. The "FAV_model_funcs.py" file must be in the same
#folder as this notebook
import FAV_model_funcs as fmf
#Load Data
pdb.set_trace()
LDV = pd.read_excel('LDV_Data.xlsx', index_col = 2)
uLDV,FAVs = fmf.loaddata(LDV)

#IDV = pd.read_excel('MDV_Data.xlsx', index_col = 2)
#uIDV,IAVs = fmf.loaddata(IDV)

CDV = pd.read_excel('LDVMDV_Data.xlsx', index_col = 2)
uCDV,FAVRs = fmf.loaddata(CDV)
#Define the compounds to calculate. We only need to do the new ones here.
#numUs_absent = 3
#comps = uCDV.index
#If you want to run a specific compound/list of compounds, comment out the above line and run the following:
#comps = ['TnBP', 'TPPO', 'TCiPP', 'TDCiPP','TmCP', 'ToCP', 'TpCP', 'TBOEP','TEHP', 'TTBPP', 'T2iPPP', 'EHDPP', 'TDBPP']
'''
comps = ['TnBP']
uCDV.loc[comps,'LogKOA_absent'] = True
#uCDV.loc[comps,'LogKOA'] = np.NaN
#mu = 7.25817, sd = 

for comp in comps:
    #pdb.set_trace()
    #For some compounds we can't make an FAV. This will raise a "ValueError", and we will skip that compound.
    try:
        FAVRs, enth_trace = fmf.run_model(comp,FAVRs,model_type ='KS',uDV = uCDV,savepath = 'Traces/FAVRs/KS/',
                                          trace = 1000, tune=1000,target_accept=0.8)
                                          #trace = 1000, tune=1000,target_accept=0.8,LogKOA='Lognormal') ,LogKOA='SkewNormal'
    except ValueError:
        pass
    

directory = 'Traces/FAVRs/KS/LoopDir/'
tracesumms = {}
for filename in os.listdir(directory):
    #Define the compound from the filename
    pdb.set_trace()
    comp = filename[0:filename.find("_")]
    if comp == "benzokfluoranthene": #Error from the square brackets in loading files
        comp = "benzo[k]fluoranthene"
    elif comp == "benzoapyrene":
        comp = "benzo[a]pyrene"
    try: 
        fig, tracesumm = fmf.plot_trace(comp,FAVRs,filename=directory+filename,uDV=uCDV,model_type = 'KS')
        #Put it in the FAV dataframe at the same time, just in case the kernel was reloaded at some point.
        FAVRs = fmf.trace_to_FAVs(comp,FAVRs,filename = directory+filename,uDV=uCDV,model_type='KS')
        #Print the compound name and the traceplot summary tables
        #comp
        tracesumms[comp] = tracesumm
    except KeyError:
        #Subfolders in the directory will return a keyerror.
        pass
'''
#Check the traceplots
directory = 'Traces/FAVRs/dUs/ManyMums/'
InteractiveShell.ast_node_interactivity = 'last_expr'   #"all"
#Loop through all the files in the directory and display plots. You may want to change the directory to a
#sub-folder if you are running lots of compounds. This may take ~minutes for lots of plots
tracesumms = {}
tracesumm = None
#for filename in os.listdir(directory):
uCDV.loc[:,'dUO_absent'] = True
for filename in ['TBOEP_dUs_1215_2259']:
    print(filename)
    #Define the compound from the filename
    pdb.set_trace()
    comp = filename[0:filename.find("_")]
    if comp == "benzokfluoranthene": #Error from the square brackets in loading files
        comp = "benzo[k]fluoranthene"
    elif comp == "benzoapyrene":
        comp = "benzo[a]pyrene"
    try: 
        fig, tracesumm = fmf.plot_trace(comp,FAVRs,filename=directory+filename,uDV=uCDV,model_type = 'dU',fig=False)
        #Put it in the FAV dataframe at the same time, just in case the kernel was reloaded at some point.
        FAVRs = fmf.trace_to_FAVs(comp,FAVRs,filename = directory+filename,uDV=uCDV,model_type='dU')
        #Print the compound name and the traceplot summary tables
        tracesumms[comp] = tracesumm
    except KeyError:
        #Subfolders in the directory will return a keyerror.
        pass
