#This file holds the functions that run the FAV model in the Bayesian FAV Tutorial. You can change parameterizations
#here.
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy
import pandas as pd
import numpy as np
import math
import pymc3 as pm
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
import os

def loaddata(DV):
    '''
    This function will load and process data from the DV dataframe.

    Attributes:
    ----------
    DV = Derived value data read from the Excel 
    uDV = Uncertain derived values. Should be a pandas dataframe of unumpy entries, formatted using the loaddata function above
    FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, formatted using the loaddata function above
    '''
    #Now, we are going to make each of our DVs into uncertain variables with the uncertaintites through this package.
    #We will make a separate dataframe for the uncertain values
    startcol = 3 #First column with DVs in it. Column order should have all data identifiers first, followed by DVs,
    #their standard deviations and then the number of literature values for each property
    #pdb.set_trace()
    uDV = pd.DataFrame(index = DV.index)
    #Name the columns
    colnames =['dVAPH','dUW','dUO','dUOA','dUAW','dUOW','LogKOA','LogKOW','LogKAW','LogPL','LogSW','LogSO','dfusS','Tm']
    i = 0
    #Loop through the columns to get the values, skipping the columns with standard deviations & number of lit values
    for idx,cols in enumerate(DV.columns[startcol:]):
        if idx/3 != math.ceil(idx/3):
            pass
        else:
            colname = colnames[i]
            #Add the uncertain values. Format is (nominal value, standard deviation)
            uDV.loc[:,colname] = unumpy.uarray(DV.loc[:,cols],DV.iloc[:,idx+startcol+1])#adding 1 to get to first SD value
            i+=1
    #Run again for the number of values - this puts the number of values columns at the end rather than interspersed.
    i = 0
    for idx,cols in enumerate(DV.columns[startcol:]):
        if idx/3 != math.ceil(idx/3):
            pass
        else:
            #The number of values will be labelled colname_NumVals e.g. 'LogKOW_NumVals'
            colname = colnames[i]+str('_NumVals') 
            uDV.loc[:,colname] = DV.iloc[:,idx+startcol+2] #Column where number of values is stored
            uDV.loc[uDV.loc[:,colname]==0,colname]=0.5
            i+=1
    #We will also make numvals for dUa, Sa, Kowd
    uDV.loc[:,'dUA_NumVals'] = uDV.loc[:,'dVAPH_NumVals']
    uDV.loc[:,'LogSA_NumVals'] = uDV.loc[:,'LogPL_NumVals']
    uDV.loc[:,'LogKOWd_NumVals'] = uDV.loc[:,'LogKOW_NumVals']
    #mask = [uDV.loc[:,'dUA_NumVals':'LogKOWd_NumVals']==0]
    #uDV.loc[mask,'dUA_NumVals':'LogKOWd_NumVals'] = 0.5
    #Next, we will convert our vapour pressures to a solubility: Sa = VP/RT
    R = 8.314 #J/molK, ideal gas constant
    T = 298.15 #K, temperature
    uDV.loc[:,'LogSA'] = unumpy.log10(10**(uDV.LogPL)/(R*T))#Convert to mol/m³
    uDV.loc[uDV.LogPL==0,'LogSA'] = np.nan #Convert zeros to nan\
    #Here we will also convert the wet octanol partition coefficient to dry so that we are all on the same page
    #Using the relationships from Beyer (2002)
    #DV.loc[:,'LogKOWd'] = np.nan#initialize
    mask = uDV.loc[:,'LogKOW']<=4
    uDV.loc[mask,'LogKOWd'] = uDV.LogKOW - 0.117 #If Kow<=4
    uDV.loc[mask==False,'LogKOWd'] = 1.35*uDV.LogKOW - 1.58
    uDV.loc[uDV.LogKOW==0,'LogKOWd'] = np.nan #Convert zeros to nan\
    #convert to internal energy of phase change (dU) rather than enthalpies (dH) (see Goss 1996 DOI 10.1021/es950508f) using rel.
    #found by Beyer et al. (2002) for PCBs, namely dAU = dvapH - 2,391 J/mol. Like Beyer, we will assume that all enthalpies of
    #phase change in the water phase are actually internal energies, as they were measured volumetrically.
    uDV.loc[:,'dUA'] = uDV.dVAPH - 2391/1000
    uDV.loc[uDV.dVAPH==0,'dUA'] = np.nan #Convert zeros to nan\
    uDV = uDV.replace(0,np.nan) #Convert all zeros to nans for the next part
    #Now. lets make the dataframe where we will store our FAVs once they have been estimated! 
    FAVs = pd.DataFrame(index = uDV.index)
    #Need to initialize with a value or it doesn't like the ufloat objects
    FAVs.loc[:,'Class'] = DV.loc[:,'Class']
    dUnames = ['dUA','dUW','dUO','dUAW','dUOW','dUOA']
    KSnames = ['LogSA','LogSW','LogSO','LogKAW','LogKOW','LogKOA','LogKOWd']
    var_names = dUnames+KSnames
    #Initialize the FAVs
    for var in var_names:
        FAVs.loc[:, var] = unumpy.uarray(0,0)            
    #Count the number of absent values for both dUs and KS. Separate loop to keep the FAVs together for easier
    #indexing later.
    for dUs in dUnames:
        dUabs = dUs+'_absent'
        uDV.loc[:, dUabs] = unumpy.isnan(uDV.loc[:,dUs])
    uDV.loc[:,'nUs_absent'] = uDV.loc[:,'dUA_absent':'dUOA_absent'].sum(axis=1)
    for KS in KSnames: #Again, separate loop to keep things together. 
        KSabs = KS+'_absent'
        uDV.loc[:, KSabs] = unumpy.isnan(uDV.loc[:,KS])
    uDV.loc[:,'nKS_absent'] = uDV.loc[:,'LogSA_absent':'LogKOA_absent'].sum(axis=1)
    #dU misclosure errors for all compounds, nans where not enough info
    uDV.loc[:,'w_dUAW'] = uDV.loc[:,'dUA'] - uDV.loc[:,'dUW'] - uDV.loc[:,'dUAW']
    uDV.loc[:,'w_dUOW'] = uDV.loc[:,'dUW'] - uDV.loc[:,'dUO'] + uDV.loc[:,'dUOW']
    uDV.loc[:,'w_dUOA'] = uDV.loc[:,'dUA'] - uDV.loc[:,'dUO'] + uDV.loc[:,'dUOA']
    #dUij misclosure errors for all compounds, nans where not enough info is present.
    uDV.loc[:,'w_dUij'] = uDV.loc[:,'dUAW'] - uDV.loc[:,'dUOW'] + uDV.loc[:,'dUOA']
    #KS misclosure errors for all compounds, nans where not enough info is present.
    uDV.loc[:,'w_Ks'] = uDV.loc[:,'LogKAW']-uDV.loc[:,'LogKOW']+uDV.loc[:,'LogKOA']
    #4 Compound misclosures
    uDV.loc[:,'w4']= uDV.loc[:,'LogSA']-uDV.loc[:,'LogSW']-uDV.loc[:,'LogKOW']+uDV.loc[:,'LogKOA']
    uDV.loc[:,'w5']= (uDV.loc[:,'LogSA'] - uDV.loc[:,'LogSW'] - uDV.loc[:,'LogKOWd']+ uDV.loc[:,'LogKOA'])
    #Equations 10-12
    uDV.loc[:,'w_KAW']= uDV.loc[:,'LogSA']-uDV.loc[:,'LogSW']-uDV.loc[:,'LogKAW']
    uDV.loc[:,'w_KOW']= uDV.loc[:,'LogSW']-uDV.loc[:,'LogSO']+uDV.loc[:,'LogKOW']
    uDV.loc[:,'w_KOA']= uDV.loc[:,'LogSA']-uDV.loc[:,'LogSO']+uDV.loc[:,'LogKOA']
    arr = uDV.loc[:,'w_Ks':'w_KOA']
    arr = np.ma.filled(np.ma.masked_where(unumpy.isnan(arr),arr),np.nan)
    uDV.loc[:,'w_Ks':'w_KOA'] = arr
    
    return uDV, FAVs


def setup_model(comp,uDV,model_type = 'KS'):
    '''
    This function sets up Bayesian models to harmonize compound enthalpy (dU) or partition coefficient and solubility (KS) values.
    Overall, our goal is to solve one of the following set of equations:
    dU:    
    dUa - dUW - dUAW = 0 [1 -1 -1 0 0] = 0
    dUw - dUo + dUOW = 0 [0 1 -1 0 1 0] = 0
    dUa - dUo + dUoa = 0 [1 0 -1 0 0 1] = 0
    Or, for KS:
    LogSa - LogSw - LogKAW = 0 [1 -1 0 -1 0 0] = 0
    LogSw - LogSo + LogKOW = 0 [0 1 -1 0 1 0] = 0
    LogSA - LogSo  + LogKOA = 0 [1 0 -1 0 0 1] = 0
    We need to reduce this system of equations if certain compounds are missing.
    
    Attributes:
    ----------
    comp = Compound to be harmonized. Should be a string e.g. TCEP = 'TCEP'
    uDV = Uncertain derived values. Should be a pandas dataframe of unumpy entries, formatted using the loaddata function above
    Optional:
        model_type = 'dU' or 'KS'.
    Outputs:
        X = input matrix for Bayesian model
        base_sd = base standard deviation for compound, used as an input for the Bayesian model
    See Rodgers et al. (2021) for full details on model paramaterization.
    '''
    #pdb.set_trace()
    #Set up the matrix, order is: dUa, dUw, dUo, dUaw, dUow, dUoa or logSa, logSw, logSo, logKaw, logKow, logKoa
    enthbasemat = np.array([[1, -1, 0, -1, 0, 0] ,[0, 1, -1, 0, 1, 0],[1, 0, -1, 0, 0, 1]])  
    #X will be the system equation (enthbasemat), y is the misclosure error - which is 0 in this case
    X = enthbasemat
    #y = 0
    #Then, we set up our Bayesian model with the measurements as priors for the model regression parameters. 
    #We are going to adjust the priors of our system based on the misclosure error, as an estimate of the 
    #certainty of the system. Basically, we are going to take the largest misclosure error and divide it by the
    #number of parameters in each eqauation and use that as our baseline minimum standard deviation.
    #Then, we are going to take that value and divide it by the number of literature values that were used to
    #determine a property, with the reasoning that we are more sure about our prior unertainty (the lit SD) with
    #more measurements added together
    #Define the cases for the base_standard deviation based on the properties that are absent
    #For zero or 1 property absent, we will calculate all properties (infer the missing value)
    #But, we still need to define the misclosure error.
    if model_type == 'dU':
        absvec = np.array(uDV.loc[comp,'dUA_absent':'dUOA_absent']==True)
        propvec = np.array([uDV.loc[comp,'dUA'],uDV.loc[comp,'dUW'],uDV.loc[comp,'dUO'],uDV.loc[comp,'dUAW'],
                          uDV.loc[comp,'dUOW'],uDV.loc[comp,'dUOA']])
    else:
        absvec = np.array(uDV.loc[comp,'LogSA_absent':'LogKOA_absent']==True)
        propvec = np.array([uDV.loc[comp,'LogSA'],uDV.loc[comp,'LogSW'],uDV.loc[comp,'LogSO'],uDV.loc[comp,'LogKAW'],
                          uDV.loc[comp,'LogKOW'],uDV.loc[comp,'LogKOA']])

    #To calculate the misclosure error (enthmat*propsvec) we need to ensure that any system with a missing value 
    #will not be included. To do this, we set the missing value arbitrarily high, then only take the misclosure
    #errors of compounds below some threshold
    propvec[absvec] = 1e9
    threshold = 1e8
    threeparam = False
    #Now, we will go through cases to determine which equations to use for the misclosure error, from which we calculate the base SD
    #There is probably a good way to do this with matrix algebra, I have mostly done a brute force method which works but isn't ideal
    #Firs, we are going to check if any of dUA, dUW and dUO are missing. If any of them are we need to introduce a fourth equation,
    #dUAW - dUOW + dUOA = 0 [0 0 0 1 -1 1] = 0 or LogKAW - LogKOW + LogKOA = 0 [0 0 0 1 -1 1] = 0 
    if sum(absvec[0:3]) == 0:
        enthbasemat2 = enthbasemat
    else:
        enthbasemat2 = np.vstack([enthbasemat,[0 ,0 ,0 ,1, -1, 1]])
    #Then, we go case-by-case through the number of missing values to determine the base_sd
    #For zero or one missing, it is fairly simple - always solve full system, with one value missing.
    if sum(absvec)<=1:
        mask = abs(np.dot(enthbasemat2,propvec)) < threshold
        base_sd = max(abs(np.dot(enthbasemat2,propvec))[mask]).n/3
        #For this case, we will always be solving the full system so no X = enthbasemat
        X = enthbasemat  
    #For 2 missing we only have three unique cases, where a partition coefficint can be inferred 
    #from the other two solubilities. We will always infer that missing property with a system of two equations.
    elif (sum(absvec)==2):
        if (absvec[0]) & (absvec[4]) == True:
            X = np.array([enthbasemat2[1],enthbasemat2[3]])
            base_sd = abs(np.dot(enthbasemat[0] - enthbasemat[2],propvec)).n/4
        elif (absvec[1]) & (absvec[5]) == True:
            X = np.array([enthbasemat2[2],enthbasemat2[3]])
            base_sd = abs(np.dot(enthbasemat[0] + enthbasemat[1],propvec)).n/4
        elif (absvec[2]) & (absvec[3]) == True:
            X = np.array([enthbasemat2[0],enthbasemat2[3]])
            base_sd = abs(np.dot(enthbasemat[1] - enthbasemat[2],propvec)).n/4
        else: #If it isn't one of the three cases, we will be able to solve at most 3 properties below.
            threeparam = True
    #With three properties missing, we can solve one of the four equations. No free variables here - only solve with misclosure error
    if (sum(absvec)==3) | (threeparam == True):
        if sum(uDV.loc[comp,'dUAW_absent':'dUOA_absent']) == 0:
            X = [0 ,0 ,0 ,1, -1, 1]
            base_sd = abs(np.dot(X,propvec)).n/3
        elif sum(absvec[[0,1,3]]) == 0:
            X = enthbasemat[0] 
            base_sd = abs(np.dot(X,propvec)).n/3
        elif sum(absvec[[1,2,4]]) == 0:
            X = enthbasemat[1]
            base_sd = abs(np.dot(X,propvec)).n/3
        elif sum(absvec[[0,2,5]]) == 0:
            X = enthbasemat[2]
            base_sd = abs(np.dot(X,propvec)).n/3
        else:
            raise ValueError('No FAV possible with current compound')
        #Need to keep the shape of these as a single row for further processing
        X = X.reshape(1,6)
    elif (sum(absvec) > 3):
        #Can't solve with >3
        raise ValueError('No FAV possible with current compound')
    return X, base_sd    

    
def define_model(comp,uDV,X=None,base_sd=None,model_type = 'KS',**kwargs):
    '''
    This function will define the Bayesian model to harmonize compound dU or KS values.
    
    Attributes:
    ----------
    comp = Compound to be harmonized. Should be a string e.g. TCEP = 'TCEP'
    uDV = Uncertain derived values. Should be a pandas dataframe of unumpy entries, formatted using the loaddata function above
    Optional (if not present will be calculated from setup_dUmodel):
        X = input matrix for Bayesian model
        base_sd = base standard deviation for compound, used as an input for the Bayesian model
        model_type = 'dU' or 'KS'.
        **kwargs contains additional optional arguments that can be passed to the model. In this case,
        we are using kwargs to define what distribution we want for a particular 
    Output:
        enth_model = pyMC3 model object for the enthalpies. This is used to run the trace.
    See Rodgers et al. (2021) for full details on model paramaterization.
    '''        
    #pdb.set_trace()
    if X == None:
        X,base_sd = setup_model(comp,uDV,model_type =model_type)
    beta = [0,0,0,0,0,0]
    #We only want to parameterize the properties that are in our matrix X. So first, we see which rows are all zeros
    presprops = np.where(X.any(axis=0))[0]
    #Then, we need to set up the names, and the uniform distribution boundaries we will be drawing from for the unkown property.
    if model_type == 'dU':
        varnames = ['dUA','dUW','dUO','dUAW','dUOW','dUOA']
        #lower bound of uniform distribution
        unilow = [0,-100,-100,0,-100,-150]
        #upper bound of uniform distribution
        unihi = [200,100,100,150,100,0]
        #The dU value is set at 0.1 as that led to a good balance of accuracy and model speed.
        sigmatest = 0.1
    else: 
        varnames = ['LogSA','LogSW','LogSO','LogKAW','LogKOWd','LogKOA']
        unilow = [-10,-10,-10,-10,-10,-10]
        unihi = [20,20,20,20,20,20]
        #The KS sigmatest value is set at 0.01 as that led to a good balance of accuracy and model speed.
        sigmatest = 0.01   
    #Define the model in pyMC3
    FAV_model = pm.Model()    
    with FAV_model:
        #Now, we define the priors. These are based on the uncertainty from the derived values, as limited by the 
        #base_sd. X, our input matrix, is the system we will solve.
        #sigma is the error on the FAV equation - essentially, this is the prior for the misclosure error.
        sigma = pm.HalfNormal('sigma', sigmatest, testval=sigmatest) 
        #For the physicochemical properties, Rodgers et al. (2021) used a normal distribution with the mean as the central tendency
        #and the standard deviation of the DVs as the standard deviation, subject to a minimum derived from base_SD. To adjust the priors,
        #read the pymc3 documentation about what distributions are available and then change this code.
        for props in presprops:
            #Check if the property has a different distribution, as defined elsewhere. Need to add these manually
            if varnames[props] in kwargs.keys():
                if kwargs[varnames[props]] == 'Lognormal':
                    beta[props] = pm.Lognormal(varnames[props], mu = np.log(uDV.loc[comp,varnames[props]].n),\
                        sigma=2*max(base_sd/uDV.loc[comp,varnames[props]+'_NumVals'],uDV.loc[comp,varnames[props]].s))
                elif kwargs[varnames[props]] == 'SkewNormal':
                    beta[props] = pm.SkewNormal(varnames[props], mu = uDV.loc[comp,varnames[props]].n,\
                        sigma=2*max(base_sd/uDV.loc[comp,varnames[props]+'_NumVals'],uDV.loc[comp,varnames[props]].s),
                        alpha = 1.)#alpha Chosen by professional judgement
            
            #See if the property is present. If not, give it a uniform distribution.
            #print(varnames[props])
            elif uDV.loc[comp,varnames[props]+'_absent'] == True:
                beta[props] = pm.Uniform(varnames[props], lower = unilow[props], upper = unihi[props])
            else: #Otherwise, the DV is the prior in a normal distribution.
                beta[props] = pm.Normal(varnames[props], mu = uDV.loc[comp,varnames[props]].n,\
                            sigma=max(base_sd/uDV.loc[comp,varnames[props]+'_NumVals'],uDV.loc[comp,varnames[props]].s))
                            #testval=np.array([uDV.loc[comp,varnames[props]].n])) #Added this line 20220204 to fix an initialization problem
        epsilon = np.dot(X,beta) #This gives us the misclosure error for each of the three equations
        #The model tries to fit to an observation, in this case that the misclosure should be 0.
        #We add the sum of squared errors to ensure that we don't over-fit to one equation.
        mu = np.sum(epsilon**2)
        #For the likelihood distribution we will use a Student's T regression, fatter tails 
        #We set our observed value as 0 - this represents an observed squared misclosure error of 0
        Y_obs = pm.StudentT('Y_obs',nu = len(beta)-1, mu=mu, sigma=sigma, observed=0)
    return FAV_model
    
def run_model(comp,FAVs,uDV=None,bayes_model = None,model_type ='KS',savepath = 'Traces/',
              trace=10000,tune=10000,chains=5,cores=5,target_accept=0.9,max_treedepth = 15,**kwargs):
    '''
    This function run the Bayesian model using the NUTS sampler, then saves the output FAVs to the FAVs input.
    It will accept either the dU or the Bayesian model, as "enth_model"
    For more information on the optional inputs trace, tune, chains, cores, target_accept and max_treedepth, 
    consult the pyMC3 documentation. To use other samplers you can simply use the pm.sample command directly, as described in the docs.
    
    Attributes:
    ----------
    comp = Compound to be harmonized. Should be a string e.g. TCEP = 'TCEP'
    FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, formatted using the loaddata function
    Optional Arguments:
        uDV = Uncertain derived values. Should be a pandas dataframe of unumpy entries, formatted using the loaddata function 
        bayes_model = pyMC3 model object. This is used to run the trace.
        model_type = 'dU' or 'KS'. If no model is directly input, this will define and then run the defined type of model
        savepath = path to save traces. Default is a folder called Traces
        trace (int) = number of traces to keep. Default is 10,000, increase if the model does not converge
        tune (int) = number of tuning samples to discard. Default is 10,000, increase if the model does not converge
        chains (int) = number of independent chains to run. Default is 5
        cores (int) = number of procesor cores to run the chains. pyMC3 supports parallel operations, ensure this number is supported by your system
        target_accept = target acceptance probability. Default is 0.9, increase if the model fails to converge
        max_treedepth = maximum depth of the NUTS sampler tree. Increase if you see "maximum tree depth" warning.
    Outputs:
        FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, now with the new FAVs inserted.
        FAV_trace = pyMC3 multitrace object. Contains the probability distribution functions for the FAV
    See Rodgers et al. (2021) for full details on model paramaterization.
    '''
    #pdb.set_trace()
    #First, initialize the desired model type if bayes_model is not provided. 
    if (bayes_model == None):
        bayes_model = define_model(comp,uDV,model_type=model_type,**kwargs)

        #bayes_model = define_KSmodel(comp,uDV,FAVs)
    #Then, all we do is run it
    with bayes_model:
        #start={'sigma':np.array([0.]),
        FAV_trace = pm.sample(trace,tune=tune,chains=chains,cores=cores,
                              target_accept=target_accept,
                              max_treedepth=max_treedepth,return_inferencedata=False)
    #Output the FAVs to the FAV datafile
    FAVs = trace_to_FAVs(comp,FAVs,FAV_trace = FAV_trace)
    #Save the trace - this tutorial uses a folder called "Traces" and saves with the compound 
    #name and date/time. Remove square brackets from comp. name - in the example dataset just the PAHs, this causes problems for loading
    if comp == "benzo[k]fluoranthene": #Error from the square brackets in loading files
        comp = "benzokfluoranthene"
    elif comp == "benzo[a]pyrene":
        comp = "benzoapyrene"
    savename = str(savepath+comp+"_"+model_type+"_"+datetime.now().strftime("%m%d_%H%M"))
    pm.save_trace(FAV_trace,directory=savename,overwrite=True)        
    
    return FAVs, FAV_trace

def plot_trace(comp,FAVs,FAV_trace=None,filename=None,bayes_model=None,uDV=None,
               model_type = 'KS',summary=True,fig=True,**kwargs):
    '''
    This function display the results of a trace, using the trace as an input or a filename specifying a trace.
    To load a trace you need to havea model defined, if no model is defined we will generate a model for that compound
    
    Attributes:
    ----------
    comp = Compound to be harmonized. Should be a string e.g. TCEP = 'TCEP'
    FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, formatted using the loaddata function
    Optional Attributes:
        FAV_trace = pyMC3 multitrace object. Contains the probability distribution functions for the FAV
        filename (str) = Required if no trace given. String containing the filename & path to a trace file
        bayes_model = pyMC3 model object. Needed if loading from file, if absent will be generated from uDV, FAVs, comp
        uDV = Uncertain derived values. Necessary if the model is not defined.
        model_type = 'dU' or 'KS'. Required if model isn't defined.
        summary (bool) = True/False for whether the summary table should be displayed or just the plots.
    Ouputs:
        fig = figure object, shows the traceplots. Generally, we want the different lines on the left
        hand figure to overlap.
        tracesumm = table showing a summary of the outputs for the Bayesian model. In Rodgers et al. (2021), we used a threshold of ess_bulk
        >= 5000 as a rule of thumb for convergence (above 5000 typically converged, below required looking at the traceplot). 
        See the pyMC3 documentation for further details on both outputs.
    See Rodgers et al. (2021) for full details on model paramaterization.
    '''
    #First, check if a trace or a filename was given
    #pdb.set_trace()
    if FAV_trace == None:
        #Check if a model was given & what type. Define the model if necessary.
        if (bayes_model == None):
            bayes_model = define_model(comp,uDV,model_type=model_type,**kwargs)      
        FAV_trace = pm.load_trace(filename,bayes_model)

    #The traceplot displays a graph of the poseteriors for each chain and the traceplots
    with bayes_model:
        if fig == True:
            pm.traceplot(FAV_trace);
            #To give the figure a title need to obtain figure name and then define title
            fig = plt.gcf();
            fig.suptitle(comp, fontsize=16)
        else:
            fig = None
        #If we want to display the summary table, display it
        if summary == True:
            tracesumm = pm.summary(FAV_trace)

    return fig, tracesumm
    
def trace_to_FAVs(comp,FAVs,FAV_trace=None,filename=None,bayes_model=None,uDV=None,model_type = 'KS',**kwargs):
    '''
    This function puts the trace outputs in the FAVs dataframe, using the trace as an input or a filename specifying a trace.
    To load a trace you need to havea model defined, if no model is defined we will generate a model for that compound
    
    Attributes:
    ----------
    comp = Compound to be harmonized. Should be a string e.g. TCEP = 'TCEP'
    FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, formatted using the loaddata function
    Optional Attributes:
        FAV_trace = pyMC3 multitrace object. Contains the probability distribution functions for the FAV
        filename (str) = Required if no trace given. String containing the filename & path to a trace file
        bayes_model = pyMC3 model object. Needed if loading from file, if absent will be generated from uDV, FAVs, comp
        uDV = Uncertain derived values. Necessary if the model is not defined.
        model_type = 'dU' or 'KS'. Required if model isn't defined.
    Ouputs:
        FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, now with the new FAVs inserted.
        See the pyMC3 documentation for further details on both outputs.
    See Rodgers et al. (2021) for full details on model paramaterization.
    '''
    #First, check if a trace or a filename was given
    #pdb.set_trace()
    if FAV_trace == None:
        #Check if a model was given & what type. Define the model if necessary.
        if (bayes_model == None):
            bayes_model = define_model(comp,uDV,model_type=model_type,**kwargs)         
        FAV_trace = pm.load_trace(filename,bayes_model)
    #Define the variable names for the loop    
    dUnames = ['dUA','dUW','dUO','dUAW','dUOW','dUOA']
    KSnames = ['LogSA','LogSW','LogSO','LogKAW','LogKOWd','LogKOA']
    var_names = dUnames+KSnames
    for prop in FAV_trace[0].keys():
        if prop in var_names:
            FAVs.loc[comp,prop] = ufloat(FAV_trace[prop].mean(),FAV_trace[prop].std())    
    return FAVs

def export_FAVs(FAVs,savename = 'FAVs'):
    '''
    This function puts the FAV outputs into a CSV for further processing.
    
    Attributes:
    ----------
    FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, formatted using the loaddata function
    Optional Attributes:
        savename (str) = Required if no trace given. String containing the filename & path to a trace file
        bayes_model = pyMC3 model object. Needed if loading from file, if absent will be generated from uDV, FAVs, comp
        uDV = Uncertain derived values. Necessary if the model is not defined.
        model_type = 'dU' or 'KS'. Required if model isn't defined.
    Ouputs:
        FAVs = Final Adjusted Values. Pandas dataframe of unumpy entries, now with the new FAVs inserted.
        See the pyMC3 documentation for further details on both outputs.
    See Rodgers et al. (2021) for full details on model paramaterization.
    '''
    #pdb.set_trace()
    dUnames = ['dUA','dUW','dUO','dUAW','dUOW','dUOA']
    KSnames = ['LogSA','LogSW','LogSO','LogKAW','LogKOWd','LogKOA']
    var_names = dUnames+KSnames
    FAVs_exp = pd.DataFrame(index = FAVs.index)
    for prop in var_names:
        FAVs_exp.loc[:,prop] = unumpy.nominal_values(FAVs.loc[:,prop])
        FAVs_exp.loc[:,prop+'_SD'] = unumpy.std_devs(FAVs.loc[:,prop])
    
    #Need to define a filepath - choose wherever you like, here you can see my file set up for GitHub
    directory = os.getcwd()
    FAVs_exp.to_csv(directory+'/'+savename+'.csv')
    
    return FAVs_exp





