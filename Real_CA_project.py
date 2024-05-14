#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:50:54 2024

@author: karan
"""

import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

def readfile(filename):
    '''
    Read the opacity files and store the data in numpy arrays
    store: log(R), log(T), log(k)  
    Notes:
        Had to modify opacity and eplison file
        First index Log(T) made parsing difficult i deleted it

    Parameters
    ----------
    filename : TYPE-string
        file path to opacity text file.

    Returns
    -------
    None.

    '''
    with open(filename, 'r') as file:
        lines = file.readlines()
    #first row is logR
    first_row = lines[0].strip().split()
    logR = [float(r) for r in first_row]
    logR = np.array(logR)
    
    logT = []
    logK = []
    #parse temperature and opaicty
    #first row is R, second is black space
    for line in lines[2:]:
        item = line.strip().split()
        logT.append(float(item[0]))
        #breakdown into rows otherwise causes interpolation errors
        kappa_row = [float(val) for val in item[1:]]  # List comprehension within the loop
        logK.append(kappa_row)
    #convert all to numpy arrays
    logT = np.array([logT])
    logK = np.array(logK)
    
    
    return logR, logT, logK
#create a grif with the opacity file
current_directory = '/home/karan/Documents/UvA/Computational Astrophysics/stellar-structure/'
opacity = os.path.join(current_directory,'opacity.txt')

opacity_file = readfile(opacity)

def opacity_grid(rho,T, logR, logT, logK):
    '''
    Notes:
    start in log space and later convert to cgs units
    scipy interpd2D depreciated, use scipy.interpolate.RegularGridInterpolator
    

    Parameters
    ----------
    rho : float
        density.
    T : float
       Temperature in kelvin.
    logT : np array of temperatures in logK

    logR : np array of density in log g/cm^3
    
    logK : Tnp array of opacity in log cgs
        .

    Returns
    opacity in cgs units, either interpolated or extrapolated
    -------

    '''
    logT0 = np.log10(T)
    logR0 = rho/((T/1e6)**3.0)
    #create an interpolator with scipy
    #extrapolate given by bound_error = false
    #and fill_value = None
    #make sure logT and logR are 1D array
    logR = logR.reshape(-1)
    logT = logT.reshape(-1)
    interpolate = RegularGridInterpolator((logT,logR),logK,method='linear' 
                                          ,bounds_error=False, 
                                          fill_value=None)
    #the interpolated opcaity in log nits
    interp_kappa = interpolate((logT0, logR0))
    kappa = interp_kappa**3.0
    
    #extrapolate warning
    Tmin, Tmax = np.min(logT), np.max(logT)
    Rmin, Rmax = np.min(logR), np.max(logR)
    
    if logT0  < Tmin or logT0 > Tmax:
        print('Warning:Exrtapolating along Temperature')
    if logR0< Rmin or logR0 > Rmax:
        print("Warning exrtapolating along Density")
    
    return kappa

logR, logT, logK,= opacity_file
kappa_check = opacity_grid(1e-7, 1e-7,logR,logT,logK)
#sanity checks    
    