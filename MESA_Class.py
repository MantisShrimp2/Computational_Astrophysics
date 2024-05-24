#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:43:57 2024

@author: karan
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from scipy.interpolate import RegularGridInterpolator


class MESA:
    
    def __init__(self,opacity_filename,epsilon_filename):
        self.opacity_file = opacity_filename #opacity or epsilon file
        self.epsilon_file = epsilon_filename
        #define the constants
        self.G = 6.67430e-11  # gravitational constant
        self.stefan = 5.670374419e-8  # Stefan-Boltzmann constant
        self.c = 3.00e8 # speed of light
        self.k_B = 1.380e-23  # Boltzmann constant
        self.M_h = 1.660e-27  # atomic mass unit
        #do the mass fraction calulation
        X_frac = 0.7 
        Y_frac = 0.29 + 10e-10
        Z_frac = 1 - (X_frac + Y_frac)
        self.mu = 1/(2*X_frac +0.75*Y_frac + 0.5*Z_frac)

        # Solar values
        self.L_sun = 3.828e26  # Luminosity of the Sun lum
        self.R_sun = 6.96342e8  # Radius of the Sun m
        self.M_sun = 1.989e30  # Mass of the Sun kg

        #Initial conditions
        # self.rho0 = 1.0*  1.42e-7 * 1.408e3  # initial density of sun kg/m3
        # self.T0 = 1.0 * 5770  # initial temperature k
        # self.P0 = 1.0 * self.pressure(self.T0, self.rho0) # initial pressure (ideal gas law)
        # self.L0 = 1.0 * self.L_sun  # initial luminosity
        # self.R0 = 1.0 * self.R_sun 
        # self.M0 = 1.0 * self.M_sun
        
        # best model
        self.rho0 = 2.0 *  1.42e-7 * 1.408e3  # initial density of sun kg/m3
        self.T0 = 1.0 * 5770  # initial temperature k
        self.P0 = 1.0 * self.pressure(self.T0, self.rho0) # initial pressure (ideal gas law)
        self.L0 = 1.0 * self.L_sun  # initial luminosity
        self.R0 = 1.0* self.R_sun 
        self.M0 = 1.0 * self.M_sun
        #other coeffiecents 
        self.Cp = (5*self.k_B)/(2*self.mu * self.M_h)
        #see how many euler steps with plot over
        self.end_step = 0

    def opacity(self, filename,T,rho):
        '''
        
       Read the file, the first row should be R
       the first column is T
       the rest is opacity or epsilon data 
       store all the data in arrays
       interpolate the values
       return a kappa or epsilon
       create warning if extrapolating outside of table bounds.

       Parameters
       ----------
       filename : txt file either opacity or epsilon.
       T : float, temperature
       rho : float density.

       Returns
       -------
       kappa or epsilon log cgs units
        '''

    # Read the data file
        with open(filename, 'r') as file:
            lines = file.readlines()
    
    # Extract logR values (excluding the first entry)
        logR = np.array(lines[0].split()[1:], dtype=float)
        
        # Extract logT values and the logK grid
        logT = []
        logK = []
        for line in lines[2:]:
            parts = line.split()
            logT.append(float(parts[0]))
            logK.append([float(k) for k in parts[1:]])
        
            #convert all to numpy arrays
        logT = np.array(logT)
        logK = np.array(logK)

        interpolate = RegularGridInterpolator((logT,logR),logK,method='linear' 
                                                  ,bounds_error=False, 
                                                  fill_value=None)
            #the interpolated opcaity in log units
        logT0 = np.log10(T)
        #convert rho to cgs
        logR0 = np.log10(rho * 0.001/((T/1e6)**3))
        interp_kappa = interpolate((logT0, logR0))
        log_kappa = interp_kappa
      
        #extrapolate warning
        Tmin, Tmax = np.min(logT), np.max(logT)
        Rmin, Rmax = np.min(logR), np.max(logR)
          
        if logT0  < Tmin or logT0 > Tmax:
            warnings.warn("Extrapolating out of bounds along Temperature")
        if logR0< Rmin or logR0 > Rmax:
            warnings.warn("Extrapolating out of bounds along Density")
        return log_kappa
    def drdm(self, r, rho):
        '''
    evaulate the derivative of radius WRT mass

    Parameters
    ----------
    r : float, radius coordinate.
    rho : float, density in cgs.

    Returns
    -------
    float
       madd derivative drdm at r, rho.

    '''
        
        return 1/( 4 * np.pi * r**2 * rho)
    def dpdm(self, m, r):
        '''
        evaulate the derivative of presure WRT mass

        Parameters
        ----------
        m : float - mass.
        r : float- radius

        Returns
        -------
        float
            dpdm at point m, r.

        '''
        return -self.G * m/(4 * np.pi * r**4)
    def dldm(self,T,rho):
   
        '''Calculate epsilion at T,rho
        convert to linear space
        convert cgs to SI'''
        epsilon = 10**self.opacity(self.epsilon_file,T,rho)
        epsilon = epsilon* 1e-4 #convert cgs to SI
        return epsilon
    def dtdm(self, T,r,rho,L):
        '''
        

        Parameters
        ----------
        T : float- temperature
        r : float- radius
        rho : float- density.
        L : float- luminosity.

        Returns
        -------
       dtdm when ledoux criterion not met

        '''
        coef = (-3/(256 * np.pi**2 * self.stefan))
        kappa = (10**self.opacity(self.opacity_file,T,rho)) * 0.1
        return coef * (kappa* L)/(r ** 4 * T**3)
        
        #__define the temperature gradients___
    def gravity(self, m,r):
        '''
        evaulate acceleratoin due to gravity (g)

        Parameters
        ----------
        m : float - mass.
        r : float- radius

        Returns
        -------
        float
           gravity at point m, r.
        
        '''
        return (self.G * m)/(r**2)
    def nabla_ad(self,T,rho,P):
        '''
        Calculate adiabatic temperature gradient

        Parameters
        ----------
        T : float- temperature
        P: float- pressure
        rho : float- density.

        Returns
        -------
       adiabatic temperature gradient.

        '''     
        return (P/(T*rho*self.Cp))
        
    def nabla_star(self,T,rho,m,r,nab_stab,nab_ad):
        '''
        Calculate the nabla when ledoux criterion met
        Define constants
        calculate stable and adiabatic gradients
        find the root of each polynomial
        take the real part of the root
        find the minimum as the root
        calculate nabla_star with minimum root
        Parameters
        ----------
        T,rho,P,m,r,L same as above

        Returns
        -------
        nabla_star : temperature gradient for leodoux

        '''
        #si units
        kappa = (10**self.opacity(self.opacity_file,T,rho)) * 0.1 # si units
        grav = self.gravity(m,r)
        #define constants

        scale_height = (self.k_B* T)/(self.mu * self.M_h *grav)
        U = np.sqrt(scale_height/grav)*((64 * self.stefan * T**3)/(3 * kappa* rho**2 * self.Cp))
        lm = scale_height
        omega = 4/lm
        #solve nablas
        # nab_ad = self.nabla_ad(T,rho,P)
        # nab_stab = nabla_stable
        #define each coefficent
        A = 1
        B = (U/lm**2)
        C = (U**2 * omega)/(lm**3)
        D = (U/lm**2)*(nab_ad-nab_stab)
        #root finding
        eta_coefficents =[A,B,C,D]
        #print(eta_coefficents)
        eta_roots = np.roots(eta_coefficents)
        #select the first  real root as the one to solve nabla_star
        
       
        for root in eta_roots:
            if np.imag(root) == np.min(np.abs(np.imag(eta_roots))):
                eta_star = np.real(root)
                break
        nabla_star = (eta_star**2) + (U*omega/lm)*eta_star + nab_ad
        return nabla_star
    def nabla_stable(self,T,rho,L,r,m,P):
            '''
        

        Parameters
        ----------
        T,rho,P,m,r,L same as above
        Returns
        -------
        radiative temperature gradient.

        '''
            grav  = self.gravity(m, r)
            scale_height = (self.k_B * T)/(self.mu * self.M_h *grav)
            kappa = (10**self.opacity(self.opacity_file, T, rho)) *0.1
            nabla_stable = (3* rho * kappa* L * scale_height)/(64 * np.pi * r**2 * self.stefan *  T**4)
            
            return nabla_stable
        #___define density and pressure___
        #methods to calculate rho(P,T) and P(rho,T)
    def rho(self,P,T):
            '''
        Parameters
        ----------
        P,T same as above
        Returns
        -------
        density at P, T, updated at P[i + 1], T[i + 1], si units

        '''
            coef = (self.mu* self.M_h)/(self.k_B * T)
            return (P - (4*self.stefan*T**4)/(3*self.c))*coef
    def pressure(self,T,rho):
            '''
        

        Parameters
        ----------
        T, rho : same as above

        Returns
        -------
        Pressure at point T, rho SI units

        '''
            t4 = (4*self.stefan) * T ** 4/(3*self.c)
            t1 = (rho * self.k_B * T)/(self.mu*self.M_h) 
            return t4 + t1
    def euler(self):
            '''
        Integrate parameters of a star with euler method and adaptive time stepping
        
        Steps):
        set the inital conditions to an array
        avoid appending by creating an array of zeros for each parameter
        
        First set the temperature gradient by stability
        evalulate the nabla to use
        

        Returns
         Numpy Arrays of integrated parameters see below
        
        -------
        rho : density- kg/m3
        
        T : Temperature K.
        
        P : Pressure kg/(m * s2).
        L : Luminsoty (J/s).
        m : Mass kg.
        nabla : Temperature gradient - unitless
        nabla_stable : Radiative gradient - unitless
        nabla_ad : adiabatic gradient - unitless.
        nabla_star : polynomial gradient - unitless

        '''
            N = 10000
            #initize each parameter as an array of zeros
            rho,T,P,r,L,m  = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
            #give them inistal conditions
            rho[0],T[0],P[0],r[0],L[0],m[0],=self.rho0,self.T0,self.P0,self.R0,self.L0,self.M0 
            #start from the surface and integreate inside`
            #same for the nablas
            nabla, nabla_ad, nabla_stable,nabla_star = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
            nabla_ad[0], nabla_stable[0],nabla_star[0] =self.nabla_ad(T[0],rho[0],P[0]), self.nabla_stable(T[0],rho[0],L[0],r[0],m[0],P[0]),self.nabla_star(T[0], rho[0], P[0], m[0], r[0], L[0])
            #again for Flux convective

            for i in range(N-1):
                
            #first make sure m does get too small
                if m[i] < 1e-5:
                    print("Minimum Mass Threshold")
                    self.end_step= i
                    break
                else:
                    #euler time
                    #kappa = self.opacity(self.opacity_file, T[i],rho[i])**3
                    #calculate nablas
                    nabla_stable[i] = self.nabla_stable(T[i],rho[i],L[i],r[i],m[i],P[i])
                    nabla_ad[i] = self.nabla_ad(T[i],rho[i],P[i])
                    nabla_star[i] = self.nabla_star(T[i], rho[i],m[i],rho[i],nabla_stable[i],nabla_ad[i])
                    
                    self.schwarzschild = nabla_stable[i] > nabla_ad[i]
                    #temperature gradient check
                    
                    if self.schwarzschild: #if true
                    #store the unstable value
                        nabla[i] = self.nabla_star(T[i], rho[i],m[i],rho[i],nabla_stable[i],nabla_ad[i])
                        #calculate what way the stepsize goes
                        dt = nabla_star[i]*(T[i]/P[i])*self.dpdm(m[i],r[i])
                    else:
                        nabla[i] = nabla_stable[i]
                        dt = self.dtdm(T[i],r[i],rho[i],L[i])
        
                    #calulate adaptive timestep
                    
                    #dm  = dm/dr * R or dm/dp* P or dm/dt* dm
                    f = np.abs(np.array([self.drdm(r[i],rho[i]),self.dpdm(m[i],r[i]),dt,
                                          self.dldm(T[i],rho[i])]))
                    V = np.array([r[i],P[i],T[i],L[i]])
                    per = 0.01
                    dm  = np.min(per * V/f)
                
            
                   # print(self.dpdm(m[i],r[i]))
                               
                    #subtract since we're starting form surface
                    r[i+1] = r[i] - self.drdm(r[i],rho[i])*dm
                    T[i+1] = T[i] - (dt*dm) #(dt/dm * dm)
                    L[i+1] = L[i] - self.dldm(T[i],rho[i])* dm
                    P[i+1] = P[i] - self.dpdm(m[i],r[i])*dm
                    m[i+1] = m[i] - dm
                    rho[i+1] = self.rho(P[i+1],T[i+1])
                    #update flux:
            return rho,T,P,r,L,m,nabla,nabla_stable, nabla_ad,nabla_star
    def plotting(self,rho,T,P,r,L,mass):
            '''
        

        Parameters
        ----------
        rho,T,P,R,L,mass: arrays of paramters integrated by euler method 
            SI units
        Plots are generated up to minimum mass threshold using end_step
        Returns
        Normalized plots of each parameter WRT Radius (R)

        '''
            plt.figure(figsize=(10, 6))
            plt.suptitle("Best Fit Euler Model for N= "+str(self.end_step)+" steps ")
            plt.subplot(2, 2, 1)
            plt.plot(r[0:self.end_step] / self.R_sun, mass[0:self.end_step]/ self.M_sun)
            plt.xlabel(r'Radius $R/R_{sun}$')
            plt.ylabel('Mass [M/M_sun]')
            plt.title('Mass over Radius')

            plt.subplot(2, 2, 2)
            plt.plot(r[0:self.end_step] / self.R_sun, P[0:self.end_step]/self.P0)
            plt.xlabel(r'Radius $R/R_{sun}$')
            plt.ylabel('Pressure [cgs]')
            plt.title('Pressure over Radius')

            plt.subplot(2, 2, 3)
            plt.plot(r[0:self.end_step] / self.R_sun, L[0:self.end_step] / self.L0)
            plt.xlabel(r'Radius $R/R_{sun}$')
            plt.ylabel('Luminosity [L/L_sun]')
            plt.title('Luminosity over Radius')

            plt.subplot(2, 2, 4)
            plt.plot(r[0:self.end_step] / self.R_sun, T[0:self.end_step]/self.T0)
            plt.xlabel(r'Radius $R/R_{sun}$')
            plt.ylabel(r'$T/T_0$ ')
            plt.title('Temperature over Radius')

            plt.tight_layout()
            plt.show()
            #plot density
            plt.title("Density over Radius for N = "+str(self.end_step)+ "steps")
            plt.plot(r[0:self.end_step]/self.R_sun, rho[0:self.end_step]/self.rho0)
            plt.xlabel(r'Radius $R/R_{sun}$')
            plt.ylabel(r"$ \rho/ \rho_0$")
            plt.show()
    def sanity(self,filename,compare,label):
        '''Do a sanity check on opacity
        Compare the interpolated opacity to the theoretical one
        
        filename- str: .txt that for opacity or epsilon
            note make sure current_directory is accurate
        compare- list: a list of opacity or eplsion to comepare interpolated results with
        label - str: for printing purposes either opacity or epslion
        
        Return:
            kappa_analytic: list- interpolated quanitites (not a fitting name)
            percent_arr: list - realtive error between compare and kappa_analytic
        
        '''
        logT_array = np.array([3.750,3.755,3.755,3.755,3.755,3.770,3.780,3.795,3.770,3.775,3.780,3.795,3.800])
        logR_array = np.array([-6.00,-5.95,-5.80,-5.70,-5.55,-5.95,-5.95,-5.95,-5.80,-5.75,-5.70,-5.55,-5.50])
        kappa_analytic =[]
        

    # Read the data file
        with open(filename, 'r') as file:
            lines = file.readlines()
    
    # Extract logR values (excluding the first entry)
        logR = np.array(lines[0].split()[1:], dtype=float)
        
        # Extract logT values and the logK grid
        logT = []
        logK = []
        for line in lines[2:]:
            parts = line.split()
            logT.append(float(parts[0]))
            logK.append([float(k) for k in parts[1:]])
        
            #convert all to numpy arrays
        logT = np.array(logT)
        logK = np.array(logK)
            #interperate the results
           # logT = logT.reshape(-1)
        interpolate = RegularGridInterpolator((logT,logR),logK,method='linear' 
                                                  ,bounds_error=False, 
                                                  fill_value=None)
            #the interpolated opcaity in log units

        for T, R in zip(logT_array, logR_array):
            kappa_san = interpolate((T,R))
            kappa_analytic.append(kappa_san)
        #make sure it's a numpy array
        # compare = np.array([compare])
        percent_arr = []
        print("Sanity check for " +label+':')
        for i in range(len(compare)):
            obs = compare[i]
            exp = kappa_analytic[i]
            err = np.abs((obs-exp)/exp)
            percent = err * 100
            percent_arr.append(percent)
            print(f'Theoretical : {obs}, Interpolated : {exp:.3f}, Percentage error,{percent:.3f}%')
            
        return kappa_analytic, percent_arr

    def plot_nabla(self,r,nabla_ad, nabla_star,nabla_stable):
        '''
        

        Parameters
        ----------
        r,nabla_ad,nabla_star,nabla_stable - Numpy arrays of parameters returned from 
        euler()

         Plots are generated for entire star
         Returns
         Plot of temperature gradietns over radius 
        '''
        plt.plot(r/self.R_sun, nabla_ad, label=r'$\nabla_{ad}$')
        plt.plot(r/self.R_sun, nabla_star, label=r'$\nabla_{*}$')
        plt.plot(r/self.R_sun, nabla_stable, label=r'$\nabla_{stable}$')
        plt.legend()
        plt.title("temperature Gradients over Radius")
        plt.xlabel(r'$R / R_{\odot}$')
        plt.ylabel(r'$\nabla$')
        plt.yscale('log')
        plt.show()
    def cross_section(self,L,R,nabla_stable,nabla_ad):
        '''
        

        Parameters
        ----------
       L,R,nabla_stable,nabla_ad- numpy arrays of  integrated by euler method 
       
       Plot regions based on luminosity and gradient conditoins
       
       For outside the core
       convection is where nabla_stable> nabla_ad- red
           Dominates surface
       
       Raditation where nabla_stable < nabla_ad - yellow
           Dominates shells
       
       For inside the core- luminosity is smaller than L_max
       convection where nabla_stable> nabla_ad-  cyan
           Dominates outer core
       
        Raditation where nabla_stable < nabla_ad - blue
            Dominates inner core
           

        Return:
            Cross section plot of radaiative and convective zones in star
        '''
        L_max = 0.995 *self.L_sun

        R_sun = self.R0
        n = len(R)
        rmin, rmax= -1.2*np.max(R/R_sun), 1.2*np.max(R/R_sun)
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.set_xlim(rmin,rmax)
        ax.set_ylim(rmin,rmax)
        for i in range(0, n-1):
            if L[i] >= L_max:
                #if convection
                if nabla_stable[i] > nabla_ad[i]:
                 
                    red_circle = plt.Circle((0,0),R[i]/R_sun,fc='red',fill=True,ec=None)
                    ax.add_patch(red_circle)
                else: #radiative trasport
                    yellow_circle = plt.Circle((0,0),R[i]/R_sun,fc='yellow',fill=True,ec=None)
                    ax.add_patch(yellow_circle)
            elif L[i] <= L_max:
                #convective core
                if nabla_stable[i] > nabla_ad[i]:
                    cyan_circle = plt.Circle((0,0), R[i]/R_sun, fc='blue',fill=True,ec=None)
                    ax.add_patch(cyan_circle)
                else:
                    #radiative core
                    blue_circle = plt.Circle((0,0), R[i]/R_sun, fc='cyan',fill=True,ec=None)
                    ax.add_patch(blue_circle)
    
                    core = plt.Circle((0,0), R[self.end_step]/R_sun,fc='white',fill=True,ec=None,lw=0)
                    ax.add_patch(core)
        plt.title("Cross Section of Star")
        plt.tight_layout()
        #add a legend
        ax.legend([red_circle,yellow_circle,cyan_circle,blue_circle], ['Convection outside Core','Radiation Outside Core','Core Radiation','Core Convection'],loc='upper right')
        # Show all plots
        plt.xlabel(r'$ R/R_{sun}$')
        plt.ylabel(r'$ R/R_{sun}$')
        plt.show()
            
                    
                
        return None
        

                    
                    
                    
            
current_directory = '/home/karan/Documents/UvA/Computational Astrophysics/stellar-structure/'
opacity = os.path.join(current_directory,'opacity.txt')
epsilon = os.path.join(current_directory,'epsilon.txt')



rho_sun = 1.42e-7 * 1.408e3 

epsilon_compare = [-87.995,-87.623]
opacity_compare = [-1.55,-1.51,-1.57,-1.61,-1.67,-1.33,-1.20,-1.02,-1.39,-1.35,-1.31,-1.16,-1.11]

if __name__ == "__main__":
    makestar = MESA(opacity,epsilon)
    initial_conditions_opacity = makestar.opacity(opacity,5770,rho_sun)
    
    rho,T,P,r,L,Mass,nabla,nabla_stable,nabla_ad,nabla_star, = makestar.euler()
    
    makestar.plotting(rho,T,P,r,L,Mass)
    makestar.cross_section(L, r ,nabla_stable, nabla_ad)
    kappa_check = makestar.sanity(opacity,opacity_compare,label='Opacity')
    epsilon_check = makestar.sanity(epsilon,epsilon_compare,label = 'Epsilon')
    makestar.plot_nabla(r,nabla_ad,nabla_star,nabla_stable)
    


        