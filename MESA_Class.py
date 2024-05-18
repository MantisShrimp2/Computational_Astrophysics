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
        self.L_sun = 3.828e26  # Luminosity of the Sun
        self.R_sun = 6.96342e8  # Radius of the Sun
        self.M_sun = 1.989e30  # Mass of the Sun

        # Initial conditions
        self.rho0 = 1.42e-7 * 1.408e3  # initial density of sun
        self.T0 = 5770  # initial temperature
        self.P0 = (self.rho0 * self.k_B * self.T0) / self.mu  # initial pressure (ideal gas law)
        self.L0 = self.L_sun  # initial luminosity
        self.R0 = self.R_sun 
        self.M0 = self.M_sun
        
        #other coeffiecents 
        self.Cp = (5*self.k_B)/(2*self.mu * self.M_h)
        #stability criterion
        self.ledoux = True
    def opacity(self, filename,T,rho):
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
        file.close()
        #convert all to numpy arrays
        logT = np.array([logT])
        logK = np.array(logK)
        #interperate the results
        logT = logT.reshape(-1)
        interpolate = RegularGridInterpolator((logT,logR),logK,method='linear' 
                                              ,bounds_error=False, 
                                              fill_value=None)
        #the interpolated opcaity in log units
        logT0 = np.log10(T)
        logR0 = np.log10(rho/((T/1e6)**3))
        interp_kappa = interpolate((logT0, logR0))
        log_kappa = interp_kappa
        
        #extrapolate warning
        Tmin, Tmax = np.min(logT), np.max(logT)
        Rmin, Rmax = np.min(logR), np.max(logR)
        
        if logT0  < Tmin or logT0 > Tmax:
            warnings.warn("Extrapolating out of bounds along Temperature")
        if logR0< Rmin or logR0 > Rmax:
            warnings.warn("Extrapolationg out of bounds along Density")
        
        return log_kappa**10
        #___define the derivates___
    def drdm(self, r, rho):
            return 1/( 4 * np.pi * r**2 * rho)
    def dpdm(self, m, r):
            return (-self.G * m)/(4 * np.pi * r**4)
    def dldm(self,T,rho):
            '''Calculate epsilion at T,rho'''
            return self.opacity(self.epsilon_file,T,rho)
    def dtdm(self, T,r,rho,L):
            coef = (-3/(256 * np.pi**2 * self.stefan))
            kappa = (self.opacity(self.opacity_file,T,rho))
            return coef * (kappa* L)/(r ** 4 * T**3)
        
        #__define the temperature gradients
    def gravity(self, m,r):
            return self.G * m/(r**2)
    def nabla_ad(self,T,rho,P):
                
            return (P/(T*rho*self.Cp))
        
    def nabla_star(self,T,rho,P,m,r,L):
            kappa = self.opacity(self.opacity_file,T,rho)
            grav = self.gravity(m,r)
            #define constants
            
            scale_height = (self.k_B* T)/(self.mu * self.M_h *grav)
            U = np.sqrt(scale_height/grav)*(64 * self.stefan * T**3)/(3 * kappa* rho**2 * self.Cp)
            lm = scale_height
            omega = 4/lm
            #solve nablas
            nab_ad = self.nabla_ad(T,rho,P)
            nab_stab = self.nabla_stable(T, rho, L, r, m)
            #define each coefficent
            A = 1
            B = (U/lm**2)
            C = (U**2 * omega/(lm**3))
            D = (U/lm**2)*(nab_ad - nab_stab)
            
            #root finding
            eta_coefficents =[A,B,C,D]
            
            eta_roots = np.roots(eta_coefficents)
            #select the first  real root as the one to solve nabla_star
            
            eta_real = [root.real for root in eta_roots]
            #select the minimum root
            eta_star= np.min(eta_real)
            nabla_star = (eta_star**2) * (U*omega/lm)*eta_star + nab_ad
            return nabla_star
    def nabla_stable(self,T,rho,L,r,m):
            coef = (3/(64 * np.pi * self.stefan))
            grav = self.gravity(m,r)
            scale_height = (self.k_B* T)/(self.mu * self.M_h *grav)
            return (coef * scale_height * L)/(r**2 * T**4)
        #___define density and pressure___
        #methods to calculate rho(P,T) and P(rho,T)
    def rho(self,P,T):
            coef = (self.mu* self.M_h/(self.k_B * T))
            return (P - (4*self.stefan*T**4)/(3*self.c))*coef
    def pressure(self,T,rho):
            t4 = (4*self.stefan/(3*self.c)) * T**4
            t1 = (self.rho * self.k_B/(self.mu*self.M_h))*T
            return t4 -t1
    def euler(self):
            N = int(1e3)
            Mass = np.linspace(1e-5*self.M0, self.M0,num=N)
            #initize each parameter as an array of zeros
            rho,T,P,r,L,m = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),
            #give them inistal conditions
            rho[0],T[0],P[0],r[0],L[0],m[0] =self.rho0,self.T0,self.P0,self.R0,self.L0,self.M0
            #start from the surface and integreate inside
            
            #same for the nablas
            nabla, nabla_ad, nabla_stable = np.zeros(N),np.zeros(N),np.zeros(N)
            nabla_ad[0], nabla_stable[0] =self.nabla_ad(T[0],rho[0],P[0]), self.nabla_stable(T[0],rho[0],L[0],r[0],m[0])
            for i in range(N-1):
                
                #first make sure m does get too small
                if m[i] < 1e-4:
                    print("Minimum Mass Threshold")
                    break
                else:
                    #euler time
                   # kappa = self.opacity(self.opacity_file, T[i],rho[i])
                    #calculate nablas
                    nabla_stable[i] = self.nabla_stable(T[i],rho[i],L[i],r[i],m[i])
                    nabla_ad[i] = self.nabla_ad(T[i],rho[i],P[i])
                    #temperature gradient check
                    if nabla_stable[i] < nabla_ad[i]:
                        self.ledoux = True
                    else:
                        self.ledoux = False
                    if self.ledoux: #if true
                    #store the stable nabla i think
                        nabla[i] = nabla_stable[i]
                        #calculate what way the stepsize goes
                        dt = self.dtdm(T[i],r[i],rho[i],L[i])
                    elif self.ledoux == False:
                        nabla[i] =self.nabla_star(T[i], rho[i], P[i], m[i], r[i], L[i])
                        #avoid diviing by zero
                        dt = (nabla[i])*T[i]/(1e-10 + P[i]) * self.dpdm(m[i],r[i])
                    
                    #calulate adaptive timestep FIX FIX FIX
                    dm = Mass[i+1] - Mass[i]
                    #subtract since we're starting form surface
                    r[i+1] = r[i] - self.drdm(r[i],rho[i])
                    T[i+1] = T[i] - dt*dm #(dt/dm * dm)
                    L[i+1] = L[i] - self.dldm(T[i],rho[i])* dm
                    P[i+1] = P[i] - self.dpdm(m[i],r[i])*dm
                    rho[i+1] = self.rho(P[i],T[i])
                    
                return rho,T,P,r,L,Mass,nabla,nabla_stable, nabla_ad
    def plotting(self,rho,T,P,r,L,mass):
            plt.figure(figsize=(10, 6))

            plt.subplot(2, 2, 1)
            plt.plot(mass / self.M_sun, r/ self.R_sun)
            plt.xlabel('Mass coordinate [M/M_sun]')
            plt.ylabel('Radius [R/R_sun]')
            plt.title('Radius vs Mass')

            plt.subplot(2, 2, 2)
            plt.plot(mass / self.M_sun, P/self.P0)
            plt.xlabel('Mass coordinate [M/M_sun]')
            plt.ylabel('Pressure [Pa]')
            plt.title('Pressure vs Mass')

            plt.subplot(2, 2, 3)
            plt.plot(mass / self.M_sun, L / self.L0)
            plt.xlabel('Mass coordinate [M/M_sun]')
            plt.ylabel('Luminosity [L/L_sun]')
            plt.title('Luminosity vs Mass')

            plt.subplot(2, 2, 4)
            plt.plot(mass / self.M_sun, T,self.T0)
            plt.xlabel('Mass coordinate [M/M_sun]')
            plt.ylabel('Temperature [K]')
            plt.title('Temperature vs Mass')

            plt.tight_layout()
            plt.show()

                    
                    
                    
            
current_directory = '/home/karan/Documents/UvA/Computational Astrophysics/stellar-structure/'
opacity = os.path.join(current_directory,'opacity.txt')
epsilon = os.path.join(current_directory,'epsilon.txt')

if __name__ == "__main__":
    makestar = MESA(opacity,epsilon)
    #makestar.opacity(opacity,1e5,1e5)
    rho,T,P,r,L,Mass,nabla,nabla_stable,ArithmeticErrornabla_ad = makestar.euler()
    
    makestar.plotting(rho,T,P,r,L,Mass)
    
    


        