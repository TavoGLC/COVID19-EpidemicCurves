#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2022 Octavio Gonzalez-Lugo 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo

"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks

###############################################################################
# Visualization functions
###############################################################################
plotfontsize = 30

def PlotStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=plotfontsize)
    Axes.yaxis.set_tick_params(labelsize=plotfontsize)
     
def Left(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.yaxis.set_tick_params(labelsize=plotfontsize)
    
def ImageStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

###############################################################################
# Loading packages 
###############################################################################

def NDFT(Xdata, Ydata):
    N = len(Ydata)
    k = -(N // 2) + np.arange(N)
    return np.dot(Ydata, np.exp(2j * np.pi * k * Xdata[:, np.newaxis]))

def GetDayLenght(J,lat):
    #CERES model  Ecological Modelling 80 (1995) 87-95
    phi = 0.4093*np.sin(0.0172*(J-82.2))
    coef = (-np.sin(np.pi*lat/180)*np.sin(phi)-0.1047)/(np.cos(np.pi*lat/180)*np.cos(phi))
    ha =7.639*np.arccos(np.max([-0.87,coef]))
    return ha

###############################################################################
# Loading packages 
###############################################################################

data = pd.read_csv(r'/media/tavoglc/storage/backup/main2/main/mining/continental.csv')
data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d')

fsize = (34,24)

###############################################################################
# Loading packages 
###############################################################################

breaks = 300
minlat,maxlat = data['lat'].min(),data['lat'].max()
lats = np.linspace(minlat,maxlat,num=breaks)
colors = [plt.cm.viridis(val) for val in np.linspace(0,1,num=breaks)]

localGroup = data.groupby('qry')
xdata = np.array(localGroup['long'].mean())
ydata = np.array(localGroup['lat'].mean())
sizedata = 1.5*np.array(localGroup['cases'].mean())

fig,axs = plt.subplots(1,2,figsize=fsize)

axs[0].scatter(xdata,ydata,s=sizedata,color='blue',alpha=0.5)
axs[0].set_xlim([-185,-5])
axs[0].set_ylim([-75,75])
ImageStyle(axs[0])
axs[0].set_title('Promedio diario de casos reportados de \n COVID-19 en el continente americano',fontsize=plotfontsize)

for i,sal in enumerate(lats):
    
    daydur = [GetDayLenght(j,sal) for j in range(1,366)]
    
    durtoday = {}
    
    for k,val in enumerate(daydur):
        durtoday[val]=k
    
    daydur = np.sort(daydur)
    
    xdata,ydata = np.array(daydur[2::]),np.diff(daydur,n=2) 
    fdata = np.abs(NDFT(xdata,ydata))**2
    fdata = 5*(fdata-fdata.min())/(fdata.max()-fdata.min())
    xdata = (xdata-xdata.min())/(xdata.max()-xdata.min())
    axs[1].plot(xdata,fdata+sal,color=colors[i])
    Left(axs[1])
    axs[1].set_title('Cambios en la duración del día \n en el continente americano por latitud',fontsize=plotfontsize)

plt.savefig('figure01.png',bbox_inches='tight')
plt.close()

###############################################################################
# Loading packages 
###############################################################################

def GetBounds(latitudes,factor,sign):
    
    containera = []
    containerb = []
    
    for val in latitudes:
        
        daydur = [GetDayLenght(j,val) for j in range(1,366)]
        
        durtoday = {}
        
        for k,val in enumerate(daydur):
            durtoday[val]=k
        
        daydur = np.sort(daydur)
        
        xdata,ydata = np.array(daydur[2::]),np.diff(daydur,n=2) 
        fdata = np.abs(NDFT(xdata,ydata))**2
        peaklocations,_ = find_peaks(fdata, height=np.mean(fdata),distance=20)
        npeaks = len(peaklocations)//factor
        
        peaktimes = [daydur[val] for val in peaklocations]
        difs0 = [(val-peaktimes[sign*npeaks])**2 for val in daydur]
        sortd = np.argsort(difs0)
        
        deltaLS = max([durtoday[daydur[val]] for val in sortd[0:5]])
        deltaLE = min([durtoday[daydur[val]] for val in sortd[0:5]])
        
        containera.append(deltaLS)
        containerb.append(deltaLE)
        
    return containera,containerb

latsup = np.linspace(7,maxlat,num=250)
latsdown = np.linspace(-7,minlat,num=250)

lats = [latsup,latsdown]

def MakeWavePlot(lats):
    
    latsup,latsdown = lats
    
    fig,axs = plt.subplots(1,2,figsize=fsize)

    localGroup = data.groupby('qry')
    xdata = np.array(localGroup['long'].mean())
    ydata = np.array(localGroup['lat'].mean())
    sizedata = np.array(localGroup['cases'].mean())
    
    axs[0].scatter(xdata,ydata,s=sizedata,color='blue',alpha=0.5)
    axs[0].set_xlim([-185,-5])
    axs[0].set_ylim([-75,75])
    ImageStyle(axs[0])
    axs[0].set_title('Promedio diario de casos reportados de \n COVID-19 en el continente americano.',fontsize=plotfontsize)
    
    for k,fac in enumerate([3,4,5]):
        
        a1,b1 = GetBounds(latsup,fac,1)
        a2,b2 = GetBounds(latsdown,fac,-1)    
            
        axs[1].scatter(a1,latsup,color='blue',alpha=2*(k+1)*0.05)
        axs[1].scatter(b1,latsup,color='blue',alpha=2*(k+1)*0.05)
        axs[1].scatter(a2,latsdown,color='blue',alpha=2*(k+1)*0.05)
        axs[1].scatter(b2,latsdown,color='blue',alpha=2*(k+1)*0.05)
            
        a1,b1 = GetBounds(latsup,fac,-1)
        a2,b2 = GetBounds(latsdown,fac,1)
            
        axs[1].scatter(a1,latsup,color='red',alpha=2*(k+1)*0.05)
        axs[1].scatter(b1,latsup,color='red',alpha=2*(k+1)*0.05)
        axs[1].scatter(a2,latsdown,color='red',alpha=2*(k+1)*0.05)
        axs[1].scatter(b2,latsdown,color='red',alpha=2*(k+1)*0.05)
        
        axs[1].set_xlim([0,365])
        axs[1].set_xlabel('Día del año',fontsize=plotfontsize)
        PlotStyle(axs[1])
        axs[1].set_title('Intervalos de tiempo con una mayor \n tasa de cambio en la duración del día.',fontsize=plotfontsize)
        
        
MakeWavePlot(lats)

plt.savefig('figure02.png',bbox_inches='tight')
plt.close()
