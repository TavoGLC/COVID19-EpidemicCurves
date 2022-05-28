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

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    Axes.spines['bottom'].set_visible(True)
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
# Visualization functions
###############################################################################

def ProcessSeries(series):
    
    adata = (series-series.min())/(series.max()-series.min())
    b, a = signal.butter(3, 0.05)
    cdata = signal.filtfilt(b, a,adata)
    cdata = (cdata-cdata.min())/(cdata.max()-cdata.min())
    
    return adata,cdata

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

def GetDayLengthDelta(J,lat):
    
    if J==366:
        J=365
    
    if J-1==0:
        prev = 365
    else:
        prev = J-1
        
    return GetDayLenght(J,lat)-GetDayLenght(prev,lat)
        
def GetDayLengthDelta2(J,lat):
    
    if J==366:
        J=365
    
    center = J-1
    forward = J
    
    if J-2==0:
        back = 365
    else:
        back = J-2
        
    return GetDayLenght(forward,lat)-2*GetDayLenght(center,lat)+GetDayLenght(back,lat)

###############################################################################
# Loading packages 
###############################################################################

data = pd.read_csv(r'/media/tavoglc/storage/backup/main2/main/mining/continental.csv',usecols=['date','cases','qry','lat','long','year','dayofyear','lengthofday'])
data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d')

###############################################################################
# Loading packages 
###############################################################################

unqrys = np.unique(data['qry'])

qrylat = [float(val[5:val.find(' ')]) for val in unqrys]
order = np.argsort(qrylat)

unqrys = unqrys[order]

years = np.unique(data['year'])

localGroup = data.groupby('qry')
xdata = np.array(localGroup['long'].mean())
ydata = np.array(localGroup['lat'].mean())
sizedata = 2*np.array(localGroup['cases'].mean())

def MakePlots(x,y,s,years,querys):
    
    xdata = x
    ydata = y
    sizedata = s
    
    for ii,qy in enumerate(querys):
        
        latData = data[data['qry']==qy]
        disc2 = latData[latData['year']==2022]
        
        if latData.shape[0]>300 and disc2.shape[0]>60:
            
            fig = plt.figure(figsize=(40,30))
            gs = gridspec.GridSpec(nrows=7, ncols=6) 
            
            axs0 = fig.add_subplot(gs[:,0:3])
            
            axs0.scatter(xdata,ydata,s=sizedata,color='blue',alpha=0.5)
            axs0.set_xlim([-185,-5])
            axs0.set_ylim([-75,75])
            ImageStyle(axs0)
            
            axs = [fig.add_subplot(gs[k,3:6]) for k in range(7)]
            
            xldata = np.array(latData['long'].mean())
            yldata = np.array(latData['lat'].mean())
            sizeldata = 250
            
            axs0.scatter(xldata,yldata,s=sizeldata,color='red',alpha=0.95)
            
            for j,yr in enumerate(years):
            
                cData = latData[latData['year']==yr]
                seriesData = cData.groupby('lengthofday')['cases'].mean()    
                cindex = np.array(seriesData.index)
                series = np.array(seriesData)
                
                if len(series)>20:
                    adata,cdata = ProcessSeries(series)
                else:
                    adata,cdata = np.zeros(len(cindex)),np.zeros(len(cindex))
                    
                axs[j].plot(cindex,adata,color='black',alpha=0.8,label=str(yr))
                axs[j].plot(cindex,cdata,color='blue')
                axs[j].fill_between(cindex,cdata, color='blue', alpha=0.15)
                axs[j].set_ylabel('Casos \n Normalizados',fontsize=plotfontsize)
                axs[j].legend(loc=1,fontsize=plotfontsize)
                Left(axs[j])
                
            seriesData = latData.groupby('lengthofday')['cases'].mean()
            cindex = np.array(seriesData.index)
            series = np.array(seriesData)
            
            meandaylength = [GetDayLenght(dy,yldata) for dy in range(366)]
            meandaylength = np.sort(meandaylength)
            
            xtime,ytime = np.array(meandaylength[2::]),np.diff(meandaylength,n=2) 
            fdata = np.abs(NDFT(xtime,ytime))**2
            fdata = (fdata-fdata.min())/(fdata.max()-fdata.min())
            
            peaklocations,_ = find_peaks(fdata, height=2*np.mean(fdata),distance=20)
            peaktimes = [meandaylength[val] for val in peaklocations]
            
            ytime = (ytime - ytime.min())/(ytime.max()-ytime.min())
            
            adata,cdata = ProcessSeries(series)
            
            timeseries = latData.groupby('lengthofday')['dayofyear'].mean()
            xtime0,ytime0 = np.array(timeseries.index),np.array(timeseries)
            
            axs[3].plot(cindex,adata,color='black',alpha=0.8,label = 'Datos Completos')
            axs[3].plot(cindex,cdata,color='blue')
            axs[3].fill_between(cindex,cdata, color='blue', alpha=0.15)
            axs[3].set_ylabel('Casos \n Normalizados',fontsize=plotfontsize)
            axs[3].legend(loc=1,fontsize=plotfontsize)
            Left(axs[3])
            
            
            axs[4].plot(xtime0,ytime0,'ko',label='Duración del Día',alpha=0.5)
            axs[4].vlines(x=peaktimes,ymin=0,ymax=365, color='red',alpha=0.75)
            axs[4].set_ylabel('Día del año',fontsize=plotfontsize)
            Left(axs[4])
            
            axs[5].plot(xtime,ytime,'k-',label=r'$\dfrac{\mathrm{d}^2 Duración del Día}{\mathrm{d}t^2}$')
            axs[5].set_ylabel('Doble \n Diferencia \n Normalizados',fontsize=plotfontsize)
            Left(axs[5])
            
            axs[6].plot(xtime,fdata,'k-')
            axs[6].set_ylabel('Transformada \n de Fourier',fontsize=plotfontsize)
            axs[6].set_xlabel('Duración del día',fontsize=plotfontsize)
            PlotStyle(axs[6])
            
            for ax in axs:
                ax.vlines(x=peaktimes,ymin=0,ymax=1.1, color='red',alpha=0.75)
        
            fig.suptitle('Curvas Epidémicas por Localización Geográfica', fontsize=2*plotfontsize)
            plt.savefig('fig' + str(ii) + '.png',bbox_inches='tight')
            fig.clear()
            plt.close(fig)
        
            del latData,cData,meandaylength,xtime,ytime,seriesData,timeseries,xtime0,ytime0,series,adata,cdata,fdata
            gc.collect()
        
MakePlots(xdata,ydata,sizedata,years,unqrys)
