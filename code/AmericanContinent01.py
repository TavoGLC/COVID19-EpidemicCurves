#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:38:03 2022

@author: tavoglc
"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal

###############################################################################
# Visualization functions
###############################################################################

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
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)
    
def Bottom(Axes): 
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
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.set_yticks([])
    
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

data = pd.read_csv(r'/media/tavoglc/storage/backup/main2/main/mining/continental.csv')
data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d')

###############################################################################
# Loading packages 
###############################################################################

breaks = 150
bounds = np.linspace(data['lat'].min(),data['lat'].max(),num=breaks)
colors = [plt.cm.viridis(val) for val in np.linspace(0,1,num=breaks)]
plt.figure(figsize=(15,20))    

years = np.unique(data['year'])

yeardata = []
for j,yr in enumerate(years):
    latdata = []
    for i in range(breaks-1):
         
         latData = data.query('lat > ' + str(bounds[i]) + '& lat < ' + str(bounds[i+1]))
         latData = latData[latData['year']==yr]
         latData = latData.groupby('dayofyear')['cases'].mean()
         
         if latData.shape[0]>50:
             adata = np.array(latData)
             b, a = signal.butter(3, 0.1)
             cdata = signal.filtfilt(b, a,adata)
             cdata = bounds[i] + 10*(cdata-cdata.min())/(cdata.max()-cdata.min())
             latdata.append([latData.index,cdata])
             
    yeardata.append(latdata)

counter = 0

for k in range(1,366):
    
    
    fig = plt.figure(figsize=(15,7))
    gs = gridspec.GridSpec(nrows=1, ncols=6) 
    
    axs0 = fig.add_subplot(gs[0:3])
    
    localData = data[data['dayofyear']==k]
    localGroup = localData.groupby('qry')
    
    xdata = np.array(localGroup['long'].mean())
    ydata = np.array(localGroup['lat'].mean())
    sizedata =  np.array(localGroup['cases'].mean())*0.75
    colordata = localGroup['lengthofday'].mean()
    
    axs0.scatter(xdata,ydata,c=colordata,s=sizedata,cmap='viridis')
    axs0.set_xlim([-185,-5])
    axs0.set_ylim([-75,75])
    axs0.text(-60,60,'Casos en el continente',fontsize=10)
    axs0.text(-60,57,'Día del Año = ' +str(k),fontsize=10)
    ImageStyle(axs0)
    
    axs = [fig.add_subplot(gs[3]),fig.add_subplot(gs[4]),fig.add_subplot(gs[5])]
    
    for j,val in enumerate(yeardata):
        for i,sal in enumerate(val):
            axs[j].plot(sal[0],sal[1],color=colors[i])
            axs[j].set_xlim([0,365])
        
        axs[j].vlines(k,data['lat'].min(),data['lat'].max(),color='red')
        axs[j].set_title('Casos normalizados por \n latitud ' +'('+ str(years[j])+')',fontsize=10)
        
        if j==0:
            PlotStyle(axs[j])
            axs[j].set_ylabel('Latitud',fontsize=10)
        else:
            Bottom(axs[j])
        
        axs[j].set_xlabel('Día del Año',fontsize=10)
        
    fig.subplots_adjust(wspace=0.05, hspace=0.05,top=0.95)
    plt.savefig('fig' + str(counter) + '.png',bbox_inches='tight')
    counter = counter + 1
    plt.close()
    
