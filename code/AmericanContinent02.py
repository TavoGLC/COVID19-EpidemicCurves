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
    Axes.xaxis.set_tick_params(labelsize=60)
    Axes.yaxis.set_tick_params(labelsize=60)
     
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
    Axes.xaxis.set_tick_params(labelsize=60)
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
fsize = 60

for k in range(50):
    
    breaks = (k+1)*10
    bounds = np.linspace(data['lat'].min(),data['lat'].max(),num=breaks)
    colors = [plt.cm.viridis(val) for val in np.linspace(0,1,num=breaks)]  
    
    years = np.unique(data['year'])
    
    yeardata = []
    for j,yr in enumerate(years):
        latdata = []
        for i in range(breaks-1):
             
             latData = data.query('lat > ' + str(bounds[i]) + '& lat < ' + str(bounds[i+1]))
             latData = latData[latData['year']==yr]
             latData = latData.groupby('lengthofday')['cases'].mean()
             
             if latData.shape[0]>50:
                 cdata = np.array(latData)
                 #b, a = signal.butter(3, 0.05)
                 #cdata = signal.filtfilt(b, a,adata)
                 cdata = bounds[i] + 2.5*(cdata-cdata.min())/(cdata.max()-cdata.min())
                 latdata.append([latData.index,cdata])
                 
        yeardata.append(latdata)
        
    fig = plt.figure(figsize=(80,60))
    gs = gridspec.GridSpec(nrows=1, ncols=6) 
    
    axs0 = fig.add_subplot(gs[0:3])
    localGroup = data.groupby('qry')
        
    xdata = np.array(localGroup['long'].mean())
    ydata = np.array(localGroup['lat'].mean())
    sizedata =  2*np.array(localGroup['cases'].mean())
    colordata = localGroup['lengthofday'].mean()
        
    axs0.scatter(xdata,ydata,c=colordata,s=sizedata,cmap='viridis')
    axs0.set_xlim([-185,-5])
    axs0.set_ylim([-75,75])
    axs0.text(-50,60,'Casos promedio \n en el continente',fontsize=fsize)
    ImageStyle(axs0)
        
    axs = [fig.add_subplot(gs[3]),fig.add_subplot(gs[4]),fig.add_subplot(gs[5])]
    
    for j,val in enumerate(yeardata):
        for i,sal in enumerate(val):
            xvals = np.array(sal[0])
            xvals = (xvals - xvals.min())/(xvals.max()-xvals.min())
            axs[j].plot(xvals,sal[1],color=colors[i])
            axs[j].set_title('Casos normalizados por \n latitud ' +'('+ str(years[j])+')',fontsize=fsize)
            
            if j==0:
                PlotStyle(axs[j])
                axs[j].set_ylabel('Latitud',fontsize=fsize)
            else:
                Bottom(axs[j])
            
            axs[j].set_xlabel('Duración normalizada \n del día',fontsize=fsize)
            
    fig.subplots_adjust(wspace=0.05, hspace=0.05,top=0.95)
    plt.savefig('fig' +str(k)+'.png',bbox_inches='tight')
    plt.close()
        
