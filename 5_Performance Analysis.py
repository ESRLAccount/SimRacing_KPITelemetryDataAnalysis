# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:50:04 2022

@author: Fazilat
based on the result of clustring, finding the best and worst groups, plotting the result, for sifferent cluster for different features
"""

import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil
import numpy as np
import csv
from FileOperation import *

def FFT_plot (df1,df2,featurename):
    from scipy.fftpack import fft, ifft
    
    
    fig, ax = plt.subplots(4)
    
    fig.set_figheight(15)
    fig.set_figwidth(15)
    

    ax[0].plot(df1['Calc Lap Time'], df1[featurename],'b')
    ax[0].plot(df2['Calc Lap Time'], df2[featurename],'r')
    ax[0].set(xlabel='Lap Time')
    ax[0].set(ylabel=featurename)

    #ax[0].plot(df1['Calc Lap Time'],df1[featurename]-df2[featurename],color='pink')

    N=len (df1)


    X = fft(df1[featurename].to_numpy())
    N = len(X)
    n = np.arange(N)
    # get the sampling rate
    sr =20
    T = N/sr
    freq = n/T 
    
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    

    ax[1].plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    ax[1].set(xlabel='Freq (Hz)')
    ax[1].set(ylabel='FFT Amplitude for MIN_' + featurename)
    
    
    signal = df1[featurename].to_numpy()
    signal = signal-signal.mean()
    fft_signal = np.abs(np.fft.fft(signal))
    ax[3].plot(fft_signal[0:int(len(signal)/2.)],color='g')




    N=len (df2)


    X = fft(df2[featurename].to_numpy())
    N = len(X)
    n = np.arange(N)
    # get the sampling rate
    sr =20
    T = N/sr
    freq = n/T 
    
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    
    ax[2].plot(f_oneside, np.abs(X[:n_oneside]), 'r')
    ax[2].set(xlabel='Freq (Hz)')
    ax[2].set(ylabel='FFT Amplitude for MAX_' + featurename)
    
    
    plt.show()

"""
    th_list = np.linspace(0,0.02,1000)
    th_list = th_list[0:len(th_list)]
    p_values = []
    corr_values = []
    th_example_value = 0.10
    signal = df1[featurename].to_numpy()
    signal = signal-signal.mean()
    fft_signal = np.abs(np.fft.fft(signal))
    
    example_signal = filter_signal(th_example_value)
    x=df1['Calc Lap Time'].to_numpy()
    plt.plot(x,signal,color='navy',label='Original Signal')
    plt.plot(x,example_signal,color='firebrick',label='Filtered signal (Th=%.2f)'%(th_example_value))
    plt.plot(x,signal-example_signal,color='darkorange',label='Difference')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()

"""

def Plot_MinandMax (df_stats,df) :

    #Best lap times. Best lap times by performance level.
    #the best lap is in the first row
    min_laptime=df_stats.iloc[0][2]
    df_min=df.loc[(df.LapTime==min_laptime)]
    df_MinTelemetry=Make_AVGLapRowData(df_min)

    #Worst lap times. Best lap times by performance level.
    #the worst lap is in the last row of the dataframe
    max_laptime=df_stats.iloc[len(df_stats)-1][1]
    df_max=df.loc[(df.LapTime==max_laptime)]
    df_MaxTelemetry=Make_AVGLapRowData(df_max)

    #Plotting best and worst lap data
    title='Best vs. Worst Performance for ' + Track_name
    plot_filename=outputdata_path +'BvsW.png'

    two_plot(df_MinTelemetry, df_MaxTelemetry, 'Best LapTime', 'Worst LapTime', 'r', 'c', title, 1500, plot_filename)
 
    FFT_plot (df_MinTelemetry,df_MaxTelemetry,'BRAKE')

 
def Plot_allClusters(df_stats,df):
    fig, ax = plt.subplots(6)
    for i in range (len(df_stats)):
        
        min_laptime=df_stats.iloc[i][1]

        df_min=df.loc[(df.LapTime==min_laptime)]
        print (df_min)
        df_MinTelemetry=Make_AVGLapRowData(df_min)

        #Plotting best and worst lap data
        title='Lap Performance for different groups in ' + Track_name
        plot_filename=outputdata_path +'AllGroups.png'
        color=colors[i]
        Make_plot(fig,ax,df_MinTelemetry, df_stats.iloc[i][1], color,  title, 1500, plot_filename)
    plt.show()  

def ReadandGetRowData (inputfile, Lap_No):
    df= pd.read_csv(inputfile)   
    """
    df_lap1=df.loc[(df.Lap=='Lap 1')]
    outlap_distance=df_lap1.iloc[0]['Distance']

    #print (outlap_distance)
    df=df.loc[(df.Lap==Lap_No)]

    Lap_NoDigit = [int(x) for x in Lap_No.split() if x.isdigit()] 
   # print (df['Distance'])
    Lap_DivisionFactor=Lap_NoDigit[0]-1
    #if Lap_DivisionFactor>0:
    LapDistance=(df['Distance']-(Lap_DivisionFactor*TRACK_LENGHT))-outlap_distance
    print (LapDistance)
        #print (Lap_DivisionFactor,first,df.iloc[0]['Distance'])
        #print ((df['Distance']-outlap_distance)%Lap_DivisionFactor)

    df.insert(0, 'LapDistance', LapDistance)"""
    return (df)


def Make_AVGLapRowData(df):
    #if len(df)==1:
        PID=df.iloc[0]['PID']
        Lap_No=df.iloc[0]['Lap']
        print (PID, Lap_No)
        Pid_file=Rowdata_path+PID+'.csv'
        df= pd.read_csv(Pid_file)  
        df=df.loc[(df.Lap==Lap_No)]
        #ReadandGetRowData(Pid_file, Lap_No)
    #else:
        """for i in range(1,len(df)):
            PID=df.iloc[i]['PID']
            Lap_No=df.iloc[i]['Lap']
            Pid_file=Rowdata_path+PID+'.csv'
            df_AVG=pd.concat(df_AVG,ReadandGetRowData(Pid_file, Lap_No))
         """   
        return df       

def Make_plot(fig,ax,telemetry_1, driver_1, color_1,  title, dpi_val, file_name):
    # Plotting the data using subplots; one for speed, throttle, and brake

    #fig.suptitle(title)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    
    ax[0].plot(telemetry_1['Corr Lap Dist'], telemetry_1['SPEED'], linewidth=0.6,label=driver_1, color=color_1)


    #ax[0].plot(ds['lapdistance'], ds['sectorname'], label='sector', color='g')
    ax[0].plot()
    ax[0].set(ylabel='Speed [kph]')
    ax[0].legend(loc='right')

    ax[1].plot(telemetry_1['Corr Lap Dist'], telemetry_1['THROTTLE'], linewidth=0.6, label=driver_1, color=color_1)

    ax[1].set(ylabel='Throttle [%]')

    ax[2].plot(telemetry_1['Corr Lap Dist'], telemetry_1['BRAKE'], linewidth=0.6, label=driver_1, color=color_1)

    ax[2].set(ylabel='Brake [%]')
    
    ax[3].plot(telemetry_1['Corr Lap Dist'], telemetry_1['STEERANGLE'], linewidth=0.6, label=driver_1, color=color_1)

    ax[3].set(ylabel='Steerangle [deg]')
    
    ax[4].plot(telemetry_1['Corr Lap Dist'], telemetry_1['G_LAT'], linewidth=0.6, label=driver_1, color=color_1)

    ax[4].set(ylabel='G_lat [m/s2]')
    
    ax[5].plot(telemetry_1['Corr Lap Dist'], telemetry_1['RPMS'], linewidth=0.6, label=driver_1, color=color_1)

    ax[5].set(ylabel='RPMS [1/min]')
    #ax[5].plot(telemetry_1['Corr Lap Dist'], telemetry_1['Oversteer'],linewidth=0.6, label=driver_1, color=color_1)

    #ax[5].set(ylabel='Oversteer')
    
    # Hide x-labels and tick labels for top plots and y ticks for right plots.
    for j in range (6):
        for i in range ( len(ds_S)):
            ax[j].axvline(x=ds_S.iloc[i]['lapdistance'],color='k', linestyle='--', alpha=0.3,linewidth=0.5)
            #ax[j].text(10.1,0,ds_S.iloc[i]['sectorname'],rotation=90)
        for i in range ( len(ds_T)):
             ax[j].axvline(x=ds_T.iloc[i]['lapdistance'],color='k', linestyle='--', alpha=0.3,linewidth=0.5)
         

    for a in ax.flat:
        a.label_outer()

    #plt.savefig(file_name, dpi=dpi_val)

    

    
def two_plot(telemetry_1, telemetry_2, driver_1, driver_2, color_1, color_2, title, dpi_val, file_name):
    # Plotting the data using subplots; one for speed, throttle, and brake
    fig, ax = plt.subplots(5)
    fig.suptitle(title)
    
    fig.set_figheight(15)
    fig.set_figwidth(15)

    ax[0].plot(telemetry_1['Corr Lap Dist'], telemetry_1['SPEED'], linewidth=0.9, label=driver_1, color=color_1)
    ax[0].plot(telemetry_2['Corr Lap Dist'], telemetry_2['SPEED'],linewidth=0.9, label=driver_2, color=color_2)   
    ax[0].set(ylabel='Speed [kph]')
    ax[0].legend(loc='lower right')

    ax[1].plot(telemetry_1['Corr Lap Dist'], telemetry_1['THROTTLE'], linewidth=0.9,label=driver_1, color=color_1)
    ax[1].plot(telemetry_2['Corr Lap Dist'], telemetry_2['THROTTLE'], linewidth=0.9,label=driver_2, color=color_2)
    ax[1].set(ylabel='Throttle [%]')

    ax[2].plot(telemetry_1['Corr Lap Dist'], telemetry_1['BRAKE'], linewidth=0.9,label=driver_1, color=color_1)
    ax[2].plot(telemetry_2['Corr Lap Dist'], telemetry_2['BRAKE'], linewidth=0.9,label=driver_2, color=color_2)
    ax[2].set(ylabel='Brakes [%]')
    
    ax[3].plot(telemetry_1['Corr Lap Dist'], telemetry_1['STEERANGLE'],linewidth=0.9, label=driver_1, color=color_1)
    ax[3].plot(telemetry_2['Corr Lap Dist'], telemetry_2['STEERANGLE'],linewidth=0.9, label=driver_2, color=color_2)
    ax[3].set(ylabel='Steerangle [%]')
    
    ax[4].plot(telemetry_1['Corr Lap Dist'], telemetry_1['G_LAT'], linewidth=0.9, label=driver_1, color=color_1)
    ax[4].plot(telemetry_2['Corr Lap Dist'], telemetry_2['G_LAT'], linewidth=0.9, label=driver_2, color=color_2)

    ax[4].set(ylabel='G_lat [m/s2]')
    
   # ax[5].plot(telemetry_1['Corr Lap Dist'], telemetry_1['RPMS'], linewidth=0.6, label=driver_1, color=color_1)
   # ax[5].plot(telemetry_2['Corr Lap Dist'], telemetry_2['RPMS'], linewidth=0.6, label=driver_2, color=color_2)


   # ax[5].set(ylabel='RPMS [1/min]')`
    
    
    for j in range (5):
        for i in range ( len(ds_S)):
            ax[j].axvline(x=ds_S.iloc[i]['lapdistance'],color='k', linestyle='--', alpha=0.3)
            #ax[j].text(10.1,0,ds_S.iloc[i]['sectorname'],rotation=90)
        for i in range ( len(ds_T)):
             ax[j].axvline(x=ds_T.iloc[i]['lapdistance'],color='k', linestyle='--', alpha=0.3)

    # Hide x-labels and tick labels for top plots and y ticks for right plots.
    for a in ax.flat:
        a.label_outer()

    #plt.savefig(file_name, dpi=dpi_val)

    

Track_name='brands_hatch, brands_hatch'  


collection='C5/'

colors = ['r','c','y','b','k']

inputdata_path="../../../2_DATA/OUTPUT/"+ collection+"ClusterResults- C2C3C5/" 
Rowdata_path="../../../2_DATA/OUTPUT/ "+ collection+ "RowData/"

outputdata_path="../../../2_DATA/OUTPUT/" + collection+ "Analysis/"


inputdata_sectorpath="../../../2_DATA/Included Participants - In-lab/"

files_list=['_Channel Report_AVG','_CL_Sector_']#'_Time Report_Only',]#,'_Time Report_OnlySummary','_Time Report_DetailSummary']


track_path="../../../2_DATA/OUTPUT/"+ collection+ "LapStats/"
inputsector_path="../../../2_DATA/INPUT/"
SectorInfor_file="../../../2_DATA/"+'/TrackInfo/brands_hatch, brands_hatch.csv'
ds = pd.read_csv(SectorInfor_file,header=0,parse_dates=[0])  

ds_S=ds[ds.iloc[:,3]=='S']
ds_T=ds[ds.iloc[:,3]=='T']      
           
####Creating output directory for the results
if not os.path.exists(outputdata_path):
  os.makedirs(outputdata_path)
else:
    #remove folder withh all of its files
  shutil.rmtree(outputdata_path)


trackname_file=track_path+'Trackname.csv'
with open(trackname_file, 'r') as f:
    df_tn = list(csv.reader(f, delimiter=","))
df_tn = np.array(df_tn)

df_t= pd.read_csv(trackname_file)

#open each file related to the track


for i in range(1,len(df_tn)):
  trackname=df_tn[i][0] 
  out_path=outputdata_path+trackname+'/'
  path=inputdata_path+trackname+'/'
  ####Creating output directory for the results
  if not os.path.exists(out_path):
      os.makedirs(out_path)
  else:
        #remove folder withh all of its files
    shutil.rmtree(out_path)
 
  for k in range (0, len(files_list))   :
    
    Filename_term=files_list[k]
    print ('trackname:', trackname, path)
    
    Excluded_Term=''

    listof_Files = FileOperation.getListOfFiles(path,Filename_term,Excluded_Term)
    output_file=outputdata_path +trackname + '/PerfromanceAnalysis.csv'
    output_file2=outputdata_path +trackname + '/PerfromanceAnalysis_Turn.csv'
    df_res=pd.DataFrame()

    for j in range (len (listof_Files)):
        inputfile_name=listof_Files[j]
        df= pd.read_csv(inputfile_name)
        print (inputfile_name)
        df_stats=df.groupby(['cluster_label'])
        df_res= df_stats.mean()
        if k==1:
            """
            df_stats=df_stats.agg({'LapTime':['mean','min', 'max','std','median'], 'SectorTime':['mean','min', 'max','std','median'],'Lap':'size',
            'Trail Steer phase_Avg':['mean','min', 'max','std','median'],
            'Trail braking phase_Avg':['mean','min', 'max','std','median'], 
            'Brake_Startdis_Avg':['mean','min', 'max','std','median'],
            'Brake_Peakdis_Avg':['mean','min', 'max','std','median'],
            'Brake_Enddis_Avg':['mean','min', 'max','std','median'],
            'Brake_StartTime_Avg':['mean','min', 'max','std','median'],
            'Brake_PeakTime_Avg':['mean','min', 'max','std','median'],
            'Brake_EndTime_Avg':['mean','min', 'max','std','median'],
            'Brake_StartdisperSec_Avg':['mean','min', 'max','std','median'],
            'Trail brakingDistance1_Avg':['mean','min', 'max','std','median'],
            'Trail brakingDistance2_Avg':['mean','min', 'max','std','median'],
            'Trail brakingTime_Avg':['mean','min', 'max','std','median'],
            'Trail braking phase_Avg':['mean','min', 'max','std','median'],
            'Brake Zone length _Avg':['mean','min', 'max','std','median'],
            'THROTTLE _Startdis_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _Peakdis_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _Enddis_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _StartTime_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _PeakTime_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _EndTime_Avg':['mean','min', 'max','std','median'],
            'Throttle release duration_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _StartdisperSec_Avg':['mean','min', 'max','std','median'],
            'Steer_Startvalue_Avg':['mean','min', 'max','std','median'],
            'Steer_Peakvalue_Avg':['mean','min', 'max','std','median'],
            'Steer_Endvalue_Avg':['mean','min', 'max','std','median'],
            'Steer_StartTime_Avg':['mean','min', 'max','std','median'],
            'Steer_PeakTime_Avg':['mean','min', 'max','std','median'],
            'Steer_EndTime_Avg':['mean','min', 'max','std','median'],
            'Steer Zone length _Avg' :['mean','min', 'max','std','median'],
            'Steer_StartdisperSec_Avg':['mean','min', 'max','std','median'],
            'Trail SteerDistance1_Avg':['mean','min', 'max','std','median'],
            'Trail SteerDistance2_Avg':['mean','min', 'max','std','median'],
            'Trail SteerTime_Avg':['mean','min', 'max','std','median'] ,
            'Brake':['mean','min', 'max','std','median'],
            'THROTTLE':['mean','min', 'max','std','median']
             })
            
            """
            df_stats=df_stats.mean()
            
            """ df_stats=df_stats.agg({'LapTime':['mean','min', 'max','std','median'], 'SectorTime':['mean','min', 'max','std','median'],'Lap':'size',
            'Trail Steer phase_Avg':['mean','min', 'max','std','median'],
            'Trail braking phase_Avg':['mean','min', 'max','std','median'], 
            'Brake_StartTime_Avg':['mean','min', 'max','std','median'],
            'Brake_PeakTime_Avg':['mean','min', 'max','std','median'],
            'Brake_EndTime_Avg':['mean','min', 'max','std','median'],
            'Trail brakingTime_Avg':['mean','min', 'max','std','median'],
            'Brake Zone length _Avg':['mean','min', 'max','std','median'],
            'THROTTLE _StartTime_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _PeakTime_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _EndTime_Avg':['mean','min', 'max','std','median'],
            'Throttle release duration_Avg':['mean','min', 'max','std','median'],
            'THROTTLE _StartdisperSec_Avg':['mean','min', 'max','std','median'],
            'Steer_Startvalue_Avg':['mean','min', 'max','std','median'],
            'Steer_Peakvalue_Avg':['mean','min', 'max','std','median'],
            'Steer_Endvalue_Avg':['mean','min', 'max','std','median'],
            'Steer_StartTime_Avg':['mean','min', 'max','std','median'],
            'Steer_PeakTime_Avg':['mean','min', 'max','std','median'],
            'Steer_EndTime_Avg':['mean','min', 'max','std','median'],
            'Steer Zone length _Avg' :['mean','min', 'max','std','median'],
            'Steer_StartdisperSec_Avg':['mean','min', 'max','std','median'],
            'Trail SteerDistance1_Avg':['mean','min', 'max','std','median'],
            'Trail SteerDistance2_Avg':['mean','min', 'max','std','median'],
            'Trail SteerTime_Avg':['mean','min', 'max','std','median']     
             })"""
              
            
            x1=inputfile_name.find("_CL_Sector_")
            x2=inputfile_name.find(".csv")
            Sectorname=inputfile_name[x1+len("_CL_Sector_"):x2]
            
            if Sectorname.find("Turn"):
                 sectortype='Str'
            else:
                 sectortype='Turn'
            df_stats['sector']=Sectorname
            df_stats['sectortype']=sectortype    
            
            df_res = df_res.append(df_stats)
            df_res2 = df_res[df_res['sectortype'] =='Turn'].groupby(['cluster_label'])
            #df_res2=df_res.filter(sectortype='Turn', axis=1).groupby(['cluster_label'])
            #print (df_res)
            #df_res=df_res.agg({'Trail Steer phase_Avg':['mean','min', 'max','std','median']}) 
            df_turn=df_res2.mean()
            
            df_turn.to_csv(output_file, mode='a', index=True, header=True)
            #Plot_allClusters(df_stats,df)
            
            df_res2 = df_res[df_res['sectortype'] =='Str'].groupby(['cluster_label'])
            #df_res2=df_res.filter(sectortype='Turn', axis=1).groupby(['cluster_label'])
            #print (df_res)
            #df_res=df_res.agg({'Trail Steer phase_Avg':['mean','min', 'max','std','median']}) 
            df_str=df_res2.mean()
            
            df_str.to_csv(output_file, mode='a', index=True, header=True)
            
            df_res2 = df_res.groupby(['cluster_label'])
            #df_res2=df_res.filter(sectortype='Turn', axis=1).groupby(['cluster_label'])
            #print (df_res)
            #df_res=df_res.agg({'Trail Steer phase_Avg':['mean','min', 'max','std','median']}) 
            df_str=df_res2.mean()
            
            df_str.to_csv(output_file, mode='a', index=True, header=True)
                 


        else:
            df_stats=df_stats.agg({'LapTime':['mean','min', 'max','std','median'],'Lap':'size'}) 
            print (df_stats)
            #df_stats=df_stats.mean()
        #df_stats.sort_values(by=df_stats.columns[1], inplace=True)
            df_res.to_csv(output_file, mode='a', index=True, header=True)
        #df_res = df_res.append(df_stats)
        #print (df_res)
#Plot_MinandMax(df_stats,df)



"""
df_res2 = df_res[df_res['sectortype'] =='Turn'].groupby(['cluster_label'])
#df_res2=df_res.filter(sectortype='Turn', axis=1).groupby(['cluster_label'])
#print (df_res)
#df_res=df_res.agg({'Trail Steer phase_Avg':['mean','min', 'max','std','median']}) 
df_turn=df_res2.mean()

df_turn.to_csv(output_file, mode='a', index=True, header=True)
#Plot_allClusters(df_stats,df)

df_res2 = df_res[df_res['sectortype'] =='Str'].groupby(['cluster_label'])
#df_res2=df_res.filter(sectortype='Turn', axis=1).groupby(['cluster_label'])
#print (df_res)
#df_res=df_res.agg({'Trail Steer phase_Avg':['mean','min', 'max','std','median']}) 
df_str=df_res2.mean()

df_str.to_csv(output_file, mode='a', index=True, header=True)

df_res2 = df_res.groupby(['cluster_label'])
#df_res2=df_res.filter(sectortype='Turn', axis=1).groupby(['cluster_label'])
#print (df_res)
#df_res=df_res.agg({'Trail Steer phase_Avg':['mean','min', 'max','std','median']}) 
df_str=df_res2.mean()

df_str.to_csv(output_file, mode='a', index=True, heade"r=True)


"""
#find best and worst lap
