# -*- coding: utf-8 -*-
"""
Created on Sun Mon 19 Nov 2022

processing of row data extracted from Motec:
for calulating different features
@author: Fazilat
"""
import pandas as pd
import numpy as np

from datetime import datetime
import os
import math
import shutil
from FileOperation import *

from scipy.signal import find_peaks, peak_prominences,peak_widths

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from scipy.stats import ttest_ind, mannwhitneyu

def finding_Brakepeaks(pid,lap,df):

    a_sector = []
    a_brake=[]
    time=[]
   
    df = df[df["Lap"] == lap] 
    
    #df=df[df['SectorName']==sectorname]
    df = df.reset_index(drop=True)
    a_brake = df['BRAKE']

    time=df['Calc Lap Time']
    corrlapdist=df['Corr Lap Dist']
    # (arr)
    #peaks,_= find_peaks(a_brake)#, height=10, distance=400)
    peaks, _ = find_peaks(-a_brake, prominence=0.3, width=2)

# Get widths of the peaks
    widths, _, _, _ = peak_widths(-a_brake, peaks)
    
    #print (peaks)
    if len(peaks)!=0:
    
        # Plot the brake signal and mark the identified peaks
        plt.plot(time, a_brake, label='Brake Signal')
        plt.plot(time[peaks], a_brake[peaks], 'rx', label='Peaks')

        total_braktime=0
        delta_time=0
        for peak, width in zip(peaks, widths):
            start_index = peak - int(width // 2)
            end_index = peak + int(width // 2)
            if start_index>0 and end_index>0:
                delta_time=abs(time[end_index]-time[start_index])
            total_braktime=total_braktime+delta_time
           # if start_index>0 and end_index>0:
           #     plt.plot(time[start_index], a_brake[start_index], 'rx', markersize=8, label='Start')
           #     plt.plot(time[end_index], a_brake[end_index], 'bx', markersize=8, label='End')
                

    

        print(f"Peak at {time[start_index]:.2f} seconds - {time[end_index]:.2f} seconds")
        plt.xlabel('Time')
        plt.ylabel('Brake Signal')
        plt.legend()
        plt.show()
        
        prominences = peak_prominences(a_brake, peaks)[0]
        #print (prominences)
        
        a_sector=df['SectorName']
    
    
        widths, width_heights, left_ips, right_ip=peak_widths(a_brake, peaks, rel_height=1)
        #results_half = peak_widths(a_brake, peaks, rel_height=1)
    
        for i in range (len(left_ips)):
            left_ips[i]=int(left_ips[i])
            right_ip[i]=int(right_ip[i])
            
        #print (time[left_ips],time[peaks],time[right_ip])
        #print (time[prominences])
    
        #print (time[prominences])
        n=len (widths)
        sum=0
        Trail_braking_duration=0
        Brake_length=0
        Brake_duration=0
        for j in range (n):
            Brake_length=corrlapdist[right_ip[j]]-corrlapdist[left_ips[j]]+Brake_length
            Brake_duration=time[right_ip[j]]-time[left_ips[j]]+Brake_duration
            Trail_braking_duration=((corrlapdist[right_ip[j]]-corrlapdist[peaks[j]])/Brake_length)*100+Trail_braking_duration
            
             #start_index = peak[j] - int(widths[j] // 2)
            #end_index = [j] + int(widths[j] // 2)
           # delta_time=abs(time[end_index]-time[start_index])
            #brakelenght

            #Brake time for peaks
           # print ('aa', time[left_ips[j]])
           # df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,['Brake_StartTime_Avg','Brake_PeakTime_Avg','Brake_PeakTime_Avg', 'BrakeLength_Avg']=[time[left_ips[j]],time[peaks[j]],time[right_ip[j]],widths[j]]

            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Brake_length_Avg']=corrlapdist[right_ip[j]]-corrlapdist[left_ips[j]]
            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Brake_duration_Avg']=time[right_ip[j]]-time[left_ips[j]]
            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Trail_braking_duration_Avg']=((corrlapdist[right_ip[j]]-corrlapdist[peaks[j]])/Brake_length)*100
           # df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'brake_time']=delta_time



           #df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Brake_PeakTime2_Avg']=12
        if n>0:
            
            

            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Brake_length_Avg']=Brake_length/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Brake_duration_Avg']=Brake_duration/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Trail_braking_duration_Avg']=Trail_braking_duration/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'TotalBrakeTime']=total_braktime
            
            

        
        

        
        contour_heights = a_brake[peaks] - prominences
        #plt.plot(a_brake,color= 'b', linewidth=1)
        #plt.plot(peaks, a_brake[peaks], "x",color= 'orange')
        #plt.vlines(x=peaks, ymin=contour_heights, ymax=arr[peaks])
        #plt.show()
        
 # Function to find the relative segment for each peak
def find_relative_segment(peak_distance):
    for index, row in ds.iterrows():
        if row['start'] <= peak_distance < row['end']: 
             return index
    return None
        
def finding_Brakepeakswithstart_end(pid,lap,df):

            a_sector = []
            a_brake=[]
            time=[]
            condition = (df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)
            cluster_label = df_lapchanneldata.loc[condition, 'cluster_label'].values[0] if not df_lapchanneldata.loc[condition, 'cluster_label'].empty else None
            
            if cluster_label!=None:

                df = df[df["Lap"] == lap] 
                
                #df=df[df['SectorName']==sectorname]
                df = df.reset_index(drop=True)
                a_brake = df['BRAKE']
                a_sector=df['SectorName']
    
                time=df['Calc Lap Time']
                corrlapdist=df['Corr Lap Dist']
                # (arr)
                #peaks,_= find_peaks(a_brake)#, height=10, distance=400)
                peaks, _ = find_peaks(-a_brake, prominence=0.3, width=2)
    
                peaks = [peak for peak in peaks if a_brake[peak] > 0]
            # Get widths of the peaks
                widths, _, _, _ = peak_widths(-a_brake, peaks)
                

                peak_starts = []
                peak_ends = []
                #print (peaks)
                if len(peaks)!=0:
                
                    # Plot the brake signal and mark the identified peaks
                    plt.plot(time, a_brake, label='Brake Signal')
                    plt.plot(time[peaks], a_brake[peaks], 'rx', label='Peaks')
    
                    total_braktime=0
                    total_brakewidth=0
                    total_trailbraking=0
                    delta_time=0
                    
                    for peak, width in zip(peaks, widths):
                        start_index = peak - int(width // 2)
                        peak_index=a_brake[peak]
                        end_index = peak + int(width // 2)
                        if end_index>3800:
                            end_index=3800
                        if start_index>0 and end_index>0:
                            delta_time=abs(time[end_index]-time[start_index])
                        if peak_index>0 and end_index>0:
                            delta_trailtime=abs(time[end_index]-time[peak])
                            
                        total_braktime=total_braktime+delta_time
                        total_trailbraking=total_trailbraking+delta_trailtime
                        total_brakewidth=total_brakewidth+width
    
                    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Brake_length_Avg']=total_brakewidth
                    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Brake_duration_Avg']=total_braktime
                    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Trail_braking_duration_Avg']=total_trailbraking
    
    
                    j=0       
                    pre_delta_time=0
                    pre_width=0
                    pre_trailtime=0

                    for peak, width in zip(peaks, widths):
                            relative_segment = find_relative_segment(peaks[j])
                            start_index = peak - int(width // 2)
                            peak_index=a_brake[peak]
                            end_index = peak + int(width // 2)
                            if end_index>3800:
                                end_index=3800
                            if start_index>0 and end_index>0:
                                delta_time=abs(time[end_index]-time[start_index])
                            if peak_index>0 and end_index>0:
                                delta_trailtime=abs(time[end_index]-time[peak])
                           
                                
                            if relative_segment!=None:
                                if width>pre_width:
                                    df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==ds.iloc[relative_segment]['sectorname']) ,'Brake_length_Avg']=width
                                if delta_time>pre_delta_time:
                                    df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==ds.iloc[relative_segment]['sectorname']) ,'Brake_duration_Avg']=delta_time
                                if delta_trailtime>pre_trailtime:
                                    df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==ds.iloc[relative_segment]['sectorname']) ,'Trail_braking_duration_Avg']=delta_trailtime
                            j=j+1
                            pre_delta_time=delta_time
                            pre_width=width
                            pre_trailtime=delta_trailtime
                           # print (ds.iloc[relative_segment]['sectorname'],delta_time,delta_trailtime,width)
        
                            # Append start and end indices to the lists
                            peak_starts.append(start_index)
                            peak_ends.append(end_index)
            
                           # Store information for the current brake
                    brake_info = pd.DataFrame({
                                'Brake_ID': pid,
                                'Lap':lap,
                                'Start_Index': peak_starts,
                                'Peak_value': a_brake[peaks],
                                'Peak_Index':peaks,
                                'End_Index': peak_ends,
                                'cluster':cluster_label
                    })
                        
                    # Append the brake information to the main DataFrame
                    brake_dfs.append(brake_info)
            


                
                

                

        
        
def call_normalize (lap,df) :
    time=[]
    df = df[df["Lap"] == lap] 
    df = df.reset_index(drop=True)
    narr=df['Calc Lap Time']
    maxlen=6000
    time=normalize(narr,maxlen)
    print (time)
    
def find_numberofZerocrossing(a_steer):
    s = 0
    preindex=0
    for i in range (len(a_steer)-1):
         if np.sign(a_steer[i])!=np.sign(a_steer[i+1]): 
             if i!=preindex :
                 s += 1
                 #print (i,a_steer[i],a_steer[i+1])
                 preindex=i+1
         if (a_steer[i]==0 and a_steer[i+1]==0):
                 preindex=i+1
    return (s)
def finding_SteeringMetrics (pid,lap,df):

    print (pid,lap)
    a_steer=[]
    angular_travel=[]
   
    df = df[df["Lap"] == lap] 
    df = df.reset_index(drop=True)

    #df.insert (3,'angular velosity','')
    #df.insert (4,'direction changed','')

    a_steer = df['STEERANGLE'].to_numpy()
   
    ############################
    #numberofcrossing=((a_steer[:-1] * a_steer[1:]) < 0).sum()
    

   
    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'number of crossing_Avg']=find_numberofZerocrossing(a_steer)


   # arr = np.array([5,85,185,270,355,10,75, 170, 250,345, 25])
    #Number of zero degree crossing
    darr = np.diff(a_steer)
    angular_velocity=np.mean(np.abs(df['angular velosity'].to_numpy()))
    #print(len( df['angular velocity']), len(darr))
    #darr=np.append(darr,0)

    #darr[len(darr)]=0
   
    #darr[darr<0] += 360
    angular_travel=np.cumsum(abs(darr), axis=None, dtype=None, out=None)

    
    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Cumulative angular travel_Avg']=angular_travel[len(angular_travel)-1]
    
    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'angular velosity_Avg']=angular_velocity


    ##Total number of direction changes
    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'number of direction changed_Avg']=changing_direction(a_steer)

    df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'steering duration_Avg']=len(df[(df['angular velosity']>300 ) |(df['angular velosity']<-300 )  ])


def changing_direction(elems):
    # can not compare with 0 or 1 elem => 0 changes
    if len(elems) < 2: 
        return 0

    changes = 0
    # helper, returns UP or DOWN or umplicit None based on b-a 
    def ud(a,b):
        if b-a > 0: return "UP"
        if b-a < 0: return "DOWN"
        
    # current start direction
    direction = None
    for idx, e in enumerate(elems):
        try:
            # save old direction, initially None
            ld = direction
            # get new direction, maybe None
            direction = ud(e,elems[idx+1])
        
            #print(direction) # --> comment for less output
            
            # if both are not None and different we have a change in direction
            # if either is None: same values this and last time round
            # 
            if direction is not None and ld is not None and ld != direction:
                changes += 1
            if direction is None and ld is not None:
                direction = ld # restore last direction for next time round
        except IndexError:
            pass # end of list reached
                
    return changes


  
def finding_Steerpeaks(pid,lap,df):

    a_sector = []
    a_steer=[]
    time=[]
   
    df = df[df["Lap"] == lap] 

    #df=df[df['SectorName']==sectorname]
    df = df.reset_index(drop=True)
    a_steer = df['abs Steerangle']

    time=df['Calc Lap Time']
    corrlapdist=df['Corr Lap Dist']
    # (arr)
    peaks,_= find_peaks(a_steer, height=10, distance=400)
    #print (peaks)
    if len(peaks)!=0:
        prominences = peak_prominences(a_steer, peaks)[0]
        #print (prominences)
        
        a_sector=df['SectorName']
    
    
        widths, width_heights, left_ips, right_ip=peak_widths(a_steer, peaks, rel_height=1)
        #results_half = peak_widths(a_brake, peaks, rel_height=1)
    
        for i in range (len(left_ips)):
            left_ips[i]=int(left_ips[i])
            right_ip[i]=int(right_ip[i])
            
        #print (time[left_ips],time[peaks],time[right_ip])
        #print (time[prominences])
    
        #print (time[prominences])
        sum=0
        Steer_Peakvalue=0
        Steer_PeakTime=0
        Steer_duration=0
        Steer_reductionSpeed=0
        n=len (widths)
        for j in range (n):
            sum=sum+widths[j]
            #brakelenght

            #Brake time for peaks
           # print ('aa', time[left_ips[j]])
           # df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,['Brake_StartTime_Avg','Brake_PeakTime_Avg','Brake_PeakTime_Avg', 'BrakeLength_Avg']=[time[left_ips[j]],time[peaks[j]],time[right_ip[j]],widths[j]]

            Steer_Peakvalue=a_steer[peaks[j]] +Steer_Peakvalue
            Steer_PeakTime=time[peaks[j]] +Steer_PeakTime
            Steer_duration=time[right_ip[j]]-time[left_ips[j]] +Steer_duration
            Steer_reductionSpeed=time[right_ip[j]]-time[peaks[j]] + Steer_reductionSpeed
            
            
            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Steer_Peakvalue_Avg']=a_steer[peaks[j]]
            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Steer_PeakTime_Avg']=time[peaks[j]]
   

           #df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Brake_PeakTime2_Avg']=12
        if n>0:

            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Steer_Peakvalue_Avg']=Steer_Peakvalue/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Steer_PeakTime_Avg']=Steer_PeakTime/n
            #df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Steer_duration']=Steer_duration/n
            #df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Steer_reductionSpeed']=Steer_reductionSpeed/n
    
        

        
        contour_heights = a_steer[peaks] - prominences
        plt.plot(a_steer,color= 'g',linewidth=1)
        plt.plot(peaks, a_steer[peaks], "x",color= 'orange',linewidth=0.6)

        #plt.vlines(x=peaks, ymin=contour_heights, ymax=arr[peaks])
        plt.show()   
        #return df_lapchanneldata
def finding_Trottlepeaks(pid,lap,df):

    a_sector = []
    a_brake=[]
    time=[]

    df = df[df["Lap"] == lap] 
    
    #df=df[df['SectorName']==sectorname]
    df = df.reset_index(drop=True)
    a_brake = df['THROTTLE']

    time=df['Time']
    corrlapdist=df['Corr Lap Dist']
    # (arr)
    peaks,_= find_peaks(a_brake, height=10, distance=400)
    #print (peaks)
    if len(peaks)!=0:
        prominences = peak_prominences(a_brake, peaks)[0]
        #print (prominences)
        
        a_sector=df['SectorName']
    
    
        widths, width_heights, left_ips, right_ip=peak_widths(a_brake, peaks, rel_height=1)
        #results_half = peak_widths(a_brake, peaks, rel_height=1)
    
        for i in range (len(left_ips)):
            left_ips[i]=int(left_ips[i])
            right_ip[i]=int(right_ip[i])
            
        #print (time[left_ips],time[peaks],time[right_ip])
        #print (time[prominences])
    
        #print (time[prominences])
        sum=0
        Throttle_length=0
        Throttle_time=0
        n=len (widths)
        for j in range (n):
            
            sum=sum+widths[j]
            Throttle_length=corrlapdist[right_ip[j]]-corrlapdist[left_ips[j]]+Throttle_length
            Throttle_time=time[right_ip[j]]-time[left_ips[j]]+Throttle_time

            
            #brakelenght

            #THROTTLE  time for peaks
           # print ('aa', time[left_ips[j]])
           # df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,['THROTTLE _StartTime_Avg','THROTTLE _PeakTime_Avg','THROTTLE _PeakTime_Avg', 'THROTTLE Length_Avg']=[time[left_ips[j]],time[peaks[j]],time[right_ip[j]],widths[j]]

            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'THROTTLE width_Avg']=widths[j]
            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Throttle_duration_Avg']=time[right_ip[j]]-time[left_ips[j]]

            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Throttle application speed_Avg']=widths[j]/time[right_ip[j]]-time[left_ips[j]]

            df_sectorchanneldata.loc[(df_sectorchanneldata['PID'] == pid) & (df_sectorchanneldata['Lap'] == lap) & (df_sectorchanneldata['Sector'] ==a_sector[peaks[j]]) ,'Throttle release application_Avg']=corrlapdist[right_ip[j]]-corrlapdist[left_ips[j]]

 
        if n>0:
            avg_lap=sum/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'THROTTLE width_Avg']=avg_lap
    
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Throttle_duration_Avg']=Throttle_time/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Throttle application speed_Avg']=(sum/Throttle_time)/n
            df_lapchanneldata.loc[(df_lapchanneldata['PID'] == pid) & (df_lapchanneldata['Lap'] == lap)  ,'Throttle release application_Avg']=Throttle_length/n
            
         
        contour_heights = a_brake[peaks] - prominences
        plt.plot(a_brake,color= 'r',linewidth=1)
        plt.plot(peaks, a_brake[peaks], "x",color= 'orange',linewidth=0.6)
        #plt.vlines(x=peaks, ymin=contour_heights, ymax=arr[peaks])
        #plt.show()
        
def find_lap(df,pid):

    df.drop(df[df.Lap=='Out Lap'].index, inplace=True)
    df.drop(df[df.Lap=='In Lap'].index, inplace=True)
    df_lap=df.groupby(['Lap'])

    for  lp in df_lap:
       lap=lp[0]

       print (pid,lap)
       plt.rcParams["figure.figsize"] = (40,10)
       #finding_Brakepeaks(pid,lap,df)
       finding_Brakepeakswithstart_end(pid,lap,df )
      # finding_Trottlepeaks(pid,lap,df)
      # finding_Steerpeaks(pid,lap,df)
      
       print ('pid',pid,lap)

      # finding_SteeringMetrics (pid, lap, df)
    brake_df = pd.concat(brake_dfs, ignore_index=True)   
    brake_df.to_csv(outputfileBrake,index=False) 



def createPlot_Brakeinfo(brake_df):
    # Create a plot
    plt.figure(figsize=(20, 6))

    
    df_fast=brake_df[brake_df.cluster=='Fast']
    df_slow=brake_df[brake_df.cluster=='Slow']   
    
    
    # Create a new DataFrame to store the averaged values
    averaged_fastdf = pd.DataFrame(columns=['Peak_Segment', 'Averaged_Start_Index', 'Averaged_End_Index','Averaged_Peak_Index','Averaged_Peak_Value'])
    
    # Create a new DataFrame to store the averaged values
    averaged_slowdf = pd.DataFrame(columns=['Peak_Segment', 'Averaged_Start_Index', 'Averaged_End_Index','Averaged_Peak_Index','Averaged_Peak_Value'])
    
    # Identify the segment for each start and end index
    df_fast['Peak_Segment'] = np.digitize(df_fast['Peak_Index'], ds['end'], right=True)

    
    # Group the start indices by the segment and calculate the average
   # averaged_fastdf['Segment'] = ds['Sector']
    averaged_fastdf['Averaged_Start_Index'] = df_fast.groupby('Peak_Segment')['Start_Index'].mean().values
    averaged_fastdf['Averaged_End_Index'] = df_fast.groupby('Peak_Segment')['End_Index'].mean().values
    
    averaged_fastdf['Averaged_Peak_Index'] = df_fast.groupby('Peak_Segment')['Peak_Index'].mean().values
    
    averaged_fastdf['Averaged_Peak_Value'] = df_fast.groupby('Peak_Segment')['Peak_value'].mean().values
    
    df_fast=df_fast.groupby('Peak_Segment').agg({'Start_Index':['mean','size'],
                                                 'End_Index':['mean'],
                                                 'Peak_Index':['mean'],
                                                 'Peak_value':['mean']
                                                 })
  



   # Identify the segment for each start and end index
    df_slow['Peak_Segment'] = np.digitize(df_slow['Peak_Index'], ds['end'], right=True)

   
   # Group the start indices by the segment and calculate the average
  # averaged_fastdf['Segment'] = ds['Sector']
    averaged_slowdf['Averaged_Start_Index'] = df_slow.groupby('Peak_Segment')['Start_Index'].mean().values
    averaged_slowdf['Averaged_End_Index'] = df_slow.groupby('Peak_Segment')['End_Index'].mean().values
   
    averaged_slowdf['Averaged_Peak_Index'] = df_slow.groupby('Peak_Segment')['Peak_Index'].mean().values
   
    averaged_slowdf['Averaged_Peak_Value'] = df_slow.groupby('Peak_Segment')['Peak_value'].mean().values
   
    
    
    df_slow=df_slow.groupby('Peak_Segment').agg({'Start_Index':['mean','size'],
                                                 'End_Index':['mean'],
                                                 'Peak_Index':['mean'],
                                                 'Peak_value':['mean']
                                                 })
    
      # Overlay markers for averaged start indices in green
    plt.scatter(averaged_fastdf['Averaged_Start_Index'], averaged_fastdf['Averaged_Peak_Value'],
                  color='red', label='Averaged Start Indices', marker='o', s=100)
      
      # Overlay markers for averaged end indices in purple
    plt.scatter(averaged_slowdf['Averaged_Start_Index'], averaged_slowdf['Averaged_Peak_Value'],
                  color='blue', label='Averaged End Indices', marker='s', s=100)
      
      # Plot segment lines
    for i, segment_end in enumerate(ds['end']):
          color = 'gray' if i % 2 == 0 else 'white'
          plt.axvline(x=segment_end, color=color, linestyle='--', linewidth=1)
      
      # Customize the plot
    plt.xlabel('Index')
    plt.ylabel('Brake Signal Value')
    plt.title('Brake Signal with Peaks and Averaged Indices')
    plt.legend()
      #plt.grid(True)
      
      # Show the plot
    plt.show() 

   # Overlay markers for averaged start indices in green
    plt.scatter(averaged_fastdf['Averaged_End_Index'], averaged_fastdf['Averaged_Peak_Value'],
               color='red', label='Averaged Start Indices', marker='o', s=100)
   
   # Overlay markers for averaged end indices in purple
    plt.scatter(averaged_slowdf['Averaged_End_Index'], averaged_slowdf['Averaged_Peak_Value'],
               color='blue', label='Averaged End Indices', marker='s', s=100)
   
   # Plot segment lines
    for i, segment_end in enumerate(ds['end']):
       color = 'gray' if i % 2 == 0 else 'white'
       plt.axvline(x=segment_end, color=color, linestyle='--', linewidth=1)
   
   # Customize the plot
    plt.xlabel('Index')
    plt.ylabel('Brake Signal Value')
    plt.title('Brake Signal with Peaks and Averaged Indices')
    plt.legend()
    #plt.grid(True)
   
   # Show the plot
    plt.show() 
    
    # Calculate mean start time for each group
    mean_start_time_fast = averaged_fastdf['Averaged_Start_Index'].mean()
    mean_start_time_slow = averaged_slowdf['Averaged_Start_Index'].mean()
    
    # Calculate median start time for each group
    median_start_time_fast = averaged_fastdf['Averaged_Start_Index'].median()
    median_start_time_slow = averaged_slowdf['Averaged_Start_Index'].median()
    
    


    # Perform t-test for mean comparison
    t_stat, p_value = ttest_ind(averaged_fastdf[1:-1]['Averaged_Start_Index'], averaged_slowdf[1:-1]['Averaged_Start_Index'])
    print("T-Test p-value:", p_value)


    # Perform t-test for mean comparison
    t_stat, p_value = ttest_ind(averaged_fastdf[1:-1]['Averaged_End_Index'], averaged_slowdf[1:-1]['Averaged_End_Index'])
    print("T-Test p-value:", p_value)
    
    # Perform t-test for mean comparison
    t_stat, p_value = ttest_ind(averaged_fastdf['Averaged_Peak_Value'], averaged_slowdf['Averaged_Peak_Value'])
    print("T-Test p-value:", p_value)
    
    # Perform Mann-Whitney U test for median comparison
    u_stat, p_value_mw = mannwhitneyu(averaged_fastdf['Averaged_Start_Index'], averaged_slowdf['Averaged_Start_Index'])
    print("Mann-Whitney U Test p-value:", p_value_mw)


    # Perform Mann-Whitney U test for median comparison
    u_stat, p_value_mw = mannwhitneyu(averaged_fastdf['Averaged_Peak_Index'], averaged_slowdf['Averaged_Peak_Index'])
    print("Averaged_Peak_Index Mann-Whitney U Test p-value:", p_value_mw)


    return (df_fast,df_slow)

##############################
collection='C5/'


inputdata_path="../../../2_DATA/OUTPUT/"+  collection+ "RowData_C2C5C3/" 



inputdata_sectorstatfile="../../../2_DATA/OUTPUT/"+  collection+ "ClusterResults - C2C3C5/brands_hatch, brands_hatch/_CL_Channel Report_Detail.csv" 

inputdata_lapstatfile="../../../2_DATA/OUTPUT/"+  collection+ "ClusterResults - C2C3C5/brands_hatch, brands_hatch/_CL_Channel Report_Only.csv" 

outputdata_path="../../../2_DATA/OUTPUT/"+  collection+ "NewFeatureRowData-test/"
outputfilelap=outputdata_path + '_Channel Report_Only.csv'

outputfilelap2=outputdata_path + '_Channel Report_AVG.csv'

outputfilelap_Avg=outputdata_path + '_Channel Report_Only.csv'
outputfilesector=outputdata_path + '_Channel Report_Detail.csv'

outputfileBrake=outputdata_path + '_Brake_Info.csv'
outputfileBrakeFast=outputdata_path + '_Brake_InfoFast.csv'
outputfileBrakeSlow=outputdata_path + '_Brake_InfoSlow.csv'

#read sectorinfo data from sectorinfo.csv
SectorInfor_file="../../../2_DATA/Included Participants - In-lab/"+'/TrackInfo/brands_hatch, brands_hatch.csv'
ds = pd.read_csv(SectorInfor_file,header=0,parse_dates=[0])  

brake_dfs=[]


columns = ['Brake_ID','Lap', 'Start_Index', 'Peak_value', 'Peak_Index', 'End_Index','cluster']
brake_df = pd.DataFrame(columns=columns)

df_lapchanneldata= pd.read_csv(inputdata_lapstatfile,header=0,parse_dates=[0])  
print (len(df_lapchanneldata))
#df_lapchanneldata.drop(df_lapchanneldata[df_lapchanneldata.outlier>0].index, inplace=True)
print (len(df_lapchanneldata))
df_sectorchanneldata= pd.read_csv(inputdata_sectorstatfile,header=0,parse_dates=[0])  
#df_sectorchanneldata.drop(df_sectorchanneldata[df_sectorchanneldata.outlier>0].index, inplace=True)

pd.set_option('mode.use_inf_as_na',True)

for i in range (len(ds)):
    sector_name=ds.loc[i,'sectorname']
    sector_len=ds.loc[i,'sector_len']
    sector_dis=ds.loc[i,'lapdistance']
    df_sectorchanneldata.loc[df_sectorchanneldata['Sector'] ==sector_name,'Sector_Length'] =sector_len
    df_sectorchanneldata.loc[df_sectorchanneldata['Sector'] ==sector_name,'Sector_Start'] =sector_dis

#remove folder with all of its files
#if  os.path.exists(outputdata_path):
#   shutil.rmtree(outputdata_path)

####Creating output directory for the partcipant
#if not os.path.exists(outputdata_path):
#  os.makedirs(outputdata_path)

df_lap=df_lapchanneldata.groupby(['PID'])

"""
for  cl in df_lap:
    pid=cl[0]

    if (pid!='P_29-1') and (pid!='D_064'):
        inputfile_name=inputdata_path + pid +'.csv'
        if os.path.exists(inputfile_name):
          df= pd.read_csv(inputfile_name)
     
          find_lap(df,pid)

        #df.to_csv(outputcsvfile,index=False)
"""       
brake_df= pd.read_csv(outputfileBrake)
fastdf,slowdf=createPlot_Brakeinfo(brake_df)
   

fastdf.to_csv(outputfileBrakeFast,index=True)
slowdf.to_csv(outputfileBrakeSlow,index=True)
     
df_sectorchanneldata.replace([np.inf, -np.inf], np.nan, inplace=True)    
df_sectorchanneldata.fillna(0, inplace=True)
df_sectorchanneldata.to_csv(outputfilesector,index=False)
df_sectorchanneldata.to_csv(inputdata_sectorstatfile,index=False)



df_lapchanneldata.fillna(0, inplace=True)
print (len(df_lapchanneldata))
df_lapchanneldata.to_csv(outputfilelap2,index=False)
df_lapchanneldata.to_csv(inputdata_lapstatfile,index=False)

#st=''
#for i in range (7,len(df_sectorchanneldata.columns)):
   
#    st=st+ df_sectorchanneldata.columns[i]+','
#print (st)    

#df_lapgrouped = df_sectorchanneldata.groupby(['PID', 'TrackName', 'CarName', 'EventDate','Lap']).mean(st).reset_index()

#df_lapgrouped.drop('SectorTime', axis=1, inplace=True)




#df_lapgrouped.to_csv(outputfilelap_Avg,index=False)

