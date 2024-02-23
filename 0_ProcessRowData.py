# -*- coding: utf-8 -*-
"""
Created on Sun Mon 19 July 2022

processing of row data extracted from Motec:
    1- remove the heading of file
    2- create LapNo column based on lap beacon
    3- create sector name column based on the sector info and lap distance
@author: Fazilat
"""
import pandas as pd
import numpy as np
import math
from datetime import datetime
import os

import shutil
from FileOperation import *


def Map_LapData(item,df,Beacon_Markers):
    if(item < Beacon_Markers[0]): 
        return(0)
    else:
        for j in range(len(Beacon_Markers)):
            if(j +1 <len(Beacon_Markers)):
                if(item>=  Beacon_Markers[j] and item < Beacon_Markers[j+1]):  
                    return (j+1)
            else:
               return(-1) 
 
def Map_SectorData(item,df):

        if(item < ds['lapdistance'][0]): 
             return(ds['sectorname'][0])
        else:
            for j in range(len(ds['sectorname'])):
                if(j +1 <len(ds['sectorname'])):
                    if(item>=  ds['lapdistance'][j] and item <  ds['lapdistance'][j+1]):    
                        return(ds['sectorname'][j+1])
                else:
                        return(ds['sectorname'][j])
 

def find_nearestSorted(array,value):
    import math
    #find=False
    #while find!=True:
    idx = np.searchsorted(array, value, side="left")
    #if df.iloc[idx]['Corr Lap Dist']!=0:
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
       # else:
       #     idx=idx-1
       #     value=df.iloc[idx]['Time']
      #      print (idx, value)
    
def find_nearest(array,value):                 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx




def Set_LapandSectorname(df,Beacon_Markers,damaged_outlap):
    sign = lambda x: math.copysign(1, x)
    array=df['Time'].to_numpy()
    pre_idx=0
    df.insert(1, 'Lap', 0)
    df.insert(2, 'SectorName', '')

    #df.insert (4,'direction changed','')
    #df.loc[0, 'angular velosity'] = 0

    a_steer = df['STEERANGLE'].to_numpy()
    a_time = df['Time'].to_numpy()
    a_steerdiff = np.diff(a_steer)

    a_timediff = np.diff(a_time)

    a_velocity = np.divide(a_steerdiff, a_timediff)

    a_velocity = np.insert(a_velocity, 0, 0)
    df.insert (3,'angular velosity',a_velocity)

   
    """for i in range (1,len(df)):
        print (i)
        dir_changed=0  
        df.loc[i, 'angular velosity'] = (df.loc[i,'STEERANGLE']-df.loc[i-1,'STEERANGLE'])/(df.loc[i,'Time']-df.loc[i-1,'Time'])
        if sign(df.loc[i, 'angular velosity'])==sign(df.loc[i-1, 'angular velosity']):
            dir_changed=0
        else:
            dir_changed=1

        df.loc[i, 'direction changed'] = dir_changed
    print ('velosity finished....')    """
    counter_lap=0
    for j in range(len(Beacon_Markers)):
        idx=find_nearestSorted(array, value=Beacon_Markers[j])
       
        GetandSet_Sectorname(df,pre_idx, idx)
        if pre_idx==0 :
          if damaged_outlap==False:
            df.loc[pre_idx:idx, 'Lap'] = "Out Lap"
          else:           
            counter_lap=counter_lap+1
            df.loc[pre_idx:idx, 'Lap'] = "Lap " + str(counter_lap)


            #GetandSet_Sectorname(df,pre_idx, idx)
        else:
            if j==len(Beacon_Markers)-1:
                df.loc[pre_idx+1:idx, 'Lap'] = "Lap " + str(counter_lap)
                df.loc[idx+1:len(df), 'Lap'] = "In Lap"
                
                GetandSet_Sectorname(df,idx+1, len(df))               

            else:
                df.loc[pre_idx+1:idx, 'Lap'] = "Lap " + str(counter_lap)
                #GetandSet_Sectorname(df,pre_idx, idx)
        pre_idx=idx
        counter_lap=counter_lap+1
    return df
    """for n in range(len(df['Time'])):
        time = df['Time'][n]
        distance=df['Distance'][n]

        lap_No=Map_LapData(time,df,Beacon_Markers)
        if lap_No==0:
            lap_name='Out Lap'  
        elif lap_No==-1:
            lap_name='In Lap'
            distance=distance-len(Beacon_Markers)*track_len
        else:
            distance=distance-lap_No*track_len
            lap_name='Lap ' +str(lap_No)
            
        arrLap=np.append(arrLap,lap_name)
        arrSector=np.append(arrSector,Map_SectorData(distance,df))
    return (arrLap,arrSector)"""

def GetandSet_Sectorname(df,startpos, endpos): 
    min_idx=df.loc[startpos:endpos,'Corr Lap Dist'].idxmin()


    #TODO: for now threshold =200 
    #for some resaon beacone array dosnt have right position of starting lap
    if min_idx!=startpos:
        if abs(min_idx-startpos)<200 :
            df.loc[startpos:min_idx,'Corr Lap Dist']=0
    array=df['Corr Lap Dist'].to_numpy()


   
    pre_idx=0
    idx_in_df=startpos
    for j in range(len(ds['sectorname'])):

        idx=find_nearest(array[startpos:endpos],ds['lapdistance'][j])
        #print (idx,startpos,endpos)

        if pre_idx==0:
            df.loc[pre_idx+idx_in_df:idx+idx_in_df, 'SectorName'] = ds['sectorname'][j]
            #print( idx,df.loc[pre_idx+idx_in_df:idx, 'SectorName'])
        else:
            if j==len(ds['sectorname'])-1:
               # df.loc[pre_idx+idx_in_df+1:idx+idx_in_df, 'SectorName'] = ds['sectorname'][j-1]
              #  df.loc[idx+idx_in_df+1:endpos, 'SectorName'] = ds['sectorname'][j]
                df.loc[pre_idx+idx_in_df+1:endpos, 'SectorName'] = ds['sectorname'][j]

            else:

                df.loc[pre_idx+idx_in_df+1:idx+idx_in_df, 'SectorName'] = ds['sectorname'][j]

        pre_idx=idx

    return df    
  


def Read_RowData (file):

     firstrow=18
     f1=FileOperation.OpenAndReadCSVFile(file)

     track_name=f1[1][1]  
     if track_name!=TRACKENAME:
        df=pd.DataFrame()
        return(-1,df, '') 

     num_cols= 83 ######NUMBER OF COLUMNS
     lastrow=len(f1)


     num_rows=lastrow-firstrow

     np_teledata = np.empty((num_rows,num_cols))
        
     #for header coulmn including names of channels
     headerrow=14
     columns = [f1[headerrow][i] 
              for i in range(0,num_cols)]

     Beacon_Markers=f1[11][1].split() 
     Beacon_Markers =list(map(float, Beacon_Markers))
     ##if the first index is zero we need to exlude the first lap

     for k in range(0,num_rows):
        for b in range (0,num_cols):
                 if f1[k+firstrow][b] !='':
                    np_teledata[k, b] = f1[k+firstrow][b]  
                 else:
                    np_teledata[k, b]=0

    # generating the Pandas dataframe
    # from the Numpy array and specifying
    # details of index and column headers
     damaged_outlap=False
     df = pd.DataFrame(np_teledata , columns = columns)
     if len(Beacon_Markers)==0:
         return (1,df,Beacon_Markers,damaged_outlap)
     if Beacon_Markers[0]==0:
         damaged_outlap=True
         Beacon_Markers=np.delete(Beacon_Markers,0,0)
         if Beacon_Markers[0]==0:
             Beacon_Markers=np.delete(Beacon_Markers,0,0)
         print (Beacon_Markers)
         #until next index in Beacon we  need to delete data
         """delindex=Beacon_Markers[0]
         array=df['Time'].to_numpy()
         idx=find_nearestSorted(array, value=delindex)

         #for i in range(idx):
          #   print (i)
           #  df.drop(axis=0,index=[i],inplace=True)"""
         
     return (0,df,Beacon_Markers,damaged_outlap) 
     
     
    
def Process_RowData (listof_Files,Outputname_file,Outputname_fileOriginal):
    
    for a in listof_Files:
       print ('proceesing for ', a)
       Beacon_Markers=[]
       valid_file,df,Beacon_Markers,damaged_outlap=Read_RowData (a)
      
       if valid_file==0:  ####IF there is valid file for the specified track
          #arrLap,arrSector=Set_LapandSectorname(df,Beacon_Markers)

           df=Set_LapandSectorname(df,Beacon_Markers,damaged_outlap)
           
           df['Steering_bin'] = df.apply(lambda row: calc_steeringbin(row), axis=1)  
           

           df.drop(df[df['Corr Lap Dist']==0].index, inplace=True)
           df.drop(df[df.Lap=='Out Lap'].index, inplace=True)
           df.drop(df[df.Lap=='In Lap'].index, inplace=True)
           #write orginal dataframe without averaging on lapdistance to csv 
           df.to_csv(Outputname_fileOriginal,index=False)
           
           df=MakeAvg_lapdata(df)
           df.to_csv(Outputname_file,index=False)


#####   
def  MakeAvg_lapdata(df):

    df_lap=df.groupby(['Lap'])
       
    df_avg=pd.DataFrame(columns=df.columns)  

    for  cl in df_lap:
        lap= cl[0]
        df_PID= df[df["Lap"] == lap] 
        """
        df_PID1 = df[df["Lap"] == lap] 
        
        df_PID=pd.merge(
                    df_PID1,
                    df_map,
                    how="right",
                    on=['Corr Lap Dist'],
                    left_on=None,
                    right_on= None,
                    sort=False,
                    suffixes=("_x", "_y"),
                    copy=True,
                    indicator=False,
                    validate=None,
             )
        """
        df_PID['Lap']=lap
        df_PID['SectorName'].fillna(method='ffill')
        df_groupd=df_PID.groupby(['Corr Lap Dist'])
        
        for key, item in df_groupd:
            df_groupkey=df_groupd.get_group(key)   
            if len(df_groupkey)>1:
                df_str=df_groupkey.select_dtypes(include='object').head(1)
                
                df_numeric = df_groupkey.select_dtypes(include='number').mean()  
    
                df_totalRow=df_numeric.to_frame()
                df_totalRow=df_totalRow.transpose()
                df_totalRow.insert(1,'Lap',df_str.iloc[0]['Lap'])
                df_totalRow.insert(2,'SectorName',df_str.iloc[0]['SectorName'])
                df_totalRow.insert(len(df_totalRow.columns),'Steering_bin',df_str.iloc[0]['Steering_bin'])
                df_avg=pd.concat([df_avg,df_totalRow])
            else:
                df_avg=pd.concat([df_avg,df_groupkey])
    return (df_avg)

##prepare bin for steering angle
def round_up(n, decimals,m):
   
    if n>0 : 
        multiplier = m ** decimals
        up= math.ceil(n * multiplier) / multiplier
        down=up-m
        bin_value=str(int(down)) + '_' + str(int(up))
        return (bin_value)
    else:
        multiplier = m ** decimals
        up= math.ceil(abs(n) * multiplier) / multiplier
        down=up-m
        bin_value=str(int(-1*up)) + '_' + str(int(-1*down))
        return (bin_value)
 #######   
def calc_steeringbin(row):  
   st=row['STEERANGLE'] 
   if abs(st)<=20:
       return(round_up(st,-1,2))
   else:
       return(round_up(st,-1,10))



print ('Start of execution . . . . .')  
print ('start time: ' , datetime.now())



collection='C3/'

inputdata_path="../../../2_DATA/Included Participants - In-lab/" + collection 
outputdata_path="../../../2_DATA/OUTPUT/"+  collection+ "RowData/"

outputdata_path_original="../../../2_DATA/OUTPUT/"+  collection+ "RowData_original/"

N=3911
df_map = pd.DataFrame({ 'Corr Lap Dist' : range(1, N + 1 ,1)})  ##used for merging with dataframe to have all lap distance
df_map['Corr Lap Dist']=df_map['Corr Lap Dist'].astype('int')


#read sectorinfo data from sectorinfo.csv
#for collection 1
#if collection=='C4/':
#    SectorInfor_file="../../../2_DATA/Included Participants - In-lab"+'/TrackInfo/brands_hatch, brands_hatch new.csv'
#else:
SectorInfor_file="../../../2_DATA/Included Participants - In-lab"+'/TrackInfo/brands_hatch, brands_hatch.csv'
    
ds = pd.read_csv(SectorInfor_file,header=0,parse_dates=[0])  

#remove folder with all of its files
if  os.path.exists(outputdata_path):
   shutil.rmtree(outputdata_path)

####Creating output directory for the partcipant
if not os.path.exists(outputdata_path):
  os.makedirs(outputdata_path)


#remove folder with all of its files
if  os.path.exists(outputdata_path_original):
   shutil.rmtree(outputdata_path_original)

####Creating output directory for the partcipant
if not os.path.exists(outputdata_path_original):
 os.makedirs(outputdata_path_original)
  
  
  
TRACKENAME='brands_hatch'

######RowData ##################
# Get the list of all files in directory tree at given path

#=['P_018_1','P_018_2','P_019_1','P_019_2','P_019_3','P_028_2','P_028_3','P_029_3','P_029_4','P_035_2','P_035_3', 'P_036_2']
pid=[]
pid=FileOperation.GetParticipantList(inputdata_path,'P_')

#for i in range (5,45):
#   _pid='P_' + str(i).zfill(3)
#   pid=np.append (pid,_pid)


for i in range (len(pid)):
    #PID_idx = str(i).zfill(3)
    #PID='P_'+PID_idx
    PID=pid[i]
    PID2='D'+PID[1:len(PID)]
    print (PID)
    
    path=inputdata_path+ PID+'/'  
    print (path)

    listof_Files=FileOperation.getListOfFilesWithExtension(path,'.ld')
    print (listof_Files)
    Outputname_file=outputdata_path+PID2+'.csv' 
    Outputname_fileOriginal=outputdata_path_original+PID2+'.csv' 

    Process_RowData (listof_Files,Outputname_file,Outputname_fileOriginal)


print ('End of execution . . . . .')  
print ('End time: ' , datetime.now())
