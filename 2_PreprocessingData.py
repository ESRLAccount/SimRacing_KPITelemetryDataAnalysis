# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:47:06 2022

@author: Fazilat

preprocessing tasks on  lap data:
    1-creating sumary of data (for statslap and StatsLapdetail)

    2-outlier detection
      outliers will be stored in the same stats files as the column named z and outlier
output:
 LapStatsSummary.csv
 LapStatsSummarydetails.csv   
"""

import pandas as pd
import csv

from datetime import datetime
import numpy as np
import os


from FileOperation import *


#make summary for lapstats data summary which includes just total lap time
def MakeSummary_ForLapTime(listof_Files,Outputname_LapSummary):

    #Read summary file to find outlier
    for a in listof_Files:
       print (a)
       df= pd.read_csv(a,usecols=('PID','TrackName','CarName','EventDate','Lap','LapTime'))
       df=Outlier_Detection(listof_Files[0],1,'LapTime')
       df_car=df.groupby(['PID','TrackName','CarName','EventDate','EventTime'])
       df_carAvg=df_car.agg({'LapTime':['mean','min', 'max','median','std'],'Lap':'size'})
       df_carAvg.fillna(0, inplace=True)
       df_carAvg.columns=['_'.join(col).strip() for col in df_carAvg.columns.values]
       #df_carAvgfiltered = df_carAvg[df_carAvg['Lap_size'] >1]  
       if  os.path.exists(Outputname_LapSummary):
           df_carAvg.to_csv(Outputname_LapSummary, mode='a', index=True, header=False)

       else:
          df_carAvg.to_csv(Outputname_LapSummary,index=True,header=True)

#make summary for lapstats data summary which includes sector times
def MakeSummary_ForLapTimeDetail(listof_Files,Outputname_LapSummaryDetail):

    #Read summary file to find outlier
    df_total = pd.DataFrame()
    for a in listof_Files:

       df= pd.read_csv(a)
       df=Outlier_Detection(listof_Files[0],1,'LapTime')
       df_car=df.groupby(['PID','TrackName','CarName','EventDate','EventTime'])

       df_carAvg=df_car.agg({'Lap':'size'})
       df_total = pd.concat([df_total, df_carAvg], axis=1)
       for i in range(6,len(df.columns)):
         c=df.columns.values[i]
         df_carAvg=df_car.agg({c:['mean','min', 'max','median','std']})

         df_total = pd.concat([df_total, df_carAvg], axis=1)
       
       df_total.columns=['_'.join(col).strip() for col in df_total.columns.values]  
       df_total.fillna(0, inplace=True)
       df_total.rename(columns={df_total.columns[5]: 'Lap_size'},inplace=True)

       #df_totalfiltered = df_total[df_total['Lap_size'] >1]
       if  os.path.exists(Outputname_LapSummaryDetail):

          outfile = open(Outputname_LapSummaryDetail, 'a')
          df_total.to_csv(outfile)
          outfile.close()   
          df_total.to_csv(Outputname_LapSummaryDetail, mode='a', index=True, header=False)
       else:

          df_total.to_csv(Outputname_LapSummaryDetail,index=True,header=True)
    
 
def Outlier_Detection (input_file, threshold,field):
    from scipy import stats

    df= pd.read_csv(input_file)

    df['z']= np.abs(stats.zscore(df[field]))

    #df['outlier']= np.where((abs(df['z'])> threshold  ), 1,0) 

    #df['outlier']= np.where(((df[field]==0 ) | (abs(df['z'])> threshold)), 1,0)#for lap time=0 
    df['outlier']= np.where(((df[field]==0 ) | ((df['z'])> threshold)), 1,0)#for lap time=0 
    
    #df.drop(['z'], axis=1, inplace=True)
    if 'missvalue_00' in df.columns:
         df.drop(['missvalue_00'], axis=1, inplace=True)
 
    #df.drop(df[df.outlier>0].index, inplace=True) 
    df.to_csv(input_file, index=False)
    return (df)

######################################
def Outlier_DetectionofChannels (input_file, threshold,field):
        from scipy import stats

        df= pd.read_csv(input_file)

        df['z']= np.abs(stats.zscore(df[field]))


        df['outlier1']= np.where(((df[field]==0 ) | ((df['z'])> threshold)), 1,0)#for lap time=0 
        
        #df.drop(df[df.outlier>0].index, inplace=True)  
        df['outlier2']=0
        if ('SectorTime') in df.columns:
            field='SectorTime'
            df['zs']= np.abs(stats.zscore(df[field]))
            df['outlier2']= np.where(( (df[field]<=0 ) | ((df['zs'])> threshold)), 1,0)#for sector time=0 
        
        #df.drop(df[df.outlier>0].index, inplace=True)     
        #df.drop(['z'], axis=1, inplace=True)
        if 'missvalue_00' in df.columns:
             df.drop(['missvalue_00'], axis=1, inplace=True)

        df['outlier3']=0
        #df['outlier3']= np.where(((df['WHEEL_SPEED_LF ms_Min']<25 ) & (df['WHEEL_SPEED_LR ms_Min']<25 ) & (df['WHEEL_SPEED_RF ms_Min']<25 ) & (df['WHEEL_SPEED_RR ms_Min']<25 ) & (df['abs ROTY _Max']>100) ), 1,0)#for lap time=0 
        df['outlier4']=0
        #df['outlier4']= np.where(((df['WHEEL_SPEED_LF ms_Min']<25 ) & (df['WHEEL_SPEED_LR ms_Min']<25 ) & (df['WHEEL_SPEED_RF ms_Min']<25 ) & (df['WHEEL_SPEED_RR ms_Min']<25 ) & (df['abs Steerangle_Max']>180) ), 1,0)#for lap time=0 

        filtered_values = np.where((df['outlier1']>0) | (df['outlier2']>0) |  (df['outlier3']>0 ) | (df['outlier4']>0) )
        df['outlier']=0
        #print (filtered_values)
        df_filtered=df.loc[filtered_values]

        for i in range (len(df_filtered)):
            #df['outlier']=np.where(((df['PID']==df_filtered.iloc[i]['PID']) & (df['Sector']==df_filtered.iloc[i]['Sector'])),1,0)
            df.loc[(df['PID']==df_filtered.iloc[i]['PID']) & (df['Lap']==df_filtered.iloc[i]['Lap']),'outlier']=1
            #df.loc[df['Fee'] > 22000, 'Fee'] = 1
        #df.drop(df[df.outlier>0].index, inplace=True)
        df.to_csv(input_file, index=False)
        return (df)
     
##########################################################
  
#### main
print ('Start of execution . . . . .')  
print ('start time: ' , datetime.now())

Term_list=['Time Report','Channel Report']
#Term_list=['Channel Report']

collection = 'C5/'

#FileName_term1='_LapsStatsOnly'
#FileName_term2='_LapsStatsDetail'

#FileName_term3='_LapStatsSummary'
#FileName_term4='_LapStatsDetailSummary'

inputdata_path="../../../2_DATA/OUTPUT/" + collection + "LapStats/"




#remove folder withh all of its files
#shutil.rmtree(outputdata_path)
#
####Creating output directory for the partcipant
#if not os.path.exists(outputdata_path):
#  os.makedirs(outputdata_path)
  

#read track name file
trackname_file=inputdata_path+'Trackname.csv'
with open(trackname_file, 'r') as f:
    df_tn = list(csv.reader(f, delimiter=","))
df_tn = np.array(df_tn)



#open each summary stats file related to the track
for i in range(1,len(df_tn)):
    trackname=df_tn[i][0] 
    term1='_'+Term_list[0]+'_Only' #Time Report
    term2='_'+Term_list[0]+'_Detail'
    print ('trackname:', trackname)
    #summary of the laps for each cases and summary of laps based on the sctors are stored in two follwing files
    Outputname_LapSummary=inputdata_path+trackname+'/'+ term1+'Summary' +'.csv'
    Outputname_LapSummaryDetail=inputdata_path+trackname+ '/'+term2+'Summary' +'.csv'
    if  os.path.exists(Outputname_LapSummaryDetail):
        os.remove(Outputname_LapSummaryDetail)
        
    if  os.path.exists(Outputname_LapSummary):
        os.remove(Outputname_LapSummary)

###########################_LapStatsSummary#####################
    #1- Get the list of all stats lap files in directory tree at given path
    Excluded_Term=''
    listof_Files = FileOperation.getListOfFiles(inputdata_path,term1,Excluded_Term)

    #2- call for making summary of laps not sector time
    #MakeSummary_ForLapTime(listof_Files,Outputname_LapSummary)

    #3-call for outlier detecion in summary file just for laptime not sector time
   #threshold=2
    #Outlier_Detection(Outputname_LapSummary,3,'LapTime_std')  #call for LapSummary

###########################_LapStatsDetailSummary#####################    
    # 1- Get the list of all stats lap files in directory tree at given path
    listof_Files = FileOperation.getListOfFiles(inputdata_path,term2,Excluded_Term)

    #2- call for making summary of laps with sector time
    #MakeSummary_ForLapTimeDetail(listof_Files,Outputname_LapSummaryDetail)

     #3- call for outlier detecion in summary file just for laptime with sector time   
     #threshold =1
    #Outlier_Detection(Outputname_LapSummaryDetail,1,'Totals_std')  #cal for LapSummaryDetail

    
    Outputfilename_Only=inputdata_path+trackname+'/'+term1 +'.csv'
    Outputfilename_Detail=inputdata_path+trackname+ '/'+term2 +'.csv'
    
#####################################_Time Report Only#######################
    #- call for outlier detecion in  file just for laptime 
    ##NOTE: the two other steps are not needed
    #threshold =2 
    Outlier_Detection(Outputfilename_Only,2,'LapTime')  #cal for _LapStatsOnly   
 
    
 #####################################_Time Report  Detail#######################
    #- call for outlier detecion in  file just for laptime with sector time  
    ##NOTE: the two other steps are not needed
    #threshold =2
    Outlier_Detection(Outputfilename_Detail,2,'LapTime')  #cal for _LapStatsDetail   
    
    
    #######################_Channel Report########################################
    term1='_'+Term_list[1]+'_Only'
    term2='_'+Term_list[1]+'_Detail'
    
    Outputfilename_Only=inputdata_path+trackname+'/'+term1 +'.csv'


    #####################################_Channel Report Only#######################
    #- call for outlier detecion in  file just for laptime with sector time  
    ##NOTE: the two other steps are not needed
    #threshold =2 
    #1- Get the list of all stats lap files in directory tree at given path
    listof_Files = FileOperation.getListOfFiles(inputdata_path,term1,Excluded_Term)
    file_counts=len(listof_Files)
    if file_counts>0:
    
       Outlier_DetectionofChannels(listof_Files[0],2,'LapTime')  #call for _LapStatsOnly   
   
    #####################################_Channel Report Track sections #######################
    #- call for outlier detecion in  file just for laptime with sector time  
    ##NOTE: the two other steps are not needed
    #threshold =2 
    #1- Get the list of all stats lap files in directory tree at given path
    listof_Files = FileOperation.getListOfFiles(inputdata_path,term2,Excluded_Term)
    file_counts=len(listof_Files)
    if file_counts>0:
       Outlier_DetectionofChannels(listof_Files[0],2,'LapTime')  #cal for _LapStatsOnly   
       