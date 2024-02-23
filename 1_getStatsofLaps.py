# -*- coding: utf-8 -*-
"""
This script is working with MOTEC data collected from online sources
The script reads all inputfiles, then creating a csv files including all of the data from the file
The script also provid a summary file, for each file a record in summary file is addedcrea
project: KPI telemetry data analysis
output: 
csvfile for lap_stats data
csvfile for summary of stats data
csvfile for channel report 

@author: Fazilat
"""


import pandas as pd
import numpy as np

from datetime import datetime
import os
import csv

import shutil
from FileOperation import *


##### Getting total lap time per participant -FOR TIME REPOPRT
#####################################################
def Make_StatsForTimeReport(outputdir,outputdata_path,listof_Files, Term,df_nanvalues):

    np_track=[]
    Cols_num=-1
    
    for a in listof_Files:
        print (a)  
        PID=os.path.split(os.path.dirname(a))[1]

        df= pd.read_csv(a)
            

        df.insert(0, 'PID', PID)

         #drop the first 4 cols
        #df1.drop(df1.columns[[1, 2, 3,4]], axis = 1, inplace = True)
        #rename lap column
        df.rename(columns={df.columns[1]: 'TrackName'},inplace=True)
        df.rename(columns={df.columns[2]: 'CarName'},inplace=True)
        df.rename(columns={df.columns[3]: 'EventDate'},inplace=True)
        df.rename(columns={df.columns[4]: 'EventTime'},inplace=True)
        df.rename(columns={df.columns[5]: 'Lap'},inplace=True)
        df.rename(columns={'Totals': 'LapTime'},inplace=True)
        
        #remove last row of the dataframe which include rolling minimum
        if len(df)>1:
            df.drop(axis=0 ,index=df.index[-1], inplace=True)
        

          
        #save track names in an array to store in a csv file
        np_track=np.append(np_track, df.iloc[0]['TrackName'])
        
        if df.iloc[0]['TrackName']=='mount_panorama, mount_panorama' :
            tr='mount_panorama'
        else:
            tr=df.iloc[0]['TrackName']

        track_path=outputdir+tr+'/'

        df.drop(df[df.Lap=='Out Lap'].index, inplace=True)
        df.drop(df[df.Lap=='In Lap'].index, inplace=True)

        
        if len (df)>0:
            #handling missing value for rows        
            nan_values = df[df.isna().any(axis=1)]  
            #for k in range (len(nan_values)):
            df_nanvalues = nan_values.append(df_nanvalues, ignore_index = True)
            
            df.dropna(thresh=len(df.columns),inplace=True)
    

            
            outputfilename_only=track_path+ '_'+ Term + '_Only' +'.csv'
           
            outputfilename_detail=track_path+ '_' +Term+ '_Detail' +'.csv'
            
            if not os.path.exists(track_path):
              os.makedirs(track_path)
            
            outputname_trackname=outputdir+'/'+ 'Trackname' +'.csv'
            #####Creating summary row data for e ach participant
            
            #JUST TOTAL AND LAP_INDEX WILL NE REMAINED AT THE END OF FILE
            df_LapsStatsOnly = df.drop(columns=df.iloc[:, 6:len(df.columns)-1],axis = 1)    
            
            if  os.path.exists(outputfilename_detail):
                #TODO: just for now that we dont use sector data
                ##################################################################################
               df_existing= pd.read_csv(outputfilename_detail)
               Cols_num=len(df_existing.columns)
               # we exclude the files that have different number of track sectors, so we compare with data in the previuse file

               if Cols_num==len(df.columns):
                   df.to_csv(outputfilename_detail, mode='a', index=False, header=False)
               df_LapsStatsOnly.to_csv(outputfilename_only, mode='a', index=False, header=False)
                   #dfs_avg.to_csv(outputname_summary, mode='a', index=False, header=False)
            else:
              #result=pd.concat(df1,axis=0)
               df.to_csv(outputfilename_detail, mode='a', index=False, header=True)
               df_LapsStatsOnly.to_csv(outputfilename_only, mode='a', index=False, header=True)
               
######df_laptimedetail is used for adding specific sector time to channel report track section data
    df_laptimedetail=pd.read_csv(outputfilename_detail)
    #storeing track names  
    # Get the list of all files in directory tree at given path
    term=Term + '_Only'

    listof_Files = FileOperation.getListOfFiles(outputdata_path,term,'')

    for a in listof_Files:
       df= pd.read_csv(a)

       #num_laps=len(df)-1
       #df_track=df.groupby(['PID']).sum()

       if df.iloc[0][1]=='mount_panorama, mount_panorama' :
            tr='mount_panorama'
       else:
            tr=df.iloc[0][1]

            
       df_track=df.groupby(['PID'])
       numberof_laps=len(df['Lap'])
       # Creating an empty Dataframe with column names only
       #dfObj = pd.DataFrame(columns=['Track_name', 'Numberof_Cases', 'Numberof_laps'])
       dfObj = pd.DataFrame({'Track_name': tr, 'Numberof_Cases': len(df_track), 'Numberof_laps': numberof_laps,'SVR':-1,'XG':-1,'RF':-1},index=[0])
       #df_carAvg.columns=['_'.join(col).strip() for col in df_carAvg.columns.values]
       
       
       if  os.path.exists(outputname_trackname):
           dfObj.to_csv(outputname_trackname, mode='a', index=False, header=False)
       else:
          dfObj.to_csv(outputname_trackname,index=False,header=True)
   # df = pd.DataFrame(np_track)
   # df=df.drop_duplicates()
   # df.to_csv(outputname_trackname,index=False, header=False)
    
    return (df_nanvalues,df_laptimedetail)
##### Getting total lap time per participant -FOR TIME REPOPRT
#####################################################
def Make_StatsForChannelReport_ForLaps(outputdata_path,listof_Files,Term,Channel_lists,df_nanvalues):


    for a in listof_Files:
        print (a)
        PID=os.path.split(os.path.dirname(a))[1]
        df= pd.read_csv(a)#', usecols=Channel_lists)
        
        #need it for finding name of columns
        #for j in range(6,len((df.columns))):
         #   print (df.columns[j])
        
        df.insert(0, 'PID', PID)

        
        df.rename(columns={df.columns[1]: 'TrackName'},inplace=True)
        df.rename(columns={df.columns[2]: 'CarName'},inplace=True)
        df.rename(columns={df.columns[3]: 'EventDate'},inplace=True)
        df.rename(columns={df.columns[4]: 'EventTime'},inplace=True)
        df.rename(columns={df.columns[5]: 'Lap'},inplace=True)
        df.rename(columns={df.columns[6]: 'LapTime'},inplace=True)
        
        df.drop(['EventTime'], axis=1, inplace=True)

        df.drop(df[df.Lap=='Out Lap'].index, inplace=True)
        df.drop(df[df.Lap=='In Lap'].index, inplace=True)

        
        dfcopy=df.copy()

        for i in range(len(dfcopy)):
            for j in range (len(df_nanvalues)):
                pid=df_nanvalues.iloc[j]['PID'] 
                lap=df_nanvalues.iloc[j]['Lap']
                if dfcopy.iloc[i]['PID']==pid and dfcopy.iloc[i]['Lap']==lap:
                    #non_valueidx=np.append(non_valueidx,i)
                    #df.drop(df[(df['PID'] == pid) & (df['Lap'] == lap)].index, inplace = True)  
                    df=df.drop(df[(df['PID'] == pid) & (df['Lap'] == lap)].index)
                    #print (pid, lap)
                    break
        #creaying header of columns (merging tow first header rows)
        df=Make_HeaderDataFrame(df,6)
                
        if len(df)>0:
        
        #handling missing value for rows
        #thresh=len(df.columns)/2
        #df.dropna(thresh=thresh,inplace=True)
        #for remaining channels
        
        # to interpolate the missing values
        #df.interpolate(method ='linear', limit_direction ='forward',inplace=True)
            df.fillna(value=0,inplace=True)
    
            if df.iloc[0]['TrackName']=='mount_panorama, mount_panorama' :
                tr='mount_panorama'
            else:
                tr=df.iloc[0]['TrackName']
    
    
            if len(df)>0:
                track_path=outputdata_path+tr+'/'
                output_filename_only=track_path+ '_'+ Term + '_Only' +'.csv'
               
    
                
                if not os.path.exists(track_path):
                  os.makedirs(track_path)
             
                if  os.path.exists(output_filename_only):
    
                   df.to_csv(output_filename_only, mode='a', index=False, header=False)
                 
                else:
                   df.to_csv(output_filename_only, index=False, header=True)



##############Channel data track sections
def Make_StatsForChannelReport_ForSectors(outputdata_path,listof_Files,Term,Channel_lists,df_nanvalues,df_laptimedetail):

    
    for a in listof_Files:
        print (a)  
        PID=os.path.split(os.path.dirname(a))[1]
        df= pd.read_csv(a)#', usecols=Channel_lists)
       
        #need it for finding name of columns
        #for j in range(6,len((df.columns))):
         #   print (df.columns[j])
        
        df.insert(0, 'PID', PID)

        
        df.rename(columns={df.columns[1]: 'TrackName'},inplace=True)
        df.rename(columns={df.columns[2]: 'CarName'},inplace=True)
        df.rename(columns={df.columns[3]: 'EventDate'},inplace=True)
        df.rename(columns={df.columns[4]: 'EventTime'},inplace=True)
        df.rename(columns={df.columns[5]: 'Lap'},inplace=True)
        df.rename(columns={df.columns[6]: 'LapTime'},inplace=True)
        df.rename(columns={df.columns[7]: 'Sector'},inplace=True)
        
        df.drop(['EventTime'], axis=1, inplace=True)
        df.drop(df[df.Lap=='Out Lap'].index, inplace=True)
        df.drop(df[df.Lap=='In Lap'].index, inplace=True)
        
        df['Sector']=df['Sector'].apply (lambda x:'Str 0-1 (End)' if x=='Str 0-1(End)' else x)


        df_timeSector=pd.DataFrame()
        np_sectortime=[]
        #df.insert(7, 'SectorTime', 0.0)
        df.insert(len(df.columns), 'missvalue', 0.0)

        #print ('before for')
        for i in range(0,len(df)):
            #finding sector time from df_laptimedetail

            df_timeSector=df_laptimedetail[(df_laptimedetail['PID']==df.iloc[i]['PID']) & (df_laptimedetail['Lap']==df.iloc[i]['Lap']) ]
            #df_timeSector=df_laptimedetail[(df_laptimedetail['PID']==df.iloc[i]['PID']) & (df_laptimedetail['Lap']==df.iloc[i]['Lap']) & (df_laptimedetail['Sector']==sector_name)]


            
            #print (sector_name,'i am here')
            sector_time=0
            sector_name=str(df.iloc[i][6]).replace(" ","")
            if len(df_timeSector)>0:
                #As some of the sector names are wrong (with more or less space), I update here the names
                for k in range (len(df_timeSector.columns)):
                   df_timeSector.rename(columns={df_timeSector.columns[k]: str(df_timeSector.columns[k]).replace(" ","")},inplace=True) 
                   
                sector_time=df_timeSector.iloc[0][sector_name]       

            np_sectortime=np.append(np_sectortime,sector_time)
            
                #df.at[i,'SectorTime']=sector_time
  
            #finding missing value
            for j in range (len(df_nanvalues)):
                pid=df_nanvalues.iloc[j]['PID'] 
                lap=df_nanvalues.iloc[j]['Lap']
                #if df.iloc[i]['PID']==pid and df.iloc[i]['Lap']==lap:
                    #non_valueidx=np.append(non_valueidx,i)
                    #df.drop(df[(df['PID'] == pid) & (df['Lap'] == lap)].index, inplace = True)  
                    #df=df.drop(df[(df['PID'] == pid) & (df['Lap'] == lap)].index)
                df[(df['PID'] == pid) & (df['Lap'] == lap)][len(df)]=1.0
                #print (pid, lap)
                break
        
        df.insert(7, 'SectorTime', np_sectortime)
        df=df.drop(df[(df['missvalue'] == 1)].index)
        #
        #creaying header of columns (merging tow first header rows)
        df=Make_HeaderDataFrame(df,8)
        
        #handling missing value for rows
        #thresh=len(df.columns)/2
        #df.dropna(thresh=thresh,inplace=True)
        #for remaining channels
        
        # to interpolate the missing values
        #df.interpolate(method ='linear', limit_direction ='forward',inplace=True)
        df.fillna(value=0,inplace=True)
        if len(df)!=0:
            if df.iloc[0]['TrackName']=='mount_panorama, mount_panorama' :
                tr='mount_panorama' 
            else:
                tr=df.iloc[0]['TrackName']
                
            df=df.drop(df[(df['SectorTime'] == 0.0)].index)
            for k in range (len(df)):
               df.iloc[k]['Sector']=df.iloc[k]['Sector'].replace(" ","") 
               
            if len(df)>0:
                track_path=outputdata_path+tr+'/'
              
                output_filename_detail=track_path+ '_' +Term+ '_Detail' +'.csv'

                if not os.path.exists(track_path):

                  os.makedirs(track_path)
             
                if  os.path.exists(output_filename_detail):
                   df.to_csv(output_filename_detail, mode='a', index=False, header=False)
                else:
                   df.to_csv(output_filename_detail, mode='a', index=False, header=True)
        
####Merg two Column name of headers if we have channel lists
def Make_HeaderDataFrameWithChannelList(df,Channel_lists,startindex):

   n=len(df.columns)       
   for j in range(startindex,n):
    
        new_name=df.columns[j]

        if cur_col in Channel_lists: 
           for char in cur_col:
                if char in "[%.?/]°123456789":
                    new_name=new_name.replace(char,'')
                else:
                    new_name=cur_col
          #add all stats for a column          
           for h in range (len(StatLists)):
               new_name=new_name+ '_'+ StatLists[h]
               df.rename(columns={df.columns[j]: new_name},inplace=True)
        else:
           df.rename(columns={df.columns[j]: 'DELLL'},inplace=True)
         
     
   df.drop(axis=0 ,index=df.index[0], inplace=True)
   df.drop(labels='DELLL', axis=1, inplace=True)

   return (df)
##########################################################

####Merg two Column name of headers 
def Make_HeaderDataFrame(df,startindex):

   n=len(df.columns)       
   for j in range(startindex,n):
        #print (df.iloc[1,0])
        
        new_name=df.columns[j]+'_'+ str(df.iloc[0,j])
         
  
        for char in new_name:
           if char in "[%.?/]°123456789":
                    new_name=new_name.replace(char,'')

        df.rename(columns={df.columns[j]: new_name},inplace=True)

         
     
   df.drop(axis=0 ,index=df.index[0], inplace=True)
   return (df)
##########################################################
#def Make_StatsForChannelReport_ForSectors(outputdata_path,listof_Files,Term):



  
#### main
print ('Start of execution . . . . .')  
print ('start ti         me: ' , datetime.now())




Term_list=['Time Report','Channel Report'] #- Track Sections

collection='C5/'

inputdata_path="../../../2_DATA/Included Participants - In-lab/"+ collection


outputdata_path="../../../2_DATA/OUTPUT/" + collection+ "LapStats/"

df_nanvalues = pd.DataFrame()

StatLists=['Min','Max','Avg','Start','End','Change','Std Dev']
#
####Creating output directory for the partcipant
if not os.path.exists(outputdata_path):
  os.makedirs(outputdata_path)
else:
#remove folder withh all of its files
    shutil.rmtree(outputdata_path)    
###############
#read track name file

##NOTE: we dont use channel list in this script,we include all channels in states
Channel_file="../../../2_DATA/Channel list.csv"

with open(Channel_file, 'r') as f:
    ch_ = list(csv.reader(f, delimiter=","))
Channel_lists = np.array(ch_)
  

######Time Report##################
# Get the list of all files in directory tree at given path
Excluded_Term=''
listof_Files = FileOperation.getListOfFiles(inputdata_path,Term_list[0],Excluded_Term)
rowcount=len(listof_Files)
print ('Number of files for time report:', rowcount)

##### Getting total lap time per participant
print ('Total lap time per participant . . . . ')
df_nanvalues,df_laptimedetail=Make_StatsForTimeReport(outputdata_path,outputdata_path,listof_Files,Term_list[0],df_nanvalues)

######Channel Report##################
# Get the list of all files in directory tree at given path

Excluded_Term='Track Sections' #for collection 1 & 5 & 3
#Excluded_Term='- Laps' #for collection 2 & 4

listof_Files = FileOperation.getListOfFiles(inputdata_path,Term_list[1],Excluded_Term)

rowcount=len(listof_Files)
print ('Number of files for Channel Report only : ', rowcount)

##### Getting total lap time per participant
print ('Total lap time per participant in Channel report. . . . ')
Make_StatsForChannelReport_ForLaps(outputdata_path,listof_Files,Term_list[1],Channel_lists,df_nanvalues)

######Channel Report track sections##################
# Get the list of all files in directory tree at given path
Excluded_Term='- Laps'  #for collection 1 & 5 & 3
#Excluded_Term='Track Sections'  # for collectio 2 & 4
listof_Files = FileOperation.getListOfFiles(inputdata_path,Term_list[1],Excluded_Term)
rowcount=len(listof_Files)
print ('Number of files for Channel Report deatils:', rowcount)

##### Getting total lap time per participant
print ('Total lap time per participant in Channel report Track section. . . . ')
Make_StatsForChannelReport_ForSectors(outputdata_path,listof_Files,Term_list[1],Channel_lists,df_nanvalues,df_laptimedetail)