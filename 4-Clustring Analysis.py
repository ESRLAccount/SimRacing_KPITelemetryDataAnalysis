# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:36:11 2022

# applying clustring algorithm to find number of cluster (elbow, silhouette), and based on the optimmum 
number of cluster, we do clustring the data, nad save the result in folder ClusterResults for each track
@author: Fazilat
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime
import csv
from FileOperation import *
import shutil
#sklearn 
from kneed import KneeLocator

os.environ["OMP_NUM_THREADS"] = '4'
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn import mixture
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer  

def  Cluster_LapData(listof_Files,output_file,trackname,Filename_term):
   kmeans_kwargs = {
       "init": "k-means++",
       "n_init": 2,
       "max_iter": 50000,
       "random_state": 42,
   }
   for a in listof_Files:

      df= pd.read_csv(a)#,usecols=('Totals_mean','Totals_max','Totals_min','Totals_median','Totals_std','outlier'))  
      

      #handling missing value for rows
      ##TO DO if need change the rule ???????????????????
      df.dropna(inplace=True)

      #df.drop(df[df.outlier>0].index, inplace=True)

      df.drop_duplicates(inplace=True)
      if 'Lap_index' in df.columns:
          df.drop(['Lap_index'], axis=1, inplace=True)



      df_original=df.copy()
      
      if df.columns[5]=='L_a_p' or df.columns[5]=='Lap_size' or df.columns[5]=='Lap':
        df.rename(columns={df.columns[5]: 'Lap_size'},inplace=True)
        df.drop(['Lap_size','EventTime'], axis=1, inplace=True)
      else:
        df.drop(df.columns[4], axis=1, inplace=True) #drop lap column from channel report
        
      if 'lap' in df:
          df.drop ('Lap',axis=1, inplace=True)
          
      if 'Sector' in df:
          df.drop(['PID','TrackName','CarName','EventDate','Sector'], axis=1, inplace=True)
      else:
          df.drop(['PID','TrackName','CarName','EventDate'], axis=1, inplace=True)
          
      if Clusterwithonlylaptime==True:
         X=df['LapTime'].values.reshape(-1,1)
      else:
         X=df
      #Call elbow algoritm to find number of cluster
      num_cluster_el=elbow(X,kmeans_kwargs,trackname)
      num_cluster_si=silhouette(X,kmeans_kwargs,trackname)
      #num_cluster_db=make_clustring_DBSCAN (X)
      print ('num_cluster for elbow ', trackname , num_cluster_el)
      print ('num_cluster for silhouette ', trackname , num_cluster_si)
      
      #Call Clustring algoritm to clustr the data and label the data
      if num_cluster_si is not None:
          #num_cluster_si=4
          make_clustring(df_original,X,kmeans_kwargs,num_cluster_si,output_file,Filename_term)
      else:
          if  num_cluster_el is not None:
              #num_cluster_si=4
              make_clustring(df_original,X,kmeans_kwargs,num_cluster_el,output_file,Filename_term)
              
      #in case that we want use Kernel Density 
      #univariate_data(df)
      
      return()

def get_sectorcode(sectorname):
    x=SectorMap[sectorname]
    return(x)

def analysis_clusteringSectors(df,file):
   df_cl=df.groupby(['cluster_label','Sector'])['Sector'].count()
   df_cl.to_csv(file,index=True)
   print (df_cl)
##################################################
#this used for finding simsilarity bnetween different sectors in a lap , differtent sectors by considering some driving metrics
def  Cluster_SectorDataforFindingDifferentkindofSectors(listof_Files,output_file,trackname,Filename_term):
   kmeans_kwargs = {
       "init": "k-means++",
       "n_init": 1,
       "max_iter": 1000,
       "random_state": 0,
   }
   for a in listof_Files:
      
      df= pd.read_csv(a)#,usecols=('Totals_mean','Totals_max','Totals_min','Totals_median','Totals_std','outlier'))  
      

      
      #handling missing value for rows
      ##TO DO if need change the rule ???????????????????
      df.dropna(inplace=True)
      
      df.insert(len(df.columns), 'Sectorcode', '')

      df['Sectorcode']=df['Sector'].apply(lambda x: get_sectorcode(x))
      
          
      df['Sector']=df['Sector'].apply (lambda x:'Str 0-1 (End)' if x=='Str 0-1(End)' else x)
      
      ds.rename(columns={'sectorname':'Sector'},inplace=True)  

      df=pd.merge(
        df,
        ds,
        how="inner",
        on=['Sector'],
        left_on=None,
        right_on= None,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
                 )    
      #we consider all of data including outlier to have more data
      #df.drop(df[(df.outlier>0)].index, inplace=True)
      #df=df[(df.Sector==SectorName)]
   
      df.drop_duplicates(inplace=True)
      if 'Lap_index' in df.columns:
          df.drop(['outlier','Lap_index'], axis=1, inplace=True)
          df.drop(['Lap_index'], axis=1, inplace=True)
      else:
          df.drop(['outlier'], axis=1, inplace=True)

      df = df[df["Type"] == 'T'] 
      df_original=df.copy()
      
      if df.columns[5]=='L_a_p' or df.columns[5]=='Lap_size' or df.columns[5]=='Lap':
        df.rename(columns={df.columns[5]: 'Lap_size'},inplace=True)
        df.drop(['Lap_size','EventTime'], axis=1, inplace=True)
      else:
        df.drop(df.columns[4], axis=1, inplace=True) #drop lap column from channel report
        

      
      #df.drop(['PID','TrackName','CarName','EventDate','Sector'], axis=1, inplace=True)
      df.drop(['PID','Sector'], axis=1, inplace=True)
      
     
      if Clusterwithonlylaptime==True:
         X=df[{'SectorTime'}]
      else:
         X=df[channellist]
    
      #Call elbow algoritm to find number of cluster
      num_cluster_el=elbow(X,kmeans_kwargs,trackname)
      num_cluster_si=silhouette(X,kmeans_kwargs,trackname)
      #num_cluster_db=make_clustring_DBSCAN (X)
      print ('num_cluster for elbow ', trackname , num_cluster_el)
      print ('num_cluster for silhouette ', trackname , num_cluster_si)
      
      #Call Clustring algoritm to clustr the data and label the data
      if num_cluster_si is not None:
          #num_cluster_si=4
         df_cl= make_clustring_Sector(df_original,X,kmeans_kwargs,num_cluster_si,output_file,Filename_term)
      else:
          if  num_cluster_el is not None:
              #num_cluster_si=4
              df_cl=make_clustring_Sector(df_original,X,kmeans_kwargs,num_cluster_el,output_file,Filename_term)
              
      #in case that we want use Kernel Density 
      #univariate_data(df)
      return(df_cl)

######################
def  Cluster_SectorData(listof_Files,output_file,trackname,Filename_term,SectorName):
   kmeans_kwargs = {
       "init": "k-means++",
       "n_init": 1,
       "max_iter": 1000,
       "random_state": 0,
   }
   for a in listof_Files:
      
      df= pd.read_csv(a)#,usecols=('Totals_mean','Totals_max','Totals_min','Totals_median','Totals_std','outlier'))  
      

      
      #handling missing value for rows
      ##TO DO if need change the rule ???????????????????
      df.dropna(inplace=True)
      
      
      
      df.drop(df[(df.outlier>0)].index, inplace=True)
      df=df[(df.Sector==SectorName)]
   
      df.drop_duplicates(inplace=True)
      if 'Lap_index' in df.columns:
          df.drop(['outlier','Lap_index'], axis=1, inplace=True)
          df.drop(['Lap_index'], axis=1, inplace=True)
      else:
          df.drop(['outlier'], axis=1, inplace=True)


      df_original=df.copy()
      
      if df.columns[5]=='L_a_p' or df.columns[5]=='Lap_size' or df.columns[5]=='Lap':
        df.rename(columns={df.columns[5]: 'Lap_size'},inplace=True)
        df.drop(['Lap_size','EventTime'], axis=1, inplace=True)
      else:
        df.drop(df.columns[4], axis=1, inplace=True) #drop lap column from channel report
        

      
      df.drop(['PID','TrackName','CarName','EventDate','Sector'], axis=1, inplace=True)
      
      
      if Clusterwithonlylaptime==True:
         X=df[{'SectorTime'}]
      else:
         X=df[channellist]
      """
      num_pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy="median")),
         ('std_scaler', StandardScaler()),
         ])

      X= num_pipeline.fit_transform(X)
      """      
      #Call elbow algoritm to find number of cluster
      num_cluster_el=elbow(X,kmeans_kwargs,trackname)
      num_cluster_si=silhouette(X,kmeans_kwargs,trackname)
      #num_cluster_db=make_clustring_DBSCAN (X)
      print ('num_cluster for elbow ', trackname , num_cluster_el)
      print ('num_cluster for silhouette ', trackname , num_cluster_si)
      
      #Call Clustring algoritm to clustr the data and label the data
      if num_cluster_si is not None:
          #num_cluster_si=4
          make_clustring_Sector(df_original,X,kmeans_kwargs,num_cluster_si,output_file,Filename_term)
      else:
          if  num_cluster_el is not None:
              #num_cluster_si=4
              make_clustring_Sector(df_original,X,kmeans_kwargs,num_cluster_el,output_file,Filename_term)
              
      #in case that we want use Kernel Density 
      #univariate_data(df)
      return()
      
#*******************************************************
def  silhouette(X, kmeans_kwargs, trackname):
    
 # A list holds the silhouette coefficients for each k
   silhouette_coefficients = []
     
   num_cluster=int(len(X)-2)
   max_cluster=6  # no need to have more than 6 cluster
   n=min(num_cluster, max_cluster)
   max_score=-1
   b_cluster=0
   if num_cluster>=3 : 
   # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, n):
        gmm=mixture.GaussianMixture(n_components=k)
        labels =gmm.fit_predict(X)
     
        #kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        #kmeans.fit(X)
        #score = silhouette_score(X, kmeans.labels_)
        score = silhouette_score(X, labels)
        if max_score<score:
            max_score=score
            b_cluster=k
        silhouette_coefficients.append(score)  

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, n), silhouette_coefficients)
    plt.xticks(range(2, n))
    plt.title('The silhouette_coefficients for :' + trackname + str(len(X)) +Filename_term)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    print ('Silhouette',b_cluster,max_score)
    return (b_cluster)


def  silhouette_Tree(X, kmeans_kwargs, trackname):
 # A list holds the silhouette coefficients for each k
   silhouette_coefficients = []
     
   num_cluster=int(len(X)-2)
   max_cluster=6  # no need to have more than 6 cluster
   n=min(num_cluster, max_cluster)
   max_score=-1
   b_cluster=0
   if num_cluster>=3 : 
   # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, n):
        gmm=mixture.GaussianMixture(n_components=k)
        labels =gmm.fit_predict(X)
     
        #kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        #kmeans.fit(X)
        #score = silhouette_score(X, kmeans.labels_)
        score = silhouette_score(X, labels)
        if max_score<score:
            max_score=score
            b_cluster=k
        silhouette_coefficients.append(score)  

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, n), silhouette_coefficients)
    plt.xticks(range(2, n))
    plt.title('The silhouette_coefficients for :' + trackname + str(len(X)) +Filename_term)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    return (b_cluster)

#*******************************************************
def elbow (X, kmeans_kwargs,trackname):
    cs = []
    num_cluster=int(len(X)-2)
    max_cluster=6  # no need to have more than 6 cluster
    n=min(num_cluster, max_cluster)

    if num_cluster>=3 : 
       
     for i in range(1, n):
            kmeans = KMeans(n_clusters=i, **kmeans_kwargs)
            kmeans.fit(X)
            cs.append(kmeans.inertia_)
     plt.plot(range(1, n), cs)
     plt.title('The Elbow Method for :' + trackname + str(len(X)) + Filename_term)
     plt.xlabel('Number of clusters')
     plt.ylabel('CS')
     plt.show()    
     kl = KneeLocator(range(1, n), cs, curve="convex", direction="decreasing")
     return(kl.elbow)

#*******************************************************
def  make_clustring_Sector(df_original,df,kmeans_kwargs,k,output_file,Filename_term)  :

    k=2
    OMP_NUM_THREADS=4
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    #X=df['LapTime'].values.reshape(-1,1)
    X=df

    # this is for using EM clustering algorithm
    """gmm=mixture.GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=1e-06,
                            max_iter=100, n_init=1, init_params='kmeans', 
                            random_state=None, warm_start=False, verbose=0, verbose_interval=10)
    #gmm=mixture.GaussianMixture(n_components=3)
    labels =gmm.fit_predict(X)"""

    print (len(X))
    kmeans.fit(X)
  
    labels=kmeans.fit_predict(X)
    # Adding the results to a new column in the dataframe
    df_original["cluster"] =labels #kmeans.labels_

    #making readable labels for cluster
    df_original=(Create_clusterLabel(df_original))

    df_original.to_csv(output_file,index=False)

    
      
    #plotting the results:

    plt.scatter(df_original['SectorTime'], df_original['Sector'],c=labels, s=40,cmap='viridis')
    title='Clusters for ' + Filename_term
    plt.title(title)        
    #plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'red')
    plt.legend()
    plt.show()
    return (df_original)

#*******************************************************
def  make_clustring(df_original,df,kmeans_kwargs,k,output_file,Filename_term)  :

    k=2
   
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    #X=df['LapTime'].values.reshape(-1,1)
    X=df

    # this is for using EM clustering algorithm
    """gmm=mixture.GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=1e-06,
                            max_iter=100, n_init=1, init_params='kmeans', 
                            random_state=None, warm_start=False, verbose=0, verbose_interval=10)
    #gmm=mixture.GaussianMixture(n_components=3)
    labels =gmm.fit_predict(X)"""


    kmeans.fit(X)

    labels=kmeans.fit_predict(X)

    # Adding the results to a new column in the dataframe
    df_original["cluster"] =labels #kmeans.labels_

    #making readable labels for cluster
    df_original=(Create_clusterLabel(df_original))
    
    df_original.to_csv(output_file,index=False)

    
      
    #plotting the results:

    plt.scatter(df_original['LapTime'], df_original['cluster'],c=labels, s=40,cmap='viridis')
    title='Clusters for ' + Filename_term
    plt.title(title)        
    #plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'red')
    plt.legend()
    plt.show()

def make_clustring_tree (X):
    
   from sklearn.cluster import DBSCAN
   from sklearn import metrics

   
   db = DBSCAN(eps=0.3, min_samples=10).fit(X)  
   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_

   
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)
   print("Estimated number of clusters: %d" % n_clusters_)
   print("Estimated number of noise points: %d" % n_noise_)

   print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
   return (n_clusters_)
def univariate_data(df):

    from sklearn.neighbors import KernelDensity



    a = df['LapTime']

    kde = KernelDensity(kernel='epanechnikov', bandwidth=2).fit(a.values.reshape(-1,1))
    print (kde.score_samples(a.values.reshape(-1,1)))
    s = linspace(0,50)
    e = kde.score_samples(s.reshape(-1,1))
    plt.fill(s.reshape(-1,1), np.exp(e), c='cyan')
    plt.show()
    #plot(s, e)
    yhat = kde.predict(a)
    # retrieve unique clusters
    clusters = unique(yhat)
    print (clusters)
    from scipy.signal import argrelextrema
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    print ("Minima:", s[mi])
    print ("Maxima:", s[ma])
 
    print (a[a < mi[0]], a[(a >= mi[0]) * (a <= mi[1])], a[a >= mi[1]])
    
    
##########################################################

def Create_clusterLabel(df_original):
    clusterlists=[]
    df_sorted=df_original.groupby(['cluster'])
    df_sorted=df_sorted.agg(mean_laptime=('LapTime', 'mean'), 
                               min_laptime=('LapTime', 'min'))
    

    df_sorted.sort_values(by=['mean_laptime'],inplace=True)
    clusterlists=df_sorted.index.tolist()
    print (df_sorted)

    num_cluster=len(clusterlists)
    
    for i in range(num_cluster):
        if num_cluster==2:
           df_original.loc[df_original['cluster'] == clusterlists[i], 'cluster_label'] = ClusterName_2[i]
        if num_cluster==3:
            df_original.loc[df_original['cluster'] == clusterlists[i], 'cluster_label'] = ClusterName_3[i]           
        if num_cluster==4:
            df_original.loc[df_original['cluster'] == clusterlists[i], 'cluster_label'] = ClusterName_4[i]           
        if num_cluster==5:
            df_original.loc[df_original['cluster'] == clusterlists[i], 'cluster_label'] = ClusterName_5[i]           
        if num_cluster==6:
            df_original.loc[df_original['cluster'] == clusterlists[i], 'cluster_label'] = ClusterName_6[i]           
        if num_cluster>6    :
            print ('num_cluster bigger than 6')
    return (df_original)       
  
#### main
print ('Start of execution . . . . .')  
print ('start time: ' , datetime.now())

ClusterName_2=['Fast','Slow']
ClusterName_3=['Fast','Medium','Slow']
ClusterName_4=['Fast','Fast','Medium','Slow']
ClusterName_5=['Fast','Very good','Good','Medium','Slow']
ClusterName_6=['Fast','Very good','Good','Medium','Slow','Slow']

collection = 'C5/'

inputdata_path="../../../2_DATA/OUTPUT/" + collection + "LapStats - C2C3C5/"

#inputdata_path="../../../2_DATA/OUTPUT/" + collection + "ChannelReportData/"


inputsector_path="../../../2_DATA/Included Participants - In-lab/TrackInfo/"
SectorInfor_file=inputsector_path+'brands_hatch, brands_hatch.csv'
ds = pd.read_csv(SectorInfor_file,header=0,parse_dates=[0])  
ds.set_index('sectorname')

#files_list=['_Channel Report_Only','_Time Report_Only','_Time Report_Detail','_Channel Report_Detail']#,'_Time Report_Detail']#,'_Time Report_OnlySummary','_Time Report_DetailSummary']
files_list=['_Channel Report_Only','_Channel Report_Detail','_Channel Report_Detail_NewSector']#,'_Channel Report_AVG']#,'_Time Report_Detail']#,'_Time Report_OnlySummary','_Time Report_DetailSummary']

#

SectorMap={'Str 0-1(End)':0,'Turn 1':1,'Str 1-2':2,'Turn 2':3,'Str 2-3':4,'Turn 3':5,'Str 3-4':6,
'Turn 4':7,'Str 4-5':8,'Turn 5':9,'Str 5-6':10,'Turn 6':11,'Str 6-7':12,'Turn 7':13,'Str 7-8':14,
'Turn 8':15,'Str 8-9':16,'Turn 9':17,'Str 0-1 (Start)':18,'Str 0-1 (End)':0}

#channellist=['SectorTime','Sectorcode','sector_len','BRAKE _Avg','THROTTLE _Avg','STEERANGLE _Avg','lane deviation_Avg','glat G_Avg','glong G_Avg','Corr Speed kmh_Avg']
channellist=['SectorTime','Sectorcode','sector_len','BRAKE','THROTTLE','STEERANGLE','lane deviation','glat','glong','SPEED']

Clusterwithonlylaptime=True


if Clusterwithonlylaptime==True :
  outputdata_path="../../../2_DATA/OUTPUT/"+ collection + "ClusterResults - C2C3C5/"
else:
    outputdata_path="../../../2_DATA/OUTPUT/"+ collection + "ClusterResults/"

####Creating output directory for the results
#if not os.path.exists(outputdata_path):
#  os.makedirs(outputdata_path)
#else:
    #remove folder withh all of its files
#  shutil.rmtree(outputdata_path)
  

    
#read track name file
trackname_file=inputdata_path+'Trackname.csv'
with open(trackname_file, 'r') as f:
    df_tn = list(csv.reader(f, delimiter=","))
df_tn = np.array(df_tn)

#open each summary stats file related to the track
for i in range(1,len(df_tn)):
  for k in range (0, len(files_list))   :
    print (k)
  #for k in range (0, 1)   :
    trackname=df_tn[i][0] 

    Filename_term=files_list[k]

    
    path=inputdata_path+trackname+'/'
    ###########################################
    out_path=outputdata_path+trackname+ '/'

    output_filename=out_path+'_CL' + Filename_term+'.csv'

    if  os.path.exists(output_filename):
        os.remove(output_filename)
      
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    ##########################################  


      
    Excluded_Term='Summary'  
    # Get the list of all stats lap files in directory tree at given path
    listof_Files = FileOperation.getListOfFiles(path,Filename_term,Excluded_Term)
    print (path,Filename_term,Excluded_Term)
    Cluster_LapData(listof_Files, output_filename, trackname, Filename_term)
    
    if k==1 :
        #just for channel report detail needs to be run
        # Get the list of all stats lap files in directory tree at given path
        listof_Files = FileOperation.getListOfFiles(path,Filename_term,Excluded_Term)
        #clustring sector of channel data to categorize particpant based on sector time (specific corner)
        for j in range (len (ds)):
            print (listof_Files)
            SectorName=ds.iloc[j,0]
            #SectorName='Turn 1' 
            output_filename=out_path+'_CL_Sector_' + SectorName + '.csv'
            if  os.path.exists(output_filename):
                os.remove(output_filename)
              
            if not os.path.exists(out_path):
              os.makedirs(out_path)
              
    
            #Cluster_SectorData(listof_Files,output_filename,trackname,Filename_term,SectorName)
        
        #cluster whole laps by considering sectors to find different groups of sectors
    if k==1:    
        output_filename=out_path+'_CL' + Filename_term+'Different_Sectors.csv'
        clustering_statfile=   out_path+'_CL' + Filename_term+'clustering_STATS.csv'
        df_cl=Cluster_SectorDataforFindingDifferentkindofSectors(listof_Files,output_filename,trackname,Filename_term)
        analysis_clusteringSectors (df_cl,clustering_statfile)
    