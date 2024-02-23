# -*- coding: utf-8 -*-
"""
Created on 16 July 2022

applying 4 different algorithm to clustring result to find what are the most important features
the results are in FeatureRanking folder
the MAE are saved for each track /algorithm in track file
@author: Fazilat

"""

import numpy as np
import pandas as pd
import os
import shutil
import pickle

import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
 
#from scikeras.wrappers import KerasClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import classification_report

from sklearn.inspection import permutation_importance

from sklearn.svm import SVR
    

from scipy.stats import uniform
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE  
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier

import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer    
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import normalize
from statsmodels.stats.outliers_influence import variance_inflation_factor



import sys


#from imblearn.under_sampling import TomekLinks
#from imblearn.pipeline import Pipeline, make_pipeline

import csv
from FileOperation import *
import seaborn as sns
import sklearn_relief as relief
import shap

finall_importances = pd.DataFrame(columns =['cluster','Attribute','Importance'])
#################################################
def Variance_inflationfactor_Filtering (df,features):
    from patsy import dmatrices
    df2=df.copy()
    df2.drop(['LapTime'],axis=1,inplace=True)


    count=0

    done=False
    while done==False:
        
        X=df[features]
        features_afterVIF=[]
        #calculate VIF for each explanatory variable
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print (vif)
        maxvif=0
        #finding max vif
        for i in range (len(vif)):
            if vif.iloc[i,0]>maxvif:
               maxvif= vif.iloc[i,0]
               maxvariable= vif.iloc[i,1]
        print (maxvif,maxvariable) 
        if maxvif<10000:
            done=True

            
        #features.remove(maxvariable)

        features = np.delete(features, np.where(features == maxvariable))
        count=count+1
        print ('after  round', count)
            #features_afterVIF=np.append(features_afterVIF,vif.iloc[i,1])
    #view VIF for each explanatory variable
    print ('after VIF')
   # print (features)
    return (features)
#################################################
def  Wrapper_FilterFeatureSelection (y,df,df_scaled):
    featurelist_afterLV=list()
    featurelist_afterMI=df_scaled.columns
    #print (featurelist_afterMI)
    #featurelist_afterMI=MI_Filtering (y,df_scaled)
    #ReliefF_Filtering (X,y,df)
    #features_afterVIF= [    'understeer_Avg',
 # 'BRAKE _Avg',    'Corr Speed kmh_Avg', 
  #  'abs Steerangle_Avg', 'lane deviation_Avg','Oversteer _Avg',   
  #   'Brake_duration_Avg', 'Trail_braking_duration_Avg', 'Throttle release application_Avg', 'Steer_PeakTime_Avg', 'steering duration_Avg']
    
    
    
    """features_afterVIF= [ 'LapTime','abs glat_Min', 'abs glat_Avg', 'abs glat_Start', 
                        'abs glat_End', 'abs ROTY _Avg', 'BRAKE _Max', 'BRAKE _Start', 
   'BRAKE _End', 'Corr Speed kmh_Min', 'Corr Speed kmh_Max',
   'Corr Speed kmh_Avg', 'Corr Speed kmh_Start', 
     'Corr Speed kmh_End', 'Corr Speed kmh_Std Dev', 'RPMS min_Min', 
     'RPMS min_Max', 'RPMS min_Avg', 'RPMS min_Start', 
    'RPMS min_End', 'RPMS min_Std Dev', 'THROTTLE _Max', 'THROTTLE _Avg', 
    'THROTTLE _Std Dev', 
     'abs Steerangle_Max', 'BRAKE _Max', 
     'Corr Speed kmh_Min', 'Corr Speed kmh_Max', 'Corr Speed kmh_Start', 
     'RPMS min_Min', 'RPMS min_Max', 
    'RPMS min_Start', 'RPMS min_End', 'RPMS min_Change', 'RPMS min_Std Dev', 
     'Brake_length', 'Brake_duration', 'Trail_braking_duration',
     'Throttle release application', 'Throttle_time',  'Steer_Peakvalue', 
     'Steer_PeakTime', 'Throttle application speed','Steer_duration', 
     'Steer_reductionSpeed', 'number of crossing', 'angular travel', 
     'number of direction changed']
    """
    
   # featurelist_afterLV=lowVariance_Filtering(y,df_scaled)
    #featurelist_afterunion=featurelist_afterMI+featurelist_afterLV
    features_afterVIF=featurelist_afterMI
    print (features_afterVIF)
   # print (featurelist_afterMI)
    #Find the union of two NumPy arrays
    #featurelist_afterunion=np.union1d(featurelist_afterMI, featurelist_afterLV)
    #print (len(featurelist_afterunion),    'number of features after Filtering ' )
    #print (featurelist_afterunion)
    #features_afterVIF=correlation(df_scaled, 0.75,features_afterVIF)
    #featurelist_afterunion.remove('LapTime')
    #featurelist_afterunion.remove('cluster')
    #featurelist_afterPearson=featurelist_afterunion
    #features_afterVIF=Variance_inflationfactor_Filtering(df,features_afterVIF)
    #print (features_afterVIF,len(features_afterVIF))
    #features_afterVIF=correlation_analysis(features_afterVIF,df_scaled)
   # print (len(features_afterVIF),'before person')
    features_afterVIF=Pearson_Filtering(features_afterVIF,y,df_scaled)
    #print(featurelist_afterPearson)    
    #return (features_afterVIF)
    #features_afterVIF=['abs glat_Min', 'abs glat_Avg', 'abs ROTY _Min', 'Corr Speed kmh_Avg', 'understeer_Min', 
     #               'abs Steerangle_Max', 'BRAKE _Max', 'lane deviation_Avg','abs Steerangle_Std Dev','BRAKE _Avg','RPMS min_Avg', 'RPMS min_Std Dev',
      #             'Throttle release duration_Avg', 'Steer_Peakvalue_Avg','Steer_PeakTime_Avg','Trail braking phase_Avg','THROTTLE _Avg']
    
    #features_afterVIF=featurelist_afterMI


  # featurelist_afterMI=['LapTime', 'abs glat_Max', 'abs glat_Avg', 'abs glat_Start', 'abs glat_End', 'abs glat_Change', 'abs glat_Std Dev', 'abs glong_Min', 'abs glong_Max', 'abs glong_Avg', 'abs glong_Start', 'abs glong_End', 'abs glong_Change', 'abs glong_Std Dev', 'abs ROTY _Min', 'abs ROTY _Max', 'abs ROTY _Avg', 'abs ROTY _Start', 'abs ROTY _End', 'abs ROTY _Change', 'abs ROTY _Std Dev', 'abs Steerangle_Min', 'abs Steerangle_Max', 'abs Steerangle_Avg', 'abs Steerangle_Start', 'abs Steerangle_End', 'abs Steerangle_Change', 'abs Steerangle_Std Dev', 'BRAKE _Max', 'BRAKE _Avg', 'BRAKE _Start', 'BRAKE _End', 'BRAKE _Change', 'BRAKE _Std Dev', 'Corr Speed kmh_Min', 'Corr Speed kmh_Max', 'Corr Speed kmh_Avg', 'Corr Speed kmh_Start', 'Corr Speed kmh_End', 'Corr Speed kmh_Change', 'Corr Speed kmh_Std Dev', 'G_LAT ms_Min', 'G_LAT ms_Max', 'G_LAT ms_Avg', 'G_LAT ms_Start', 'G_LAT ms_End', 'G_LAT ms_Change', 'G_LAT ms_Std Dev', 'G_LON ms_Min', 'G_LON ms_Max', 'G_LON ms_Change', 'G_LON ms_Std Dev', 'GEAR no_Min', 'GEAR no_Max', 'GEAR no_Avg', 'GEAR no_Start', 'GEAR no_End', 'GEAR no_Change', 'glat G_Min', 'glat G_Max', 'glat G_Avg', 'glat G_Start', 'glat G_End', 'glat G_Change', 'glat G_Std Dev', 'glong G_Min', 'glong G_Max', 'glong G_Avg', 'glong G_Start', 'glong G_End', 'glong G_Change', 'lane deviation_Avg', 'lane deviation_Start', 'lane deviation_End', 'lane deviation_Std Dev', 'Oversteer _Min', 'Oversteer _Max', 'Oversteer _Avg', 'Oversteer _Start', 'Oversteer _End', 'Oversteer _Change', 'Oversteer _Std Dev', 'ROTY s_Min', 'ROTY s_Max', 'ROTY s_Avg', 'ROTY s_Start', 'ROTY s_End', 'ROTY s_Change', 'ROTY s_Std Dev', 'RPMS min_Min', 'RPMS min_Max', 'RPMS min_Avg', 'RPMS min_Start', 'RPMS min_End', 'RPMS min_Change', 'RPMS min_Std Dev', 'SPEED kmh_Min', 'SPEED kmh_Max', 'SPEED kmh_Avg', 'SPEED kmh_Start', 'SPEED kmh_End', 'SPEED kmh_Change', 'SPEED kmh_Std Dev', 'STEERANGLE _Min', 'STEERANGLE _Max', 'STEERANGLE _Avg', 'STEERANGLE _Start', 'STEERANGLE _End', 'STEERANGLE _Change', 'STEERANGLE _Std Dev', 'steered angle rad_Min', 'steered angle rad_Max', 'steered angle rad_Avg', 'steered angle rad_Start', 'steered angle rad_End', 'steered angle rad_Change', 'steered angle rad_Std Dev', 'steering reversal rate per lap_Min', 'steering reversal rate per lap_Max', 'steering reversal rate per lap_Avg', 'steering reversal rate per lap_Start', 'steering reversal rate per lap_End', 'steering reversal rate per lap_Change', 'steering reversal rate per lap_Std Dev', 'steering reversal rate per sector_Max', 'steering reversal rate per sector_Avg', 'steering reversal rate per sector_Start', 'steering reversal rate per sector_End', 'steering reversal rate per sector_Change', 'TC _Start', 'TC _Change', 'THROTTLE _Avg', 'THROTTLE _Start', 'THROTTLE _End', 'THROTTLE _Change', 'THROTTLE _Std Dev', 'understeer_Max', 'understeer_Avg', 'understeer_Start', 'understeer_End', 'understeer_Change', 'understeer_Std Dev', 'BrakeLength_Avg', 'Brake_length', 'Brake_duration', 'Trail_braking_duration', 'Throttle_length', 'Steer_Peakvalue', 'Steer_PeakTimeg', 'Steer_duration', 'Steer_reductionSpeed', 'number of crossing', 'angular travel', 'number of direction changed']
  
    return (features_afterVIF)


################################################
def Pearson_Filtering(name_dataset,y,df):

    n=len(name_dataset)
    part1_dataset=[]
    part2_dataset=[]

    #correlation_mat = df[name_dataset].corr()
   # heatmap = sns.heatmap(correlation_mat, cmap="Blues", annot = True)  
   # heatmap.set (xlabel = 'IRIS values on x axis',ylabel = 'IRIS values on y axis\t', title = "Correlation matrix of IRIS dataset\n")  
    #plt.show()
    #sns.heatmap(correlation_mat, annot = True)
   # fig, ax = plt.subplots(figsize=( 70,30))  
   # mask = np.triu(np.ones_like(correlation_mat, dtype=np.bool))
   #heatmap=sns.heatmap(correlation_mat, vmin=-1, mask=mask, vmax=1,annot=True, linewidths=.5, ax=ax)
   # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':15}, pad=12);
    #plt.show()
    
    

    for i in range(n):
        if i<n/2:
            part1_dataset=np.append(part1_dataset,name_dataset[i])
        else:
            part2_dataset=np.append(part2_dataset,name_dataset[i])
    df_part1_dataset=df[part1_dataset]
   # print (df_part1_dataset.isnull().sum()
    df_part2_dataset=df[part2_dataset]
   # print(len(df_part1_dataset),len(df_part2_dataset),'gggg')
    featurelist_correlated=[]

    for i in range(0,len(df_part1_dataset.columns)):
        for j in  range(0,len(df_part2_dataset.columns)):
                #print (df_part2_dataset[df_part2_dataset.columns[j]],df_part1_dataset[df_part1_dataset.columns[i]])
                corr_1=np.abs(df_part1_dataset[df_part1_dataset.columns[i]].corr(df_part2_dataset[df_part2_dataset.columns[j]]))
                #if df_part1_dataset.columns[i] not in featurelist_afterPearson:
                 #   featurelist_afterPearson=np.append(featurelist_afterPearson,df_part1_dataset.columns[i])
                #corr_1=np.abs(reduced_dataset[df_part1_dataset.columns[i]].corr(df_part2_dataset[df_part1_dataset.columns[j]]))
                if corr_1 >0.75:
                   #print( df_part1_dataset.columns[i] , " is correlated  with ", df_part2_dataset.columns[j])
                    if df_part2_dataset.columns[j] not in featurelist_correlated:
                        featurelist_correlated=np.append(featurelist_correlated,df_part2_dataset.columns[j])
                #else:
                 #   if df_part2_dataset.columns[j] not in featurelist_afterPearson:
                  #      featurelist_afterPearson=np.append(featurelist_afterPearson,df_part2_dataset.columns[j])
                    
                #elif corr_1>0.75:
                    #print( reduced_dataset.columns[i] , " is highly  correlated  with ", reduced_dataset.columns[j])
    print (len(featurelist_correlated), 'number of features after pearson correlation' )
    featurelist_afterPearson=[]
    for i in range (len(name_dataset)):
        if name_dataset[i] not in featurelist_correlated:
            featurelist_afterPearson=np.append(featurelist_afterPearson,name_dataset[i])
   
    reduced_dataset=df[featurelist_afterPearson]
    mi = mutual_info_regression(reduced_dataset, y)
    mi = pd.Series(mi)
    mi.index = reduced_dataset.columns
    mi.sort_values(ascending=False)
    
    #mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
    
    return(featurelist_afterPearson)

################################################
#Mutal information filtering
def MI_Filtering (y,df):

    #full_data=df.drop("cluster_label", axis=1)

    importances = df.drop("cluster", axis=1).apply(lambda x: x.corr(df.cluster))

    indices = np.argsort(importances)
    print (importances)
    names=df.columns

    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [names[i] for i in indices])
    plt.xlabel('Relative Importance')
    #plt.show()
    
    name_dataset=list()
    for i in range(0, len(indices)):
        if np.abs(importances[i])>0.00001:
            #print(names[i])
            name_dataset.append(names[i])
                
    print(len(name_dataset),'MI_Filtering')
    #print (name_dataset)
    return(name_dataset)


###########################
def lowVariance_Filtering(y,data):
    #storing the variance and name of variables

    norm = normalize(data)
    data_scaled = pd.DataFrame(norm)
    variance=data_scaled.var()


    columns = data.columns

  #saving the names of variables having variance more than a threshold value
    
    variable = list()
    print (len(variance),'low varience')
    print (variance)
    for i in range(0,len(variance)):

        if variance[i]>=0:#setting the threshold as 1%
            variable.append(columns[i])
            #print (variance[i],columns[i]) 
            
    print (len(variable),'len')
    return (variable)



   
##############FeacureSeclection for Sectors
def Feature_SelectionSector(listof_Files,output_filename,trackname,channel_list,finall_importances):

 mae_PCA=0
 mae_Regall=0
 mae_xgall=0
 mae_RFall=0


    #Read summary file to find outlier
 for a in listof_Files:  
     print (a)
     if a.find("_CL_Sector"):
       x1=a.find("_CL_Sector_")
       x2=a.find(".csv")
       Sectorname=a[x1+len("_CL_Sector_"):x2]
       
       featurelist_forML=[]

       df= pd.read_csv(a,usecols=channel_list)

       num_dataAll = df.select_dtypes(include='number')
       if 'LapTime' in num_dataAll :       
           num_dataAll.drop(['LapTime'], axis=1, inplace=True)    
           
       if 'SectorTime' in num_dataAll :
           num_dataAll.drop(['SectorTime'], axis=1, inplace=True)
  

       y=num_dataAll['cluster']
       if 'cluster' in num_dataAll :
           num_dataAll.drop(['cluster'], axis=1, inplace=True)  
           
       featurelist_forML=Wrapper_FilterFeatureSelection (y,df,num_dataAll)



       num_dataAll=num_dataAll[featurelist_forML]


       #num_dataAll.drop(['cluster','SectorTime'], axis=1, inplace=True)
       
       #outputname=output_filename+'_'+Sectorname +'_'+ str(IS_SORTED)+'.xlsx'
       #writer = pd.ExcelWriter(outputname, engine = 'xlsxwriter')

       X=num_dataAll[featurelist_forML]
       
    
       for i in range (len (featurelist_forML))   :
           num_dataAll.rename(columns={featurelist_forML[i]: Channelname_dict.get(featurelist_forML[i])},inplace=True)
           
           
       best_importance=run_models(X,y,featurelist_forML)
       print (best_importance)
       if 'Str' in Sectorname :
          
               finall_importances=pd.concat([finall_importances, best_importance], axis=0)

          
                  
      # writer.save()
      # writer.close()

 return (mae_PCA, mae_Regall, mae_xgall,mae_RFall,finall_importances)




def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal [...] # Add axis labels and grid



# Creating function for scaling
def Standard_Scaler (df, col_names):
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features
    
    return df

  
#run feature selection for all cluster
def run_models (X,y,featurelist_forML ):
    cluster='All'
    importance_score=[]
    best_importance = pd.DataFrame(columns =['cluster','Attribute','Importance'])
     #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)
    
    #Feature Scaling
    X_train = Standard_Scaler (X_train, featurelist_forML)
    X_test = Standard_Scaler (X_test,featurelist_forML)   

    results = []
    names = []
    df_mae=pd.DataFrame()
    seed=50
    scoring = 'accuracy'
    for name, model in models:

        
         print (name,model)
         #kfold = model_selection.KFold(n_splits=10, random_state=seed)
         #kfold=kfold_crossvalidation(X,y,model,name)
         kfold=10
         cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)#, scoring=scoring)
         model.fit(X_train, y_train)
         
         #find number of features
         #select_optimalnumberofFeatures(X_train,y_train,kfold,model)
         #accuracy
         preds = model.predict(X_test) 
         mae=r2_score(y_test, preds)
         #confusion matrix
         predictions=model.predict(X_test).round()
         print( confusion_matrix(y_test, predictions))
         print (classification_report(y_test, predictions)) 
         perm_importance=permutation_importance(model, X_test, y_test)
         
         plot_percision_recall_curve(X_train,y_train,X_test,y_test)
         #shap summary
         shap.initjs()
         vect = CountVectorizer(stop_words='english', max_features=10000,
                               token_pattern=r'[a-zA-Z]{3,}' , ngram_range=(1,2))
         X_train_dtm = vect.fit_transform(X_train)
         X_test_dtm = vect.transform(X_test)
            
         if name=='SVR': 
              
              kfold=12
         
              importance_score=perm_importance.importances_mean
              

              
              
             #explainer = shap.KernelExplainer(model.predict,X_train,feature_names=featurelist_forML)
             #shap_values = explainer.shap_values(X_train)
         if name=='LR':
             kfold=12
             importance_score=model.coef_[0]
             

             # Visualize one value
             explainer = shap.Explainer(model,X_train,feature_names=get_featurename(featurelist_forML))
             single_shap_value = explainer(X_train)



             shap.plots.heatmap(single_shap_value[:1000])
             clustering = shap.utils.hclust(X, y)
             shap.plots.bar(single_shap_value, clustering=clustering,clustering_cutoff=0.8)

             shap.plots.beeswarm(single_shap_value,max_display=15)
             
             shap.plots.bar(single_shap_value.abs.mean(0),max_display=15)
             
             shap_values = explainer(X_train)
             shap.plots.waterfall(single_shap_value[0],max_display=15)
    
             #shap.plots.bar(shap_values,show=False)
             shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type='bar',max_display=15)
             
             shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type="layered_violin", color='coolwarm',max_display=15)
             
              
             plt.title('for Cluster All- Logistic Regression')
             plt.show()

             
         if name=='RF':
             kfold=15
             importance_score=model.feature_importances_
             
             # Our Code

             explainer = shap.Explainer(model,X_train,feature_names=featurelist_forML)
             single_shap_value = explainer(X_test)

             shap.plots.beeswarm(single_shap_value,max_display=15)
             
             shap_values = explainer(X_test)
             shap.plots.waterfall(shap_values[0],max_display=15)
             

             shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type='bar')
             
             shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type="layered_violin", color='coolwarm')


             shap.plots.bar(shap_values)             

             plt.show()
             
         results.append(cv_results)
         names.append(name)
         msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
         print(msg)
         

         importances = pd.DataFrame(data={
            'cluster': cluster,
            'Attribute': featurelist_forML,
            'Importance':importance_score
         })
         
         if name=='LR':
             best_importance=importances
             
         new_row = pd.Series(data={'cluster':cluster,
                    'Attribute': name,
                    'Importance': mae},name='x')
         df_mae = df_mae.append(new_row, ignore_index=False)
       
         if IS_SORTED==True:

          importances = importances.sort_values(by='Importance', ascending=ascending)

         plt.bar(importances['Attribute'], importances['Importance'], color='#087E8B')

       

         new_row = pd.Series(data={'cluster':cluster,
                    'Attribute': name,
                    'Importance': mae},name='x')
         df_mae = df_mae.append(new_row, ignore_index=False)
        
        ####
         #importances.to_excel(writer, sheet_name=name)
         output_filenamecsv=output_filename+'_'+name + '.csv'
         print (importances)
         importances.to_csv('test.csv')
         importances.to_csv(output_filenamecsv)
    #df_mae.to_excel(writer, sheet_name=name)
    #df_mae.to_csv(output_filenamecsv)
    return (best_importance)
        
      
def plot_percision_recall_curve(X_train,y_train,X_test,y_test):
    from sklearn import metrics
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(0).clf()
    
    pred = np.random.rand(1000)
    label = np.random.randint(2, size=1000)
    fpr, tpr, thresh = metrics.roc_curve(label, pred)
    auc = metrics.roc_auc_score(label, pred)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    
    pred = np.random.rand(1000)
    label = np.random.randint(2, size=1000)
    fpr, tpr, thresh = metrics.roc_curve(label, pred)
    auc = metrics.roc_auc_score(label, pred)
    plt.plot(fpr,tpr,label="data 2, auc="+str(auc))
    
    plt.legend(loc=0)

def run_models_forCluster (X,y,featurelist_forML ,cluster):
   
    importance_score=[]
     #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)
    
    #Feature Scaling
    X_train = Standard_Scaler (X_train, featurelist_forML)
    X_test = Standard_Scaler (X_test,featurelist_forML)   

    results = []
    names = []
    df_mae=pd.DataFrame()
    if cluster=='Fast':
       cutoff = 93  
    else:
      cutoff = 120  
    scoring = 'accuracy'
    for name, model in models:
        print (name,model)
        if (name!='LR'and name!='XG'):
         #kfold = model_selection.KFold(n_splits=10, random_state=seed)
         #kfold=kfold_crossvalidation(X,y,model,name)
         kfold=10
         #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)#, scoring=scoring)
         model.fit(X_train, y_train)
         
         #find number of features
         #select_optimalnumberofFeatures(X_train,y_train,kfold,model)
         #accuracy

         y_pred = model.predict(X_test)                            # decide on a cutoff limit
         y_pred_classes = np.zeros_like(y_pred)    # initialise a matrix full with zeros
         y_pred_classes[y_pred > cutoff] = 1  

         y_test_classes = np.zeros_like(y_pred)
         y_test_classes[y_test > cutoff] = 1


         print(confusion_matrix(y_test_classes, y_pred_classes))
         print (classification_report(y_test_classes, y_pred_classes)) 

         
         mae=r2_score(y_test, y_pred)
         #confusion matrix
         #predictions=model.predict(X_test).round()
         #print( confusion_matrix(y_test, predictions))
         #print (classification_report(y_test, predictions)) 
         perm_importance=permutation_importance(model, X_test, y_test)
         
         
         #shap summary
         shap.initjs()
         
         if name=='SVR': 
             kfold=12
             importance_score=perm_importance.importances_mean
             
            # model =  create_model(SVC, C=300, probability=True) #user defined function works right
            # model.fit(X_train, y_train)
            
             explainer = shap.KernelExplainer(model.predict,X_train,feature_names=featurelist_forML)
             #explainer = shap.Explainer(model)
             shap_values = explainer.shap_values(X_test)
             
             shap.initjs()
             shap.force_plot(explainer.expected_value,  X_train)


             #shap_values = explainer.shap_values(X_train)
             
         if name=='RF':
             kfold=15
             importance_score=model.feature_importances_
             
             # Our Code
             explainer = shap.TreeExplainer(model) 
            
            # Visualize one value
             single_shap_value = explainer(X_test.sample(n=1))
             shap.summary_plot(single_shap_value, feature_names=get_featurename(X_test.columns), plot_type='bar')
             #shap.plots.waterfall(single_shap_value[0])
             #plt.title('for Cluster All- Random Forest')
             #plt.show()
             #explainer = shap.Explainer(model,X_train,feature_names=featurelist_forML)
            #shap_values = explainer(X_train)
             #shap.plots.waterfall(shap_values[0])
    
            # shap.plots.bar(shap_values)    
             
         if name=='XG'  :
             importance_score=model.feature_importances_
             explainer = shap.Explainer(model,X_train,feature_names=featurelist_forML)
             single_shap_value = explainer(X_test.sample(n=1))
             shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type='bar')
             #shap_values = explainer(X_train)
             shap.plots.waterfall(single_shap_value[0])
             
         #results.append(cv_results)
        # names.append(name)
         #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print(msg)
         

         importances = pd.DataFrame(data={
            'cluster': cluster,
            'Attribute': featurelist_forML,
            'Importance':importance_score
          })
       
         new_row = pd.Series(data={'cluster':cluster,
                    'Attribute': name,
                    'Importance': mae},name='x')
         df_mae = df_mae.append(new_row, ignore_index=False)
       
         if IS_SORTED==True:

          importances = importances.sort_values(by='Importance', ascending=ascending)

         plt.bar(importances['Attribute'], importances['Importance'], color='#087E8B')

       

         new_row = pd.Series(data={'cluster':cluster,
                    'Attribute': name,
                    'Importance': mae},name='x')
         df_mae = df_mae.append(new_row, ignore_index=False)
        
        ####
        # importances.to_excel(writer, sheet_name=name)
         output_filenamecsv=output_filename+'_'+name + '_'+cluster + '.csv'
         print (output_filenamecsv)
         importances.to_csv(output_filenamecsv)
   # df_mae.to_csv(output_filenamecsv)
    #df_mae.to_excel(writer, sheet_name=name)
    
        
       
       

    
       # save the model to disk
     #pickle.dump(model, open(modelfilename, 'wb'))


def  get_featurename(f_list):
    feature_names=[]
    for i in range(len(f_list)):
        feature_names.append(Channelname_dict.get(f_list[i]))
    return (feature_names)
def Make_channelList2(dc,is_all,is_forsector):
   
    channel_list=[]
    n=len(dc)
    if is_all==True:        
        for j in range(n):
            channel_list.append(dc.iloc[j,0])
    else:
        for j in range(n):
            st=dc.iloc[j,0]
            if st.find('Avg')!=-1:
               channel_list.append(dc.iloc[j,0])
               
    channel_list.append('cluster')
    channel_list.append('cluster_label')
    channel_list.append('LapTime')
    if is_forsector:
        channel_list.append('SectorTime')
    #print (channel_list)


    return (channel_list)

def Make_channelList(dc,is_all,is_forsector):
    channel_stats=['Avg','Max','Min','Start', 'End', 'Change','Std Dev']
    if is_all==True:
        numberof_stats=len(channel_stats)
    else:
        numberof_stats=1  #just for AVG
        
    channel_list=list()    
    
    n=len(dc)       

    for j in range(n):
        col_name=dc.iloc[j][0]

        for k in range (numberof_stats):
            col_namestats=col_name+'_' + channel_stats[k]

            found=False
            for char in col_namestats:
                 if char in "[%.?/]°123456789":
                     found=True
                     
                     col_namestats=col_namestats.replace(char,'')

            if found==True:
                channel_list.append(col_namestats)
                #channel_list=np.append(channel_list,col_namestats) 

            else:

                #channel_list=np.append(channel_list,col_namestats) 
                channel_list.append(col_namestats)

    #channel_list=np.append(channel_list,'cluster') 
    #channel_list=np.append(channel_list,'cluster_label') 
    channel_list.append('cluster')
    channel_list.append('cluster_label')
    if is_forsector==True:
        #channel_list=np.append(channel_list,'SectorTime')  
        channel_list.append('SectorTime')
    else:
        #channel_list=np.append(channel_list,'LapTime')
        channel_list.append('LapTime')

    return (channel_list)

    
#############################
def select_optimalnumberofFeatures(X_train,y_train,kf,model):
    
       hyper_params =[{'n_features_to_select': list(range(1, len(X_train.columns)+1))}]
       plot_optimalnumberofFeatures(model,kf,hyper_params,X_train,y_train)
       
       
##################################
def plot_optimalnumberofFeatures(model,kf,hyper_params,X_train,y_train):
    
       rfe = RFE(model)
       model_cv = GridSearchCV(estimator = rfe, param_grid = hyper_params, scoring= 'r2', cv = kf, verbose = 1, return_train_score=True)
       model_cv.fit(X_train, y_train)   
       cv_results = pd.DataFrame(model_cv.cv_results_)
       #print (cv_results)
       
       plt.figure(figsize=(16,6))
       plt.plot(cv_results['param_n_features_to_select'], cv_results['mean_test_score'])
       plt.plot(cv_results['param_n_features_to_select'], cv_results['mean_train_score'])
       plt.xlabel('number of features')
       plt.ylabel('r-squared')
       plt.title('Optimal Number of Features')
       plt.legend(['test score', 'train score'], loc='upper left')
       plt.show()
       
def  check_dataStatus(df):
       ####class_distribution_ratio
       df['cluster'].value_counts()
       fast = len(df[df['cluster'] == 0])
       slow = len(df[df['cluster'] == 1])
       class_distribution_ratio = slow/fast
       print ('class_distribution_ratio',fast,slow,class_distribution_ratio) 
       
       
def Feature_Selection(listof_Files,output_filename,trackname,channel_list):

    mae_PCA=0
    mae_Regall=0
    mae_xgall=0
    mae_RFall=0

    #Read summary file to find outlier
    for a in listof_Files:
      # print (a)
       df= pd.read_csv(a,usecols=channel_list)
       num_dataAll = df.select_dtypes(include='number')
       
       num_dataAll.drop(['cluster','LapTime'], axis=1, inplace=True)


       if 'SectorTime' in num_dataAll :
           num_dataAll.drop(['SectorTime'], axis=1, inplace=True)
       if 'LapTime' in num_dataAll :       
           num_dataAll.drop(['LapTime'], axis=1, inplace=True)    
       
       y=df['cluster']

       ##Check_datastatus
       check_dataStatus(df)
       
       #feature filtering before applying ML models
       featurelist_forML=Wrapper_FilterFeatureSelection (y,df,num_dataAll)
       num_dataAll=num_dataAll[featurelist_forML]

       
       X=num_dataAll[featurelist_forML]    
       for i in range (len (featurelist_forML))   :
            X.rename(columns={featurelist_forML[i]: Channelname_dict.get(featurelist_forML[i])},inplace=True)


     
       
       run_models(X,y,get_featurename(featurelist_forML))

       print ('run nmodels for cluster ')
    
       num_cluster=df['cluster'].nunique()
      
       """for i in range (num_cluster):
           cluster=i
         
           df2=df[(df.cluster==cluster)]

           cluster_label=df2.iloc[0]['cluster_label']

           if len(df2)>1:
               num_dataAll = df2.select_dtypes(include='number')
               y=df2['LapTime']
               
               
               num_dataAll.drop(['LapTime','cluster'], axis=1, inplace=True)
               featurelist_forML=Wrapper_FilterFeatureSelection (y,df2,num_dataAll)
               num_dataAll=num_dataAll[featurelist_forML]
               
             
               
               X=num_dataAll[featurelist_forML]
               print (len(X),len(y))
               run_models_forCluster(X,y,featurelist_forML,cluster_label)

"""
            

   

       return (mae_PCA, mae_Regall, mae_xgall,mae_RFall)
       
      


       
       
       RF = RandomForestClassifier()

       
       
      
       param_grid = [ {'n_estimators': [3, 10, 30], 
                        'max_features': [ 12, 15, 20]}, 
                      {'bootstrap': [False], 
                       'n_estimators': [3, 10], 
                       'max_features': [12, 15, 20]},
                      ]

       grid_search = GridSearchCV(RF, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
       grid_search.fit(X, y)
       
       model=RF
       scores = cross_val_score(RF, X_train, y_train, cv=kf)
       cvres = grid_search.cv_results_
       for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
           print(np.sqrt(-mean_score), params)
     
       print (grid_search.best_params_)
       print (grid_search.best_estimator_)
       train_yhat = grid_search.predict(X_train)
       train_acc = accuracy_score(y_train, train_yhat)

       # evaluate on the test dataset
       test_yhat = grid_search.predict(X_test)
       test_acc = accuracy_score(y_test, test_yhat)
       print (train_acc,test_acc)
       print("RandomForest  : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
       
       #RF = RandomForestClassifier(grid_search.best_params_).fit(X, y)
       RF=RandomForestClassifier(bootstrap=False, max_features=20, n_estimators=10).fit(X_train, y_train)
       y_train_F = (y_train == 0)
       #print(cross_val_score(classifier, X_train, y_train, cv=3, scoring="accuracy"))
       #y_train_pred = cross_val_predict(classifier, X_train, y_train_F, cv=5,scoring="Root Mean Squared Error")
      # print(confusion_matrix(y_train, round(y_train_pred)))
       predictions=RF.predict(X_test).round()
       
       print(confusion_matrix(y_test, predictions))
       print (classification_report(y_test, predictions))
      # print(precision_score(y_train, round(y_train_pred),average='binary')) 
       #print( recall_score(y_train, round(y_train_pred),average='binary'))        
       
       importances = pd.DataFrame(data={
            'Attribute': num_dataAll.columns,
            'Importance': RF.feature_importances_
        })
       
       importances.to_csv('testimportanceRF.csv')
       plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')


           
       import sklearn as sk
       from sklearn.neural_network import MLPClassifier
        
       NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
       NN.fit(X, y)

       scores = cross_val_score(NN, X, y, cv=15)
       
       print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

       #for i in range (1, 10):
       #    clf = svm.SVC(kernel='linear', C=i).fit(X_train, y_train)
       #    print(clf.score(X_test, y_test))
        
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
       NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
       NN.fit(X_train, y_train)
       print (round(NN.score(X_train, y_train), 4))

def make_recallPlotfor_multiclass ( X_train, X_test, y_train, Y_test):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score       
    from sklearn.metrics import PrecisionRecallDisplay
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from itertools import cycle
    
    n_classes = 3
    random_state = np.random.RandomState(0)

    classifier = OneVsRestClassifier(
    make_pipeline(StandardScaler(), LinearSVC(random_state=random_state)))
    classifier.fit(X_train,y_train)
    y_score = classifier.decision_function(X_test)
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    
    display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")
    
    
    


    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    
    _, ax = plt.subplots(figsize=(7, 8))
    
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")
    
    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
    
    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")
    
    plt.show()
    
    
def kfold_crossvalidation(X,y,model,modelname):

    #kf=KFold(n_splits=15,random_state=None, shuffle=False)
    
    list_training_error=[]
    list_testing_error=[]
    
   # kf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    kf=split(X, y, groups=None)

    for train_index,test_index in kf.split(X,y):

      # X_train,X_test=X['index1'=train_index],X['index1'=test_index]
       X_train,X_test=X.iloc[train_index],X.iloc[test_index]

       y_train,y_test=y.iloc[train_index],y.iloc[test_index]
       
       model.fit (X_train,y_train)
       y_train_data_pred=model.predict(X_train)
       y_test_data_pred=model.predict(X_test)
       fold_training_error=mean_absolute_error(y_train,y_train_data_pred)
       fold_testing_error=mean_absolute_error(y_test,y_test_data_pred)
       list_training_error.append(fold_training_error)
       list_testing_error.append(fold_testing_error)
       #print (fold_training_error,fold_testing_error)

    plt.subplot(1,2,1)
    plt.plot(range(1,kf.get_n_splits()+1), np.array(list_training_error).ravel(),'o-')
    plt.xlabel ('number of fold')
    plt.ylabel('training error')
    plt.title('Training error across folds %s'% modelname)
    plt.tight_layout()
    plt.subplot(1,2,2)
    plt.plot(range(1,kf.get_n_splits()+1), np.array(list_testing_error).ravel(),'o-')
    plt.xlabel ('number of fold')
    plt.ylabel('testing error')
    plt.title('Testing error across folds %s' % modelname)
    plt.tight_layout()
    plt.show()
    return (kf)   

def featureselectiontest_multiclass(listof_Files,output_filename,trackname,channel_list):
    from sklearn.preprocessing import label_binarize
    mae_PCA=0
    mae_Regall=0
    mae_xgall=0
    mae_RFall=0


    #Read summary file to find outlier
    for a in listof_Files:
      # print (a)
       df= pd.read_csv(a,usecols=channel_list)
       num_dataAll = df.select_dtypes(include='number')
       
       num_dataAll.drop(['cluster','LapTime'], axis=1, inplace=True)


       if 'SectorTime' in num_dataAll :
           num_dataAll.drop(['SectorTime'], axis=1, inplace=True)
       if 'LapTime' in num_dataAll :       
           num_dataAll.drop(['LapTime'], axis=1, inplace=True)    
       
       y=df['cluster']
       
       
       ####class_distribution_ratio
       df['cluster'].value_counts()
       fast = len(df[df['cluster'] == 0])
       slow = len(df[df['cluster'] == 1])
       med = len(df[df['cluster'] == 2])
       class_distribution_ratio1 = slow/(fast+slow+med)
       class_distribution_ratio2 = fast/(fast+slow+med)
       class_distribution_ratio3 = med/(fast+slow+med)
       print ('kkk',fast,slow,med,class_distribution_ratio1,class_distribution_ratio2,class_distribution_ratio3)
       
       #feature filtering before applying ML models
       featurelist_forML=Wrapper_FilterFeatureSelection (y,df,num_dataAll)
       num_dataAll=num_dataAll[featurelist_forML]
       X=num_dataAll[featurelist_forML]
       Y = label_binarize(y, classes=[0, 1, 2])
       n_classes = Y.shape[1]
       #Train test split
       
       X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

       #Feature Scaling
       col_names = num_dataAll.columns
       X_train = Standard_Scaler (X_train, col_names)
       X_test = Standard_Scaler (X_test, col_names)





       
       kfold = StratifiedKFold(n_splits=5, shuffle=False)
       
       #Training Model
      
       make_recallPlotfor_multiclass ( X_train, X_test, y_train, y_test)


       clf = svm.SVC(kernel='linear', C=1, random_state=42).fit(X_train,y_train)
       scores = cross_val_score(clf, X_train,y_train, cv=kfold)
       print("SVM  %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


       y_train_F = (y_train == 0)
       print(cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy"))
       y_train_pred = cross_val_predict(clf, X_train, y_train_F, cv=3)
       print(confusion_matrix(y_train, y_train_pred))
       
 
       print(precision_score(y_train, y_train_pred,average='samples')) # == 4096 / (4096 + 1522) 0.7290850836596654 
       print( recall_score(y_train, y_train_pred,'samples')) # == 4096 / (4096 + 1325)
       
       
       y_scores = cross_val_predict(clf, X_train, y_train, cv=3)
    
       from sklearn.metrics import roc_curve 
       fpr, tpr, thresholds = roc_curve(y_train, y_scores)
       plot_roc_curve(fpr, tpr) 
       plt.show()
    
       #f_importances(clf.coef_, num_dataAll.columns)
         
       SVM_regressor = svm.SVR(cache_size = 800)
       
       loo = LeaveOneOut()

       parm_grid=dict(C=uniform(loc=0, scale=4),
                    gamma=[0.0005, 0.0002, 0.0001, 0.001, 0.01, 0.1, 1, 2])
      # parm_grid={'C': [0.1, 0.5, 1, 3, 5, 7, 9, 11, 13, 15],
      #      'gamma':[0.0005, 0.0002, 0.0001, 0.001, 0.01, 0.1, 1, 2]}

      # grid_search = GridSearchCV(c,
      #                     param_grid=parm_grid,
        #                   scoring='neg_mean_squared_error',
      #                     cv=loo)
         
       #clf  = RandomizedSearchCV( SVM_regressor,parm_grid, scoring='neg_mean_squared_error', cv=loo)
       
      # rgr = clf.fit(X, y)
    
     #  r = rgr.best_estimator_

      

       
     #  print (r,'best stmator for svm')
       #for i in range (len(num_dataAll.columns)):
       #  print (r.feature_importances_[i],num_dataAll.columns[i])

       #scores = cross_val_score(rgr, X, y, cv=kfold)
       
       #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
       
       LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
       scores = cross_val_score(LR, X, y, cv=kfold)
       print("LogisticRegression  : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
       train_yhat = LR.predict(X)
       train_acc = accuracy_score(y, train_yhat)

       # evaluate on the test dataset
       test_yhat = LR.predict(X)
       test_acc = accuracy_score(y, test_yhat)
       print (train_acc,test_acc)
       print ( LR.coef_[0])
       importances = pd.DataFrame(data={
            'Attribute': num_dataAll.columns,
            'Importance': LR.coef_[0]
       })
       
       print(cross_val_score(LR, X, y, cv=3, scoring="accuracy"))
       y_train_pred = cross_val_predict(LR, X, y_train_F, cv=3)
       print(confusion_matrix(y, y_train_pred))
       
 
       print(precision_score(y, y_train_pred)) # == 4096 / (4096 + 1522) 0.7290850836596654 
       print( recall_score(y, y_train_pred)) # == 4096 / (4096 + 1325)
       #explainer = shap.Explainer(clf,X,feature_names=featurelist_forML)
       
       #shap.plots.waterfall(shap_values[0])
       #shap.plots.bar(shap_values)
       #shap.plots.beeswarm(shap_values)
      # print (num_dataAll.columns)

       #for i in range (len(num_dataAll.columns)):
        # print (LR.coef_[0][i],num_dataAll.columns[i])
       #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

       values = [i for i in range(1, 21)]
       train_scores=[]
       test_scores=[]
       for i in values:
         # configure the model
         model = DecisionTreeClassifier(max_depth=i)
         # fit model on the training dataset
         model.fit(X_train, y_train)
         # evaluate on the train dataset
         train_yhat = model.predict(X_train)
         train_acc = accuracy_score(y_train, train_yhat)
         train_scores.append(train_acc)
         #evaluate on the test dataset
         test_yhat = model.predict(X_test)
         test_acc = accuracy_score(y_test, test_yhat)
         test_scores.append(test_acc)
         # summarize progress
        # print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

    # plot of train and test scores vs tree depth
       plt.plot(values, train_scores, '-o', label='Train')
       plt.plot(values, test_scores, '-o', label='Test')
       plt.legend()
       plt.show()
    

       importances.to_csv('testimportanceLR.csv')
       plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
       RF = RandomForestClassifier()
       
       scores = cross_val_score(LR, X, y, cv=kfold)
       
       
       RF = RandomForestClassifier()
       #scores = cross_val_score(LR, X, y, cv=kfold)
       
       
      
       param_grid = [ {'n_estimators': [3, 10, 30], 
                        'max_features': [ 12, 15, 20]}, 
                      {'bootstrap': [False], 
                       'n_estimators': [3, 10], 
                       'max_features': [12, 15, 20]},
                      ]

       grid_search = GridSearchCV(RF, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
       grid_search.fit(X, y)
       
       
       cvres = grid_search.cv_results_
       for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
           print(np.sqrt(-mean_score), params)
        
       print (grid_search.best_params_)
       print (grid_search.best_estimator_)
       train_yhat = model.predict(X_train)
       train_acc = accuracy_score(y_train, train_yhat)

       # evaluate on the test dataset
       test_yhat = model.predict(X_test)
       test_acc = accuracy_score(y_test, test_yhat)
       print (train_acc,test_acc)
       print("RandomForest  : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
       
       RF = RandomForestClassifier(grid_search.best_params_).fit(X, y)
       importances = pd.DataFrame(data={
            'Attribute': num_dataAll.columns,
            'Importance': RF.feature_importances_
        })
       
       importances.to_csv('testimportanceRF.csv')
       plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')


           
       import sklearn as sk
       from sklearn.neural_network import MLPClassifier
        
       NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
       NN.fit(X, y)

       scores = cross_val_score(NN, X, y, cv=kfold)
       
       print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

       #for i in range (1, 10):
       #    clf = svm.SVC(kernel='linear', C=i).fit(X_train, y_train)
       #    print(clf.score(X_test, y_test))
        
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
       NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
       NN.fit(X_train, y_train)
       print (round(NN.score(X_train, y_train), 4))
# Load data
collection='C5/'
track_path="../../../2_DATA/OUTPUT/"+ collection+ "LapStats/"



inputdata_path="../../../2_DATA/OUTPUT/"+ collection+ "ClusterResults/"
outputdata_path="../../../2_DATA/OUTPUT/"+ collection+ "FeaturesRanking/"

files_list=['_CL_Channel Report_Only']#,'_CL_Sector_']#,,'_CL_Channel Report_Only']#,'_Time Report_OnlySummary','_Time Report_DetailSummary']

#files_list=['_CL_Channel Report_DetailDifferent_Sectors']

modelfilename="../SavedModels/bestmodel.sav"
inputsector_path="../../../2_DATA/TrackInfo/"
SectorInfor_file=inputsector_path+'brands_hatch, brands_hatch.csv'
ds = pd.read_csv(SectorInfor_file,header=0,parse_dates=[0])  


Channelname_dict = {
 'abs glat_Avg': 'absolute lateral acceleration',
 'abs glong_Avg': 'absolute longitudinal acceleration',
 'abs Steerangle_Avg': 'absolute steering angle', 
 'BRAKE _Avg' : 'brake',
 'Corr Speed kmh_Avg': 'speed',
 'lane deviation_Avg': 'lane deviation',
 'Oversteer _Avg': 'oversteer',
        'ROTY s_Avg': 'ROTY ',
        'RPMS min_Avg': 'RPMS', 
        'STEERANGLE _Avg': 'steering angle',
        'steering reversal rate per lap_Avg': 'steering reversal rate per lap',
        'steering reversal rate per sector_Avg': 'steering reversal rate per sector',
        'THROTTLE _Avg':'throttle'  ,      
        'understeer_Avg':   'understeer',
        'Brake_length_Avg':        'brake length',
        'Brake_duration_Avg': 'brake duration',
        'Trail_braking_duration_Avg':  'trail braking duration', 
        'THROTTLE width_Avg': 'throttle width',
        'Throttle_duration_Avg': 'throttle duration', 
        'Throttle application speed_Avg': 'throttle application speed',
        'Throttle release application_Avg':        'throttle release application',
        'Steer_Peakvalue_Avg': 'maximum value of steering',
        'Steer_PeakTime_Avg':        'time of maximum value of steering',
        'number of crossing_Avg': 'number of zero crossing', 
        'Cumulative angular travel_Avg': 'Cumulative angular travel',
        'number of direction changed_Avg': 'number of direction changed', 
        'steering duration_Avg': 'steering duration',
        'angular velosity_Avg': 'angular velocity'

}


models = []

models.append(('LR', LogisticRegression(max_iter=10000)))

models.append(('SVR', SVR(kernel = 'rbf')))

models.append(('RF', RandomForestRegressor( oob_score = True, n_jobs = -1,random_state =50,  max_features = 12, min_samples_leaf = 50,n_estimators=30)))

plt.rcParams['figure.figsize'] = [20, 20]


#models.append(('LR', LinearRegression())) #LinearRegression()
"""xgb = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1, _label_encoder=False)
"""
#xgb= xgboost.XGBRegressor(n_estimators=100, max_depth=2)
#models.append (('XG',xgb))

       #lm = LinearRegression()
       #LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
      
      #

  #
       
       


    
MAE_reg=[]
MAE_xg=[]
MAE_PCA=[]
MAE_RF=[]

inputChannelList_path="../../../2_DATA/"
ChannelInfor_file=inputChannelList_path+'Channel list.csv'
dc = pd.read_csv(ChannelInfor_file,header=0,parse_dates=[0])  

ChannelInfor_file=inputChannelList_path+'Channel list for sector.csv'
dc_sector= pd.read_csv(ChannelInfor_file,header=0,parse_dates=[0])  


    
is_all=False

ascending=True
#channel_list=Make_channelList(dc,is_all,False)

###if we want to include all of the stats for each channel this var should be True , othervise False
plt.rcParams["figure.figsize"] = (20,20)
###if we want tto use data for heatmap, it is False , otherwise True
IS_SORTED=False
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


plt.rcParams.update({'font.size': 8})

for i in range(1,len(df_tn)):
  trackname=df_tn[i][0] 
  path=inputdata_path+trackname+'/'
  out_path=outputdata_path+trackname+ '/'
  ####Creating output directory for the results
  if not os.path.exists(out_path):
      os.makedirs(out_path)
  else:
        #remove folder withh all of its files
    shutil.rmtree(out_path)
 
  for k in range (0, len(files_list))   :

    
    Filename_term=files_list[k]
    print ('trackname:', trackname)
    
    Excluded_Term=''

    listof_Files = FileOperation.getListOfFiles(path,Filename_term,Excluded_Term)
    mae_Reg=0
    mae_xg=0
    mae_RF=0

    if k==2: # channel_report_only
       
        channel_list=Make_channelList(listof_Files[0],is_all,False)
        channel_list=Make_channelList(dc,is_all,False)


        print ('number of channels before filtering ' , Filename_term, len(channel_list))
        output_filename=out_path+'_FR' + Filename_term+ '_'+ str(IS_SORTED)+'.xlsx'
        writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
        mae_PCA, mae_Reg, mae_xg,mae_RF=Feature_Selection(listof_Files,output_filename,trackname,channel_list)

        writer.save()
        writer.close()


    else:
     if  k==1 :  # channel_report_detail
        #for channel report detail data that includes sector
        #channel_list1=Make_channelList(dc,is_all,False)
        #channel_list2=Make_channelList(dc_sector,False,True)
        #channel_list=np.append(channel_list1,channel_list2)
       # print (dc_sector)

        channel_list=Make_channelList2(dc_sector,is_all,True)
        #print (channel_list)
        print ('number of channels before filtering', Filename_term, len(channel_list))
        output_fileSectorname=out_path+'_FR'
       # writer = pd.ExcelWriter(output_fileSectorname, engine = 'xlsxwriter')
        mae_PCA, mae_Reg, mae_xg,mae_RF,finall_importances=Feature_SelectionSector(listof_Files,output_fileSectorname,trackname,channel_list,finall_importances)
       # writer.save()
       # writer.close()
        output_filename=out_path+'_FR final average for Str' +'.csv'
        print (finall_importances)
        finall_importances=finall_importances.groupby(['Attribute']).mean('Importance').reset_index()
     
        finall_importances.to_csv(output_filename,index=False)
     else:
         # channel_report_AVG

          #channel_list1=Make_channelList(dc,is_all,False)
          #channel_list2=Make_channelList(dc_sector,False,True)
          #channel_list=np.append(channel_list1,channel_list2)

          #channel_list=np.delete(channel_list,len(channel_list)-1)
          
          channel_list1=Make_channelList2(dc,is_all,False)
          print ('number of channels before filtering', Filename_term, len(channel_list1))
          # output_filename=out_path+'_FR' + Filename_term+ '_'+ str(IS_SORTED)+'.xlsx'
          output_filename=out_path+'_FR' + Filename_term+ '_'+ str(IS_SORTED)+'.xlsx'
         # writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
          mae_PCA, mae_Reg, mae_xg,mae_RF=Feature_Selection(listof_Files,output_filename,trackname,channel_list1)

         # writer.save()
         # writer.close()

    df_tn[i][3]=mae_Reg
    df_tn[i][4]=mae_xg
    df_tn[i][5]=mae_RF
    

print (finall_importances)
finall_importances.to_csv(output_filename,index=False)
df_t=pd.DataFrame(df_tn)
df_t.to_csv(trackname_file,index=False,header=False)