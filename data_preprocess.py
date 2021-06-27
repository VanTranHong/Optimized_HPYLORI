import pandas as pd 
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def remove_invalid_column(data, target):
    """Function that removes columns that aren't suitable for machine learning.
    This includes features with more than 5% missing values, wrong data type,
    and the indices.

    Args:
        data (Pandas DataFrame): The DataFrame that contains data that hasn't been preprocessed.

    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis = 1)
    data1 = data[data.columns[data.isnull().mean()<0.05]]
    data1 = data1.select_dtypes(exclude=['object'])
    if target not in data1.columns:
        data1 = pd.concat([data1,data[target]], axis =1)
    data1 = data1[data1[target].notna()]  
    return data1

def recategorize(data):
    """Recategorizes the data according to the standards in the SPSS file.

    Args:
        data (DataFrame): DataFrame containing data that hasn't been preprocessed.

    Returns:
        DataFrame: A DataFrame with replaced values.
    """
    
    data["ADD"].replace({2: 0}, inplace=True)
    data["AnyPars3"].replace({2: 0, 9: 0}, inplace=True)
    data["CookArea"]=data["CookArea"]-1
    data["GCOW"]=data["GCOW"]-1 
    data["DEWORM"].replace({0: 'a', 1: 0}, inplace=True)
    data["DEWORM"].replace({'a': 1}, inplace=True) 
    data["GCAT"]=data['GCAT']-1
    data["GDOG"]=data['GDOG']-1
    data["GELEC"].replace({3: 0, 2: 'a', 1: 2}, inplace=True)
    data["GELEC"].replace({'a': 1}, inplace=True)
    data["GFLOOR6A"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["GWASTE"].replace({4: 0}, inplace=True)
    data["HCIGR6A"].replace({1: 0, 2: 1}, inplace=True)
    data["HPYLORI"].replace({1: 0, 2: 1}, inplace=True)
    data["AgeGroups"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["FamilyGrouped"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["ToiletType"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["WaterSource"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    return data

def getnominal(data,nominal):
    """Finds the features that contain nominal values.

    Args:
        data (DataFrame): DataFrame containing the dataset.

    Returns:
        List: A list that contains the nominal features.
    """
    returnarr = []
    non_dummy =[]
    number_distinct=[]
    for col in nominal:
        if col in data.columns:
            distinct = np.sort(data[col].dropna().unique())
            if len(distinct) > 2:
                returnarr.append(col)    
                number_distinct.append(len(distinct)-1)
            else:
                non_dummy.append([col])
    return returnarr, number_distinct, non_dummy


def create_dummies (data):
    """Creates dummy variables.

    Args:
        data (DataFrame): DataFrame containing the dataset

    Returns:
        DataFrame: DataFrame containing the dataset with dummy variables.
    """
    dummy = pd.get_dummies(data, columns = data.columns, drop_first= True) 
    return dummy

def normalize_data(data, continuous):
    returnarr =[]
    for col in continuous:
        if col in data.columns:
            data[col].replace({-9999:None}, inplace = True)
            returnarr.append(col)
            
    for col in returnarr:
        mean = np.mean(data[col])
        std = np.std(data[col])
        normed = (data[col]-mean)/std
        data[col] = normed 
    return data
def impute(data):
    """Multivariate imputer that estimates each feature from all the others

    Args:
        data (Numpy Array): A numpy array containing the dataset.

    Returns:
        Numpy Array: A matrix with imputed values.
    """
    imputed_data = []
    for i in range(5):
        imputer = IterativeImputer(sample_posterior=True, random_state=i)
        imputed_data.append(imputer.fit_transform(data))
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    
    return returned_data

def group_nominals(names, num_distinct):
    start = 0
    groups = []
    for num in num_distinct:
        group =[]
        for i in range(num):
            group.append(names[start])
            start+=1
        groups.append(group)
    return groups
        
    
    

            

def modify_data(data, numerical, nominal,target):
    """Runs all the preprocessing functions on the dataset.

    Args:
        data (DataFrame): DataFrame containing the dataset with no preprocessing.
        numerical (List): List containing all the features that are ordinal.
        nominal (List): List containing all the features that are nominal.

    Returns:
        DataFrame: DataFrame with all the preprocessing done.
    """
    data1 = remove_invalid_column(data)
  
    data2 = recategorize(data1)
    nominal, num_distinct, non_dummy = getnominal(data2,nominal)  
    nominal_data = create_dummies (data2[nominal])
    new_nominal_groups = group_nominals(nominal_data.columns, num_distinct) 
    groups = new_nominal_groups + non_dummy
    groups.remove([target])
    data3 = data2.drop(nominal, axis = 1)
    data4 = pd.concat([data3,nominal_data], axis =1)
    data4 = normalize_data(data4, numerical)
    
 
    return data4, groups
