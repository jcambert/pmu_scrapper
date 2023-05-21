import sys
import re
import numpy as np
import pandas as pd
import os.path
import logging
from operator import indexOf
from joblib import dump,load
from sklearn.compose import make_column_transformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.tree import DecisionTreeClassifier

dir_path = os.path.dirname(os.path.realpath(__file__))

HEADER_COLUMNS=['date','reunion','course','nom']
NUMERICAL_FEATURES=['numPmu','rapport','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente','sexe','musique']
# CATEGORICAL_FEATURES=['hippo_code','deferre']
CATEGORICAL_FEATURES=['deferre']
CALCULATED_FEATURES=[]

TARGETS=['ordreArrivee']
FEATURES=NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES

MODEL_PATH='models_test'

def load_classifier(name,type_course):
    if not has_classifier(name,type_course):
        return False
    return load(os.path.join(MODEL_PATH, f'{name}_{type_course}.model'))

def save_classifier(name,type_course,classifier):
    dump(classifier,os.path.join(MODEL_PATH, f'{name}_{type_course}.model'))

def has_classifier(name,type_course):
    return os.path.isfile(os.path.join(MODEL_PATH, f'{name}_{type_course}.model'))


music_pattern='([0-9,D,T,A,R][a,m,h,s,c,p,o]){1}'
music_prog = re.compile(music_pattern,flags=re.IGNORECASE)
music_penalities={'0':11,'D':6,'T':11,'A':11,'R':11}
DEFAULT_MUSIC=music_penalities['0']

def musique_converter(music,speciality='a'):
    points=0
    results=[]
    try:
        results = music_prog.findall(music)
        for result in results:
            point= music_penalities[result[0]] if result[0] in music_penalities else int(result[0])
            points+=point
    except:
        pass
    finally:
        res= points/len(results) if len(results)>0 else DEFAULT_MUSIC
        return res

SEXES= ['MALES','FEMELLES','HONGRES']
def sexe_converter(sexe):
    if not sexe in SEXES:
        return -1
    return indexOf(SEXES,sexe)

DEFERRES=['','DEFERRE_ANTERIEURS_POSTERIEURS','DEFERRE_POSTERIEURS','DEFERRE_ANTERIEURS','REFERRE_ANTERIEURS_POSTERIEURS','PROTEGE_ANTERIEURS_DEFERRRE_POSTERIEURS','PROTEGE_ANTERIEURS','PROTEGE_ANTERIEURS_POSTERIEURS','DEFERRE_ANTERIEURS_PROTEGE_POSTERIEURS','PROTEGE_POSTERIEURS']
def deferre_converter(value):
    try:
        return DEFERRES.index(value)
    except:
        return 0
    
def place_converter(row):
    return 1 if row['ordreArrivee'] in range(1,3) else 0

def load_csv_file(filename,nrows=None):

    usecols=HEADER_COLUMNS+FEATURES+TARGETS
    types={key:np.number for key in NUMERICAL_FEATURES if key not in ['sexe','musique','deferre']}
    converters={'musique':musique_converter,'sexe':sexe_converter,'deferre':deferre_converter}
    
    df=pd.read_csv(filename,sep=";",header=0,usecols=usecols,dtype=types,converters=converters,nrows=nrows,low_memory=False,skip_blank_lines=True)

    df[TARGETS].fillna(0,inplace=True)
    targets=df.apply (lambda row: place_converter(row), axis=1)
    features=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES]
    return features,targets

def create_pipelined_model(model):
    numerical_pipeline=make_pipeline(SimpleImputer(fill_value=0), RobustScaler())
    categorical_pipeline=(make_pipeline(OneHotEncoder(handle_unknown = 'ignore')))
    preprocessor=make_column_transformer(
        (numerical_pipeline,NUMERICAL_FEATURES),
        (categorical_pipeline,CATEGORICAL_FEATURES))
    model= make_pipeline(preprocessor,PolynomialFeatures(),VarianceThreshold(0.05),model)
    return model

def search_best(model,params,features_train,targets_train,cv=5,use_pipeline=True):
    name=model.__class__.__name__.lower()
    params_grid={}

    if(use_pipeline):
        this_model=create_pipelined_model(model)
        this_params=dict(( name+"__"+k,v) for k,v in params.items())
    else:
        this_model=model
        this_params=params
    this_params.update(params_grid)
    grid=GridSearchCV(this_model,this_params,cv=cv)
    grid.fit(features_train,targets_train)
    p=grid.best_params_
    # print(p)
    model=grid.best_estimator_
    return model,p,name


def get_models_to_find_best():
    models=[]

    # ALLREADY TESTED
    # {'sgdclassifier__learning_rate': 'optimal', 'sgdclassifier__loss': 'modified_huber', 'sgdclassifier__penalty': 'elasticnet'})
    # {'kneighborsclassifier__metric': 'kulsinski', 'kneighborsclassifier__n_neighbors': 18}
    models.append([SGDClassifier(),{'learning_rate':['constant','optimal','invscaling','adaptive'], 'loss':['hinge','log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'penalty':['l2', 'l1', 'elasticnet',]}])
    models.append([KNeighborsClassifier(),{'metric': ['euclidean','kulsinski','manhattan'],'n_neighbors':np.arange(1,20),}])

    # TO TESTING
    # models.append([SGDClassifier(),{'learning_rate':['constant','optimal','invscaling','adaptive'], 'penalty':['l2', 'l1', 'elasticnet',], 'loss':['hinge','log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}])
    # models.append([KNeighborsClassifier(),{'n_neighbors':np.arange(1,20),'metric':['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']}])

    models.append([AdaBoostClassifier(),{'n_estimators':np.arange(1,100,5),'learning_rate':np.arange(0.5,10,0.5),'algorithm':['SAMME','SAMME.R']}])
    models.append([DecisionTreeClassifier(),{'criterion':['gini','entropy','log_loss'],'max_features':['auto','sqrt','log2']}])



    return models


def find_best_models(models,features,targets,test_size=0.2,random_state=200,shuffle=False):
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)

    # model=create_pipelined_model( KNeighborsClassifier())
    # model.fit(features_train,targets_train)
    # print('train score',model.score(features_test,targets_test))

    # params_grid={'n_neighbors':np.arange(1,20),'metric':['seuclidean', 'braycurtis', 'sokalsneath', 'kulsinski', 'sqeuclidean', 'nan_euclidean', 'euclidean', 'chebyshev', 'wminkowski', 'sokalmichener', 'minkowski', 'l2', 'jaccard', 'dice', 'russellrao', 'p', 'cosine', 'correlation', 'matching', 'l1', 'pyfunc', 'canberra', 'precomputed', 'rogerstanimoto', 'infinity', 'haversine', 'manhattan', 'cityblock', 'mahalanobis', 'hamming', 'yule']}
    # params_grid={'n_neighbors':np.arange(1,20),'metric':['euclidean', 'manhattan']}
    # model=search_best(KNeighborsClassifier(),params_grid,features_train,targets_train)
    # print(model.score(features_test,targets_test))

    bests=[]
    for model in models:
        best_model,best_params,best_name=search_best(model[0],model[1],features_train,targets_train)
        print('best name',best_name)
        print('best model:',best_model.score(features_test,targets_test))
        print('best params:',best_params)
        
        bests.append({'name':best_name ,'model':best_model,'params':best_params})
    return bests

def train(model,features,targets,test_size=0.2,random_state=200,shuffle=False):
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)
    this_model=create_pipelined_model(model)
    this_model.fit(features_train,targets_train)
    score=this_model.score(features_test,targets_test)
    return this_model,score

def predict_place(model,row):
    numPmu=int(row.numPmu)
    prediction= model.predict(row)
    result=prediction[0]==1
    return numPmu,result,prediction


def get_history_files():
    files={}
    ff= os.listdir(os.path.join( dir_path,'history')) 
    for f in ff:
        n=os.path.basename(f)
        name=re.search('participants_(\w*).csv',n)
        if not name is None:
            files[name[1]]=os.path.join(dir_path,'history', f)
    return files
if __name__=="__main__":

    files=get_history_files()
    for this_type_course,file in files.items():
        features,targets=load_csv_file(file,nrows=100000)
        models=find_best_models( get_models_to_find_best(),features=features,targets=targets)
        for model in models:
            this_model=model.model
            this_name=model.name
            save_classifier(this_name,this_type_course,this_model)

    # features,targets=load_csv_file('E:\projets\pmu_scrapper\history\participants_trot_attele.csv',nrows=10000)
    # models=find_best_models( get_models_to_find_best(),features=features,targets=targets)
    pass

