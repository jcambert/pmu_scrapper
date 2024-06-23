import json
import os
import sys
import re
import numpy as np
from joblib import dump,load
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, RobustScaler,StandardScaler
from common import PATHES,DEFAULT_NROWS,execution_time_tostring

SUPPORTED_CLASSIFIERS=(AdaBoostClassifier,RidgeClassifier,SGDClassifier,MLPClassifier,KNeighborsClassifier,DecisionTreeClassifier)
INT32_FEATURES=['numPmu','rapport','age','sexe','musique']
FLOAT64_FEATURES=['nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente',]
# NUMERICAL_FEATURES=['numPmu','rapport','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente','sexe','musique']
NUMERICAL_FEATURES=INT32_FEATURES+FLOAT64_FEATURES
# Remove rapport column for testing weight of this values
# NUMERICAL_FEATURES=['numPmu','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente','sexe','musique']
CATEGORICAL_FEATURES=['hippo_code','deferre']
CALCULATED_FEATURES=[]
# NUMERICAL_FEATURES=['numPmu']
# CATEGORICAL_FEATURES=['hippo_code']
FEATURES=NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES

def load_classifier(name,type_course,from_params=False):
    if not from_params:
        file=has_classifier(name,type_course,'model')
        if file:
            return load(file),True
    
    file= has_classifier(name,type_course,'json')
    if file:
        with open(file, "r") as fp:
            this_params=json.load(fp)
            this_params=dict(( k.replace(f'{name}__',''),v) for k,v in this_params.items())
        cls= [ c for c in SUPPORTED_CLASSIFIERS if c.__name__.lower()==name]
        this_cls=cls[0](**this_params)
        return create_pipelined_model(this_cls),False
    return False,False

def save_classifier(name,type_course,classifier):
    path=PATHES['model'] 
    if(not os.path.exists(path)):
        os.mkdir(path)
    file=os.path.join(path, f'{name}_{type_course}.model')
    if(os.path.isfile(file)):
        os.remove(file)
    dump(classifier,file)

def has_classifier(name,type_course,model_or_params='model'):
    file=os.path.join(PATHES['model'], f'{name}_{type_course}.{model_or_params}')
    return file if os.path.isfile(file) else False




def json_type_converter(value):
    if isinstance(value, np.generic): return value.item()  
    return value

def save_classifier_params(name,type_course,params):
    path=PATHES['model']
    if(not os.path.exists(path)):
        os.mkdir(path)
    file=os.path.join(path, f'{name}_{type_course}.json')
    if(os.path.isfile(file)):
        os.remove(file)
    try:
        with open(file, "w") as fp:
            json.dump(params,fp, default=json_type_converter) 
    except :
        print('ERROR====>')
        print( sys.exc_info()[0] )

def load_classifiers_with_params(classifier=None,specialite=None):
    
    path=PATHES['model']
    classifiers = {}
    ff= os.listdir(path) 
    classifier__=classifier if classifier is not None else r'(\w*)'
    specialite__=specialite if specialite is not None else r'([a-z_a-z*]*)'
    s=f'{classifier__}classifier_{specialite__}.json'
    for f in ff:
        n=os.path.basename(f)
        name=re.search(s ,n)
        if name:
            this_classifier=name[1] if classifier is None else classifier
            this_course=name[2] if specialite is None else specialite
            c=load_classifier(f'{this_classifier}classifier',this_course,True)
            if c:
                classifiers[this_classifier]=c[0]
    
    return classifiers

def create_pipelined_model(model):
    numerical_pipeline=make_pipeline(SimpleImputer(strategy='constant', fill_value=-999), StandardScaler())
    categorical_pipeline=make_pipeline(OneHotEncoder(handle_unknown = 'ignore'))
    preprocessor=make_column_transformer(
        (numerical_pipeline,NUMERICAL_FEATURES),
        (categorical_pipeline,CATEGORICAL_FEATURES))
    model= make_pipeline(preprocessor,PolynomialFeatures(),VarianceThreshold(0.05),model)
    return model
