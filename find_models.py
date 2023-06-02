import json
import sys
import re
import numpy as np
import pandas as pd
import os.path
from logger import configure_logging
from operator import indexOf
from joblib import dump,load
from sklearn.compose import make_column_transformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.tree import DecisionTreeClassifier



HEADER_COLUMNS=['date','reunion','course','nom']
NUMERICAL_FEATURES=['numPmu','rapport','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente','sexe','musique']
# CATEGORICAL_FEATURES=['hippo_code','deferre']
CATEGORICAL_FEATURES=['deferre']
CALCULATED_FEATURES=[]

TARGETS=['ordreArrivee']
FEATURES=NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES


_base_dir= os.path.dirname(os.path.realpath(__file__))
PATHES={'base':_base_dir, 'model':os.path.join(_base_dir, 'models_test'),'input':os.path.join(_base_dir, 'input'),'output':os.path.join(_base_dir, 'output'),'history':os.path.join(_base_dir, 'history')}

SUPPORTED_CLASSIFIERS=(AdaBoostClassifier,RidgeClassifier,SGDClassifier,MLPClassifier,KNeighborsClassifier,DecisionTreeClassifier)
SUPPORTED_COURSES=['trot_attele','trot_monte','plat','obstacle']




def load_classifier(name,type_course):
    file=has_classifier(name,type_course,'model')
    if file:
        return load(file),True
    
    file= has_classifier(name,type_course,'json')
    if file:
        with open(file, "r") as fp:
            this_params=json.load(fp)
            this_params=dict(( k.replace(f'{name}__',''),v) for k,v in this_params.items())
        cls= [ c for c in SUPPORTED_CLASSIFIERS if c.__name__.lower()==name]
        this_cls=cls[0](this_params)
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
    this_name=model.__class__.__name__.lower()
    params_grid={}

    if(use_pipeline):
        this_model=create_pipelined_model(model)
        this_params=dict(( this_name+"__"+k,v) for k,v in params.items())
    else:
        this_model=model
        this_params=params
    this_params.update(params_grid)
    grid=GridSearchCV(this_model,this_params,cv=cv,verbose=2,n_jobs=2)
    grid.fit(features_train,targets_train)
    this_params=grid.best_params_
    this_model=grid.best_estimator_
    return this_model,this_params,this_name


def get_models_to_find_best():
    models=[]

    # models.append([KNeighborsClassifier(),{'n_neighbors':np.arange(1,20),'metric':['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']}])
    
    # models.append([SGDClassifier(),{'eta0':np.arange(0.1,0.9,0.3), 'learning_rate':['constant','optimal','invscaling','adaptive'], 'penalty':['l2', 'l1', 'elasticnet',], 'loss':['hinge','log', 'modified_huber', 'squared_hinge', 'perceptron', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}])
    # models.append([AdaBoostClassifier(),{'n_estimators':np.arange(10,100,10),'learning_rate':np.arange(0.5,5,0.5,np.single),'algorithm':['SAMME','SAMME.R']}])
    # models.append([MLPClassifier(),{'hidden_layer_sizes':[100],'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam'],'learning_rate':['constant', 'invscaling', 'adaptive'],'early_stopping':[True]}])
    # models.append([RidgeClassifier(),{'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'] }])


    # models.append([DecisionTreeClassifier(),{'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2']}])



    return models


def find_best_models(models,features,targets,test_size=0.2,random_state=200,shuffle=False):
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)


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

def get_files(path,filter='',courses=SUPPORTED_COURSES):
    if isinstance(courses, str):
        courses=courses.split(',')
    files={}
    ff= os.listdir(path) 
    for f in ff:
        n=os.path.basename(f)
        name=re.search(f'{filter}_({ "|".join(courses)}).csv',n)
        if not name is None:
            files[name[1]]=os.path.join(path, f)
    return files

# def get_history_files(courses=SUPPORTED_COURSES):
#     files={}
#     ff= os.listdir(PATHES['history']) 
#     for f in ff:
#         n=os.path.basename(f)
#         name=re.search(f'participants_({ "|".join(courses)}).csv',n)
#         if not name is None:
#             files[name[1]]=os.path.join(PATHES['history'], f)
#     return files

# def get_input_files(courses=SUPPORTED_COURSES):
#     files={}
#     ff= os.listdir(PATHES['input']) 
#     for f in ff:
#         n=os.path.basename(f)
#         name=re.search(f'topredict_({ "|".join(courses)}).csv',n)
#         if not name is None:
#             files[name[1]]=os.path.join(PATHES['input'], f)
#     return files

def finder(**args):
    files=get_files(PATHES['history'],'participants',args['courses'] if 'courses' in args else SUPPORTED_COURSES)
    nrows=args['nrows'] if 'nrows' in args else DEFAULT_NROWS
    for this_type_course,file in files.items():
        features,targets=load_csv_file(file,nrows=nrows)
        models=find_best_models( get_models_to_find_best(),features=features,targets=targets)
        for model in models:
            this_model=model['model']
            this_name=model['name']
            this_params=model['params']
            save_classifier(this_name,this_type_course,this_model)
            save_classifier_params(this_name,this_type_course,this_params)

def predicter(**args):
    this_classifiers={}
    files=get_files(PATHES['input'],'topredict','trot_attele,plat')
    files1=get_files(PATHES['input'],'topredict',args['courses'] if 'courses' in args else SUPPORTED_COURSES)
    for index,(course,file) in enumerate(files.items()):
        for classifier_name in [c.__name__.lower() for c in SUPPORTED_CLASSIFIERS ]:
            log.warning(f'Load classifier {classifier_name}-{course}')
            classifier,fitted=load_classifier(classifier_name,course)
            if classifier:
                this_classifiers[f'{classifier_name}_{course}']=classifier

    if not fitted:
        #train
        pass
    pass

def trainer(**args):
    pass

MODES=['predicter' , 'trainer' , 'finder']
DEFAULT_NROWS=200000
if __name__=="__main__":
    args=dict(arg.split('=') for arg in sys.argv[1:])

    mode=args['mode'] if 'mode' in args else 'predicter'
    log= configure_logging(mode)

    try:
        if mode not in MODES:
            raise KeyError(f'le mode {mode} n\'est pas supporté')
        locals()[mode](**args)
    except KeyError as e:
        log.fatal(str(e))
    except Exception as e:
        log.fatal("Un erreur innatendue est survenue")
        log.fatal(str(e))
    pass

