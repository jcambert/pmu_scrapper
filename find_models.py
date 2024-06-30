
import sys,traceback
import re
import numpy as np
import pandas as pd
import os
import time


from logger import configure_logging
from operator import indexOf
from common import PATHES,DEFAULT_NROWS,execution_time_tostring
from classifier import *

from pandas import DataFrame
global log,verbose
HEADER_COLUMNS=['date','reunion','course','nom']
TARGETS=['ordreArrivee']
SUPPORTED_COURSES=['trot_attele','trot_monte','plat','obstacle']

DEFAULT_GRIDSEARCH_NJOB=1
DEFAULT_VERBOSE=1

music_pattern='([0-9,D,T,A,R][a,m,h,s,c,p,o]){1}'
music_prog = re.compile(music_pattern,flags=re.IGNORECASE)
music_penalities={'0':11,'D':6,'T':11,'A':11,'R':11}
DEFAULT_MUSIC=music_penalities['0']
NON_PLACE_NUMBER = -999
PLACE_NUMBER=1
QUINTE_NUMBER=5

def get_directory(directory,**args):
    use_folder=get_args('usefolder',None,**args)
    folder= os.path.join(PATHES[directory],use_folder) if use_folder else PATHES[directory]
    return folder

def get_model_name(model):
    return model.__class__.__name__.lower()

def get_args(name,default_,**args):
    return args[name] if name in args else default_

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
    if row['ordreArrivee'] in range(1,3):
        return PLACE_NUMBER
    return NON_PLACE_NUMBER
def quinte_converter(row):
    if row['ordreArrivee'] in range(4,5):
         return QUINTE_NUMBER
    return NON_PLACE_NUMBER

def load_csv_file(filename,nrows=None, is_for_prediction=False):

    usecols=None if is_for_prediction else HEADER_COLUMNS+FEATURES+TARGETS
    
    types={key:np.float64 for key in INT32_FEATURES if key not in ['sexe','musique']}
    types.update({key:np.float64 for key in FLOAT64_FEATURES })

    converters={'musique':musique_converter,'sexe':sexe_converter,'deferre':deferre_converter}
    df=pd.read_csv(filename,sep=";",header=0,usecols=usecols,dtype=types,converters=converters,nrows=nrows,low_memory=False,skip_blank_lines=True)

    if is_for_prediction:
        return df
    df.fillna({'ordreArrivee':0},inplace=True)
    targets=df.apply (lambda row: place_converter(row), axis=1)
    features=df[FEATURES]
    return features,targets

def get_nrows(**args) :
    """Return nrows from args line else DEFAULT_NROWS"""
    nrows = get_args('nrows',None,**args)
    nrows = nrows if isinstance(nrows,int) else (None if nrows=='max' else DEFAULT_NROWS)
    return nrows


def search_best(model,params,features_train:DataFrame,targets_train:DataFrame,cv=5,use_pipeline=True,**args):
    this_name=get_model_name( model)
    params_grid={}
    n_job =get_args('n_jobs',DEFAULT_GRIDSEARCH_NJOB,**args) 
    
    if(use_pipeline):
        this_model=create_pipelined_model(model)
        this_params=dict(( this_name+"__"+k,v) for k,v in params.items())
        # this_params=dict(( k,v) for k,v in params.items())
    else:
        this_model=model
        this_params=params
    this_params.update(params_grid)
    if verbose>1:
        log.info(f"features train:{features_train.shape}")
        log.info(features_train.head())
        log.info(type(features_train))

        log.info(f"targets train:{features_train.shape}")
        log.info(targets_train.head())
        log.info( type(targets_train))
    grid=GridSearchCV(this_model,this_params,cv=cv,verbose=verbose,n_jobs=n_job,refit=True,pre_dispatch='2*n_jobs',scoring='accuracy')
    grid.fit(features_train,targets_train)
    this_params=grid.best_params_
    this_model=grid.best_estimator_
    return this_model,this_params,this_name


def get_models_to_find_best(classifers):
    models=[]

    for classifier in classifers:
        if classifier is KNeighborsClassifier:
            models.append([classifier(),{'n_neighbors':np.arange(1,20),'metric':['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']}])
        elif classifier is AdaBoostClassifier:
            models.append([classifier(),{'n_estimators':np.arange(10,100,10),'learning_rate':np.arange(0.5,5,0.5,np.single),'algorithm':['SAMME']}])
        elif classifier is MLPClassifier:
            models.append([classifier(),{'hidden_layer_sizes':[20],'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam'],'learning_rate':['constant', 'invscaling', 'adaptive'],'early_stopping':[True]}])
        elif classifier is RidgeClassifier:
            models.append([classifier(),{'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'] }])
        elif classifier is SGDClassifier:
            models.append([classifier(),{'eta0':np.arange(0.1,0.9,0.3), 'learning_rate':['constant','optimal','invscaling','adaptive'], 'penalty':['l2', 'l1', 'elasticnet',], 'loss':['hinge','log', 'modified_huber', 'squared_hinge', 'perceptron', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}])
        elif classifier is DecisionTreeClassifier:
            models.append([classifier(),{'criterion':['gini','entropy'], 'max_depth':np.arange(1,2),'min_samples_split':np.arange(1,2),'min_samples_leaf':np.arange(1,2), 'max_features':np.arange(1,2)    }])



    return models


def find_best_models(models,features,targets,test_size=0.2,random_state=200,shuffle=False,**args):
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)


    bests=[]
    for model in models:
        best_model,best_params,best_name=search_best(model[0],model[1],features_train,targets_train,args=args)
        print('best name',best_name)
        print('best model:',best_model.score(features_test,targets_test))
        print('best params:',best_params)
        bests.append({'name':best_name ,'model':best_model,'params':best_params})
    return bests

def train(model,features,targets,test_size=0.2,random_state=200,shuffle=False,**args):
    log.info(f"start training {get_model_name(model)}")
    test_size=get_args('test_size',test_size, **args)
    random_state=get_args('random_state',random_state,**args)
    shuffle=get_args('shuffle',shuffle,**args)
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)
    this_model=model
    this_model.fit(features_train,targets_train)
    score=this_model.score(features_test,targets_test)
    log.info(f"{get_model_name(model)} training finished")
    return this_model,score

def predict_place(model,row):
    numPmu=int(row.numPmu)
    prediction= model.predict(row)
    result=prediction[0]==1
    return numPmu,result,prediction

def get_classifiers(**args):
    if 'classifiers' in args:
        tmp=args['classifiers'].split(',') 
        tmp=[ f'{c.lower()}classifier' for c in tmp]
        classifiers=[ c for c in SUPPORTED_CLASSIFIERS if c.__name__.lower() in tmp]
        return classifiers
    return   SUPPORTED_CLASSIFIERS

def get_files(path,filter='',courses=SUPPORTED_COURSES):
    """get files of supported courses"""
    if isinstance(courses, str):
        courses=courses.split(',')
    files={}
    ff= os.listdir(path) 
    for f in ff:
        n=os.path.basename(f)
        f=f'{filter}_({ "|".join(courses)}).csv' if courses else f'{filter}.csv'
        match=re.search(f,n)
        if  match :
            key= match[1] if len(match.groups())>0 else match[0]
            files[key]=os.path.join(path, match.string)
    return files


def save_df_to_csv(df,filename,mode):
    try:
        
        writeHeader=not os.path.exists(filename) or mode=='w'
        df.to_csv(filename,header=writeHeader,sep=";",mode=mode,index=False,na_rep='')
    except Exception as e:
        log.error(f"Un probleme est survenu lors de la sauvegarde de {filename}")
        log.error(str(e))

def scorer(**args):
    """ Get predicted and resultat for courses"""

    def create_key(row):
         """Compute Key"""
         return row['date']+'-'+str(row['reunion'])+'-'+str(row['reunion'])+'-'+str(row['course'])+'-'+str(row['numPmu'])
    
    def merge(left,right,mode='inner',key='key'):
        """Merge Panda DataFrame By Mode and Key"""
        result = pd.merge(left,right, on =key,how=mode).drop('specialite',axis=1).drop('date_y',axis=1).drop('reunion_y',axis=1).drop('course_y',axis=1).drop('numPmu_y',axis=1).drop(key,axis=1)
        return result
    
    def print_statistics(result,classifier=None,synthese:DataFrame=None):
        if classifier is None:
            col=result ['rapport']
        else:
            col=result.query(f"index_classifier=='{classifier}'")['rapport']
        somme = col.sum()
        reussite=col.notna().sum()
        compte=col.shape[0]
        pct = reussite/compte

        log.info(f"{classifier if classifier else ''} -> Somme des gain: {somme:0.2f} pour {compte} joués [ {pct:.2%} ]")
        log.info("*"*60)

        if synthese is not None and classifier is not None:
            synthese.loc[len(synthese.index)] = [usefolder.split(' ')[1],usefolder.split(' ')[0],classifier,somme,compte,pct]
            

    folder=get_directory('output',**args)
    score_file=os.path.join(folder,'score.csv')
    if os.path.exists(score_file):
        os.remove(score_file)

    usefolder=get_args('usefolder', None,**args)
    nrows=get_nrows(**args)
    courses = get_args('courses',SUPPORTED_COURSES,**args)
    # history_files=get_files(PATHES['input'],'participants',args['courses'] if 'courses' in args else SUPPORTED_COURSES)
    resultat_files= get_files( folder,'resultats',courses=courses)
    predicted_files= get_files(folder ,'predicted',None)
  
    predicted_file=predicted_files['predicted.csv']
    predicted_df=load_csv_file(predicted_file,nrows=nrows,is_for_prediction=True)
    classifiers=predicted_df['index_classifier'].unique().tolist()
    # date;reunion;course;numPmu;
    predicted_df['key']=predicted_df.apply(create_key,axis=1)
    # resultat_df.to_csv('d:/predicted.csv',sep=';')
    log.info("Load prediction file")
    log.info(predicted_df.shape)
    log.info(predicted_df.head())
    print("-"*60)

    synthese = pd.DataFrame({'year':pd.Series(dtype='int'),
                             'month':pd.Series(dtype='int'),
                             'index_classifier':pd.Series(dtype='string'),
                             'gain':pd.Series(dtype='float64'),
                             'played':pd.Series(dtype='float64'),
                             'pct':pd.Series(dtype='float64')})

    for key,file in resultat_files.items():
        resultat_df=load_csv_file(file,nrows=nrows,is_for_prediction= True).query("pari != 'E_SIMPLE_GAGNANT' ")
        resultat_df['key']=resultat_df.apply(create_key,axis=1)
        log.info(f"Load history file {key}")
        log.info(resultat_df.shape)
        log.info(resultat_df.head())

        result = merge(predicted_df, resultat_df,mode='left')
        result.to_csv(score_file,sep=';',mode='a',decimal=',')

        for classifier in classifiers:
            print_statistics(result,classifier=classifier,synthese=synthese)

        print_statistics(result)

    synthese_file=os.path.join( get_directory('output',args={}),'scores.csv')
    synthese.to_csv(synthese_file,sep=';',mode='a',decimal=',')
    pass

def finder(**args):
    """Finder best classifier function with various args"""
    files=get_files(PATHES['history'],'participants',args['courses'] if 'courses' in args else SUPPORTED_COURSES)
    nrows=get_nrows(**args)
    
    classifiers=get_classifiers(**args)
    for this_type_course,file in files.items():
        features,targets=load_csv_file(file,nrows=nrows)
        models=find_best_models( get_models_to_find_best(classifiers),features=features,targets=targets,args=args)
        for model in models:
            this_model=model['model']
            this_name=model['name']
            this_params=model['params']
            save_classifier(this_name,this_type_course,this_model)
            save_classifier_params(this_name,this_type_course,this_params)

def predicter(**args):
    """ Predicter Function"""
    nrows=get_nrows(**args)
    usefolder=get_args('usefolder', None,**args)
    # output_columns=['index_classifier','date','reunion','course','numPmu','place','nom','rapport','specialite','hippo_code']
    # remove rapport
    output_columns=['index_classifier','date','reunion','course','numPmu','place','nom','specialite','hippo_code']
    
    this_classifiers={}


    classifiers=get_classifiers(**args)
    directory=os.path.join(PATHES['input'],usefolder)
    files=get_files( directory,'topredict',args['courses'] if 'courses' in args else SUPPORTED_COURSES)
    output_df=pd.DataFrame(columns=output_columns)
    def internal_save(df):
        mode=args['mode'] if 'mode' in args else 'a'
        # folder= os.path.join(PATHES['output'],usefolder) if usefolder else PATHES['output'] 
        folder=get_directory('output',**args)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename=os.path.join(folder,f"predicted.csv")
        save_df_to_csv(df,filename=filename,mode=mode)

    for index,(course_name,file) in enumerate(files.items()):
        log.info(str("*"*20))
        log.info(f'Chargement de  {file}')
        df=load_csv_file(file,nrows=nrows, is_for_prediction=True)
        column_headers = list(df.columns.values)
        log.info(column_headers)
        log.info(f'Nombre de participants à predire: {df.shape[0]}')
        courses=df[['date','reunion','course','hippo_code']].drop_duplicates().sort_values(by=['date','reunion','course'])

        #TODO:Remove only for testing
        # log.info(f'Before filtering {courses.shape[0]}')
        # courses=courses[(courses['date']=='2023-02-28') & (courses['reunion']==2) & (courses['course']==3)]
        # log.info(f'After filtering {courses.shape[0]}')

        participants=df[['date','nom','numPmu']]

        for classifier_name in [c.__name__.lower() for c in classifiers ]:
            log.info(str("-"*20))
            
            log.info(f'Chargement de {classifier_name}-{course_name}')
            classifier,fitted=load_classifier(classifier_name,course_name)
            if classifier:
                this_classifiers[(classifier_name,course_name)]=classifier

                if not fitted:
                    log.info(f'the classifier {classifier_name} has not been fitted. Start training')
                    #train
                    history=get_files(PATHES['history'],'participants',classifier_name)
                    if len(history)==1:
                        features,targets=load_csv_file(history,nrows=nrows)
                        train(classifier,features=features,targets=targets)
                        #TODO:save the classifier
                        save_classifier(classifier_name,course_name,classifier)
                        pass

                for course in courses.iterrows():
                    x = np.asarray(course[1]).reshape(1,len(course[1]))
                    d,r,c,h=x[0,0],x[0,1],x[0,2],x[0,3]
                    participants_=df[(df['date']==d) & (df['reunion']==r) & (df['course']==c)].sort_values(by=['date','reunion','course','numPmu'])
                    # for z in range(participants_.shape[0]):
                    #     t=participants_.iloc[z]
                    #     log.info(f' {int(t.numPmu)} {t.nom}[{t.rapport}]')
                    
                    participants=participants_[NUMERICAL_FEATURES+  CATEGORICAL_FEATURES+CALCULATED_FEATURES]
                    log.info(str("+"*10))
                    log.info(f'Calcul de la Prediction pour Date:{d} - R{r}C{c}')
                    place=classifier.predict(participants)
                    res=participants.assign(place=place,
                                            hippo_code=h,
                                            reunion=r,
                                            course=c,
                                            state='place',
                                            nom=participants_['nom'],
                                            date=participants_['date'],
                                            specialite=course_name,
                                            resultat_place=0,
                                            resultat_rapport=0,
                                            gain_brut=0,
                                            gain_net=0,
                                            index_classifier=classifier_name)
                    log.info(f'Nombre de Prediction total:{res.shape[0]}')
                    res=res.loc[res['place']==1][output_columns]
                    log.info(f'Nombre de chevaux place:{res.shape[0]} pour {classifier_name}')
                    
                    for z in range(res.shape[0]):
                        t=res.iloc[z]
                        log.info(f' {int(t.numPmu)} {t.nom}')
                    output_df=pd.concat([output_df,res.copy()])
                # predict_place(classifier)
            else:
                log.warning(f'Le classifier {classifier_name.replace("classifier","")} n\'a pus etre chargé ')
    internal_save(output_df)

def trainer(**args):
    """Trainer function"""
    log.info(str("*"*15))
    log.info('Demarrage des entrainements')

    log.info("Chargements des classifiers avec leurs parametres par défaut")
    classifier_name=args['classifier'] if 'classifier' in args else None
    specialite=args['courses'] if 'courses' in args else None
    classifiers=load_classifiers_with_params(classifier_name,specialite)
    log.info(f'Chargement des clasifiers terminé: ({len(classifiers)} trouvé(s))')

    log.info("Recuperation des chemins des fichiers d\'historique")
    files=get_files(PATHES['history'],'participants',args['courses'] if 'courses' in args else SUPPORTED_COURSES)
    log.info(f"{len(files)} Fichiers d\'historique trouvé(s)")

    nrows=get_nrows(**args)
    for this_type_course,file in files.items():
        log.info(str("-")*35)
        log.info(f"Chargement des historiques de course de {this_type_course}")
        features,targets=load_csv_file(file,nrows=nrows)
        for index,(this_name,this_model) in enumerate(classifiers.items()):
            log.info(f"{index+1} --- Demarrage de l\'entrainement {this_name.upper()} sur {this_type_course}" )
            train(this_model,features=features,targets=targets,args=args)
            log.info(f"Entrainement {this_name.upper()} sur {this_type_course} terminé" )
            log.info("*"*20)
            log.info(f"Demarrage de la sauvegarde du classifier  {this_name.upper()} sur {this_type_course}" )
            save_classifier(f"{this_name}classifier",this_type_course,this_model)
            log.info(f"Sauvegarde du classifier  {this_name.upper()} sur {this_type_course} terminé" )
            log.info("+"*20)
    log.info("Entrainements des classifiers par course terminé !!")
    pass

MODES=['predicter' , 'trainer' , 'finder','scorer']

if __name__=="__main__":
    
    args=dict(arg.split('=') for arg in sys.argv[1:])

    func=args['func'] if 'func' in args else 'trainer'
    log = configure_logging(func,**args)
    verbose = int(args['verbose']) if 'verbose' in args else DEFAULT_VERBOSE
    try:
        if func not in MODES:
            raise KeyError(f'le mode {func} n\'est pas supporté')
        start_time = time.time()
        locals()[func](**args)
        end_time=time.time()
        log.info(f"Temps d'execution: {execution_time_tostring(start=start_time,end=end_time)}")
    except KeyError as e:
        log.fatal(str(e))
    except Exception as e:
        log.fatal("Un erreur innatendue est survenue")
        log.fatal(str(e))
        if verbose>1:
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
    pass

