from operator import indexOf
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import logging
import os.path
from joblib import dump,load
from os import error, path
from sklearn.base import TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,OneHotEncoder,LabelEncoder,PolynomialFeatures,FunctionTransformer
from sklearn.model_selection import cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_selection import RFE,RFECV,VarianceThreshold
from sklearn.model_selection import  learning_curve
from sklearn.metrics import confusion_matrix

from logger import configure_logging

HEADER_COLUMNS=['date','reunion','course','nom']
NUMERICAL_FEATURES=['numPmu','rapport','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente','sexe','musique']
CATEGORICAL_FEATURES=['hippo_code','deferre']
CALCULATED_FEATURES=[]
MODEL_PATH='models'
DATA_PATH='datas'
encoders={}
encoder=OneHotEncoder(handle_unknown = 'ignore')

music_pattern='([0-9,D,T,A,R][a,m,h,s,c,p,o]){1}'
music_prog = re.compile(music_pattern,flags=re.IGNORECASE)
music_penalities={'0':11,'D':6,'T':11,'A':11,'R':11}

DEFAULT_MUSIC=11


def load_classifier(name,type_course):
    if not has_classifier(name,type_course):
        return False
    return load(os.path.join(MODEL_PATH, f'{name}_{type_course}.model'))

def save_classifier(name,type_course,classifier):
    dump(classifier,os.path.join(MODEL_PATH, f'{name}_{type_course}.model'))

def has_classifier(name,type_course):
    return os.path.isfile(os.path.join(MODEL_PATH, f'{name}_{type_course}.model'))


class FuncTransformer(TransformerMixin):
    def __init__(self, func):
        self.func = func
    def fit(self,X,y=None, **fit_params):
        return self
    def transform(self,X, **transform_params):
        return self.func(X)
        
def process_music_transform(df):
    if "musique" in df:
        df_=df["musique"].map(lambda r:calculate_music(r))
        df.drop(['musique'],axis=1)
        
        return pd.concat([df,df_], ignore_index=True)
    return df

def calculate_music(music,speciality='a'):
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

def encode(df,categories):
    for category in categories:
        if not category in encoders:
            encoders[category]=LabelEncoder()
            encoders[category].fit(df[category])
        df[category]= encoders[category].transform(df[category])

def place_converter(row):
    return 1 if row['ordreArrivee'] in range(1,3) else 0

def load_file(filename,is_predict=False):
    types={key:np.number for key in NUMERICAL_FEATURES if key not in ['sexe','musique','deferre']}
    df=pd.read_csv(filename,sep=";",header=0,usecols=HEADER_COLUMNS+NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES+['ordreArrivee'],dtype=types,low_memory=False,converters={'musique':calculate_music,'sexe':sexe_converter,'deferre':deferre_converter})

    if not is_predict:
        df['ordreArrivee'].fillna(0,inplace=True)
        places=df.apply (lambda row: place_converter(row), axis=1)
        targets = places
        features = df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES]
        return features,targets
    else:
        courses=df[['date','reunion','course','hippo_code']].drop_duplicates().sort_values(by=['date','hippo_code','reunion','course'])
        participants=df[['date','nom','numPmu']]
        return df[HEADER_COLUMNS+NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES],courses,participants

def load_to_predict_file(filename):
    df=pd.read_csv(filename,sep=";")
    courses=df[['reunion','course']].drop_duplicates()
    return df,courses

def learning_curve_data(model,X_train,y_train,train_sizes=None,cv=None):
    if not train_sizes:
        train_sizes=np.linspace(0.2,1.0,5)
    N,train_score,val_score=learning_curve(model,X_train,y_train,train_sizes=train_sizes,cv=cv)
    return N,train_score,val_score

classifiers = [
    ('sgdclassifier',None),
    ('kneighborsclassifier',KNeighborsClassifier(3)),
    ('decisiontreeclassifier',DecisionTreeClassifier(max_depth=5)),
    ('randomforestclassifier', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ('adaboostclassifier',AdaBoostClassifier( )),
    ]
def load_classifiers_for_type_course(type_course):
    models={}
    result=False
    for classifier in classifiers:
        res=load_classifier(classifier[0],type_course)
        models[classifier[0]]=res
        result=result or res
    return result,models
def train(features,targets,test_size=0.15,random_state=200,shuffle=False):
    
    # GridSearchCv Result
    #{'sgdclassifier__eta0': 0.05, 'sgdclassifier__learning_rate': 'optimal', 'sgdclassifier__loss': 'squared_hinge', 'sgdclassifier__max_iter': 5000, 'sgdclassifier__n_jobs': 1, 'sgdclassifier__shuffle': True}
    classifier=SGDClassifier(random_state=random_state,loss='squared_hinge',shuffle=True,learning_rate='optimal')
    # classifier=SGDClassifier(random_state=random_state)
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)

    numerical_pipeline=make_pipeline(SimpleImputer(fill_value=0), RobustScaler())
    categorical_pipeline=(make_pipeline(OneHotEncoder(handle_unknown = 'ignore')))
    preprocessor=make_column_transformer(
        (numerical_pipeline,NUMERICAL_FEATURES),
        (categorical_pipeline,CATEGORICAL_FEATURES))

    _models={}
    _models['sgdclassifier']=make_pipeline(preprocessor,PolynomialFeatures(),VarianceThreshold(0.05),classifier)
    
    for classifier in classifiers:
        if classifier[1] is not None:
            model_=make_pipeline(preprocessor,PolynomialFeatures(),VarianceThreshold(0.05),classifier[1])
            _models[classifier[0]]=model_

    
    for index, (key, model) in enumerate(_models.items()):
        logging.info(f"Start fitting for model {index+1}/{len(_models)}-{key}")
        model.fit(features_train,targets_train)

    logging.info("Fiting is finished")
    return _models,features_train, features_test, targets_train, targets_test 

def predict_place(model,row):
    
    numPmu=int(row.numPmu)
    prediction= model.predict(row)
    place=prediction[0]==1
    return numPmu,place,prediction

models={

}



if __name__=='__main__':
    args=dict(arg.split('=') for arg in sys.argv[1:])
    log= configure_logging("Predicter")
    
    filter_by_hippo_code=False

    save_to_file=True
    print_confusion_matrix=False
    print_training_score=True
    print_result=True
    training_files={'trot attele':'trot_attele','plat':'plat','trot monte':'trot_monte','obstacle':'obstacle'}
    filter_max_horse_count_by_course=3
    usefolder=args['usefolder'] if 'usefolder' in args else ''

    output={'date':[],'reunion':[],'course':[],'nom':[],'rapport':[],'numPmu':[],'state':[]}    
    output_columns=['index_classifier','date','reunion','course','numPmu','place','nom','rapport','specialite','hippo_code']
    output_df=pd.DataFrame(columns=output_columns)
    for key,file in training_files.items():
        try:
            log.info(f"Start prediction for {key}")

            to_predict,courses,chevaux=load_file(os.path.join("input", usefolder,f"topredict_{file}.csv"),is_predict=True)
            has_models,saved_models=load_classifiers_for_type_course(file)
            if not has_models:
                log.info(f"{file} has no model=> create model train")
                features,targets=load_file(os.path.join("history", f"participants_{file}.csv"))
                if not filter_by_hippo_code:
                    models_,features_train, features_test, targets_train, targets_test =train(features,targets,test_size=0.05,shuffle=True)
                    
                    for idx_, (name_, model_) in enumerate(models_.items()):
                        save_classifier(name_,file,model_)
                    hippo_code='all'
                    models[hippo_code]={}
                    models[hippo_code]['model']=models_
                    models[hippo_code]['features_train']=features_train
                    models[hippo_code]['features_test']=features_test
                    models[hippo_code]['targets_train']=targets_train
                    models[hippo_code]['targets_test']=targets_test
            else:
                log.info(f"{file} has a model to predict")
                if not filter_by_hippo_code:
                    hippo_code='all'
                    models[hippo_code]={}
                    models[hippo_code]['model']=saved_models
               
                
            
            for course in courses.iterrows():
                x = np.asarray(course[1]).reshape(1,len(course[1]))
                d,r,c,h=x[0,0],x[0,1],x[0,2],x[0,3]
                if filter_by_hippo_code:
                    hippo_code=h
                if filter_by_hippo_code  and h not in models:
                    if not has_models:
                        models_,features_train, features_test, targets_train, targets_test =train(features[features['hippo_code']==h],targets[features['hippo_code']==h],test_size=0.2,shuffle=True)
                        for idx_, (name_, model_) in enumerate(models_.items()):
                            save_classifier(name_,file,model_)
                        models[hippo_code]={}
                        models[hippo_code]['model']=models_
                        models[hippo_code]['features_train']=features_train
                        models[hippo_code]['features_test']=features_test
                        models[hippo_code]['targets_train']=targets_train
                        models[hippo_code]['targets_test']=targets_test
                    else:
                        models[hippo_code]={}
                        models[hippo_code]['model']=saved_models

                participants_=to_predict[(to_predict['date']==d) & (to_predict['reunion']==r) & (to_predict['course']==c)]
                log.info(f"Try to predict some Number for Date:{d} - Reunion:{r} - Course:{c}")
                participants=participants_[NUMERICAL_FEATURES+  CATEGORICAL_FEATURES+CALCULATED_FEATURES]
                
                try:
                    
                    for idx, (key_, model) in enumerate(models[hippo_code]['model'].items()):
 
                        place=model.predict(participants)
                    
                        res=participants.assign(place=place,
                                            hippo_code=h,
                                            reunion=r,
                                            course=c,
                                            state='place',
                                            nom=participants_['nom'],
                                            date=participants_['date'],
                                            specialite=key,
                                            resultat_place=0,
                                            resultat_rapport=0,
                                            gain_brut=0,
                                            gain_net=0,
                                            index_classifier=key_)
                        res=res.loc[res['place']==1][output_columns]
                        count=res.shape[0]
                        if (filter_max_horse_count_by_course and count<=filter_max_horse_count_by_course) or not filter_max_horse_count_by_course:
                            output_df=pd.concat([output_df,res.copy()])
                            if print_result:
                                for z in range(count):
                                    t=res.iloc[z]
                                    log.info(f"R{r}/C{c} -> {t.nom}[{t.rapport}] {t.numPmu} placé -> Classifier:{key_}" )
                        elif count==0 and print_result:      
                            log.info(f"R{r}/C{c} -> aucune prediction")
                except Exception as ex:
                    log.warning(ex)
                
        except FileNotFoundError as fnf_error:
            log.warning(f'{fnf_error}. Prediction is impossible')
            pass
        except Exception as ex:
            log.warning(ex)
    if save_to_file:
        

        output_df=output_df.sort_values(by=['date','reunion','course'])
        def save_fn():
            folder=os.path.join("output",usefolder)
            if not os.path.exists(folder):
                os.makedirs(folder)

            filename=os.path.join(folder,f"predicted.csv")
            mode=args['mode'] if 'mode' in args else 'a'
            
            writeHeader=not path.exists(filename) or mode=='w'

           

            output_df.to_csv(filename,header=writeHeader,sep=";",mode=mode,index=False,na_rep='')
            
            output_df.to_html(os.path.join(folder, f"predicted.html"),header=True,justify='left',border=1)

        try:
            save_fn()
        except PermissionError as e:
            save_fn()

