import json
from pandas.core.base import IndexOpsMixin
from pandas.core.frame import DataFrame
import requests
import pandas as pd
import numpy as np
import logging
import time
import threading
from os import error, path
from datetime import datetime,  timedelta,date

PROXIES = {
    'http': 'socks5://localhost:9050',
    'https': 'socks5://localhost:9050'
}
#PMU_MIN_DATE='01032013'
PMU_DATE_FORMAT='%d%m%Y'

PREDICT_FILENAME_PREFIX="to_predict_"

prg_url="https://online.turfinfo.api.pmu.fr/rest/client/1/programme/%s?meteo=false&specialisation=INTERNET"
ptcp_url="https://online.turfinfo.api.pmu.fr/rest/client/1/programme/%s/R%s/C%s/participants?specialisation=INTERNET"
resultat_url="https://turfinfo.api.pmu.fr/rest/client/1/programme/%s/R%s/C%s/rapports-definitifs?specialisation=INTERNET&combinaisonEnTableau=true"
courses_filename="courses.csv"
ptcp_filename="%sparticipants_%s.csv"
def get_tommorow(fmt=False):
    yesterday = datetime.now() + timedelta(1)
    if fmt:
        return yesterday.strftime(fmt)
    return yesterday
def get_yesterday(fmt=False):
    yesterday = datetime.now() - timedelta(1)
    if fmt:
        return yesterday.strftime(fmt)
    return yesterday

def get_today(fmt=False):
    today = datetime.now()
    if fmt:
        return today.strftime(fmt)
    return today

def get_pmu_date(d):
    if isinstance(d,str):
        d=d.replace('/','')
        if len(d)==8:
            d=date(int(d[-4:]),int(d[2:4]),int(d[:2]))
        elif len(d)==6:
            century=int(datetime.now().year/100)*100
            d=date(century+int(d[-2:]),int(d[2:4]),int(d[:2]))
        else:
            raise ValueError("PMU Date string must match 010221 or 01022021")
    if isinstance(d,date):
        return d.strftime(PMU_DATE_FORMAT)
    raise ValueError(f"PMU date {d} is not supported format")

def get_date_from_pmu(d):
    if not isinstance(d,str) and len(d)!=8:
        raise ValueError(f"{d} must be a pmu date string format")
        
    return date(int(d[-4:]),int(d[2:4]),int(d[:2]))

yesterday,_yesterday=get_yesterday('%d%m%Y'),get_yesterday('%Y%m%d')


class AbstractScrapper():
    def __init__(self,use_proxy=True,use_threading=True,test=False,**kwargs):
        # self._use_proxy= kwargs['use_proxy'] if 'use_proxy' in kwargs else True
        # self._use_threading= kwargs['use_threading'] if 'use_threading' in kwargs else True
        # self._test=kwargs['test'] if 'test' in kwargs else False
        self._use_proxy,self._use_threading,self._test=use_proxy,use_threading,test
        self._fname= kwargs['fname'] if 'fname' in kwargs else None
        if('logger' in kwargs):
            self.logger=kwargs['logger']
        else:
            self.logger=logging
        self._mode=kwargs['mode'] if 'mode' in kwargs else 'a'
        self._usefolder=kwargs['usefolder'] if 'usefolder' in kwargs else ''
    def get_save_mode(self):
        return self._mode
    def _origins(self):
        url = 'http://httpbin.org/ip'
        o_p=json.loads(self._request(url,use_proxy=True).text)['origin']
        o_o=json.loads(self._request(url,use_proxy=False).text)['origin']
        return ( o_p,o_o)
    
    def _request(self,url,**kwargs):
        __use_proxy= kwargs['use_proxy'] if 'use_proxy' in kwargs else self._use_proxy
        return  (requests.get(url, proxies=PROXIES) if __use_proxy else  requests.get(url) )
    
    def _save(self,df,filename,mode='a'):
        self.logger.info(f"Saving to {filename}")
        if not self._test:
            writeHeader=not path.exists(filename) or mode=='w'
            df.to_csv(filename,sep=";",na_rep='',mode=mode,index=False,header=writeHeader)
        else:
            self.logger.info("Mode Test=> No saving file action")
        self.logger.info(f"{filename} saved")
    def _check(self):
        if self._use_proxy:
            o=self._origins()
            if o[0]==o[1]:
                raise AssertionError(f"When using proxy {o[0]} must be different than {o[1]} \ensure you are set proxy to Internet Options")
    def _scrap(self,specialites, day):
        raise NotImplementedError(f"You cant run {self.__class__.__name__}")
    def get_default_start_date(self):
        return get_yesterday()
    def get_save_mode(self):
        return 'a'
    def get_filename(self):
        return self._fname if isinstance(self._fname,str) else self._filename
    def get_end(self,current,**kwargs):
        if 'end' in kwargs:
            result= get_date_from_pmu( kwargs['end'])
        elif 'count' in kwargs:
            result=current+timedelta( int(kwargs['count']))
        else:
            result=get_date_from_pmu(yesterday)
        return result
    def start(self,start=None,**kwargs):
        if(self.__class__.__name__ == type(AbstractScrapper).__name__):
            raise NotImplementedError(f"You cant run {self.__class__.__name__}")
        self._check()
        if not start:
            start=self.get_default_start_date()
        start=get_pmu_date(start)
        current=get_date_from_pmu( start)
        step=int(kwargs['step']) if 'step' in kwargs else 1
        
        if 'count' in kwargs:
            count = int(kwargs['count'])
        else:
            if start==get_pmu_date(get_yesterday()) or start==get_pmu_date(get_today()):
                kwargs['count']=0
        # end=get_date_from_pmu( kwargs['end']) if 'end' in kwargs  else ( current+timedelta( int(kwargs['count'])+1) if 'count' in kwargs else get_date_from_pmu(yesterday))
        end=self.get_end(current,**kwargs)
        sleep=int(kwargs['sleep']) if 'sleep' in kwargs else 500
        specialites = kwargs['specialites'] if 'specialites' in kwargs else None
        self.logger.info(f"Start scrapping {self.__class__.__name__} from {get_date_from_pmu( start)} to {end} exclude")
        while current<end:
            try:
                self.logger.info(f"Start day {current}")
                self._scrap(specialites, get_pmu_date(current ))
                time.sleep(sleep/1000)
                current=current+ timedelta(step)
            except Exception  as ex:
                self.logger.warn(ex,exc_info=True)
                self.logger.warn(f"an error happened while scrap {current}. go to next day")
                current=current+ timedelta(step)
        return (start,current,step)

    def _get_reunions(self,date,**kwargs):

        self.logger.info(f"Get Reunion:{prg_url%date}")
        
        try:
            resp=self._request(prg_url % date)
            if resp.status_code!=200:
                return False
            as_json=kwargs['as_json'] if 'as_json' in kwargs else False
            if as_json:
                return resp.json()
            df=pd.DataFrame(resp.json())
            return df
        except Exception as ex:
            self.logger.warning(ex)
            return False

    def _get_participants(self,reunion,course,date,**kwargs):
    #     print(ptcp_url % (date,reunion,course))
        self.logger.info(f"Get Participants {ptcp_url % (date,reunion,course)}")
        resp=self._request(ptcp_url % (date,reunion,course))
        as_json=kwargs['as_json'] if 'as_json' in kwargs else False
        if  as_json:
            return resp
        df=pd.DataFrame(resp.json()['participants'])
        return df

    def _get_resultats(self,date,reunion,course,**kwargs):
        self.logger.info(f"Get Resultat {resultat_url % (date,reunion,course)}")
        resp=self._request(resultat_url % (date,reunion,course)).json()
        as_json=kwargs['as_json'] if 'as_json' in kwargs else False
        if  as_json:
            return resp
        df=pd.DataFrame(resp)
        
        return df

class ResultatScrapper(AbstractScrapper):
    def __init__(self,use_proxy=True,use_threading=True,test=False,**kwargs):
        super().__init__(use_proxy,use_threading,test,**kwargs)
        self._filename="resultats_%s.csv"
    # def get_save_mode(self):
    #     return 'a'
    def _scrap(self,specialites,day):
        df_reunions=self._get_reunions(day,as_json=True)
        if isinstance(df_reunions,bool) and not df_reunions:
            return
        courses=pd.DataFrame()
        list_reunions= df_reunions['programme']['reunions']
        resultats={'TROT_MONTE':[],'TROT_ATTELE':[],'PLAT':[],'OBSTACLE':[]}
        for reunion in list_reunions:
            threads = list()
            for course in reunion['courses']:
                num_reunion=int(reunion['numExterne'])
                num_course=int(course['numExterne'])
                specialite=course['specialite']
                c=pd.DataFrame.from_dict(course, orient = 'index').transpose()
                hippo=c['hippodrome'].to_dict()[0]
                c['hippoCode']=hippo['codeHippodrome']
                c['hippoCourt']=hippo['libelleCourt']
                c['hippoLong']=hippo['libelleLong']
                if(courses.shape[0]==0):
                    courses=c
                else:
                    courses=pd.concat([c,courses], axis=0)
                # print(courses.head())
                if specialites is  not None and specialite not in specialites:
                    continue
                if self._use_threading:
                    self.logger.info(f"Start reading resultat R{num_reunion}/C{num_course}")
                    
                    x = threading.Thread(target=self.__scrap_resulats, args=(day,num_reunion,num_course,resultats[specialite]))
                    threads.append(x)
                    x.start()
                else:

                    scrap_resultats=self.__scrap_resulats(day,num_reunion,num_course)
                    if isinstance( scrap_resultats,DataFrame):
                        resultats[specialite].append(scrap_resultats)
                    self.logger.info(f"End reading  resultat R{num_reunion}/C{num_course}")
            for index, thread in enumerate(threads):
                thread.join()
        for spec in resultats:
            if len(resultats[spec])>0:
                df_resultats=pd.concat(resultats[spec])
                self._save(df_resultats,path.join("output",self._usefolder, self.get_filename() % spec.lower()),self.get_save_mode())

        courses=courses[['numReunion','numOrdre','libelle','libelleCourt','montantPrix','distance','distanceUnit','discipline','specialite','nombreDeclaresPartants','ordreArrivee','hippoCode','hippoCourt','hippoLong']]
        courses['date']=get_date_from_pmu(day)
        self._save(courses,path.join("output",self._usefolder, "courses.csv"),self.get_save_mode())
    def __scrap_resulats(self,day,reunion,course,result=False):
        lines=[]
        try:
            resultats=self._get_resultats(day,reunion,course,as_json=True)
            
            for resultat in resultats:
                mise_base=int(resultat['miseBase'])
                libelle=resultat['typePari']
                for rapport in resultat['rapports']:
                    dividende=rapport['dividendePourUneMiseDeBase']
                    for combinaison in rapport['combinaison']:
                        try:
                            numPmu=int(combinaison)
                            line=(get_date_from_pmu(day) ,reunion,course,libelle,numPmu,dividende/mise_base)
                            lines.append(line)
                        except ValueError:
                            pass
                
            if isinstance(result,list):
                df_lines=pd.DataFrame(lines,columns=['date','reunion','course','pari','numPmu','rapport'])
                df_lines = df_lines[df_lines.pari.isin (['E_SIMPLE_GAGNANT','E_SIMPLE_PLACE'])]
                result.append( df_lines)
            else:
                df= pd.DataFrame(lines,columns=['date','reunion','course','pari','numPmu','rapport'])
                df = df[df.pari.isin( ['E_SIMPLE_GAGNANT','E_SIMPLE_PLACE'])]
                return df
        except Exception as ex:
            self.logger.error(ex)
            return False

class HistoryScrapper(AbstractScrapper):
    def __init__(self,use_proxy=True,use_threading=True,test=False,**kwargs):
        super().__init__(use_proxy,use_threading,test,**kwargs)
        self._filename="participants_%s.csv"
        self._directory="history"
        
      
    def _scrap(self,specialites,day):
        day=get_pmu_date(day)
        self.logger.info(f"Start scraping {get_date_from_pmu( day)}")
        participants={'TROT_MONTE':[],'TROT_ATTELE':[],'PLAT':[],'OBSTACLE':[]}
        df_reunions=self._get_reunions(day)
        if isinstance(df_reunions,bool) and not df_reunions:
            return
        df_reunions=df_reunions['programme']['reunions']
        for r_index,reunion in enumerate(df_reunions):
            sub=pd.DataFrame.from_dict(reunion,orient="index")[0]
            subdf=pd.json_normalize(sub['courses'],max_level=1)

            threads = list()
            for c_index,course in subdf.iterrows():
                subdf_ptcp=None
                specialite=course['specialite']
                if specialites is  not None and specialite not in specialites:
                    continue
                if self._use_threading:
                    self.logger.info(f"START REUNION {course['numReunion']}/{course['numExterne']}")
                    self.logger.debug("Main    : create and start thread %d.", c_index)
                    x = threading.Thread(target=self.__scrap_participants, args=(day,course,sub,participants[specialite]))
                    threads.append(x)
                    x.start()
                else:

                    subdf_ptcp=self.__scrap_participants(day,course,sub)
                    if subdf_ptcp:
                        participants[specialite].append(subdf_ptcp)
                    self.logger.info(f"END REUNION {course['numReunion']}/{course['numExterne']}")
                
            for index, thread in enumerate(threads):
                self.logger.debug("Main    : before joining thread %d.", index)
                thread.join()
                self.logger.debug("Main    : thread %d done", index)

        for spec in participants:
            if len(participants[spec])>0:
                df_participants=pd.concat(participants[spec])
                self._save( df_participants,path.join(self._directory,self.get_filename() % spec.lower(),self._usefolder),self.get_save_mode())

        self.logger.info(f"End scrapping day:{day}")
    def __scrap_participants(self,day,course,sub,result=False):
        try:
            subdf_ptcp=self._get_participants(course['numReunion'],course['numExterne'],day)
            if 'dernierRapportDirect' in subdf_ptcp:
                subdf_ptcp = subdf_ptcp[subdf_ptcp['dernierRapportDirect'].notna()]
            subdf_ptcp['date']=get_date_from_pmu(day)
            subdf_ptcp['reunion']=course['numReunion']
            subdf_ptcp['course']=course['numExterne']
            subdf_ptcp['hippo_code']=sub['hippodrome']['code']
            subdf_ptcp['hippo_nom']=sub['hippodrome']['libelleCourt']
            subdf_ptcp['distance']= course['distance']
            subdf_ptcp['distanceUnit']= course['distanceUnit']
            
            # TODO CHECK
            # if 'gainsParticipant' in subdf_ptcp and not 'gainsCarriere' in subdf['gainsParticipant']:
            #     subdf_ptcp=subdf_ptcp.assign(gain_carriere=[0])
            subdf_ptcp=subdf_ptcp.assign(gain_carriere=[value['gainsCarriere'] for value in subdf_ptcp['gainsParticipant']])
            subdf_ptcp=subdf_ptcp.assign(gain_victoires=[value['gainsVictoires'] for value in subdf_ptcp['gainsParticipant']])
            subdf_ptcp=subdf_ptcp.assign(gain_places=[value['gainsPlace'] for value in subdf_ptcp['gainsParticipant']])
            subdf_ptcp=subdf_ptcp.assign(gain_annee_en_cours=[value['gainsAnneeEnCours'] for value in subdf_ptcp['gainsParticipant']])
            if 'gainsParticipant' in subdf_ptcp:
                subdf_ptcp=subdf_ptcp.assign(gain_annee_precedente=[value['gainsAnneePrecedente'] for value in subdf_ptcp['gainsParticipant']])
            else:
                subdf_ptcp['gainsParticipant']=0

            if 'dernierRapportDirect' in subdf_ptcp:
                subdf_ptcp=subdf_ptcp.assign(rapport=[value['rapport'] for value in subdf_ptcp['dernierRapportDirect']])
            else:
                subdf_ptcp['rapport']=0

            if not 'placeCorde' in subdf_ptcp:
                subdf_ptcp['placeCorde']=0

            if not 'handicapValeur' in subdf_ptcp:
                subdf_ptcp['handicapValeur']=0

            if not 'handicapPoids' in subdf_ptcp:
                subdf_ptcp['handicapPoids']=0

            if not 'deferre' in subdf_ptcp:
                subdf_ptcp['deferre']=0

            if not 'handicapDistance'   in subdf_ptcp:
                subdf_ptcp['handicapDistance']=0    

            col_ex=subdf_ptcp.columns.tolist()
            col_to=['date','reunion','course','hippo_code','hippo_nom', 'nom','numPmu','rapport','age','sexe','race','statut','oeilleres','deferre','indicateurInedit','musique','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','ordreArrivee','distance','handicapDistance','gain_carriere'	,'gain_victoires'	,'gain_places'	,'gain_annee_en_cours',	'gain_annee_precedente','placeCorde','handicapValeur','handicapPoids']
            for col in filter(lambda x: x not in col_ex ,col_to) :
                self.logger.warning(f"{col} does not exist in dataframe")
            
            if not 'ordreArrivee' in subdf_ptcp:
                subdf_ptcp['ordreArrivee']=0
            subdf_ptcp=subdf_ptcp[col_to]
            if isinstance(result,list):
                result.append(subdf_ptcp)
                self.logger.info(f"END REUNION {course['numReunion']}/{course['numExterne']}")
            else:
                return subdf_ptcp
        except Exception as ex:
            self.logger.error(ex)
            return False

class ToPredictScrapper(HistoryScrapper):
    def __init__(self,use_proxy,use_threading,test,**kwargs):
        super().__init__(use_proxy,use_threading,test,**kwargs)
        self._filename="topredict_%s.csv"
        self._directory="input"
    def get_default_start_date(self):
        return get_today()
    # def get_save_mode(self):
    #     return 'a'