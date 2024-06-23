import logging
import time
import sys
from scrapper import  ResultatScrapper,get_yesterday,get_pmu_date
from logger import configure_logging

if __name__=="__main__":

    args=dict(arg.split('=') for arg in sys.argv[1:])
    log=configure_logging("Resultat",**args)
   

    specialites=None
    use_proxy=False
    start_time = time.time()

    args=dict(arg.split('=') for arg in sys.argv[1:])
    if not 'start' in args:
        args['start']=get_pmu_date(get_yesterday())

    #specialites=['PLAT']
    scrapper=ResultatScrapper(use_proxy=use_proxy,use_threading=True,test=False,logger=log,**args)

    scrapper.start(specialites=specialites,**args)

    log.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
