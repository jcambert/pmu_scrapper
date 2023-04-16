import logging
import time
import sys,getopt
from sympy import true

from logger import configure_logging
from scrapper import ToPredictScrapper,get_pmu_date,get_today

if __name__=="__main__":

    log=configure_logging("Scrapper")

    specialites=None
    use_proxy=False
    start_time = time.time()

    
    args=dict(arg.split('=') for arg in sys.argv[1:])
    if not 'start' in args:
        args['start']=get_pmu_date(get_today())
        
    scrapper=ToPredictScrapper(use_proxy=use_proxy,use_threading=True,test=False,logger=log,**args)
    scrapper.start(specialites=specialites,**args)

    log.info(f"it's took {(time.time() - start_time)} seconds")
    log.info("Bye...")
