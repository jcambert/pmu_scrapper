import logging
import time
import sys,getopt
from sympy import true


from scrapper import HistoryScrapper,get_pmu_date,get_today,get_tommorow

if __name__=="__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    specialites=None
    use_proxy=False
    start_time = time.time()

    
    args=dict(arg.split('=') for arg in sys.argv[1:])
    if not 'start' in args:
        args['start']=get_pmu_date(get_today())


    scrapper=HistoryScrapper(use_proxy=use_proxy,use_threading=True,test=False,**args)
    scrapper.start(specialites=specialites,**args)

    logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
