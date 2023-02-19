import logging
import time

from sympy import true


from scrapper import  ResultatScrapper,get_yesterday

if __name__=="__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    specialites=None
    use_proxy=False
    start_time = time.time()

    #specialites=['PLAT']
    scrapper=ResultatScrapper(use_proxy=use_proxy,use_threading=True,test=False)

    scrapper.start(specialites=specialites,start=get_yesterday())

    logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
