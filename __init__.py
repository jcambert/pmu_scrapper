import logging
import time

from sympy import true


from scrapper import HistoryScrapper, ResultatScrapper, ToPredictScrapper

if __name__=="__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    specialites=None
    use_proxy=False
    start_time = time.time()

    # s=Scrapper(use_proxy=use_proxy,USE_THREADING=True,to_predict=True)
    # days=s.start("23072021",count=0)
    # days=scrapdays("01012017",end="31122017")
    # days=s.start("01012019",end="31122019")
    # logging.info(f"Scrapping from {days[0]} to {days[1]} by step {days[2]} ")

    # s=Scrapper(use_proxy=use_proxy,USE_THREADING=True,to_check_results=True)
    # s.start(predict_filename="predicted")



    #specialites=['PLAT']
    # scrapper=ResultatScrapper(use_proxy=use_proxy,use_threading=True,test=True)

    # scrapper=HistoryScrapper(use_proxy=use_proxy,use_threading=True,test=False)
    # scrapper.start(start="01012018",end="31122018", specialites= specialites)

    scrapper=ToPredictScrapper(use_proxy=use_proxy,use_threading=True,test=False)
    scrapper.start(specialites=specialites)

    logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
    print('Bye...')
