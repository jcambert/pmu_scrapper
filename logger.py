import logging
import sys


def configure_logging(name):
    format = "[%(asctime)s: %(levelname)s] %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S",handlers=[])
    formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s',"%H:%M:%S")
    log = logging.getLogger(name)                                          
    handler = logging.StreamHandler(sys.stdout)                             
    handler.setLevel(logging.INFO)                                        
    handler.setFormatter(formatter)                   
    log.addHandler(handler)  
    return log