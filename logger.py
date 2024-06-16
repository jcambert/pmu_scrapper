import logging
import sys


def configure_logging(name,**args):
    level_name = args['log_level'] if 'log_level' in args else logging.ERROR
    format = "[%(asctime)s: %(levelname)s] %(message)s"
    logging.basicConfig(format=format, datefmt="%H:%M:%S",handlers=[])
    formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s',"%H:%M:%S")
    log = logging.getLogger(name)   
    log.setLevel(level_name)   

    handler = logging.StreamHandler(sys.stdout)                             
    handler.setLevel(level_name)                                        
    handler.setFormatter(formatter)                   
    
    log.addHandler(handler)  
    return log