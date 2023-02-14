import logging
import time
import sys



if __name__=="__main__":
    start_time = time.time()
    format = "%(asctime)s: %(message)s"
    # logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    print("CECI EST UN TEST")

    # logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
    print(sys.exit(1))
