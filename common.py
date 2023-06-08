import os
_base_dir= os.path.dirname(os.path.realpath(__file__))
PATHES={'base':_base_dir, 'model':os.path.join(_base_dir, 'models'),'input':os.path.join(_base_dir, 'input'),'output':os.path.join(_base_dir, 'output'),'history':os.path.join(_base_dir, 'history')}
DEFAULT_NROWS=200000

def execution_time_tostring(start,end):
    seconds=end-start
    if seconds>3600:
        return f"{seconds/3600:.2f} heures"
    if seconds>60:
        return f"{seconds/60:.2f} minutes"
    return f"{seconds:.2f} secondes"