from uitils import *

import psutil

@timer
def get_cpu_num():
  
  cpu_num=psutil.cpu_count()
  
  return cpu_num
