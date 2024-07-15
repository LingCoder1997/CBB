# -*- coding: utf-8 -*-
"""
-------------------------------------------------
@ Author ：L WANG
@ Email: lingxuanwang123@163.com
@ Date ： 2024/2/27
@ Description: The code below is done by myself
-------------------------------------------------
"""

import time

def timer(func):
     def wrapper(*args, **kwargs):
         start_time = time.time()
         result = func(*args, **kwargs)
         end_time = time.time()
         print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
         return result
     return wrapper

def log_results(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        with open("results.log", "a") as log_file:
            log_file.write(f"{func.__name__} - Result: {result}\n")
        return result
    return wrapper

import warnings

def deprecated(func):
     def wrapper(*args, **kwargs):
         warnings.warn(f"{func.__name__} is deprecated and will be removed in future versions.", DeprecationWarning)
         return func(*args, **kwargs)
     return wrapper