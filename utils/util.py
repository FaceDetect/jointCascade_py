import os
import sys

def raw_input_t(*args):
    old_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        return raw_input(*args)
    finally:
        sys.stdout = old_stdout

def getTimeByStamp(beg, end, mode):
    t = end - beg
    if 'SEC' == mode.upper():
        return t*1.0
    elif 'MIN' == mode.upper():
        return t/60.0
    elif 'HOUR' == mode.upper():
        return t/3600.0
    return t

def getTime(beg, end, mode):
    t=0
    t += (end.day-beg.day)*24*60*60
    t += (end.hour-beg.hour)*60*60
    t += (end.minute-beg.minute)*60
    t += (end.second-beg.second)
    t += (end.microsecond-beg.microsecond)/1000000.0

    if 'MS' == mode.upper():
        return t*1000
    elif 'SEC' == mode.upper():
        return t*1.0
    elif 'MIN' == mode.upper():
        return t/60.0
    elif 'HOUR' == mode.upper():
        return t/3600.0
    
    return t*1.0
