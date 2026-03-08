
import io
import pandas as pd
import numpy as np
import re
from xrd_engine import parse_xrd_file

# Test with the sample provided by user
sample_data = """/////////////////////////////////////////////////////////////////////////////////
/// Profile Data Ascii Dump (XRD)                                             ///
/////////////////////////////////////////////////////////////////////////////////

  Group     : Standard
  Data      : Sample-1
  File Name : Sample-1.RAW

# Profile Datafile
          comment             = RUET 
          date & time         = 02-16-26 14:59:05 

# Measurement Condition
    X-ray tube
          target              = Cu 
          voltage             = 40.0 (kV)
          current             = 30.0 (mA)
    Slits
          divergence slit     = 1.00000 (deg) 
          scatter slit        = 1.00000 (deg) 
          receiving slit      = 0.30000 (mm)  
    Scanning
          drive axis          = Theta-2Theta 
          scan range          =   10.000 -   80.000 
          scan mode           = Continuous Scan 
          scan speed          =   2.0000 (deg/min) 
          sampling pitch      =   0.0200 (deg) 
          preset time         =   0.60 (sec) 

# Data      [ Total No. = 3501 ]
  <2Theta>   <   I   >
    10.0000       356
    10.0200       406
    10.0400       342
    10.0600       360
    10.0800       352
    10.1000       406
    10.1200       366
    10.1400       362
    10.1600       376
    10.1800       366
    10.2000       356
    10.2200       348
    10.2400       378
    10.2600       384
"""

try:
    df = parse_xrd_file(sample_data)
    print("Successfully parsed!")
    print(df.head(5))
    print(f"Total points: {len(df)}")
except Exception as e:
    import traceback
    traceback.print_exc()
