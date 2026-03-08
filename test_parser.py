
import io
import pandas as pd
import numpy as np
import re

def smart_parse_xrd(file_content):
    # Convert to string if bytes
    if isinstance(file_content, bytes):
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            text = file_content.decode('latin-1')
    else:
        text = file_content

    lines = text.splitlines()
    
    # Heuristic 1: Look for common header markers
    data_start_idx = -1
    for i, line in enumerate(lines):
        clean_line = line.strip().lower()
        # Look for the legend line just before data
        if '<2theta>' in clean_line and '<' in clean_line and 'i' in clean_line:
            data_start_idx = i + 1
            break
        if '2theta' in clean_line and 'intensity' in clean_line:
            data_start_idx = i + 1
            break
            
    # Heuristic 2: If no header found, find the first line that is purely numeric with 2+ columns
    if data_start_idx == -1:
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    [float(p.replace(',', '')) for p in parts[:2]]
                    # Check if the next line is also numeric to be sure
                    if i + 1 < len(lines):
                        next_parts = lines[i+1].strip().split()
                        if len(next_parts) >= 2:
                            [float(p.replace(',', '')) for p in next_parts[:2]]
                            data_start_idx = i
                            break
                except ValueError:
                    continue

    if data_start_idx == -1:
        raise ValueError("Could not find start of XRD data in file.")

    # Read from data_start_idx onwards
    data_lines = lines[data_start_idx:]
    # Join and read with pandas, handling any whitespace separator
    data_str = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(data_str), sep=r'\s+', header=None, engine='python')
    
    # Cleanup: take first two columns, drop NaNs, convert to float
    df = df.iloc[:, :2]
    df.columns = ["two_theta", "intensity"]
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    return df

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
    df = smart_parse_xrd(sample_data)
    print("Successfully parsed!")
    print(df.head(10))
    print(df.tail(5))
    print(f"Total points: {len(df)}")
except Exception as e:
    print(f"Failed: {e}")
