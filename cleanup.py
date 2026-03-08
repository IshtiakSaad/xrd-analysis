
import os
files = [
    "C:/Users/DELL/Desktop/xrd_app/xrd_engine.py",
    "C:/Users/DELL/Desktop/xrd_app/xrd_plots.py",
    "C:/Users/DELL/Desktop/xrd_app/main-script.py",
    "C:/Users/DELL/Desktop/xrd_app/peak-shift.py",
    "C:/Users/DELL/Desktop/xrd_app/test_fix.py",
    "C:/Users/DELL/Desktop/xrd_app/test_parser.py",
    "C:/Users/DELL/Desktop/xrd_app/test_theme.py"
]
for f in files:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")
    except Exception as e:
        print(f"Failed to delete {f}: {e}")
