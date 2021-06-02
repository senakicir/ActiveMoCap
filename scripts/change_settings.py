"""
    This little script allows you to change the ApiServerPort of your settings.json file for AirSim.
"""

import json
import sys

#specify the path of your AirSim settings.json file here
settings_path = "/home/kicirogl/Documents/AirSim/settings.json"

def main():
    port_name=sys.argv[1]
    with open(settings_path, 'r+') as f:
        data = json.load(f)
        data["ApiServerPort"] = int(port_name) 
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()     # remove remaining part

if __name__ == "__main__":
    main()