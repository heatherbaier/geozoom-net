import landsat_prep as lp
import geopandas as gpd
import os

from joblib import Parallel, delayed
from functools import partial
import multiprocessing



def main(shapeID):
    
    print(shapeID)

    try:

        # Download the imagery of the municipality bounding box
        lp.download_boundary_imagery(gb_path, shapeID, year, ic, month, iso, v = False)
        lp.save_boundary_pngs(shapeID, iso, v = False)
        
    except Exception as e:
        print(e)
    

if __name__ == "__main__":
    
    # Set up variables
    gb_path = "./data/MEX/ipumns_bbox.shp"
    year = "2010"
    month = "all"
    iso = "MEX"
    ic = "LANDSAT/LT05/C01/T1"

    # Read in the shapefile and grab the unique shapeID's
    shp = gpd.read_file(gb_path)
    ids = shp["shapeID"].unique()
    a = 1
    
    print("Number of ID's to download: ", len(ids))
        
    num_cores = 32
    output = Parallel(n_jobs = num_cores)(delayed(main)(shapeID = i) for i in ids)
    
