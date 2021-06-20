import landsat_prep as lp
import geopandas as gpd
import os


# Set up variables
gb_path = "./data/MEX2/ipumns_bbox.shp"
year = "2010"
month = "1"
iso = "MEX2"
ic = "LANDSAT/LT05/C01/T1"


# Read in the shapefile and grab the unique shapeID's
shp = gpd.read_file(gb_path)
ids = shp["shapeID"].unique()[9:]
a = 1


# For each shapeID...
for i in ids:
    
    print(a, i, " out of ", len(ids))
    
    try:

        # Download the imagery of the municipality bounding box
        lp.download_boundary_imagery(gb_path, i, year, ic, month, iso, v = False)
        lp.save_boundary_pngs(i, iso, v = False)
        a += 1
        
    except:
        pass