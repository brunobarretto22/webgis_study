import math
import geopandas as gpd

def calcular_utm(gdf):
    gdf = gdf.dissolve(by=None)
    centroid = gdf.centroid.iloc[0]
    zona = math.ceil((centroid.x+180)/6)
    if centroid.y > 0:
        epsg = zona + 32600
    else:
        epsg = zona + 32700
    return(epsg);
    # print(epsg);

# calcular_utm(gpd.read_file(r'D:\webgis_study\webgis_study\data\MT53492022.kml'))
# calcular_utm(gpd.read_file(r'D:\webgis_study\webgis_study\data\MT1361982018.kml'))

