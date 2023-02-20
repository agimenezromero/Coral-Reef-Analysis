import numpy as np
import geopandas as gpd
import os

from osgeo import gdal

gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '2000MB')

from shapely.strtree import STRtree

def harmonize_data(input_folder, output_folder):

    os.chdir(input_folder)

    filenames = os.listdir()

    for filename in filenames:

        if filename[-4:] == "gpkg":

            name = filename[0:-5]

        else:

            name = filename[0:-8]

        print("Processing %s..." % name, end="")

        gdf = gpd.read_file(filename)

        gdf_reprojected = gdf.to_crs(epsg=3395)

        gdf_reprojected["centroid"] = gdf_reprojected["geometry"].centroid

        gdf["area (m2)"] = gdf_reprojected.area

        gdf["centroid"] = gdf_reprojected["centroid"].to_crs(crs=4326)

        #gdf_measures = gdf[(gdf["class"] == "Rock") | (gdf["class"] == "Coral/Algae")]
        gdf_measures = gdf

        lons = np.array([item.x for item in gdf_measures["centroid"].values])
        lats = np.array([item.y for item in gdf_measures["centroid"].values])

        gdf_measures["cntr_lon"] = lons
        gdf_measures["cntr_lat"] = lats

        gdf_measures_final = gdf_measures[["class", "geometry", "area (m2)", "cntr_lon", "cntr_lat"]]

        gdf_measures_final.to_file(output_folder + "/%s.geojson" % name, driver='GeoJSON')

        print("done!")

def combine_polygons(name, input_folder, output_folder, class_save="Coral/Algae"):

    if not os.path.exists(output_folder): 
        
        os.mkdir(output_folder)

    #Read data
    data = gpd.read_file("%s/%s.geojson" % (input_folder, name), driver="GeoJSON")
    
    gdf = data[data["class"] == class_save]

    N_gdf = len(gdf)

    #Compute tree
    tree = STRtree(gdf["geometry"])

    #index_by_id = dict((id(pt), i) for i, pt in enumerate(geometries)) -> This was needed with an older version of STRtree

    labels = np.arange(0, N_gdf, 1)

    new_label = N_gdf

    #Create clusters
    for i in range(len(gdf)):

        intersections = tree.query(gdf["geometry"].iloc[i])

        idx_intersect = []

        if len(intersections) > 1:

            for item in intersections:

                if item == i: #gdf["geometry"][i] == item: -> This was needed with an older version of STRtree

                    pass

                elif gdf["geometry"].iloc[i].intersects(gdf["geometry"].iloc[item]): #gdf["geometry"][i].intersects(item): -> This was needed with an older version of STRtree

                    #Index that intersect with polygon i                      
                    idx_intersect.append(item) #idx_intersect.append(index_by_id[id(item)]) -> This was needed with an older version of STRtree

        #Labels of the clusters that intersect cluster i
        labels_idx_intersect = labels[idx_intersect]

        #All clusters with same label get new label
        for old_label in np.unique(labels_idx_intersect):

            labels[labels == old_label] = new_label

        #Cluster i gets new label
        labels[i] = new_label

        #Update new label
        new_label += 1

    combined_polygons = gdf.dissolve(by=labels, aggfunc="sum")

    combined_polygons.to_file("%s/combined_polygons_%s.geojson" % (output_folder, name), driver="GeoJSON")
        
def combine_polygons_all(input_folder, output_folder, class_save="Coral/Algae"):

    if not os.path.exists(output_folder): 
        
        os.mkdir(output_folder)
    
    names = os.listdir(input_folder)[-12:]

    for name in names:

        print("Computing %s..." % name)

        data = gpd.read_file("%s/%s" % (input_folder, name), driver="GeoJSON")
        
        gdf = data[data["class"] == class_save]

        N_gdf = len(gdf)

        tree = STRtree(gdf["geometry"])

        #index_by_id = dict((id(pt), i) for i, pt in enumerate(geometries)) -> This was needed with an older version of STRtree

        labels = np.arange(0, N_gdf, 1)

        new_label = N_gdf

        for i in range(len(gdf)):

            intersections = tree.query(gdf["geometry"].iloc[i])

            idx_intersect = []

            if len(intersections) > 1:

                for item in intersections:

                    if item == i: #gdf["geometry"][i] == item: -> This was needed with an older version of STRtree

                        pass

                    elif gdf["geometry"].iloc[i].intersects(gdf["geometry"].iloc[item]): #gdf["geometry"][i].intersects(item): -> This was needed with an older version of STRtree

                        #Index that intersect with polygon i                      
                        idx_intersect.append(item) #idx_intersect.append(index_by_id[id(item)]) -> This was needed with an older version of STRtree

            #Labels of the clusters that intersect cluster i
            labels_idx_intersect = labels[idx_intersect]

            #All clusters with same label get new label
            for old_label in np.unique(labels_idx_intersect):

                labels[labels == old_label] = new_label

            #Cluster i gets new label
            labels[i] = new_label

            #Update new label
            new_label += 1

        combined_polygons = gdf.dissolve(by=labels, aggfunc="sum")

        combined_polygons.to_file("%s/combined_polygons_%s" % (output_folder, name), driver="GeoJSON")

