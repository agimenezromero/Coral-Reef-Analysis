import numpy as np
import geopandas as gpd
import os

from osgeo import gdal
from shapely.strtree import STRtree

gdal.SetConfigOption("OGR_GEOJSON_MAX_OBJ_SIZE", "2000MB")


def benthic_preprocessing(
    province, input_folder, output_folder, class_save="Coral/Algae"
):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Read data
    data = gpd.read_file("%s/%s/Benthic-Map/benthic.gpkg" % (input_folder, province))

    gdf = data[data["class"] == class_save]

    N_gdf = len(gdf)

    # Compute tree
    tree = STRtree(gdf["geometry"])

    labels = np.arange(0, N_gdf, 1)

    new_label = N_gdf

    # Create clusters
    for i in range(len(gdf)):
        intersections = tree.query(gdf["geometry"].iloc[i])

        idx_intersect = []

        if len(intersections) > 1:
            for item in intersections:
                if item == i:
                    pass

                elif gdf["geometry"].iloc[i].intersects(gdf["geometry"].iloc[item]):
                    # Index that intersect with polygon i
                    idx_intersect.append(item)

        # Labels of the clusters that intersect cluster i
        labels_idx_intersect = labels[idx_intersect]

        # All clusters with same label get new label
        for old_label in np.unique(labels_idx_intersect):
            labels[labels == old_label] = new_label

        # Cluster i gets new label
        labels[i] = new_label

        # Update new label
        new_label += 1

    gdf_new = gdf.dissolve(by=labels, aggfunc="sum")

    # Rename class column to class name (because of dissolve)
    gdf_new["class"] = class_save

    # Compute area and perimeter
    gdf_new["area (m2)"] = gdf_new.to_crs(epsg="6933").area
    gdf_new["perimeter (m)"] = gdf_new.to_crs(epsg="6933").boundary.length

    # Remove small polygons which are not well classified
    gdf_final = gdf_new[gdf_new["area (m2)"] > 10**3].to_crs(epsg="6933")

    # Save data
    gdf_final.to_crs(epsg="4326").to_file(
        "%s/%s.geojson" % (output_folder, province), driver="GeoJSON"
    )


def reef_extent_preprocessing(province, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Read data
    gdf = gpd.read_file("%s/%s/Reef-Extent/reefextent.gpkg" % (input_folder, province))

    N_gdf = len(gdf)

    # Compute tree
    tree = STRtree(gdf["geometry"])

    labels = np.arange(0, N_gdf, 1)

    new_label = N_gdf

    # Create clusters
    for i in range(len(gdf)):
        intersections = tree.query(gdf["geometry"].iloc[i])

        idx_intersect = []

        if len(intersections) > 1:
            for item in intersections:
                if item == i:
                    pass

                else:
                    try:
                        bool_intersect = (
                            gdf["geometry"]
                            .iloc[i]
                            .intersects(gdf["geometry"].iloc[item])
                        )

                    except Exception:
                        bool_intersect = (
                            gdf["geometry"]
                            .iloc[i]
                            .buffer(0)
                            .intersects(gdf["geometry"].iloc[item].buffer(0))
                        )

                    if bool_intersect:
                        # Index that intersect with polygon i
                        idx_intersect.append(item)
        # Labels of the clusters that intersect cluster i
        labels_idx_intersect = labels[idx_intersect]

        # All clusters with same label get new label
        for old_label in np.unique(labels_idx_intersect):
            labels[labels == old_label] = new_label

        # Cluster i gets new label
        labels[i] = new_label

        # Update new label
        new_label += 1

    gdf_new = gdf.dissolve(by=labels, aggfunc="sum")

    # Rename class column to class name (because of dissolve)
    gdf_new["class"] = "Reef"

    # Compute area and perimeter
    gdf_new["area (m2)"] = gdf_new.to_crs(epsg="6933").area
    gdf_new["perimeter (m)"] = gdf_new.to_crs(epsg="6933").boundary.length

    # Remove small polygons which are not well classified
    gdf_final = gdf_new[gdf_new["area (m2)"] > 10**3].to_crs(epsg="6933")

    # Save data
    gdf_final.to_crs(epsg="4326").to_file(
        "%s/%s.geojson" % (output_folder, province), driver="GeoJSON"
    )
