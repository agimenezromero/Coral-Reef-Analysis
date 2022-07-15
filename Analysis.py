#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import geopandas as gpd

import pygeos
from rtree import index
from shapely.strtree import STRtree

import powerlaw

def power_law(x, p, A):
    
    return A * np.power(x, p)

def power_law_norm(x, α, xmin):
    
    return np.power(x, -α) * (α-1) * xmin**(α-1)

def power_law_CCDF(x, α, xmin):
    
    return np.power(x/xmin, 1-α)

def truncated_power_law_norm(x, α, λ, xmin):
    
    from mpmath import gammainc
       
    C = ( λ**(1-α) / float(gammainc(1-α,λ*xmin)))
    
    return np.power(x, -α) * np.exp(-λ * x) * C


# In[ ]:


def compute_size_distribution(input_folder, output_filename, plot=False, figname="Figures/size_dist"):
    
    filenames = os.listdir(input_folder)

    optimal_alphas = []
    optimal_xmins = []
    optimal_Ds = []

    names = []

    for filename in filenames:

        name = filename[18:-8]

        print("Reading %s..." % name, end="")

        gdf = gpd.read_file("%s/%s" % (input_folder, filename))

        print("computing powerlaw...", end="")

        cluster_sizes = gdf["area (m2)"].values

        results = powerlaw.Fit(cluster_sizes, xmin=np.amin(cluster_sizes))

        plt.figure(figsize=(8, 6))

        fig = results.plot_pdf(ls='', marker='o', color='k', markerfacecolor='w', markersize=15, 
                               label="")

        #Get approx xmin:
        Ds = []
        xmins = np.array([10**i for i in range(7)])

        for x_min in xmins:

            results = powerlaw.Fit(cluster_sizes, xmin=x_min)

            if str(results.power_law.D) != "nan":

                Ds.append(results.power_law.D)

            else:

                Ds.append(100000)

        x_min = xmins[np.argmin(Ds)]

        results = powerlaw.Fit(cluster_sizes, xmin=x_min)

        alpha = results.power_law.alpha
        sigma = results.power_law.sigma
        xmin = results.power_law.xmin

        optimal_alphas.append(alpha)
        optimal_xmins.append(xmin)
        optimal_Ds.append(np.amin(Ds))
        names.append(name)

        x = [10**i for i in range(len(str(int(np.amin(cluster_sizes)))), 
                                  len(str(int(np.amax(cluster_sizes))))+1)]

        if plot == True:
        
            plt.plot(x, power_law_norm(x, alpha, xmin), lw=3, color='r', 
                     label=r"$y\sim x^{\alpha}, \alpha=%.2f$" % alpha)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.xlabel(r"$Km^2$", fontsize=30)
            plt.ylabel("PDF", fontsize=30)

            plt.yscale("log")
            plt.xscale("log")

            plt.legend(loc="upper right", fontsize=20);

            plt.savefig(figname + "_%s.png" % name, bbox_inches='tight', dpi=300)

            plt.close()

        print("Done!")

    df = pd.DataFrame({'Name':names, "Alpha":optimal_alphas, "Xmin":optimal_xmins, "D":optimal_Ds})

    df.to_csv(output_filename)

def create_square_grid_gdf(lons, lats):

    points = pygeos.creation.points(lons.ravel(), lats.ravel())

    gdf_temp = gpd.GeoDataFrame(geometry=points)

    coords = pygeos.get_coordinates(pygeos.from_shapely(gdf_temp.geometry))

    nrow = lons.shape[0]-1
    ncol = lons.shape[1]-1
    n = nrow * ncol
    nvertex = (nrow + 1) * (ncol + 1) 
    assert len(coords) == nvertex

    # Make sure the coordinates are ordered into grid form:
    x = coords[:, 0]
    y = coords[:, 1]
    order = np.lexsort((x, y))
    x = x[order].reshape((nrow + 1, ncol + 1))
    y = y[order].reshape((nrow + 1, ncol + 1))

    # Setup the indexers
    left = lower = slice(None, -1)
    upper = right = slice(1, None)
    corners = [
        [lower, left],
        [lower, right],
        [upper, right],
        [upper, left],
    ]

    # Allocate output array
    xy = np.empty((n, 4, 2))

    # Set the vertices
    for i, (rows, cols) in enumerate(corners):
        xy[:, i,  0] = x[rows, cols].ravel()
        xy[:, i, 1] = y[rows, cols].ravel()

    # Create geodataframe and plot result
    mesh_geometry = pygeos.creation.polygons(xy)
    mesh_gdf = gpd.GeoDataFrame(geometry=mesh_geometry, crs=4326)
    
    return mesh_gdf

def box_counting_dimension(filename, epsilons):

    #Load coral data
    print("Reading file...")
    gdf = gpd.read_file("Coral_clusters/Coral_Rock/%s" % filename)

    #Compute centroids (again...)
    centroids = gdf["geometry"].to_crs(epsg="3035").centroid.to_crs(epsg="4326")

    cntr_lon = [item.x for item in centroids]
    cntr_lat = [item.y for item in centroids]

    gdf["cntr_lon"] = cntr_lon
    gdf["cntr_lat"] = cntr_lat

    #Tree for corals geometry
    tree = STRtree(gdf["geometry"])

    #Create index for Tree-labeled objects

    #Get maximum and minimum latlon values to create a mesh
    max_latlon = gdf["geometry"].bounds

    max_lat = np.amax(max_latlon["maxy"])
    min_lat = np.amin(max_latlon["miny"])
    max_lon = np.amax(max_latlon["maxx"])
    min_lon = np.amin(max_latlon["minx"])

    t0 = time.time()

    N_boxes = []

    for ϵ in epsilons:
        
        print("Computing ϵ=%s" % ϵ)

        #Create mesh with length ϵ
        x_c = np.arange(min_lon-ϵ, max_lon, ϵ)
        y_c = np.arange(min_lat-ϵ, max_lat, ϵ)

        x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ/2
        y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ/2

        yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij") 

        mesh = create_square_grid_gdf(xx_v, yy_v)

        #Compute number of overlapping boxes
        idxs = []

        for i in range(len(mesh)):

            query = tree.query(mesh["geometry"][i])

            for item in query:

                if item.intersects(mesh["geometry"][i]):

                    idxs.append(i)

                    break

        N_boxes.append(len(idxs))

    np.savetxt("Box_Counting_Dimensions/" + filename[18:-8] + ".txt", np.transpose([epsilons, N_boxes]),
               header="ϵ N")

    print("\nFinished in ", time.time()-t0, "s")

