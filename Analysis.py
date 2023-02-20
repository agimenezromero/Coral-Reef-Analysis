import numpy as np
import pandas as pd
import geopandas as gpd

import pygeos
from rtree import index
from shapely.strtree import STRtree

from osgeo import gdal

gdal.SetConfigOption('OGR_GEOJSON_MAX_OBJ_SIZE', '2000MB')

import powerlaw

import os
import time

import matplotlib.pyplot as plt

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

def reef_size_distribution_txt(input_folder, output_folder_pdf, output_folder_ccdf):

    filenames = os.listdir(input_folder)

    for filename in filenames:

        t0 = time.time()

        name = filename[18:-8]

        print("Reading %s..." % name, end="")

        gdf = gpd.read_file("%s/%s" % (input_folder, filename))

        print("computing powerlaw...", end="")

        cluster_sizes = gdf.to_crs(epsg="3035").area #gdf["area (m2)"].values

        results = powerlaw.Fit(cluster_sizes, xmin=np.amin(cluster_sizes))

        bins_pdf, values_pdf = results.pdf()
        bins_ccdf, values_ccdf = results.ccdf()

        center_bins_pdf = bins_pdf[0:-1] + (bins_pdf - np.roll(bins_pdf, shift=1))[1:] / 2

        np.savetxt("%s/%s.txt" % (output_folder_pdf, name), np.transpose([center_bins_pdf, values_pdf]))
        np.savetxt("%s/%s.txt" % (output_folder_ccdf, name), np.transpose([bins_ccdf, values_ccdf]))

        print("done in %.2f m" % ((time.time()-t0)/60.0))

def compute_size_distribution(input_folder, output_filename, plot=False, figname="Figures/"):
    
    filenames = os.listdir(input_folder)

    optimal_alphas = []
    optimal_xmins = []
    optimal_Ds = []

    names = []

    if not os.path.exists(figname + "PDF/"): os.mkdir(figname + "PDF/")
    if not os.path.exists(figname + "CCDF/"): os.mkdir(figname + "CCDF/")
    if not os.path.exists(figname + "PDF_whole_range/"): os.mkdir(figname + "PDF_whole_range/")
    if not os.path.exists(figname + "CCDF_whole_range/"): os.mkdir(figname + "CCDF_whole_range/")

    for filename in filenames:

        name = filename[18:-8]

        print("Reading %s..." % name, end="")

        gdf = gpd.read_file("%s/%s" % (input_folder, filename))

        print("computing powerlaw...", end="")

        cluster_sizes = gdf.to_crs(epsg="3035").area #gdf["area (m2)"].values

        results = powerlaw.Fit(cluster_sizes, xmin=np.amin(cluster_sizes))

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        results.plot_pdf(ax=ax1, ls='', marker='o', color='k', markerfacecolor='w', markersize=15, label="")
        results.plot_ccdf(ax=ax2, ls='', marker='o', color='k', markerfacecolor='w', markersize=15, label="")

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

        x = np.array([10**i for i in range(len(str(int(np.amin(cluster_sizes)))), 
                                  len(str(int(np.amax(cluster_sizes))))+1)])

        if plot == True:
        
            #Plot PDF whole range
            ax1.plot(x, power_law_norm(x, alpha, xmin), lw=3, color='r', 
                        label=r"$y\sim x^{-\alpha}, \alpha=%.2f$" % alpha)

            ax1.tick_params(axis='both', which='major', labelsize=16)

            ax1.set_xlabel(r"$Km^2$", fontsize=30)
            ax1.set_ylabel("PDF", fontsize=30)

            ax1.set_yscale("log")
            ax1.set_xscale("log")

            ax1.legend(loc="upper right", fontsize=20);

            fig1.savefig(figname + "PDF_whole_range/%s.png" % name, bbox_inches='tight', dpi=300)

            #Plot CCDF whole range
            ax2.plot(x, power_law_CCDF(x, alpha, xmin), lw=3, color='r', 
                        label=r"$y\sim x^{1-\alpha}, \alpha=%.2f$" % alpha)

            ax2.tick_params(axis='both', which='major', labelsize=16)

            ax2.set_xlabel(r"$Km^2$", fontsize=30)
            ax2.set_ylabel("CCDF", fontsize=30)

            ax2.set_yscale("log")
            ax2.set_xscale("log")

            ax2.legend(loc="upper right", fontsize=20);

            fig2.savefig(figname + "CCDF_whole_range/%s.png" % name, bbox_inches='tight', dpi=300)

            #Plot from xmin
            x = np.array([10**i for i in range(len(str(int(xmin)))-1, 
                                    len(str(int(np.amax(cluster_sizes))))+1)])

            #PDF
            fig3, ax3 = plt.subplots(figsize=(8, 6))

            results.plot_pdf(ax=ax3, ls='', marker='o', color='k', markerfacecolor='w', markersize=15, label="")

            ax3.plot(x, power_law_norm(x, alpha, xmin), lw=3, color='r', 
                        label=r"$y\sim x^{-\alpha}, \alpha=%.2f$" % alpha)

            ax3.tick_params(axis='both', which='major', labelsize=16)

            ax3.set_xlabel(r"$Km^2$", fontsize=30)
            ax3.set_ylabel("PDF", fontsize=30)

            ax3.set_yscale("log")
            ax3.set_xscale("log")

            ax3.legend(loc="upper right", fontsize=20);

            fig3.savefig(figname + "PDF/%s.png" % name, bbox_inches='tight', dpi=300)

            #CCDF
            fig4, ax4 = plt.subplots(figsize=(8, 6))

            results.plot_ccdf(ax=ax4, ls='', marker='o', color='k', markerfacecolor='w', markersize=15, label="")
            
            ax4.plot(x, power_law_CCDF(x, alpha, xmin), lw=3, color='r', 
                        label=r"$y\sim x^{1-\alpha}, \alpha=%.2f$" % alpha)

            ax4.tick_params(axis='both', which='major', labelsize=16)

            ax4.set_xlabel(r"$Km^2$", fontsize=30)
            ax4.set_ylabel("CCDF", fontsize=30)

            ax4.set_yscale("log")
            ax4.set_xscale("log")

            ax4.legend(loc="upper right", fontsize=20);

            fig4.savefig(figname + "CCDF/%s.png" % name, bbox_inches='tight', dpi=300)
            
            plt.close("all")


        print("Done!")

    df = pd.DataFrame({'Name':names, "Alpha":optimal_alphas, "Xmin":optimal_xmins, "D":optimal_Ds})

    df.to_csv("%s.csv" % output_filename)

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

def box_counting_dimension(input_folder, filename, epsilons, output_folder):

    if not os.path.exists(output_folder): os.mkdir(output_folder)

    #Load coral data
    print("Reading file...")
    gdf = gpd.read_file("%s/%s" % (input_folder, filename))

    #Tree for corals geometry
    tree = STRtree(gdf["geometry"])

    #Get maximum and minimum latlon values to create a mesh
    max_latlon = gdf["geometry"].bounds

    max_lat = np.amax(max_latlon["maxy"])
    min_lat = np.amin(max_latlon["miny"])
    max_lon = np.amax(max_latlon["maxx"])
    min_lon = np.amin(max_latlon["minx"])

    t0 = time.time()

    N_boxes = []

    print("ϵ\tN")

    for ϵ in epsilons:
        
        #print("Computing ϵ=%s" % ϵ)

        #Create mesh with length ϵ
        x_c = np.arange(min_lon-ϵ, max_lon, ϵ)
        y_c = np.arange(min_lat-ϵ, max_lat, ϵ)

        x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ/2
        y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ/2

        yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij") 

        mesh = create_square_grid_gdf(xx_v, yy_v)

        #Compute number of overlapping boxes
        idxs = 0

        for i in range(len(mesh)):

            #Indices of the polygons that intersect the box
            query = tree.query(mesh["geometry"][i])

            #If some polygon intersect the box, add 1
            if len(query) > 0:

                idxs += 1

        N_boxes.append(idxs)

        print(ε, idxs)

    np.savetxt("%s/%s.txt" % (output_folder, filename[0:-8]), np.transpose([epsilons, N_boxes]), header="ϵ N")

    print("\nFinished in ", time.time()-t0, "s")

def Taylor_law(filename, ϵ):

    mean_density = []
    var_density = []

    #Load coral data
    print("Reading file...")
    gdf = gpd.read_file("%s" % filename)

    #Tree for corals geometry
    tree = STRtree(gdf["geometry"])

    #Get maximum and minimum latlon values to create a mesh
    max_latlon = gdf["geometry"].bounds

    max_lat = np.amax(max_latlon["maxy"])
    min_lat = np.amin(max_latlon["miny"])
    max_lon = np.amax(max_latlon["maxx"])
    min_lon = np.amin(max_latlon["minx"])

    print("Creating mesh...")

    #Create mesh with length ϵ
    x_c = np.arange(min_lon-ϵ, max_lon, ϵ)
    y_c = np.arange(min_lat-ϵ, max_lat, ϵ)

    x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ/2
    y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ/2

    yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij") 

    mesh = create_square_grid_gdf(xx_v, yy_v)

    #Comput area of corals in each grid
    print("Computing cell areas")

    tot_its = len(mesh)

    #Iterate through the cells of the mesh
    for i in range(tot_its):

        completed = (i / tot_its) * 100

        if np.round(completed, 0) % 5 == 0:

            print("%.0f%% completed" % completed)

        #Obtain the inidces of the polygons that intersect the cell
        query = tree.query(mesh["geometry"][i])

        coral_areas = gdf["geometry"].iloc[query].intersection(mesh["geometry"][i]).to_crs(epsg="3035").area.values #In m²

        #area_cell = mesh["geometry"].to_crs(epsg="3035").iloc[i].area #In m²

        densities = coral_areas #/ area_cell

        if len(densities) > 0:

            mean_density.append(np.mean(densities))
            var_density.append(np.var(densities))

    return np.array(mean_density), np.array(var_density)

def Area_Perimeter_scaling(input_folder, outfilename):

    filenames = os.listdir(input_folder)

    df_f = pd.DataFrame({"Province":[], "Area":[], "Perimeter":[]})

    for filename in filenames:

        t0 = time.time()
        
        print("Computing %s..." % filename[18:-8], end="")

        gdf = gpd.read_file("Data/Coral/%s" % filename)

        gdf = gdf.to_crs(epsg="3035")

        df = pd.DataFrame({"Province":[filename[18:-8] for i in range(len(gdf))], "Area":gdf.area.values, "Perimeter":gdf.boundary.length.values})

        df_f = pd.concat((df_f, df), ignore_index=True)

        print("finished in %.2f m" % ((time.time() - t0) / 60.0))

    df_f.to_parquet("%s.parquet" % outfilename, engine="fastparquet")

def multifractal_dimension(qs, epsilons, filename, input_folder, output_folder):

    for ε in epsilons:

        t0 = time.time()

        print("Computing ϵ=%.2e..." % ϵ, end="")

        t0 = time.time()

        gdf = gpd.read_file("%s/%s" % (input_folder, filename))

        total_area = gdf["geometry"].to_crs(epsg="3035").area.sum()

        #Tree for corals geometry
        tree = STRtree(gdf["geometry"])

        t0 = time.time()

        #Get maximum and minimum latlon values to create a mesh
        max_latlon = gdf["geometry"].bounds

        max_lat = np.amax(max_latlon["maxy"])
        min_lat = np.amin(max_latlon["miny"])
        max_lon = np.amax(max_latlon["maxx"])
        min_lon = np.amin(max_latlon["minx"])

        #Create mesh with length ϵ
        x_c = np.arange(min_lon-ϵ, max_lon, ϵ)
        y_c = np.arange(min_lat-ϵ, max_lat, ϵ)

        x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ/2
        y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ/2

        yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij") 

        mesh = create_square_grid_gdf(xx_v, yy_v)

        #Mass measure
        p_boxes = []

        for i in range(len(mesh)):

            #Indices of the polygons that intersect the box
            query = tree.query(mesh["geometry"][i])

            if len(query) > 0:

                p_box = np.sum(gdf["geometry"].iloc[query].intersection(mesh["geometry"][i]).to_crs(epsg="3035").area.values) / total_area

                if p_box > 0:

                    p_boxes.append(p_box)

        D_q = []
        P_q = []

        for q in qs:
            
            #Compute number of overlapping boxes
            idxs = np.sum([np.power(p_box, q) for p_box in p_boxes]) 

            D = (1/(q-1)) * (np.log(idxs) / np.log(ε))

            D_q.append(D)
            P_q.append(idxs)

        print("done in %.2f m" % ((time.time() - t0) / 60.0))

        np.savetxt("%s/%s_epsilon_%s.txt" % (output_folder, filename[0:-8], ε), np.transpose([qs, P_q, D_q]), header="q\tP_q\tD_q")

def compute_province_boundaries(input_folder, outfilename):

    filenames = os.listdir(input_folder)

    min_lons = []
    max_lons = []
    min_lats = []
    max_lats = []

    provinces = []

    for filename in filenames:

        t0 = time.time()

        print("Reading %s..." % filename[:-8], end="")

        gdf = gpd.read_file("/data/bio/corals/Clusterized_data/Coral/%s" % filename)

        max_latlon = gdf["geometry"].bounds

        max_lat = np.amax(max_latlon["maxy"])
        min_lat = np.amin(max_latlon["miny"])
        max_lon = np.amax(max_latlon["maxx"])
        min_lon = np.amin(max_latlon["minx"])

        min_lons.append(min_lon)
        max_lons.append(max_lon)
        min_lats.append(min_lat)
        max_lats.append(max_lat)

        provinces.append(filename[:-8])

        print("done in %.2f m" % ((time.time()-t0) / 60.0))

    df = pd.DataFrame({'Province':provinces, 'Min lon':min_lons, 'Max lon':max_lons, 'Min lat':min_lats, 'Max lat':max_lats})

    df.to_csv("%s.csv" % outfilename)