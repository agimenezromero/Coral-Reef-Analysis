import numpy as np
import pandas as pd
import geopandas as gpd
import pygeos
from shapely.strtree import STRtree
from shapely import Polygon
from osgeo import gdal
import powerlaw
import xarray as xr
import os
import time
import matplotlib.pyplot as plt

import csv

from scipy.stats import linregress

gdal.SetConfigOption("OGR_GEOJSON_MAX_OBJ_SIZE", "2000MB")


def power_law(x, p, A):
    return A * np.power(x, p)


def power_law_norm(x, α, xmin):
    return np.power(x, -α) * (α - 1) * xmin ** (α - 1)


def power_law_CCDF(x, α, xmin):
    return np.power(x / xmin, 1 - α)


def truncated_power_law_norm(x, α, λ, xmin):
    from mpmath import gammainc

    C = λ ** (1 - α) / float(gammainc(1 - α, λ * xmin))
    return np.power(x, -α) * np.exp(-λ * x) * C


def reef_size_distribution_txt(input_folder, output_folder_pdf, output_folder_ccdf):
    filenames = os.listdir(input_folder)

    for filename in filenames:
        t0 = time.time()

        name = filename[0:-8]

        print("Reading %s..." % name, end="")

        gdf = gpd.read_file("%s/%s" % (input_folder, filename))

        print("computing powerlaw...", end="")

        cluster_sizes = gdf.to_crs(epsg="6933").area  # gdf["area (m2)"].values

        # The data has some wrong polygons of less than 1m^2, which are not reasonable. Set xmin to 100m^2.
        results = powerlaw.Fit(cluster_sizes, xmin=1000)
        bins_pdf, values_pdf = results.pdf()
        bins_ccdf, values_ccdf = results.ccdf()

        center_bins_pdf = (
            bins_pdf[0:-1] + (bins_pdf - np.roll(bins_pdf, shift=1))[1:] / 2
        )

        np.savetxt(
            "%s/%s.txt" % (output_folder_pdf, name),
            np.transpose([center_bins_pdf, values_pdf]),
        )
        np.savetxt(
            "%s/%s.txt" % (output_folder_ccdf, name),
            np.transpose([bins_ccdf, values_ccdf]),
        )

        print("done in %.2f m" % ((time.time() - t0) / 60.0))


def compute_size_distribution(
    input_folder, output_filename, plot=False, figname="Figures/"
):
    filenames = os.listdir(input_folder)

    optimal_alphas = []
    optimal_xmins = []
    optimal_Ds = []

    names = []

    if not os.path.exists(figname + "PDF/"):
        os.mkdir(figname + "PDF/")
    if not os.path.exists(figname + "CCDF/"):
        os.mkdir(figname + "CCDF/")
    if not os.path.exists(figname + "PDF_whole_range/"):
        os.mkdir(figname + "PDF_whole_range/")
    if not os.path.exists(figname + "CCDF_whole_range/"):
        os.mkdir(figname + "CCDF_whole_range/")

    for filename in filenames:
        name = filename[0:-8]

        print("Reading %s..." % name, end="")

        gdf = gpd.read_file("%s/%s" % (input_folder, filename))

        print("computing powerlaw...", end="")

        cluster_sizes = gdf.to_crs(epsg="6933").area  # gdf["area (m2)"].values

        # The data has some wrong polygons of less than 1m^2, which are not reasonable. Set xmin to 1m^2.
        results = powerlaw.Fit(cluster_sizes, xmin=1)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        results.plot_pdf(
            ax=ax1,
            ls="",
            marker="o",
            color="k",
            markerfacecolor="w",
            markersize=15,
            label="",
        )
        results.plot_ccdf(
            ax=ax2,
            ls="",
            marker="o",
            color="k",
            markerfacecolor="w",
            markersize=15,
            label="",
        )

        # Get approx xmin:
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
        # sigma = results.power_law.sigma
        xmin = results.power_law.xmin

        optimal_alphas.append(alpha)
        optimal_xmins.append(xmin)
        optimal_Ds.append(np.amin(Ds))
        names.append(name)

        x = np.array(
            [
                10**i
                for i in range(
                    len(str(int(np.amin(cluster_sizes)))),
                    len(str(int(np.amax(cluster_sizes)))) + 1,
                )
            ]
        )

        if plot:
            # Plot PDF whole range
            ax1.plot(
                x,
                power_law_norm(x, alpha, xmin),
                lw=3,
                color="r",
                label=r"$y\sim x^{-\alpha}, \alpha=%.2f$" % alpha,
            )

            ax1.tick_params(axis="both", which="major", labelsize=16)

            ax1.set_xlabel(r"$Km^2$", fontsize=30)
            ax1.set_ylabel("PDF", fontsize=30)

            ax1.set_yscale("log")
            ax1.set_xscale("log")

            ax1.legend(loc="upper right", fontsize=20)

            fig1.savefig(
                figname + "PDF_whole_range/%s.png" % name, bbox_inches="tight", dpi=300
            )

            # Plot CCDF whole range
            ax2.plot(
                x,
                power_law_CCDF(x, alpha, xmin),
                lw=3,
                color="r",
                label=r"$y\sim x^{1-\alpha}, \alpha=%.2f$" % alpha,
            )

            ax2.tick_params(axis="both", which="major", labelsize=16)

            ax2.set_xlabel(r"$Km^2$", fontsize=30)
            ax2.set_ylabel("CCDF", fontsize=30)

            ax2.set_yscale("log")
            ax2.set_xscale("log")

            ax2.legend(loc="upper right", fontsize=20)

            fig2.savefig(
                figname + "CCDF_whole_range/%s.png" % name, bbox_inches="tight", dpi=300
            )

            # Plot from xmin
            x = np.array(
                [
                    10**i
                    for i in range(
                        len(str(int(xmin))) - 1,
                        len(str(int(np.amax(cluster_sizes)))) + 1,
                    )
                ]
            )

            # PDF
            fig3, ax3 = plt.subplots(figsize=(8, 6))

            results.plot_pdf(
                ax=ax3,
                ls="",
                marker="o",
                color="k",
                markerfacecolor="w",
                markersize=15,
                label="",
            )

            ax3.plot(
                x,
                power_law_norm(x, alpha, xmin),
                lw=3,
                color="r",
                label=r"$y\sim x^{-\alpha}, \alpha=%.2f$" % alpha,
            )

            ax3.tick_params(axis="both", which="major", labelsize=16)

            ax3.set_xlabel(r"$Km^2$", fontsize=30)
            ax3.set_ylabel("PDF", fontsize=30)

            ax3.set_yscale("log")
            ax3.set_xscale("log")

            ax3.legend(loc="upper right", fontsize=20)

            fig3.savefig(figname + "PDF/%s.png" % name, bbox_inches="tight", dpi=300)

            # CCDF
            fig4, ax4 = plt.subplots(figsize=(8, 6))

            results.plot_ccdf(
                ax=ax4,
                ls="",
                marker="o",
                color="k",
                markerfacecolor="w",
                markersize=15,
                label="",
            )
            ax4.plot(
                x,
                power_law_CCDF(x, alpha, xmin),
                lw=3,
                color="r",
                label=r"$y\sim x^{1-\alpha}, \alpha=%.2f$" % alpha,
            )

            ax4.tick_params(axis="both", which="major", labelsize=16)

            ax4.set_xlabel(r"$Km^2$", fontsize=30)
            ax4.set_ylabel("CCDF", fontsize=30)

            ax4.set_yscale("log")
            ax4.set_xscale("log")

            ax4.legend(loc="upper right", fontsize=20)

            fig4.savefig(figname + "CCDF/%s.png" % name, bbox_inches="tight", dpi=300)
            plt.close("all")

        print("Done!")

    df = pd.DataFrame(
        {"Name": names, "Alpha": optimal_alphas, "Xmin": optimal_xmins, "D": optimal_Ds}
    )

    df.to_csv("%s.csv" % output_filename)


def create_square_grid_gdf(lons, lats):
    points = pygeos.creation.points(lons.ravel(), lats.ravel())

    gdf_temp = gpd.GeoDataFrame(geometry=points)

    coords = pygeos.get_coordinates(pygeos.from_shapely(gdf_temp.geometry))

    nrow = lons.shape[0] - 1
    ncol = lons.shape[1] - 1
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
        xy[:, i, 0] = x[rows, cols].ravel()
        xy[:, i, 1] = y[rows, cols].ravel()

    # Create geodataframe and plot result
    mesh_geometry = pygeos.creation.polygons(xy)
    mesh_gdf = gpd.GeoDataFrame(geometry=mesh_geometry, crs=4326)

    return mesh_gdf


# Create a function that subdivide each cell of the mesh into smaller cells
def subdivide_cell(cell, ϵ_new):
    """
    This function subdivide a cell into smaller cells of size ϵ.
    """
    # Get coordinates of the cell
    x_min, y_min, x_max, y_max = cell.bounds

    # Create a list of coordinates of the smaller cells
    x = np.arange(x_min, x_max - ϵ_new / 2, ϵ_new)
    y = np.arange(y_min, y_max - ϵ_new / 2, ϵ_new)

    # Create a list of polygons
    polygons = []
    for i in range(len(x)):
        for j in range(len(y)):
            polygons.append(
                Polygon(
                    [
                        (x[i], y[j]),
                        (x[i] + ϵ_new, y[j]),
                        (x[i] + ϵ_new, y[j] + ϵ_new),
                        (x[i], y[j] + ϵ_new),
                    ]
                )
            )

    return polygons


def create_meshes(input_folder, filename, output_folder, ϵ):
    # Load coral data
    t0 = time.time()
    print("Reading %s... " % filename[0:-8], end="")
    gdf = gpd.read_file("%s/%s" % (input_folder, filename))
    print("Done in %f seconds" % (time.time() - t0))

    # Create mesh
    t0 = time.time()
    print("Creating mesh... ", end="")

    max_latlon = gdf["geometry"].bounds

    total_mesh = gpd.GeoDataFrame()

    for i in range(len(max_latlon)):
        # Print the progress in percentage every 1%
        if i % int(len(max_latlon) / 100) == 0:
            print("%d%%" % int(i / len(max_latlon) * 100))

        max_lat = max_latlon.iloc[i]["maxy"]
        min_lat = max_latlon.iloc[i]["miny"]
        max_lon = max_latlon.iloc[i]["maxx"]
        min_lon = max_latlon.iloc[i]["minx"]

        # Create mesh with length ϵ
        x_c = np.arange(min_lon - ϵ, max_lon, ϵ)
        y_c = np.arange(min_lat - ϵ, max_lat, ϵ)

        x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ / 2
        y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ / 2

        yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij")

        mesh = create_square_grid_gdf(xx_v, yy_v)

        new_mesh = gpd.sjoin(mesh, gdf.iloc[i], how="inner", predicate="intersects")

        total_mesh = total_mesh.append(new_mesh, ignore_index=True)

    print("Done in %f minutes" % ((time.time() - t0) / 60.0))

    total_mesh.to_file(
        "%s/mesh_%s_epsilon_%s.geojson" % (output_folder, filename[0:-8], ε),
        driver="GeoJSON",
    )


# Substitute this function for the one below at some point
def box_counting_dimension(input_folder, filename, ϵ_0, points, factor, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load coral data
    gdf = gpd.read_file("%s/%s" % (input_folder, filename))

    # Get maximum and minimum latlon values to create a mesh
    max_latlon = gdf["geometry"].bounds

    max_lat = np.amax(max_latlon["maxy"]) + ϵ_0
    min_lat = np.amin(max_latlon["miny"]) - ϵ_0
    max_lon = np.amax(max_latlon["maxx"]) + ϵ_0
    min_lon = np.amin(max_latlon["minx"]) - ϵ_0

    t0 = time.time()

    N_boxes = []

    print("ϵ\tN")

    ϵ = ϵ_0

    epsilons = [ϵ]

    # Write the result to a file with with open syntax
    with open("%s/%s.txt" % (output_folder, filename[0:-8]), "w", buffering=1) as f:
        f.write("ϵ\tN\n")

        for i in range(points):
            if i == 0:
                # Create mesh with length ϵ
                x_c = np.arange(min_lon - ϵ, max_lon, ϵ)
                y_c = np.arange(min_lat - ϵ, max_lat, ϵ)

                x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ / 2
                y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ / 2

                yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij")

                mesh = create_square_grid_gdf(xx_v, yy_v)

                mesh = gpd.sjoin(mesh, gdf, how="inner", predicate="intersects")

                mesh = mesh.drop_duplicates(subset=["geometry"])

            else:
                ϵ = ϵ / factor

                epsilons.append(ϵ)

                # Subdivide each cell of the mesh into smaller cells of size ϵ
                new_mesh = (
                    mesh.apply(lambda x: subdivide_cell(x["geometry"], ϵ), axis=1)
                    .explode(ignore_index=True)
                    .values
                )

                new_mesh = [item for item in new_mesh]

                # Convert list of polygons to GeoDataFrame
                mesh = gpd.GeoDataFrame({"geometry": new_mesh}, crs=4326)

                mesh = gpd.sjoin(mesh, gdf, how="inner", predicate="intersects")

                mesh = mesh.drop_duplicates(subset=["geometry"])

            # Compute number of overlapping boxes
            idxs = len(mesh)
            N_boxes.append(idxs)

            print(ε, idxs)

            f.write("%f\t%f \n" % (ϵ, idxs))

    np.savetxt(
        "%s/%s.txt" % (output_folder, filename[0:-8]),
        np.transpose([epsilons, N_boxes]),
        header="ϵ N",
    )

    print("\nFinished in ", time.time() - t0, "s")


def box_counting_dimension_gdf(gdf, ϵ_0, points, factor, outfilename, verbose=False):
    # Get maximum and minimum latlon values to create a mesh
    max_latlon = gdf["geometry"].bounds

    max_lat = np.amax(max_latlon["maxy"]) + ϵ_0
    min_lat = np.amin(max_latlon["miny"]) - ϵ_0
    max_lon = np.amax(max_latlon["maxx"]) + ϵ_0
    min_lon = np.amin(max_latlon["minx"]) - ϵ_0

    t0 = time.time()

    N_boxes = []

    if verbose:
        print("ϵ\tN")

    ϵ = ϵ_0

    epsilons = [ϵ]

    # Write the result to a file with with open syntax
    with open("%s.txt" % (outfilename), "w", buffering=1) as f:
        f.write("ϵ\tN\n")

        for i in range(points):
            if i == 0:
                # Create mesh with length ϵ
                x_c = np.arange(min_lon - ϵ, max_lon, ϵ)
                y_c = np.arange(min_lat - ϵ, max_lat, ϵ)

                x_v = np.append(x_c, x_c[-1] + ϵ) - ϵ / 2
                y_v = np.append(y_c, y_c[-1] + ϵ) - ϵ / 2

                yy_v, xx_v = np.meshgrid(y_v, x_v, indexing="ij")

                mesh = create_square_grid_gdf(xx_v, yy_v)

                mesh = gpd.sjoin(mesh, gdf, how="inner", predicate="intersects")

                mesh = mesh.drop_duplicates(subset=["geometry"])

            else:
                ϵ = ϵ / factor

                epsilons.append(ϵ)

                # Subdivide each cell of the mesh into smaller cells of size ϵ
                new_mesh = (
                    mesh.apply(lambda x: subdivide_cell(x["geometry"], ϵ), axis=1)
                    .explode(ignore_index=True)
                    .values
                )

                new_mesh = [item for item in new_mesh]

                # Convert list of polygons to GeoDataFrame
                mesh = gpd.GeoDataFrame({"geometry": new_mesh}, crs=4326)

                mesh = gpd.sjoin(mesh, gdf, how="inner", predicate="intersects")

                mesh = mesh.drop_duplicates(subset=["geometry"])

            # Compute number of overlapping boxes
            idxs = len(mesh)
            N_boxes.append(idxs)

            if verbose:
                print(ε, idxs)

            f.write("%.16f\t%.2f \n" % (ϵ, idxs))

    if verbose:
        print("\nFinished in ", time.time() - t0, "s")

    return epsilons, N_boxes


def fractal_dimension_individual_reefs_area(province):
    principal_folder = "Processed_data/Box_Counting_Individual_Reefs/Area-based"

    if not os.path.exists("%s/%s" % (principal_folder, province)):
        os.mkdir("%s/%s" % (principal_folder, province))

    filename_D = "%s/%s.txt" % (principal_folder, province)

    # Load data
    filename = "/data/bio/corals/coral_reefs/%s.geojson" % province

    gdf = gpd.read_file("%s" % filename)

    with open(filename_D, "w", buffering=1) as file:
        file.write("#Area\tD\n")

        for i in range(len(gdf)):
            # Print the progress in percentage every 1%
            if i % int(len(gdf) / 100) == 0:
                print("%d%%" % int(i / len(gdf) * 100))

            gdf_compute = gdf.iloc[i : i + 1]

            area_ij = gdf_compute["area (m2)"]

            ϵ_0 = 16
            points = 18
            factor = 2

            outfilename = "%s/%s/%.9f" % (
                principal_folder,
                province,
                area_ij,
            )

            epsilons, N_boxes = box_counting_dimension_gdf(
                gdf_compute, ϵ_0, points, factor, outfilename
            )

            epsilons = np.array(epsilons)
            N_boxes = np.array(N_boxes)

            epsilons_fit = np.log(
                epsilons[(N_boxes > 1) & ((N_boxes - N_boxes[1]) > 0)]
            )[3:]
            N_boxes_fit = np.log(N_boxes[(N_boxes > 1) & ((N_boxes - N_boxes[1]) > 0)])[
                3:
            ]

            try:
                results_lr = linregress(epsilons_fit, N_boxes_fit)

                D = np.abs(results_lr.slope)

            except Exception:
                D = np.nan

            file.write("%.9f\t%.4f\n" % (area_ij, D))

            # print("D=%.2f" % D)


def fractal_dimension_individual_reefs_perimeter(province):
    principal_folder = "Processed_data/Box_Counting_Individual_Reefs/Perimeter-based"

    if not os.path.exists("%s/%s" % (principal_folder, province)):
        os.mkdir("%s/%s" % (principal_folder, province))

    filename_D = "%s/%s.txt" % (principal_folder, province)

    # Load data
    filename = "/data/bio/corals/coral_reefs/%s.geojson" % province

    gdf = gpd.read_file("%s" % filename)

    gdf["geometry"] = gdf.boundary

    with open(filename_D, "w", buffering=1) as file:
        file.write("#Perimeter\tD\n")

        for i in range(len(gdf)):
            # Print the progress in percentage every 1%
            if i % int(len(gdf) / 100) == 0:
                print("%d%%" % int(i / len(gdf) * 100))

            gdf_compute = gdf.iloc[i : i + 1]

            perimeter_ij = gdf_compute["perimeter (m)"]

            ϵ_0 = 16
            points = 22
            factor = 2

            outfilename = "%s/%s/%.9f" % (
                principal_folder,
                province,
                perimeter_ij,
            )

            epsilons, N_boxes = box_counting_dimension_gdf(
                gdf_compute, ϵ_0, points, factor, outfilename
            )

            epsilons = np.array(epsilons)
            N_boxes = np.array(N_boxes)

            epsilons_fit = np.log(
                epsilons[(N_boxes > 1) & ((N_boxes - N_boxes[1]) > 0)]
            )[3:]
            N_boxes_fit = np.log(N_boxes[(N_boxes > 1) & ((N_boxes - N_boxes[1]) > 0)])[
                3:
            ]

            try:
                results_lr = linregress(epsilons_fit, N_boxes_fit)

                D = np.abs(results_lr.slope)

            except Exception:
                D = np.nan

            file.write("%.9f\t%.4f\n" % (perimeter_ij, D))

            # print("D=%.2f" % D)


def Area_Perimeter_scaling(input_folder, outfilename):
    filenames = os.listdir(input_folder)

    df_f = pd.DataFrame({"Province": [], "Area": [], "Perimeter": []})

    for filename in filenames:
        t0 = time.time()
        print("Computing %s..." % filename[18:-8], end="")

        gdf = gpd.read_file("Data/Coral/%s" % filename)

        gdf = gdf.to_crs(epsg="6933")

        df = pd.DataFrame(
            {
                "Province": [filename[18:-8] for i in range(len(gdf))],
                "Area": gdf.area.values,
                "Perimeter": gdf.boundary.length.values,
            }
        )

        df_f = pd.concat((df_f, df), ignore_index=True)

        print("finished in %.2f m" % ((time.time() - t0) / 60.0))

    df_f.to_parquet("%s.parquet" % outfilename, engine="fastparquet")


def inter_reef_distance(province, verbose=True):
    gdf = gpd.read_file("/data/bio/corals/coral_reefs/%s.geojson" % province)

    gdf = gdf.to_crs("EPSG:6933")

    # Compute the tree of the polygons
    tree = STRtree(gdf.geometry)

    with open("Processed_data/inter_reef_distances/%s.csv" % province, "w") as file:
        writer = csv.writer(file)

        writer.writerow(["Distance"])

        for i in range(len(gdf)):
            # Print the progress in percentage every 2%
            if (i % int(len(gdf) / 50) == 0) & (verbose is True):
                print("%.2f %%" % (100 * i / len(gdf)))

            idxs, dists = tree.query_nearest(
                gdf.geometry.iloc[i], return_distance=True, exclusive=True
            )

            for dist in dists:
                writer.writerow([dist])


def compute_province_boundaries(provinces, folder, outfilename):
    min_lons = []
    max_lons = []
    min_lats = []
    max_lats = []

    for province in provinces:
        t0 = time.time()

        print("Reading %s..." % province, end="")

        gdf = gpd.read_file("%s/%s.geojson" % (folder, province))

        gdf = gdf.to_crs(epsg="4326")

        max_latlon = gdf["geometry"].bounds

        max_lat = np.amax(max_latlon["maxy"])
        min_lat = np.amin(max_latlon["miny"])
        max_lon = np.amax(max_latlon["maxx"])
        min_lon = np.amin(max_latlon["minx"])

        min_lons.append(min_lon)
        max_lons.append(max_lon)
        min_lats.append(min_lat)
        max_lats.append(max_lat)

        print("done in %.2f m" % ((time.time() - t0) / 60.0))

    df = pd.DataFrame(
        {
            "Province": provinces,
            "Min lon": min_lons,
            "Max lon": max_lons,
            "Min lat": min_lats,
            "Max lat": max_lats,
        }
    )

    df.to_csv("%s.csv" % outfilename)
