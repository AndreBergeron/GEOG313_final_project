import os
import numpy as np
import dask as da
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import xarray as xr
import cartopy
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely import contains_xy
import warnings
import logging
from matplotlib.patches import Rectangle

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

warnings.filterwarnings('ignore', message='.*HDF5.*')
warnings.filterwarnings('ignore', message='.*H5F*')
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import h5py
    h5py._errors.silence_errors()
except ImportError:
    pass

da.config.set({
    'logging.distributed':'error',
    'distributed.logging.distributed':'error'

})

from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=4, silence_logs=logging.ERROR)
client = Client(cluster)
print(client.dashboard_link)

BASIN_EXTENTS = {
    'NA': [-100, -10, 0, 50],     # North Atlantic
    'EP': [-160, -80, 0, 40],     # East Pacific
    'WP': [100, 180, 0, 50],      # West Pacific
    'NI': [40, 100, -5, 30],      # North Indian
    'SI': [20, 120, -40, 0],      # South Indian
    'SP': [140, -120, -40, 0],    # South Pacific
    'GL': [-180, 180, -90, 90] 
}

def get_masked_mean_precip(year_start, year_end, basin):

    precip_url = ("https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc")

    ds_precip = xr.open_dataset(precip_url, chunks="auto")

    warnings.filterwarnings('ignore', message='.*HDF5.*')
    warnings.filterwarnings('ignore', message='.*H5F*')
    
    precip_subset = ds_precip["precip"].sel(time=slice(f"{year_start}-01-01", 
                                                       f"{year_end}-12-01"))

    shapefile_path = shpreader.natural_earth(resolution="50m", 
                                             category="physical", 
                                             name="land")
    
    land_geoms = list(shpreader.Reader(shapefile_path).geometries())
    land_geom = ops.unary_union(land_geoms)

    # Convert longitudes from 0–360 to -180–180
    lon = (((precip_subset.lon + 180) % 360) - 180)
    lat = precip_subset.lat

    lon2d, lat2d = np.meshgrid(lon.values, lat.values)

    # contains_xy → flatten then reshape to (lat, lon)
    inside_flat = contains_xy(land_geom, lon2d.ravel(), lat2d.ravel())
    land_mask_2d = inside_flat.reshape(lat.size, lon.size)

    # Make a 2‑D DataArray mask with matching coords
    land_mask = xr.DataArray(land_mask_2d,
                             coords={"lat": lat, "lon": lon},
                             dims=("lat", "lon"))

    # Reassign the adjusted lon coordinate to data
    precip_subset = precip_subset.assign_coords(lon=lon)

    # Apply mask: keep ocean (~land_mask), broadcast across time
    masked_da = precip_subset.where(~land_mask)
    
    masked_da = masked_da.sortby("lon")
    masked_da = masked_da.sortby("lat", ascending=False)
    
    min_lon, max_lon, min_lat, max_lat = BASIN_EXTENTS[basin]

    if basin == "GL":
        ds_subset = masked_da
    else:
        ds_subset = masked_da.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(max_lat, min_lat))

    return ds_subset

def plot_spatial_var(masked_da, year_start, year_end, cmap, basin):
    
    masked_map = masked_da.mean("time")

    if "lat" in masked_map.coords:
        masked_map = masked_map.sortby("lat")
    if "lon" in masked_map.coords:
        masked_map = masked_map.sortby("lon")

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor("black")

    plot_result = masked_map.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        cmap=cmap
    )
        
    cbar = fig.colorbar(plot_result, 
                        ax=ax, 
                        orientation='horizontal', 
                        fraction=0.05, 
                        pad=0.05)
    
    cbar.set_label("Precipitation (mm)", 
                   fontsize=22, 
                   fontweight='bold')

    ax.add_feature(
            cfeature.LAND,
            facecolor="white",    
            edgecolor="black",
            linewidth=0.6,
            zorder=11)
        
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.7,
        zorder=11)

    ax.add_feature(
        cartopy.feature.BORDERS,
        linestyle="-",
        linewidth=1,
        zorder=11)

    ax.gridlines(
        draw_labels=True, 
        dms=True, 
        x_inline=False, 
        y_inline=False)
    
    if basin != "GL" and basin in BASIN_EXTENTS:
        min_lon, max_lon, min_lat, max_lat = BASIN_EXTENTS[basin]

        # lower-left corner and width/height in degrees
        width = max_lon - min_lon
        height = max_lat - min_lat

        rect = Rectangle(
            (min_lon, min_lat),
            width,
            height,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            transform=ccrs.PlateCarree(),  # lon/lat coords
            zorder=12
        )
        ax.add_patch(rect)

    ax.set_title(f"Mean Monthly Precipitation from {year_start} to {year_end} for the {basin} basin", 
                 fontsize=22,
                 fontweight='bold')
    plt.show()

def plot_mean_precip(masked_data, basin):

    # Monthly climatology (spatial mean)
    mean_monthly_climatology = masked_data.groupby("time.month").mean(dim=("time", "lat", "lon")).compute()

    # Mean annual precipitation (time + space mean)
    mean_annual_precip = masked_data.groupby("time.year").mean(dim=('lat', 'lon')).compute()

    fig, axes = plt.subplots(2, 1, figsize=(20, 18), 
                             height_ratios=[1, 1], 
                             gridspec_kw={'hspace': 0.4})

    mean_monthly_climatology.plot(ax=axes[0], 
                                  linestyle="-", 
                                  marker='d')
    
    axes[0].set_title(f"Mean Monthly Precipitation for {basin} basin", 
                      fontsize=20, 
                      fontweight='bold')
    axes[0].set_xlabel("Month", 
                       fontsize=22, 
                       fontweight='bold')
    axes[0].set_ylabel("Precipitation (mm)", 
                       fontsize=22, 
                       fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=22)
    
    axes[0].grid(True, alpha=0.5)

    mean_annual_precip.plot(ax=axes[1], 
                            linestyle="--", 
                            marker='d')
    
    axes[1].set_title(f"Mean Annual Precipitation for {basin} basin", 
                      fontsize=22, 
                      fontweight='bold')
    
    axes[1].set_xlabel("Year", 
                       fontsize=22, 
                       fontweight='bold')
    
    axes[1].set_ylabel("Precipitation (mm)", 
                       fontsize=22, 
                       fontweight='bold')
    
    axes[1].tick_params(axis='both', labelsize=22)
    
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

    return mean_annual_precip

# def plot_interactive_mean_precip(masked_data):

#     mean_monthly_climatology = (
#         masked_data
#         .groupby("time.month")
#         .mean(dim=("time", "lat", "lon"))
#         .compute()
#     )

#     mean_annual_precip = (
#         masked_data
#         .groupby("time.year")
#         .mean(dim=('lat', 'lon'))
#         .compute()
#     )

#     p1 = mean_monthly_climatology.hvplot.line(
#         x="month",
#         title="Mean Monthly Climatology",
#         xlabel="Month",
#         ylabel="Precipitation (mm/day)",
#         line_width=2,
#         grid=True,
#         height=300, width=800,
#     )

#     p2 = mean_annual_precip.hvplot.line(
#         x="time",  # or "year" if you changed the coord
#         title="Mean Annual Precipitation",
#         xlabel="Year",
#         ylabel="Precipitation (mm/day)",
#         line_width=2,
#         grid=True,
#         height=300, width=800,
#     )

#     layout = (p1 + p2).cols(1)
#     layout
#     return layout


# def plot_spatial_var(sst_url):
    
#     ds_sst = xr.open_dataset(sst_url, engine="netcdf4", chunks="auto")
#     sst_subset = ds_sst.sel(time=slice("2005-01-01", "2025-12-01"))
#     mean_sst_map= sst_subset.sst.mean("time")  
    
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(2)
#         spine.set_edgecolor("black")
        
#     plot_result = mean_sst_map.plot(
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     add_colorbar=False,
#     cmap="coolwarm",      
#     vmin=0,              
#     vmax=30)

#     cbar = fig.colorbar(plot_result, ax=ax, orientation='horizontal', 
#                        fraction=0.05, pad=0.05)
#     cbar.set_label("Sea Surface Temperature (°C)")
    
#     ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.6, zorder=11)
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=11)
#     ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1, zorder=11)
#     ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#     ax.set_title("Global Mean SST (2005-2025)")
#     plt.show()
    
#     return mean_sst_map

def load_ersst(start_year, end_year, basin='GL'):
    """
    Load ERSST v5 monthly SST data via OPeNDAP.
    """
    
    ersst_url = ("http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc")

    ds = xr.open_dataset(
        ersst_url,
        drop_variables=["time_bnds"]
    )
    warnings.filterwarnings('ignore', message='.*HDF5.*')
    warnings.filterwarnings('ignore', message='.*H5F*')
    
    ds = ds.sel(time=slice(f"{start_year}-01", f"{end_year}-12"))
    
    lon = (((ds.lon + 180) % 360) - 180)
    ds = ds.assign_coords(lon=lon)
    
    ds = ds.sortby("lon")
    ds = ds.sortby("lat", ascending=False)
    
    min_lon, max_lon, min_lat, max_lat = BASIN_EXTENTS[basin]

    if basin == "GL":
        ds_subset = ds
    else:
        ds_subset = ds.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(max_lat, min_lat),
            method="nearest",
        )
            
    return ds_subset.compute()
    
# def subset_basin_sst(sst_data, basin_extent):
#     lon_min, lon_max, lat_min, lat_max = basin_extent[BASIN_EXTENTS]

#     if lon_min < 0:
#         lon_min = lon_min % 360
#     if lon_max < 0:
#         lon_max = lon_max % 360

#     lat_slice = slice(lat_max, lat_min)

#     return sst_data.sel(
#         lon=slice(lon_min, lon_max),
#         lat=lat_slice
#     )
    
def compute_monthly_sst_anomalies(sst_data):
    """
    Compute monthly SST anomalies relative to a climatology period.
    """

    # monthly_climatology = sst_data.sst.groupby("time.month").mean(dim=("time", "lat", "lon"))
    # monthly_anomalies = sst_data.sst.groupby("time.month") - monthly_climatology
    # sst_data["sst_anomaly"] = monthly_anomalies
        
    # Monthly climatology (spatial mean)
    mean_monthly_sst = sst_data.sst.groupby("time.month").mean(dim=("time", "lat", "lon")).compute()

    # Mean annual precipitation (time + space mean)
    mean_annual_sst = sst_data.sst.groupby("time.year").mean(dim=('lat', 'lon')).compute()

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    mean_monthly_sst.plot(ax=axes[0], linestyle="-", marker='d')
    axes[0].set_title("Mean Monthly SST")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Temperature (Degrees Celsius)")
    axes[0].grid(True, alpha=0.5)

    mean_annual_sst.plot(ax=axes[1], linestyle="--", marker='d')
    axes[1].set_title("Mean Annual SST")
    axes[0].set_xlabel("Year")
    axes[1].set_ylabel("Temperature (Degrees Celsius)")
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()
        
    return mean_annual_sst