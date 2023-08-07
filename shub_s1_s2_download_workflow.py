"""
Name:
    shub_s1_s2_download_workflow.py
Description - short:
	This Python workflow prepares Sentinel-1 and Sentinel-2 layer stacks described in the article "Machine learning-ready
	remote sensing data for Maya archaeology" by Kokalj et al. (2023).
	Sentinel Hub is used as a source for Sentinel-1 and Sentinel-2 data, so credentials (INSTANCE_ID; CLIENT_ID;
	CLIENT_SECRET) to access it are required (https://www.sentinel-hub.com).
	Some basic processing information is displayed in the Python console during runtime.
Description - long:
	A) For Sentinel-2: Data for selected 17 cloud-free dates on selected AOIs are loaded from Sentinel Hub. For each
	date, all image bands except B10 (i.e. 12 layers) are loaded and a cloud mask layer CLM is calculated.
	All 13 layers of all 17 dates are stacked into an array of 221 layers and stored in TIFF format.
	B) For Sentinel-1: A statistical layer stack is created for the years 2017-2020. Six per-pixel statistics (mean,
	median, standard deviation, coefficient of variance, 5th percentile and 95th percentile) are calculated for each
	year separately and for the entire time span, separately for ASC and DESC orbits and separately for VV and VH
	polarisations. Existing acquisition dates are filtered to obtain only unique data.
	Data are loaded separately for each year and ASC/DES collection, and converted to dB.
	All 6 layers/parameters of all both polarizations and orbit directions (24 layers per year) for all four years and
	the entire time span are stacked into a 120-layer array and stored as TIF.
    C) The outputs are LZW compressed and contain geo-reference information. Due to the large amount of Sentinel-1 data
    considered, it takes about 3 minutes to download and process a single tile.

Uses:
    SentinelHub
    DataCollection.SENTINEL1_IW_ASC
    DataCollection.SENTINEL1_IW_DES
    DataCollection.SENTINEL2_L2A
Authors:
    Aug. 2023.
    N. Čož (ZRC SAZU): original code for downloading cloudless S-2 images in given extents and time frame,
    code review and polishing.
    P. Pehani (ZRC SAZU): updated for preparation of requested S-1 and S-2 layer-stacks.
    Ž. Kokalj (ZRC SAZU): code review and polishing.
"""

import datetime
import time

import numpy as np
from osgeo import gdal
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    SentinelHubCatalog,
    filter_times,
)
from sentinelhub import SHConfig


'''
==================================================================================
Configuration parameters
==================================================================================
'''
# Set INSTANCE_ID from configuration in SentinelHub account
INSTANCE_ID = 'INSERT-INSTANCE-ID'  # SH configuration, string parameter
CLIENT_ID = 'INSERT-CLIENT-ID'  # if already in SH config file (config.json) ($ sentinelhub.config --show), leave it as is
CLIENT_SECRET = 'INSERT-CLIENT-SECRET'  # if already in SH config file (config.json) ($ sentinelhub.config --show), leave it as is

if INSTANCE_ID and CLIENT_ID and CLIENT_SECRET:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
    config.sh_client_id = CLIENT_ID               # use only if you changed variable CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET       # use only if you changed variable CLIENT_SECRET
else:
    config = None

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")


'''
==================================================================================
Input processing parameters
==================================================================================
'''

# Input parameters
resolution = 10  # 10/20/60 m
s1_year_list = ['2017', '2018', '2019', '2020', '2017-2020']
s2_valid_dates_file = './input_files/s2_list_of_valid_dates.txt'  # file with list of S-2 cloudless dates
bboxes_file = './input_files/list_of_bbox_samples.txt'  # file with bounding boxes of several test AOIs
utm_zone = "16N"

process_sentinel_2 = True  # False
process_sentinel_1 = True  # True


'''
==================================================================================
Supporting functions
==================================================================================
'''


# Import lists from the external TXT files
def import_lists(s2_valid_dates_file_pth, bboxes_file_pth):

    # Import pre-defined cloud-free dates for S2 imagery
    list_of_dates = []
    with open(s2_valid_dates_file_pth, mode='r') as infile_dates:
        for line in infile_dates.read().splitlines():
            current_date = datetime.datetime.strptime(line, "%Y-%m-%d")
            list_of_dates.append(current_date)

    # Import pre-defined test bounding boxes
    list_of_bboxes = []
    with open(bboxes_file_pth, mode='r') as infile_bboxes:
        for line in infile_bboxes.readlines():
            line = line.split(sep=' ')
            _, xmin, ymin, xmax, ymax = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            coords = (xmin, ymin, xmax, ymax)
            # list_of_bboxes.append([id, coords])
            list_of_bboxes.append(coords)

    return list_of_dates, list_of_bboxes


# Sentinel Hub Request (aka time-series) returns only one image (i.e. mosaic)
def s2_get_all_bands(aoi_bbox, aoi_cols_rows, curr_date, time_diff):
    # Evalscript to read all bands except B10 and CLM of Sentinel-2 image
    evalscript_s2_bands = """
        //VERSION=3
        function setup() {
            return {

                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","CLM"],
                    units: "DN" }],
                output: { bands: 13, sampleType: "INT16"  }
            };
        }

        function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
            outputMetadata.userData = { "scenes":  scenes.orbits }
        }

        function evaluatePixel(sample) {
            return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
                    sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12,
                    sample.CLM];
        }
    """

    return SentinelHubRequest(
        evalscript=evalscript_s2_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(curr_date, curr_date + time_diff)
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_cols_rows,
        config=config,
    )


# Set the time interval for the current year or all years
def s1_set_time_interval(curr_year_str):
    if curr_year_str != '2017-2020':
        next_year_str = '{:4d}'.format(int(curr_year_str) + 1)
        time_interval = curr_year_str + '-01-01', next_year_str + '-01-01'
    else:
        last_year_str = '{:4d}'.format(int(curr_year_str[5:9]) + 1)
        time_interval = curr_year_str[0:4] + '-01-01', last_year_str + '-01-01'

    return time_interval


# Find all images in current timespan (from-to) for current data-collection
def s1_search_iterator(catalog, curr_data_collection, aoi_bbox, time_interval):
    return catalog.search(
        curr_data_collection,
        bbox=aoi_bbox,
        time=time_interval,
        fields={
            "include": [
                "id",
                "properties.datetime"
            ],
            "exclude": []
        }
    )


def s1_get_both_bands(aoi_bbox, aoi_cols_rows, curr_date, time_diff, curr_data_collection):
    # Evalscript to load both Sentinel-1 bands VV and VH and change it do dB scale
    evalscript_s1_asc_or_des = """
    //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["VV", "VH"]
                }],
                output: [
                    { id:"custom", bands:2, sampleType: SampleType.FLOAT32 }
                ]
            }
        }

        function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
            outputMetadata.userData = { "norm_factor":  inputMetadata.normalizationFactor }
        }

        // apply "toDb" function on input bands
         function evaluatePixel(samples) {
          var VVdB = toDb(samples.VV)
          var VHdB = toDb(samples.VH)
          return [VVdB, VHdB]
        }

        // definition of "toDb" function
        function toDb(linear) {
          var log = 10 * Math.log(linear) / Math.LN10   // VV and VH linear to decibels
          var val = Math.max(Math.min(log, 5), -30) 
          return val
        }    
    """

    return SentinelHubRequest(
        evalscript=evalscript_s1_asc_or_des,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=curr_data_collection,
                time_interval=(curr_date - time_diff, curr_date + time_diff),
                other_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID"}}
            )
        ],
        responses=[SentinelHubRequest.output_response("custom", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_cols_rows,
        config=config,
    )


# Function to calculate per pixel statistics (i.e. six parameters) of Sentinel-1 time-series
# Calculated parameters are: mean, median, standard deviation, coefficient of variance, 5th and 95th percentile
def s1_calc_yearly_stats(data_np, data_cols_rows):
    # data_shape = data_np.shape
    data_stats_np = np.zeros((data_cols_rows[1], data_cols_rows[0], 6), dtype=np.float32)
    data_stats_np[:, :, 0] = np.nanmean(data_np, axis=2)
    data_stats_np[:, :, 1] = np.nanmedian(data_np, axis=2)
    data_stats_np[:, :, 2] = np.nanstd(data_np, axis=2)
    data_stats_np[:, :, 3] = np.nanvar(data_np, axis=2)
    data_stats_np[:, :, 4] = np.nanpercentile(data_np, 5, axis=2)
    data_stats_np[:, :, 5] = np.nanpercentile(data_np, 95, axis=2)
    return data_stats_np


# Save numpy array as geotiff via the GDAL, again in GDAL axis-order: NCOLS, NROWS!!
def save_np_as_tiff(out_filename, data_np, res, ul_x, ul_y, data_type):

    data_shape = data_np.shape
    geo_transform = (ul_x, res, 0.0, ul_y, 0.0, -res)  # edit GeoTiff tags

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(out_filename, data_shape[1], data_shape[0], data_shape[2], data_type)
    out_raster.SetGeoTransform(geo_transform)

    for b in range(data_shape[2]):
        out_band = out_raster.GetRasterBand(b + 1)
        out_band.WriteArray(data_np[:, :, b])
        out_band.FlushCache()

    # out_raster_crs = osr.SpatialReference(wkt=prj)
    # out_raster.SetProjection(out_raster_crs.ExportToWkt())
    out_band = None    # dereference band     # also not needed
    out_raster = None


def normalize_lin(image, minimum, maximum) -> np.ndarray:
    # linear cut off
    image[image > maximum] = maximum
    image[image < minimum] = minimum

    # stretch to 0.0 - 1.0 interval
    image = (image - minimum) / (maximum - minimum)
    image[image > 1] = 1
    image[image < 0] = 0

    image = np.float32(image)

    return image


'''
==================================================================================
Main function
==================================================================================
'''


def process_one_bbox(bbox_indx, aoi_bbox, aoi_cols_rows, list_s2_dates):
    # Print starting parameters
    print(f"\nAOI size at {resolution} m resolution: {aoi_cols_rows} pixels")
    print("AOI centre:                 ", aoi_bbox.middle)
    print(f"AOI UTM zone:                {utm_zone}")
    print("Process Sentinel-1?:        ", process_sentinel_1)
    print("Process Sentinel-2?:        ", process_sentinel_2)

    # ==================================================================================================================
    # Prepare layer-stack of bands and cloud mask of pre-selected cloud-free Sentinel-2 images
    # ==================================================================================================================

    if process_sentinel_2:

        print("\n-------------------------------------------------------------------------------------------")
        print("# SENTINEL-2: Start preparation of layer-stack of pre-selected cloud-free Sentinel-2 images\n")

        # Create a list of requests for all 17 pre-defined dates
        one_day_diff = datetime.timedelta(hours=24)
        process_requests = [
            s2_get_all_bands(aoi_bbox, aoi_cols_rows, curr_date, one_day_diff)
            for curr_date in list_s2_dates
        ]

        # Extract download information and pass it to the download client
        list_of_requests = [curr_request.download_list[0] for curr_request in process_requests]
        dl_client = SentinelHubDownloadClient(config=config)

        # Download the selected S2 data to a list of arrays
        data_s2_list = dl_client.download(list_of_requests, max_threads=5)

        # Init numpy array and build it by appending it from the list
        data_s2_np = np.array([], dtype=np.int16).reshape(aoi_cols_rows[1], aoi_cols_rows[0], 0)
        for curr_arr in data_s2_list:
            data_s2_np = np.concatenate((data_s2_np, np.asarray(curr_arr)), axis=2)

        # Print some basic info of loaded S2 data
        print('# Some basic info of loaded S2 data:')
        print('  - number of unique acquisitions:    ', len(list_s2_dates))
        print('  - size of output S2 numpy array:    ', data_s2_np.shape)  # (3, 4, 221)
        tif_shape = (data_s2_np.shape[2], data_s2_np.shape[1], data_s2_np.shape[0])
        print('  - size of output TIF layer-stack:   ', tif_shape)         # (221, 4, 3)
        print('  - maximum of output S2 raster stack:', np.max(data_s2_np))
        bands_clm = list(range(12, data_s2_np.shape[2], 13))  # [12, 25, 38, 51, 64, ...]
        print(
            '  - maximum of all output CLM layers: ',
            np.max(data_s2_np[:, :, bands_clm]),
            ' (0/1=no/cloud; 255=nodata)'
        )

        # Save final raster as geotiff
        out_filename = f'tile_{bbox_indx}_S2_.tif'
        # out_filename = 's2_layerstack_{:2d}m.tif'.format(resolution)
        save_np_as_tiff(out_filename, data_s2_np, resolution, aoi_bbox.min_x, aoi_bbox.max_y, gdal.GDT_UInt16)
        print('  - output raster stack saved as:', out_filename)  # (120, 3, 4)

        print("\n# Finished preparation of layer-stack of pre-selected cloud-free Sentinel-2 images")

    # ==================================================================================================================
    # Prepare layer-stack of per-pixel statistics of Sentinel-1 time-series 2017-2020
    # ==================================================================================================================

    if process_sentinel_1:

        print("\n--------------------------------------------------------------------------------------------")
        print("# SENTINEL-1: Start preparation of Sentinel-1 per-pixel statistical layer-stack\n")

        # Inits
        catalog = SentinelHubCatalog(config=config)
        data_collection_list = [DataCollection.SENTINEL1_IW_ASC, DataCollection.SENTINEL1_IW_DES]

        # Init out raster, i.e. stack of 120 layers
        out_stack = np.array([], dtype=np.float32).reshape(aoi_cols_rows[1], aoi_cols_rows[0], 0)

        # Repeat for all single years YYYY or year intervals YYYY-YYYY
        for curr_year_str in s1_year_list:
            # Set the time interval by year
            time_interval = s1_set_time_interval(curr_year_str)

            # Repeat for both data-collections, ASC and DES
            for curr_data_collection in data_collection_list:

                print(
                    '# S1 process year and orbit direction: {:} - {:}'.format(
                        curr_year_str,
                        curr_data_collection.value.orbit_direction
                    )
                )

                # Get a list of all available Sentinel-1 images in a given time frame (without loading them)
                search_iterator = s1_search_iterator(catalog, curr_data_collection, aoi_bbox, time_interval)
                all_acquisitions = list(search_iterator)

                # Many acquisitions differ only for a few seconds. That is because the acquisitions are acquired in the
                # same orbit pass, and then cropped into tiles. To find unique acquisitions filter them
                one_hour_diff = datetime.timedelta(hours=1)
                all_timestamps = search_iterator.get_timestamps()
                unique_acquisitions = filter_times(all_timestamps, one_hour_diff)
                print(
                    '  - number of unique/all acquisitions: {}/{}  (excl/incl those differing for  few seconds)'.format(
                        len(unique_acquisitions),
                        len(all_acquisitions)
                    )
                )

                # To join them together use SentinelHubRequest, which produces mosaic
                if len(unique_acquisitions) > 0:

                    # Create a list of requests for all unique dates
                    process_requests = [
                        s1_get_both_bands(
                            aoi_bbox,
                            aoi_cols_rows,
                            curr_date,
                            one_hour_diff,
                            curr_data_collection
                        )
                        for curr_date in unique_acquisitions
                    ]

                    # Extract download information and pass it to the download client
                    list_of_requests = [curr_request.download_list[0] for curr_request in process_requests]
                    dl_client = SentinelHubDownloadClient(config=config)

                    # Download the selected S1 data to a list of arrays
                    data_s1_list = dl_client.download(list_of_requests, max_threads=5)

                    # Init numpy array and build it by appending data from the list
                    data_s1_np = np.array([], dtype=np.float32).reshape(aoi_cols_rows[1], aoi_cols_rows[0], 0)
                    for curr_arr in data_s1_list:
                        data_s1_np = np.concatenate((data_s1_np, np.asarray(curr_arr)), axis=2)
                    # Print some basic info of loaded S1 data
                    # print('  - shape of currently loaded S1 data:', data_s1_np.shape)  # e.g. (3, 4, 54)

                    # Prepare lists of VV and VH bands
                    bands_vv = list(range(0, data_s1_np.shape[2], 2))  # [0, 2, 4, ...]
                    bands_vh = list(range(1, data_s1_np.shape[2], 2))  # [1, 3, 5, ...]

                    # Calculate per-pixel statistics per VV and VH bands, and append it to output layer-stack
                    curr_out_stack = s1_calc_yearly_stats(data_s1_np[:, :, bands_vv], aoi_cols_rows)  # VV
                    out_stack = np.concatenate((out_stack, curr_out_stack), axis=2)
                    curr_out_stack = s1_calc_yearly_stats(data_s1_np[:, :, bands_vh], aoi_cols_rows)  # VV
                    out_stack = np.concatenate((out_stack, curr_out_stack), axis=2)

                else:
                    # If no images are found, append empty statistics to output layer-stack
                    curr_out_stack = np.zeros((aoi_cols_rows[1], aoi_cols_rows[0], 12), dtype=np.float32)
                    out_stack = np.concatenate((out_stack, curr_out_stack), axis=2)

        # Save final raster as geotiff
        print('\n# Some basic info of prepared S1 statistical layer-stack:')
        print('  - size of output S1 numpy array:    ', out_stack.shape)  # (3, 4, 120)
        tif_shape = (out_stack.shape[2], out_stack.shape[1], out_stack.shape[0])
        print('  - size of output TIF layer-stack:   ', tif_shape)        # (120, 4, 3)
        # out_filename = 's1_layerstack_{:2d}m.tif'.format(resolution)
        out_filename = f'tile_{bbox_indx}_S1_.tif'
        # Normalize arrays (min/max cutoff: -30, 5) before saving
        out_stack = normalize_lin(out_stack, -30, 5)
        save_np_as_tiff(
            out_filename,
            out_stack,
            resolution,
            aoi_bbox.min_x,
            aoi_bbox.max_y,
            gdal.GDT_Float32  # gdal.GDT_UInt16
        )
        print('  - output raster stack saved as:', out_filename)

        print("\n# Finished preparation of Sentinel-1 per-pixel statistical layer-stack")
        print("\n--------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    '''
    ==================================================================================
    Main code
    ==================================================================================
    '''

    time_start = time.time()

    # Import list of cloud-free dates for Sentinel-2, and list of test bboxes from external TXT files
    list_of_s2_dates, list_of_bbs = import_lists(s2_valid_dates_file, bboxes_file)

    # CRS for Maya sites in Mexico, Chactun: EPSG:32616 - WGS 84 / UTM zone 16N
    crs_maya = getattr(CRS, f"UTM_{utm_zone}")   # CRS.UTM_16N
    # List of supported CRSs: https://docs.sentinel-hub.com/api/latest/api/process/crs/

    # Run for all bounding boxes in the list
    for bb_i, bb in enumerate(list_of_bbs):
        aoi_bb = BBox(bbox=bb, crs=crs_maya)
        aoi_dims = bbox_to_dimensions(aoi_bb, resolution=resolution)

        process_one_bbox(
            bb_i,
            aoi_bb,
            aoi_dims,
            list_of_s2_dates
        )

    # Time of processing
    time_end = time.time()
    print("# Processing time for S-1/S-2 layer-stack preparation:     {:0.1f} min".format((time_end - time_start) / 60))
    print('# End of processing')
