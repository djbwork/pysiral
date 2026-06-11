# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

from loguru import logger
from datetime import datetime, date
from typing import List, Tuple

from pysiral.auxdata import AuxdataBaseClass, GridTrackInterpol
from pysiral.auxdata.snow._data import SnowParameterContainer
from pysiral.core.legacy_classes import ErrorStatus


class Warren99SMLGClimDataContainer(object):
    """
    A dedicated data container for the merged W99/AMSR2 snow climatology. This class has been introduced
    with the use of daily scaling that requires data to loaded also from month adjacent to the month
    of the current Level-2 data object
    """

    def __init__(self, cfg, use_daily_scaling):
        """
        Init the class
        :param cfg: A copy of the auxdata class configuration
        :param use_daily_scaling:
        """

        # Properties
        self.cfg = cfg
        self.use_daily_scaling = use_daily_scaling
        self.ds = None
        self.error = ErrorStatus()

        # Load the data on initialization
        self._load()

    def _load(self):
        """
        Load the required data. This will load the data for all winter month into memory and the return
        either a weighted fiels (if `use_daily_scaling` is True) or just the field from the corresponding month
        :return:
        """

        # Load the data of all month
        try:
            self.ds = xr.load_dataset(self.cfg.filename)
        except FileNotFoundError:
            msg = "Could not locate file: {}".format(self.cfg.filename)
            self.error.add_error("invalid-filepath", msg)
            self.error.raise_on_error()

    def get_lonlat(self):
        """
        Return longitude and latitude variables
        :return:
        """

        # The grid is the same for all month, therefore we can just retrieve the fields
        # from the first data sets
        return self.data.longitude.values, self.data.latitude.values

    def get_var(self, parameter_name, date_tuple):
        """
        Get the geophysical variable from the netCDF(s). If daily scaling is activated, the date information
        given by date tuple will be used to create output fields that are interpolated between adjacent month.
        :param parameter_name:
        :param date_tuple:
        :return:
        """

        # There are three cases that requires a different handling:
        #
        # 1. daily scaling is off
        #    -> return the single field of the single data set for the corresponding month
        if not self.use_daily_scaling:
            return self.get_monthly_field(date_tuple[1], parameter_name)

        breakpoint()

        # 2. daily scaling is on and requested date is a reference date
        #    -> return the field of the single data set for the reference date
        is_reference_date = date_tuple[1:] in self.reference_dates
        if self.use_daily_scaling and is_reference_date:
            return self.get_monthly_field(date_tuple[1], parameter_name)

        # 3. daily scaling is on and requested date is between reference dates
        #   -> return a linear interpolated field based on the distance to the two enclosing
        #      reference dates
        if self.use_daily_scaling and not is_reference_date:
            return self.get_weighted_variable(date_tuple, parameter_name)

    @staticmethod
    def get_reference_datetimes(target_date: date) -> List[date]:
        """
        Creates datetimes objects for the reference dates for the actual winter season

        :param target_date:

        :return: List of reference times for determining the two months for temporal interpolation
        """
        months_center = list(zip(range(1, 13), [15]*12))  # 15th of each month
        reference_dates = [months_center[11], *months_center, months_center[0]]
        # Get the winter id (year of October for October - April winter)
        years = [target_date.year-1, *[target_date.year]*12, target_date.year+1]
        # Get the winter id (year of October for October - April winter)
        return [date(yyyy, mm, dd) for yyyy, (mm, dd) in zip(years, reference_dates)]

    def get_monthly_field(self, month_num, parameter_name):
        """
        Return the monthly field for given parameter name
        :param month_num:
        :param parameter_name:
        :return:
        """
        index = self.month_nums.index(month_num)
        variable = getattr(self.data[index], parameter_name, None)
        if variable is None:
            msg = "Dataset has no variable: {}".format(parameter_name)
            self.error.add_error("invalid-variable", msg)
            self.error.raise_on_error()
        return variable.values

    # def _get_weighted_dataset(self, target_date: date) -> xr.Dataset:
    #     """
    #     Get the weighted dataset for the target date by interpolating between the two
    #     reference months.
    #
    #     :param target_date:
    #
    #     :return:
    #     """
    #     # Determine the two months for interpolation and the weighting factor
    #     # var = var_before + weight_factor*(var_after-var_before)
    #     month_before, month_after, weight = self.get_reference_month_nums(target_date)
    @property
    def months_centers(self) -> List[Tuple[int, int]]:
        return list(zip(range(1, 13), [15] * 12))  # 15th of each month


    def get_reference_month_nums(self, target_date: date) -> Tuple[int, int, float]:
        """
        Return the two month required for the interpolation.

        :param target_date: [year, month, day] as integer

        :return: month_left, month_right, weight_factor
        """

        ref_datetimes = self.get_reference_datetimes(target_date)
        ref_date_offset = [(target_date-dt).days for dt in ref_datetimes]

        # Find the index of the first months, where the difference in day is 0 or less (right boundary)
        ref_after_idx = int(np.argmax(np.array(ref_date_offset) <= 0))
        ref_before_idx = ref_after_idx - 1

        # Compute the weighting factor
        month_before = ref_datetimes[ref_before_idx].month
        month_after = ref_datetimes[ref_after_idx].month
        period_n_days = (ref_datetimes[ref_after_idx] - ref_datetimes[ref_before_idx]).days
        weight_factor = float(ref_date_offset[ref_before_idx])/float(period_n_days)

        # All done
        return month_before, month_after, weight_factor

    def get_reference_datetimes(self, date_tuple):
        """
        Creates datetimes objects for the reference dates for the actual winter season
        :param date_tuple:
        :return:
        """

        # Get the winter id (year of October for October - April winter)
        winter_id = date_tuple[0] - int(date_tuple[1] < 10)
        year_vals = [winter_id]*3 + [winter_id+1]*4
        ref_dts = [datetime(yyyy, mm, dd) for yyyy, (mm, dd) in zip(year_vals, self.reference_dates)]
        return ref_dts

    def get_weighted_variable(self, date_tuple, parameter_name):
        """
        Compute the weighted variable between two reference dates
        :param date_tuple:
        :param parameter_name:
        :return:
        """

        # Get the fields of both reference month
        month_num_left, month_num_right, weight_factor = self.get_reference_month_nums(date_tuple)
        var_left = self.get_monthly_field(month_num_left, parameter_name)
        var_right = self.get_monthly_field(month_num_right, parameter_name)

        # Get the relative distance (0: var_left, 1: var_right)
        var = var_left + weight_factor*(var_right-var_left)

        # Done
        return var

    @property
    def w99_weight(self):
        """
        Return the static regional mask for the merged climatology
        :return:
        """
        return self.ds.w99_weight.values if self.ds is not None else None


class CryoTempoNorthernClimatology(AuxdataBaseClass):
    """
    Class for monthly snow depth & density climatology based on merged Warren99 climatology and
    monthly filterer SnowModelLG. The source data is organized as
    netCDF files that contain a:

        1. monthly climatological snow depth and density
        2. the mask of the W99 region of influence
        3. uncertainties of geophysical parameters

    This class reads the netCDF data and applies a correction for first-year sea ice areas in regions
    solely influenced by the Warren99 climatalogy.

    REQUIREMENTS:

        - Level-2 object has attribute `sitype`
          (sea ice type classication: 0 [fyi] <) sitype <= 1 [myi])
        - options has attribute `fyi_correction_factor`
          (factor 0-1 for reduction of snow depth in FYI W99 areas)

    OPTIONAL:

        - options has boolean attribute `daily_scaling`
          (if True snow depth and density will be scaled between successive month)

    NOTES:

        - This class is mainly designed to be calles from the Level-2 processor for trajectory based data sets.
          An more generalized version is in planning.

    UPDATES:

        - [July 2020] With AWI CryoSat-2 v2.3 a change was introduced to allow a daily change of the snow depth fields
          rather than monthly (see github issue: https://github.com/shendric/pysiral/issues/40)
    """

    def __init__(self, *args, **kwargs):
        """
        Init the class for getting snow depth on sea ice with density and respective uncertainties.
        :param args: Arguments for AuxdataBaseClass
        :param kwargs: Keyword arguments for AuxdataBaseClass
        """
        super(CryoTempoNorthernClimatology, self).__init__(*args, **kwargs)

        # This step loads the data file (no more data loading needed)
        self.data = Warren99SMLGClimDataContainer(self.cfg, self.use_daily_scaling)

    def get_l2_track_vars(self, l2):
        """
        This is the method that will be evoked by the Level-2 processor. This method will extract the
        geophysical parameters

        :param l2: The Level-2 data container
        :return: None
        """

        # Extract along track snow depth and density
        snow = self.get_snow_track(l2)

        # Register Variables
        self.register_auxvar("sd", "snow_depth", snow.depth, snow.depth_uncertainty)
        self.register_auxvar("sdens", "snow_density", snow.density, snow.density_uncertainty)

    def get_snow_track(self, l2):
        """
        Extract the snow depth and density track along the l2 track
        :param l2:
        :return:
        """

        # Get the MYI fraction variable
        my_fraction_var_name = self.cfg.options.get("myi_fraction_var_name", "sitype")
        myi_fraction = getattr(l2, my_fraction_var_name, None)
        if myi_fraction is None:
            msg = f"Level-2 object has no attribute: {my_fraction_var_name}"
            self.error.add_error("missing-l2-variable", msg)
            self.error.raise_on_error()

        # Extract track data from grid
        griddef = self.cfg.options[l2.hemisphere]
        grid_lons, grid_lats = self.data.get_lonlat()
        grid2track = GridTrackInterpol(l2.track.longitude, l2.track.latitude, grid_lons, grid_lats, griddef)

        # Extract data (Map the extracted tracks directly on the snow parameter container)
        var_map = self.cfg.options.variable_map
        snow = SnowParameterContainer()
        for var_name in var_map.keys():
            source_name = var_map[var_name]
            sdgrid = self.data.get_var(source_name, self._requested_date)
            setattr(snow, var_name, grid2track.get_from_grid_variable(sdgrid))

        # Extract the W99 weight for the specific track
        w99_weight = grid2track.get_from_grid_variable(self.data.w99_weight)

        # Apply the same modification as the Warren climatology
        # Apply ice_type but this time modified by the regional weight of the Warren climatology.
        # The weight ranges from 0 to 1 and ensures that snow reduction on first-year sea ice is
        # only applied to the W99 contribution of the snow depth climatology.
        # NOTE: sea ice type here is defined as MYI fraction: 0 [fyi] <= sitype <= 1 [myi]
        scale_factor = (1.0 - myi_fraction) * self.cfg.options.fyi_correction_factor * w99_weight

        # The scaling factor affects the snow depth ...
        snow.depth = snow.depth - scale_factor * snow.depth

        # ... and the uncertainty. Here it is assumed that the uncertainty
        # is similar affected by the scaling factor.
        snow.depth_uncertainty = snow.depth_uncertainty - scale_factor * snow.depth_uncertainty

        # The uncertainty of the MYI fraction is acknowledged by adding
        # a term that depends on snow depth, the magnitude of scaling
        # and the sea ice type uncertainty
        scaling_uncertainty = snow.depth * scale_factor * myi_fraction.uncertainty * w99_weight
        snow.depth_uncertainty = snow.depth_uncertainty + scaling_uncertainty

        return snow

    @property
    def use_daily_scaling(self):
        """
        Return a flag that indicates whether to use daily scaling (absence of flag in options will be treated as no)
        :return:
        """
        if "daily_scaling" not in self.cfg.options:
            return False
        else:
            return self.cfg.options.daily_scaling
