# -*- coding: utf-8 -*-

"""
@author: Stefan Hendricks

pysiral module for estimating sea surface height (ssh) respectively sea level anomaly (sla) from along-track
radar altimeter data. The classes are designed to work with the Level-2 Processor, therefore need to be
child classes of pysiral.l2proc.procsteps.Level2ProcessorStep.

NOTES:

    1. The convention of the pysiral Level-2 processor is to compute and sla (with ssh = sla + mss)
       to the Level-2 data object. Thus, the auxiliary data set mean sea surface (mss) is a mandatory
       auxiliary data set for the functionality of all classes in this module.

"""


import typing as tp

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from pysiral.core.flags import SurfaceType
from pysiral.filter import fill_nan, idl_smooth
from pysiral.l2data import L2DataArray
from pysiral.l2proc.procsteps import Level2ProcessorStep

# Raw SLA (tie-point) measurement uncertainty model (CryoTEMPO Polar Ocean, delivery of 2026-10-07)
#
#   Open ocean (LRM, SAR, SARIn):
#       sigma = sqrt(sigma_min^2 + (alpha * SWH)^2 + (beta * eps_tilde)^2)
#   Leads in sea ice (SAR, SARIn):
#       sigma = sqrt(sigma_min^2 + (gamma * eps_tilde)^2 + (delta * (mean_square_slope / slope_ref)^0.25)^2)
#
# with eps_tilde = waveform misfit / eps_ref. LRM uses the MLE4 misfit, SAR/SARIn the SAMOSA+ misfit.
# Coefficients calibrated 2026-10-06 from the CryoTEMPO extended-parameter TDS (v4.02). Values
# can be overridden with the `uncertainty_coefficients` option of SLARaw.
SLA_RAW_UNCERTAINTY_COEFFICIENTS = {
    "open_ocean_lrm": {"sigma_min": 0.0647, "alpha": 0.018, "beta": 0.0, "eps_ref": 0.00276},
    "open_ocean_sar": {"sigma_min": 0.0272, "alpha": 0.0094, "beta": 0.0498, "eps_ref": 2.1203},
    "open_ocean_sarin": {"sigma_min": 0.0869, "alpha": 0.0348, "beta": 0.0289, "eps_ref": 3.6757},
    "lead_sar": {"sigma_min": 0.0198, "gamma": 0.0, "delta": 0.0147, "slope_ref": 1e-07, "eps_ref": 6.5297},
    "lead_sarin": {"sigma_min": 0.0323, "gamma": 0.0, "delta": 0.0183, "slope_ref": 1e-07, "eps_ref": 6.4758},
}
SLA_RAW_UNCERTAINTY_SLOPE_EXPONENT = 0.25
SLA_RAW_UNCERTAINTY_SLOPE_VALID_RANGE = (1e-9, 1e-2)
SLA_RAW_UNCERTAINTY_SWH_VALID_MAX = 25.0
SLA_RAW_UNCERTAINTY_SWH_NEGATIVE_TOLERANCE = -1.0


def sla_raw_measurement_uncertainty(
        radar_mode: npt.NDArray,
        is_ocean: npt.NDArray,
        is_lead: npt.NDArray,
        swh: npt.NDArray,
        misfit_samosa: npt.NDArray,
        mean_square_slope: npt.NDArray,
        misfit_mle4: tp.Optional[npt.NDArray] = None,
        coefficients: tp.Optional[tp.Dict[str, tp.Dict[str, float]]] = None
) -> npt.NDArray:
    """
    Measurement uncertainty of the raw sea level anomaly (sigma_meas of the CryoTEMPO
    Polar Ocean uncertainty model). The population (open ocean LRM/SAR/SARIn, leads
    SAR/SARIn) is selected per record by radar mode and surface type. Records of no
    population, or with invalid inputs, are NaN.

    :param radar_mode: pysiral radar mode flag (0: LRM, 1: SAR, 2: SARIn)
    :param is_ocean: True for open ocean records
    :param is_lead: True for lead records
    :param swh: significant wave height (m)
    :param misfit_samosa: SAMOSA+ waveform misfit
    :param mean_square_slope: SAMOSA+ mean square slope
    :param misfit_mle4: MLE4 waveform misfit (LRM only, NaN uncertainty for LRM if None)
    :param coefficients: model coefficients (default: SLA_RAW_UNCERTAINTY_COEFFICIENTS)

    :return: measurement uncertainty (m)
    """
    coefficients = SLA_RAW_UNCERTAINTY_COEFFICIENTS if coefficients is None else coefficients
    n_records = radar_mode.shape[0]
    if misfit_mle4 is None:
        misfit_mle4 = np.full(n_records, np.nan)

    # Screen significant wave height: small negatives set to zero, implausible values to NaN
    swh = np.where((swh < 0) & (swh >= SLA_RAW_UNCERTAINTY_SWH_NEGATIVE_TOLERANCE), 0.0, swh)
    swh = np.where((swh >= 0) & (swh <= SLA_RAW_UNCERTAINTY_SWH_VALID_MAX), swh, np.nan)

    # Lead roughness term, NaN for mean square slope outside the physically admissible range
    slope_min, slope_max = SLA_RAW_UNCERTAINTY_SLOPE_VALID_RANGE
    valid_slope = (mean_square_slope >= slope_min) & (mean_square_slope <= slope_max)
    mean_square_slope = np.where(valid_slope, mean_square_slope, np.nan)

    sigma = np.full(n_records, np.nan)
    for key, mask in uncertainty_population_masks(radar_mode, is_ocean, is_lead).items():
        if not mask.any():
            continue
        c = coefficients[key]
        misfit = misfit_mle4 if key == "open_ocean_lrm" else misfit_samosa
        eps_tilde = misfit[mask] / c["eps_ref"]
        if key.startswith("lead"):
            roughness = (mean_square_slope[mask] / c["slope_ref"]) ** SLA_RAW_UNCERTAINTY_SLOPE_EXPONENT
            variance = c["sigma_min"] ** 2 + (c["gamma"] * eps_tilde) ** 2 + (c["delta"] * roughness) ** 2
        else:
            variance = c["sigma_min"] ** 2 + (c["alpha"] * swh[mask]) ** 2 + (c["beta"] * eps_tilde) ** 2
        sigma[mask] = np.sqrt(variance)

    return sigma


def uncertainty_population_masks(
        radar_mode: npt.NDArray,
        is_ocean: npt.NDArray,
        is_lead: npt.NDArray
) -> tp.Dict[str, npt.NDArray]:
    """
    Boolean mask per population of the CryoTEMPO Polar Ocean uncertainty model,
    keyed as in SLA_RAW_UNCERTAINTY_COEFFICIENTS. LRM is always treated as open ocean.

    :param radar_mode: pysiral radar mode flag (0: LRM, 1: SAR, 2: SARIn)
    :param is_ocean: True for open ocean records
    :param is_lead: True for lead records

    :return: dictionary of population masks
    """
    is_lrm, is_sar, is_sarin = radar_mode == 0, radar_mode == 1, radar_mode == 2
    return {
        "open_ocean_lrm": is_lrm,
        "open_ocean_sar": is_sar & is_ocean,
        "open_ocean_sarin": is_sarin & is_ocean,
        "lead_sar": is_sar & is_lead,
        "lead_sarin": is_sarin & is_lead,
    }


def tiepoint_measurement_uncertainty(
        l2,
        ssh_tiepoint_indices: npt.NDArray,
        coefficients: tp.Optional[tp.Dict[str, tp.Dict[str, float]]] = None,
        lrm_misfit_parameter: tp.Optional[str] = None
) -> tp.Tuple[npt.NDArray, tp.Dict[str, npt.NDArray]]:
    """
    Raw sea level anomaly measurement uncertainty at the ssh tie points (see
    `sla_raw_measurement_uncertainty`). Requires the SAMOSA+ auxiliary parameters
    (samosa_swh, samosa_misfit, samosa_mean_square_slope) in the l2 data object.

    :param l2: Level-2 data container
    :param ssh_tiepoint_indices: indices of the ssh tie points
    :param coefficients: uncertainty model coefficients (default: SLA_RAW_UNCERTAINTY_COEFFICIENTS)
    :param lrm_misfit_parameter: name of the l2 parameter with the MLE4 misfit for LRM records

    :return: measurement uncertainty and population masks, both at the tie points
    """

    def get_parameter(name):
        parameter = l2.get_parameter_by_name(name, raise_on_error=False)
        return np.asarray(parameter[ssh_tiepoint_indices], dtype=float)

    missing = [name for name in ("samosa_swh", "samosa_misfit", "samosa_mean_square_slope")
               if name not in l2.auxvar_names]
    if missing:
        logger.warning(f"Missing input for raw SLA uncertainty: {missing} -> NaN uncertainty")

    radar_mode = np.asarray(l2.radar_mode[ssh_tiepoint_indices])
    misfit_mle4 = None
    if lrm_misfit_parameter is not None:
        misfit_mle4 = get_parameter(lrm_misfit_parameter)
    elif np.any(radar_mode == 0):
        logger.warning("LRM tie points but no MLE4 misfit parameter -> NaN uncertainty for LRM")

    is_ocean = l2.surface_type.ocean.flag[ssh_tiepoint_indices]
    is_lead = l2.surface_type.lead.flag[ssh_tiepoint_indices]
    sigma_meas = sla_raw_measurement_uncertainty(
        radar_mode,
        is_ocean,
        is_lead,
        get_parameter("samosa_swh"),
        get_parameter("samosa_misfit"),
        get_parameter("samosa_mean_square_slope"),
        misfit_mle4=misfit_mle4,
        coefficients=coefficients
    )
    return sigma_meas, uncertainty_population_masks(radar_mode, is_ocean, is_lead)


# Smoothed (interpolated) SLA uncertainty model (CryoTEMPO Polar Ocean, delivery of 2026-10-07)
#
#   sigma_sla^2 = mean(sigma_meas^2) / N + a^2 + (b * d)^2
#
# with N the number of tie points within SLA_INTERPOLATION_HALF_WINDOW_M of the record (the mean
# is taken over them) and d the along-track distance in km to the nearest tie point, whose
# population selects a (m) and b (m/km). Values can be overridden with the
# `interpolation_uncertainty_coefficients` option of SLASmoothedLinear.
SLA_INTERPOLATION_UNCERTAINTY_COEFFICIENTS = {
    "open_ocean_lrm": {"a": 0.0413, "b": 0.001285},
    "open_ocean_sar": {"a": 0.0151, "b": 0.001501},
    "open_ocean_sarin": {"a": 0.0151, "b": 0.001501},
    "lead_sar": {"a": 0.0237, "b": 0.000243},
    "lead_sarin": {"a": 0.0283, "b": 0.000892},
}
SLA_INTERPOLATION_HALF_WINDOW_M = 50000.0  # half of the 100 km smoothing window the model was fitted for
SLA_INTERPOLATION_SPARSE_N = 3             # flag bit 1: this many tie points or fewer in the window
SLA_INTERPOLATION_FITTED_MAX_KM = 65.0     # flag bit 2: a and b were fitted out to this distance
EARTH_RADIUS_M = 6371000.0


def along_track_distance(latitude: npt.NDArray, longitude: npt.NDArray) -> npt.NDArray:
    """
    Cumulative great-circle distance along the track

    :param latitude: latitude in degrees
    :param longitude: longitude in degrees

    :return: along-track distance in meter
    """
    lat, lon = np.radians(latitude), np.radians(longitude)
    a = (np.sin(np.diff(lat) / 2) ** 2
         + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
    step = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate(([0.0], np.cumsum(step)))


def sla_interpolation_uncertainty(
        latitude: npt.NDArray,
        longitude: npt.NDArray,
        ssh_tiepoint_indices: npt.NDArray,
        sigma_meas: npt.NDArray,
        population_masks: tp.Dict[str, npt.NDArray],
        coefficients: tp.Optional[tp.Dict[str, tp.Dict[str, float]]] = None
) -> tp.Tuple[npt.NDArray, npt.NDArray]:
    """
    Uncertainty of the smoothed (interpolated) sea level anomaly at every record of the
    track (sigma_sla of the CryoTEMPO Polar Ocean uncertainty model) and its quality flag:

        bit 1: SLA_INTERPOLATION_SPARSE_N or fewer tie points in the window
               (the noise term is overestimated)
        bit 2: more than SLA_INTERPOLATION_FITTED_MAX_KM from the nearest tie point
               (beyond the distance a and b were fitted for)

    :param latitude: latitude of all records in degrees (one track in along-track order)
    :param longitude: longitude of all records in degrees
    :param ssh_tiepoint_indices: sorted indices of the ssh tie points
    :param sigma_meas: measurement uncertainty at the tie points
    :param population_masks: population masks at the tie points (see `uncertainty_population_masks`)
    :param coefficients: a, b per population (default: SLA_INTERPOLATION_UNCERTAINTY_COEFFICIENTS)

    :return: uncertainty (m) and flag, both for all records
    """
    coefficients = SLA_INTERPOLATION_UNCERTAINTY_COEFFICIENTS if coefficients is None else coefficients
    n_records = latitude.shape[0]
    sigma_sla = np.full(n_records, np.nan)
    flag = np.zeros(n_records, dtype=np.uint8)
    n_tiepoints = len(ssh_tiepoint_indices)
    if n_tiepoints == 0:
        return sigma_sla, flag

    # Along-track distance of all records and of the tie points
    distance = along_track_distance(latitude, longitude)
    tiepoint_distance = distance[ssh_tiepoint_indices]

    # Tie points within the window (cumulative sums for the windowed mean of sigma_meas^2)
    variance = sigma_meas ** 2
    usable = np.isfinite(variance)
    variance_sum = np.concatenate(([0.0], np.cumsum(np.where(usable, variance, 0.0))))
    variance_count = np.concatenate(([0], np.cumsum(usable)))
    lower = np.searchsorted(tiepoint_distance, distance - SLA_INTERPOLATION_HALF_WINDOW_M, side="left")
    upper = np.searchsorted(tiepoint_distance, distance + SLA_INTERPOLATION_HALF_WINDOW_M, side="right")
    n_window = upper - lower

    # Nearest tie point and distance to it
    if n_tiepoints == 1:
        nearest = np.zeros(n_records, dtype=int)
    else:
        j = np.clip(np.searchsorted(tiepoint_distance, distance), 1, n_tiepoints - 1)
        is_left = np.abs(distance - tiepoint_distance[j - 1]) <= np.abs(tiepoint_distance[j] - distance)
        nearest = np.where(is_left, j - 1, j)
    distance_km = np.abs(distance - tiepoint_distance[nearest]) / 1000.

    # Noise term (beyond the window there is no tie point to average: use the nearest one alone)
    count = variance_count[upper] - variance_count[lower]
    mean_variance = np.where(
        count > 0,
        (variance_sum[upper] - variance_sum[lower]) / np.maximum(count, 1),
        variance[nearest]
    )
    noise_variance = mean_variance / np.maximum(n_window, 1)

    # Interpolation term (coefficients from the population of the nearest tie point)
    a = np.full(n_records, np.nan)
    b = np.full(n_records, np.nan)
    for key, mask in population_masks.items():
        is_population = mask[nearest]
        a[is_population] = coefficients[key]["a"]
        b[is_population] = coefficients[key]["b"]

    sigma_sla = np.sqrt(noise_variance + a ** 2 + (b * distance_km) ** 2)
    flag[n_window <= SLA_INTERPOLATION_SPARSE_N] |= 1
    flag[distance_km > SLA_INTERPOLATION_FITTED_MAX_KM] |= 2
    return sigma_sla, flag


def marine_segment_filter(l2, minimum_lead_number, footprint_size, use_ocean_wfm):
    """
    Check all sections divided by land masses for reliable information content.
    Specifically, each marine segment between two land masses must have a minimum
    number of leads
    :param l2:
    :param minimum_lead_number:
    :param footprint_size:
    :param use_ocean_wfm:
    :return: mask: True: To be masked, False: Valid SLA
    """

    # Create a mask (all valid by default)
    mask = np.full(l2.n_records, False)

    # Find sea ice clusters
    land = l2.surface_type.land

    # No land -> nothing to do
    if land.num == 0:
        return mask

    # Get indices for land sections
    lead_flag = l2.surface_type.lead.flag
    if use_ocean_wfm:
        ocean_flag = l2.surface_type.ocean.flag
        logger.info('Addded {nocean} ocean records to {nlead} lead records'.format(
            nocean=ocean_flag.sum(), nlead=lead_flag.sum())
        )
        lead_flag = np.bitwise_or(lead_flag, ocean_flag)

    land_flag = land.flag.astype(int)
    land_start = np.where(np.ediff1d(land_flag) > 0)[0]
    land_stop = np.where(np.ediff1d(land_flag) < 0)[0]

    # It is assumed here, that the l1b orbit segment never starts
    # or end with land. Thus, the number of land start and land stop
    # events need to be identical.
    if len(land_start) != len(land_stop):
        msg = "l2 segments either starts or ends with land. SLA marine segment will not perform properly."
        logger.error(msg)
        return mask

    # Add artificial large land sections on beginning and end of profile
    n_marine_segments = len(land_start) + 1
    n = l2.n_records
    land_start = np.concatenate(([-1000], land_start, [n-1]))
    land_stop = np.concatenate(([-1], land_stop, [n+1000]))

    # Loop over marine segments and collect information
    # TODO: The marine segment list does not to be kept
    marine_segments = []
    section_prop = {"i0": 0.0, "i1": 0.0,
                    "width": 0.0, "n_tiepoints": 0,
                    "land_before": (9999.0, 0),
                    "land_after": (9999.0, 0)}
    for i in np.arange(n_marine_segments):

        marine_segment = section_prop.copy()

        # Get the start stop indices for marine section
        i0 = land_stop[i]+1
        i1 = land_start[i+1]
        marine_segment["i0"] = i0
        marine_segment["i1"] = i1
        marine_segment["width"] = (i1-i0) * footprint_size

        # get the number of leads
        marine_section_indices = np.arange(i0, i1+1)
        n_tiepoints = np.where(lead_flag[marine_section_indices])[0].size
        marine_segment["n_tiepoints"] = n_tiepoints

        if marine_segment["n_tiepoints"] < minimum_lead_number:
            mask[marine_section_indices] = True

        marine_segments.append(marine_segment)

    return mask


class SLABaseFunctionality(object):
    """
    A container for methods that are independent of the interpolation algorithm, basically
    cllection of mostly static method that can be pinned to all other using inheritance.
    """

    def __init__(self):
        """
        Init the class (nothing is done here)
        """
        pass

    @staticmethod
    def get_ssh_tiepoints_indices(
            surface_type: SurfaceType,
            elevation: L2DataArray,
            use_ocean_wfm: bool = None
    ) -> npt.NDArray:
        """
        Return the index list of SSH tiepoints. These are as a minimum waveforms identified as lead in the
        surface type classifcation. Ocean waveforms can be added and an optional filter applies that removes
        SSH tiepoints based on their distance to the mean sea surface (mss).
        optional addition

        :param surface_type: Surface type flag object
        :param elevation: retracked elevation (will be checked for NaN values)
        :param use_ocean_wfm: Boolean flag whether to include ocean waveforms

        :return: A list of indices indicated valid SSH tie points for the Level-2 data object
        """

        # Use waveforms identified as leads in first iteration
        ssh_tiepoint_indices = surface_type.lead.indices

        # (Optional) Add ocean waveforms if applicable
        if use_ocean_wfm:
            ssh_tiepoint_indices = np.append(ssh_tiepoint_indices, surface_type.ocean.indices)
            ssh_tiepoint_indices = np.sort(ssh_tiepoint_indices)

        # Remove indices that point to a valid range value
        valid_range = np.isfinite(elevation[ssh_tiepoint_indices])
        ssh_tiepoint_indices = ssh_tiepoint_indices[valid_range]

        # All done, return the index list of tie points
        return ssh_tiepoint_indices

    def get_tiepoint_distance_from_l2(self, l2, smooth_filter_width_footprint_size, use_ocean_wfm=False):
        """
        Returns the distance in meter to the next ssh tiepoint for each record

        :param l2: Level-2 data container
        :param smooth_filter_width_footprint_size:
        :param use_ocean_wfm:

        :return: array(float32, shape=l2.n_records)
        """

        # prepare parameter arrays
        lead_indices = l2.surface_type.lead.indices

        # (Optional) Add ocean waveforms if applicable
        if use_ocean_wfm:
            lead_indices = np.where(np.bitwise_or(l2.surface_type.ocean.flag, l2.surface_type.lead.flag))[0]

        lead_elevation = np.full(l2.n_records, np.nan, dtype=np.float32)
        lead_elevation[lead_indices] = l2.elev[lead_indices]

        # Compute distance to next lead tie point in meter
        tiepoint_distance = self.get_tiepoint_distance(np.isfinite(lead_elevation))
        tiepoint_distance = tiepoint_distance.astype(np.float32)
        tiepoint_distance *= smooth_filter_width_footprint_size

        return tiepoint_distance

    def tiepoint_maxdist_filter(self, l2, edges_only, distance_threshold, footprint_size):
        """
        A filter that does not remove sla values which distance to
        the next ssh tiepoint exceeds a defined threshold
        :param l2: Level-2 data container
        :param edges_only:
        :param distance_threshold:
        :param footprint_size:
        :return: None
        """

        # Get options
        # filter_options = self.cfg.options.tiepoint_maxdist_filter
        # edges_only = filter_options.edges_only
        # distance_threshold = filter_options.maximum_distance_to_tiepoint

        # Compute distance to next tie point
        use_ocean_wfm = self.cfg.options.get("use_ocean_wfm", False)
        tiepoint_distance = self.get_tiepoint_distance_from_l2(l2, footprint_size, use_ocean_wfm)

        # Get indices
        invalid_indices = np.where(tiepoint_distance > distance_threshold)[0]

        # Only remove sla values at the edges (if edges_only:True)
        if edges_only and len(l2.surface_type.lead.indices) > 0:
            lead_indices = l2.surface_type.lead.indices
            before_first_lead = invalid_indices < lead_indices[0]
            after_last_lead = invalid_indices > lead_indices[-1]
            edge_condition = np.logical_or(before_first_lead, after_last_lead)
            invalid_indices = invalid_indices[np.where(edge_condition)[0]]

        # Create and return mask
        mask = np.full(l2.n_records, False)
        mask[invalid_indices] = True
        return mask

    def get_tiepoint_distance(self, is_tiepoint):
        """
        Calculates the distance to the next tie point in array entries
        :param is_tiepoint: boolean array
        :return:
        """
        distance_forward = self.get_tiepoints_oneway_distance(is_tiepoint)
        distance_reverse = self.get_tiepoints_oneway_distance(is_tiepoint, reverse=True)
        return np.minimum(distance_forward, distance_reverse)

    @staticmethod
    def tiepoint_elevation_sdev_filter(
            ssh_tiepoint_indices: npt.NDArray,
            elevation: npt.NDArray,
            footprint_spacing: float,
            elevation_filter_window_m: float = 50000.,
            upper_limit_standard_deviation: float = 2.,
            lower_limit_standard_deviation: float = 3.,
            minimum_standard_deviation_m: float = 0.1
    ) -> npt.NDArray:
        """
        Filter tie points if their elevation is outside a specified multiple of
        the local elevation standard deviation. Two thresholds for upper and lower
        elevation bound must be specified.

        :param ssh_tiepoint_indices: Indices of elevation that are a (potential)
            sea surface height tiepoint
        :param elevation: elevation array
        :param footprint_spacing: Average footprint spacing. Needed to convert
            filter lenghts in meters into number of array items.
        :param elevation_filter_window_m: The filter window used for computing the
            mean elevation and the computation of rolling standard deviation.
        :param upper_limit_standard_deviation: Sea surface height tie points
            are removed if their elevation exceeds the mean elevation plus
            the local standard deviation multiplied by this factor.
        :param lower_limit_standard_deviation: Sea surface height tie points
            are removed if their elevation is below the mean elevation minus
            the local standard deviation multiplied by this factor.
        :param minimum_standard_deviation_m: The minimum local standard devation.
            This value prevents the filter becoming to greedy in the case
            of small local standard devation.

        :return: The filtered ssh tiepoint index list
        """

        # Step 1: Convert filter size to number of array entries
        window_size_float = elevation_filter_window_m / footprint_spacing
        window_size = None if np.isnan(window_size_float) else int(int(window_size_float) // 2 * 2 + 1)

        lower_limit, upper_limit = get_rolling_standard_deviation_elevation_window(
            elevation,
            window_size,
            minimum_standard_deviation_m=minimum_standard_deviation_m,
            lower_limit_standard_deviation=lower_limit_standard_deviation,
            upper_limit_standard_deviation=upper_limit_standard_deviation
        )

        tie_points_inside_elevation_bounds = np.logical_and(
            elevation[ssh_tiepoint_indices] < upper_limit[ssh_tiepoint_indices],
            elevation[ssh_tiepoint_indices] > lower_limit[ssh_tiepoint_indices]
        )

        logger.debug(f"num tie_points_inside_elevation_bounds={len(tie_points_inside_elevation_bounds)}")

        # # Debug code
        # import matplotlib.pyplot as plt
        # x = np.arange(elevation.shape[0])
        # tie_points_outside_elevation_bounds = np.logical_not(tie_points_inside_elevation_bounds)
        # plt.figure(dpi=150)
        # plt.fill_between(x, upper_limit, lower_limit,
        #                  color="green", alpha=0.25, edgecolor="none")
        # plt.fill_between(x, lower_limit, upper_limit, color="green", alpha=0.25, edgecolor="none")
        # plt.plot(x, elevation, lw=0.5, color="0.0", label="raw elevation")
        # # plt.plot(x, elevation_rolling_mean, lw=0.75, linestyle="dashed")
        # plt.scatter(x[ssh_tiepoint_indices], elevation[ssh_tiepoint_indices], s=30,
        #             c="none", edgecolors="red", label="all tie-points")
        # plt.scatter(x[ssh_tiepoint_indices[tie_points_outside_elevation_bounds]],
        #             elevation[ssh_tiepoint_indices[tie_points_outside_elevation_bounds]], s=20,
        #             marker="x", c="red", edgecolors="red", label="filtered tie-points")
        # plt.legend()
        # plt.show()

        return ssh_tiepoint_indices[tie_points_inside_elevation_bounds]

    @staticmethod
    def get_filter_width(smooth_filter_width_m, smooth_filter_width_footprint_size):
        """
        Compute the filter width in points
        :param smooth_filter_width_m:
        :param smooth_filter_width_footprint_size:
        :return:
        """
        filter_width = smooth_filter_width_m / smooth_filter_width_footprint_size
        # Make sure filter width is odd integer
        filter_width = np.floor(filter_width) // 2 * 2 + 1
        filter_width = filter_width.astype(int)
        return filter_width

    @staticmethod
    def apply_surface_type_masks(sla, sla_unc, l2, surface_types):
        """
        Remove sla and sla uncertainty values for a set surface types
        (Names must match the surface types name definitions in the l2 data containers)
        :param sla:
        :param sla_unc:
        :param l2:
        :param surface_types:
        :return:
        """
        # Loop over all surface types and modify array in place
        for surface_type in surface_types:
            flag = l2.surface_type.get_by_name(surface_type)
            sla[flag.indices] = np.nan
            sla_unc[flag.indices] = np.nan
        return sla, sla_unc

    @staticmethod
    def get_tiepoints_oneway_distance(a, reverse=False):
        """ loops through array and determines distance to latest flag=true """
        n = len(a)
        distance = np.full(a.shape, n + 1, dtype=np.int32)
        if reverse:
            a = a[::-1]
        dist = n
        for i in np.arange(n):
            if a[i]:
                dist = 0
            elif dist != n:
                dist += 1
            distance[i] = dist
        if reverse:
            distance = distance[::-1]
        return distance


class SLAGaussianProcess(Level2ProcessorStep, SLABaseFunctionality):
    """
    Use gaussian processes of the scikit-learn module to predict optimal sla from of ssh tiepoints
    (with various filter options similar to SLASmoothedLinear).
    This class will compute the sea level anomaly (sla)
    """

    def __init__(self, *args, **kwargs):
        """
        Init the class. Options will be passed to parent class
        (pysiral.l2proc.procsteps.Level2ProcessorStep)
        :param args:
        :param kwargs:
        """

        # Init both base classes
        Level2ProcessorStep.__init__(self, *args, **kwargs)
        SLABaseFunctionality.__init__(self)

    def execute_procstep(self, l1b, l2):
        """
        Mandatory Level-2 processor method that will execute the processing step
        and modify the L2 data object in-place.
        This method will interpolate ssh tiepoints given by lead (+ ocean) elevations
        and smooth the result. Filter options are available to filter unreasonable
        sla values.
        :param l1b:
        :param l2:
        :return:
        """

        # Step 1: Get a list of valid SSH tie points
        # This method will return a list of indices for all SSH observations
        # with an optional pre-filtering step
        filter_max_mss_offset_m = self.cfg.options.get("filter_max_mss_offset_m", None)
        use_ocean_wfm = self.cfg.options.get("use_ocean_wfm", False)
        ssh_tiepoint_indices = self.get_ssh_tiepoints_indices(l2, filter_max_mss_offset_m, use_ocean_wfm)

        # Verification that there is any ssh tie points
        # -> Will return all NaN sla if not
        if len(ssh_tiepoint_indices) == 0:
            all_nans = np.full(l2.n_records, np.nan)
            l2.sla.set_value(all_nans)
            l2.sla.set_uncertainty(all_nans)
            return np.isnan(l2.sla[:])

        # Step 2: A linear interpolation between lead elevations
        # -> will add properties `sla_raw`, `ssh_tiepoints` and `sla` to the instance
        matern_kernel = self.cfg.options.get("matern_kernel", None)
        white_noise_kernel = self.cfg.options.get("white_noise_kernel", None)
        sla, sla_unc = self.sla_from_gaussian_process(l2, ssh_tiepoint_indices, matern_kernel, white_noise_kernel)

        # Step 3: Apply sea-ice and land masks
        surface_types = self.cfg.options.get("surface_types_masks", [])
        sla, sla_unc = self.apply_surface_type_masks(sla, sla_unc, l2, surface_types)

        # Step 4: Modify the Level-2 data container with the result in-place
        l2.sla.set_value(sla)
        l2.sla.set_uncertainty(sla_unc)

        return np.isnan(l2.sla[:])

    @staticmethod
    def sla_from_gaussian_process(l2, ssh_tiepoint_indices, matern_kernel=None, white_noise_kernel=None):
        """
        Compute sea level anomaly be fitting lead tie points with a gaussian process. This method uses
        an optimization process based on the assumption
        :param l2:
        :param ssh_tiepoint_indices:
        :param matern_kernel:
        :param white_noise_kernel:
        :return:
        """

        # Step 1: Get the observed (noisy) sea surface elevations
        sla_raw = l2.elev[ssh_tiepoint_indices] - l2.mss[ssh_tiepoint_indices]
        x = np.arange(l2.n_records)
        y = np.array(sla_raw)

        # Step 2: Remove the mean value
        # -> SLA prediction will converge against mean SLA in the absence of
        #    ssh tie points
        mean_sla = float(np.nanmean(sla_raw))
        y -= mean_sla

        # Step 3: Prepare the input array for fitting
        x_fit = x[ssh_tiepoint_indices].reshape(-1, 1)
        y_fit = y.reshape(-1, 1)

        # Step 4: Establish the fitting kernel
        # The assumption here is that the covariance decreases with distance (Matern kernel) and
        # that the data is noisy (white noise kernel)
        if matern_kernel is None:
            logger.warning("SLAGaussianProcess: No input for matern kernel")
            matern_kernel = {}
        if white_noise_kernel is None:
            logger.warning("SLAGaussianProcess: No input for white noise kernel")
            white_noise_kernel = {}
        kernel = Matern(**matern_kernel) + WhiteKernel(**white_noise_kernel)

        # Step 5: Execute the Gaussian Process Regressor
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(x_fit, y_fit)

        # Step 6: Predict sla for the entire track and re-add mean value.
        # The uncertainty value is also output of the prediction
        x_pred = x.reshape(-1, 1)
        sla, sla_unc = gp.predict(x_pred, return_std=True)
        sla = sla.squeeze() + mean_sla

        # Return the two parameters
        return sla, sla_unc

    @property
    def l2_input_vars(self):
        """
        Mandatory property for Level2ProcessorStep children
        :return: list (str)
        """
        return ["surface_type", "elev", "mss"]

    @property
    def l2_output_vars(self):
        """
        Mandatory property for Level2ProcessorStep children
        :return: list (str)
        """
        return ["sla"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["sla"]


class SLASmoothedLinear(Level2ProcessorStep, SLABaseFunctionality):
    """
    Default implemetation of a smoothed interpolation of ssh tiepoints (with various filter options).
    This class will compute the sea level anomaly (sla)
    """

    def __init__(self, *args, **kwargs):
        """
        Init the class. Options will be passed to parent class
        (pysiral.l2proc.procsteps.Level2ProcessorStep)
        :param args:
        :param kwargs:
        """
        Level2ProcessorStep.__init__(self, *args, **kwargs)
        SLABaseFunctionality.__init__(self)

    def execute_procstep(self, l1b, l2):
        """
        Mandatory Level-2 processor method that will execute the processing step
        and modify the L2 data object in-place.
        This method will interpolate ssh tiepoints given by lead (+ ocean) elevations
        and smooth the result. Filter options are available to filter unreasonable
        sla values.
        :param l1b:
        :param l2:
        :return:
        """

        # Step 1: Get a list of valid SSH tie points
        # This method will return a list of indices for all SSH observations
        # with an optional pre-filtering step
        use_ocean_wfm = self.cfg.options.get("use_ocean_wfm", False)
        ssh_tiepoint_indices = self.get_ssh_tiepoints_indices(l2.surface_type, l2.elev, use_ocean_wfm)

        # Legacy MSS offset filter
        # Deprecated: Not a reliable filter, `tiepoint_elevation_filter` should be used
        # instead
        filter_max_mss_offset_m = self.cfg.options.get("filter_max_mss_offset_m", None)
        if filter_max_mss_offset_m is not None:
            logger.warning("filter_max_mss_offset_m is deprecated, use tiepoint_elevation_filter instead")
            sla_observed = l2.elev[ssh_tiepoint_indices] - l2.mss[ssh_tiepoint_indices]
            valid = np.where(np.abs(sla_observed) <= filter_max_mss_offset_m)[0]
            ssh_tiepoint_indices = ssh_tiepoint_indices[valid]

        # Offset from mean elevation filter
        tiepoint_elevation_filter = self.cfg.options.get("tiepoint_elevation_filter", None)
        if tiepoint_elevation_filter is not None:
            ssh_tiepoint_indices = self.tiepoint_elevation_sdev_filter(
                ssh_tiepoint_indices,
                l2.elev[:]-l2.mss[:],
                l2.footprint_spacing,
                **tiepoint_elevation_filter
            )

        # Verification that there is any ssh tie points
        # -> Will return all NaN sla if not
        if len(ssh_tiepoint_indices) == 0:
            logger.warning("No sea surface height tiepoint -> Empty SLA")
            all_nans = np.full(l2.n_records, np.nan)
            l2.sla.set_value(all_nans)
            l2.sla.set_uncertainty(all_nans)
            self.set_interpolation_uncertainty(l2, all_nans, all_nans, np.zeros(l2.n_records))
            return np.isnan(l2.sla[:])

        # Step 2: Calculate the SLA by
        smooth_filter_width_m = self.cfg.options.get("smooth_filter_width_m", np.nan)
        filter_width = self.get_filter_width(smooth_filter_width_m, l2.footprint_spacing)
        sla, sla_raw = self.smoothed_linear_interpolation_between_tiepoints(l2, ssh_tiepoint_indices, filter_width)

        # Step 3: Compute sea level anomaly uncertainty
        max_distance = self.cfg.options.get("uncertainty_tiepoints_distance_max", np.nan)
        sla_unc_min = self.cfg.options.get("uncertainty_minimum", np.nan)
        sla_unc_max = self.cfg.options.get("uncertainty_maximum", np.nan)
        sla_unc = self.calculate_sla_uncertainty(l2, max_distance, sla_unc_min, sla_unc_max, l2.footprint_spacing)

        # Step 3b: Compute sea level anomaly uncertainty with the CryoTEMPO Polar Ocean model
        # (measurement uncertainty at the tie points propagated to the smoothed sla)
        # NOTE: Separate variable for now, eventually a configurable alternative to step 3
        sigma_meas, population_masks = tiepoint_measurement_uncertainty(
            l2,
            ssh_tiepoint_indices,
            coefficients=self.cfg.options.get("uncertainty_coefficients", None),
            lrm_misfit_parameter=self.cfg.options.get("uncertainty_lrm_misfit_parameter", None)
        )
        sla_interp_unc, sla_interp_unc_flag = sla_interpolation_uncertainty(
            np.asarray(l2.latitude, dtype=float),
            np.asarray(l2.longitude, dtype=float),
            ssh_tiepoint_indices,
            sigma_meas,
            population_masks,
            coefficients=self.cfg.options.get("interpolation_uncertainty_coefficients", None)
        )

        # Step 4 (optional): Filter small marine segments surrounded by land
        # Note: This intends to remove small segments in fjords/channels for which
        #       the SLA computation is very likely not trustworthy
        mask = np.full(l2.n_records, False)
        if "marine_segment_filter" in self.cfg.options:
            minimum_lead_number = self.cfg.options.marine_segment_filter.get("minimum_lead_number", np.nan)
            use_ocean_wfm = self.cfg.options.get("use_ocean_wfm", False)
            filter_mask = marine_segment_filter(l2, minimum_lead_number, l2.footprint_spacing, use_ocean_wfm)
            mask = np.logical_or(mask, filter_mask)

        # Step 5 (optional): Filter SLA segments that are far away from the next SSH tie point
        if "tiepoint_maxdist_filter" in self.cfg.options:
            is_tiepoint = np.full(l2.n_records, False)
            is_tiepoint[ssh_tiepoint_indices] = True
            distance_threshold = self.cfg.options.tiepoint_maxdist_filter.get("maximum_distance_to_tiepoint", np.nan)
            edges_only = self.cfg.options.tiepoint_maxdist_filter.get("edges_only", False)
            filter_mask = self.tiepoint_maxdist_filter(l2, edges_only, distance_threshold, l2.footprint_spacing)
            mask = np.logical_or(mask, filter_mask)

        # Step 6: Apply filter (if any)
        if len(np.where(mask)) > 0:
            sla[mask] = np.nan
            sla_unc[mask] = np.nan

        # The interpolation uncertainty is only defined where there is a smoothed sla
        no_sla = np.isnan(sla)
        sla_interp_unc[no_sla] = np.nan
        sla_interp_unc_flag[no_sla] = 0

        # Step 7: Modify the Level-2 data container with the result in-place
        l2.sla.set_value(sla)
        l2.sla.set_uncertainty(sla_unc)

        l2.set_auxiliary_parameter("slar", "sea_level_anomaly_unfiltered", sla_raw)
        self.set_interpolation_uncertainty(l2, sla, sla_interp_unc, sla_interp_unc_flag)

        # Return the error status
        return np.isnan(l2.sla[:])

    @staticmethod
    def smoothed_linear_interpolation_between_tiepoints(l2, ssh_tiepoint_indices, filter_width):
        """
        The main SLA computation method in this class

        :param l2: Level-2 data container
        :param ssh_tiepoint_indices:
        :param filter_width:

        :return: None
        """

        # Step 1: Get the observed (noisy) sea surface elevations
        sla_raw = np.full(l2.n_records, np.nan)
        sla_raw[ssh_tiepoint_indices] = l2.elev[ssh_tiepoint_indices] - l2.mss[ssh_tiepoint_indices]
        non_tiepoints = np.isnan(sla_raw)

        # Step 2: Create a filtered version of the raw SLA, but don't interpolate yet
        # Use python implementation of IDL SMOOTH:
        # idl_smooth(x, w) equivalent to SMOOTH(x, w, /edge_truncate, /nan)
        sla_filter1 = idl_smooth(sla_raw, filter_width)
        sla_filter1[non_tiepoints] = np.nan

        # Step 3: Fill nans with linear interpolation and constant values at borders
        # python: fill_nan(x) = IDL: FILL_NAN(x, /NEIGHBOUR)
        sla_filter2 = fill_nan(sla_filter1)

        # Step 4: The sea level anomaly is the smoothed version
        # of the gap filled sla
        return idl_smooth(sla_filter2, filter_width), sla_raw

    @staticmethod
    def set_interpolation_uncertainty(l2, sla, sla_interp_unc, sla_interp_unc_flag):
        """
        Add the CryoTEMPO Polar Ocean uncertainty of the smoothed sea level anomaly
        and its quality flag (see `sla_interpolation_uncertainty`) to the l2 data object.

        The uncertainty is stored as the uncertainty of the auxiliary parameter
        `sea_level_anomaly_interpolation` (value: smoothed sla), so that it is available
        as `sea_level_anomaly_interpolation_uncertainty` (parameter names with `_uncertainty`
        always resolve to the uncertainty of the base parameter). For the same reason the
        flag name does not contain `_uncertainty`.

        :param l2: Level-2 data container
        :param sla: smoothed sea level anomaly (m)
        :param sla_interp_unc: uncertainty of the smoothed sea level anomaly (m)
        :param sla_interp_unc_flag: quality flag of the uncertainty

        :return: None
        """
        l2.set_auxiliary_parameter("slai", "sea_level_anomaly_interpolation", sla, sla_interp_unc)
        l2.set_auxiliary_parameter("slaif", "sea_level_anomaly_interpolation_flag", sla_interp_unc_flag)

    def calculate_sla_uncertainty(self, l2, max_distance, sla_unc_min, sla_unc_max,
                                  smooth_filter_width_footprint_size):
        """
        Components that add to sea surface anomaly uncertainty
            - mss uncertainty (if known)
            - uncertainty of lead elevations
            - distance to next lead tiepoint
        :param l2:
        :param max_distance:
        :param sla_unc_min:
        :param sla_unc_max:
        :param smooth_filter_width_footprint_size:
        :return:
        """

        # get tie point distance
        use_ocean_wfm = self.cfg.options.get("use_ocean_wfm", False)
        tiepoint_distance = self.get_tiepoint_distance_from_l2(l2, smooth_filter_width_footprint_size, use_ocean_wfm)

        # Compute the influence of distance to next tie points
        # in the range of 0: minimum influence to 1: maximum influence
        # It is assumed that the uncertainty has a quadratic dependence
        # on tie point distance
        tiepoint_distance_scalefact = tiepoint_distance / max_distance
        above_distance_limit = np.where(tiepoint_distance_scalefact > 1.)[0]
        tiepoint_distance_scalefact[above_distance_limit] = 1.
        tiepoint_distance_scalefact = tiepoint_distance_scalefact**2.

        # Compute the sla uncertainty based on a min/max approach scaled by factor
        sla_unc_range = sla_unc_max - sla_unc_min
        return sla_unc_min + sla_unc_range * tiepoint_distance_scalefact

    @property
    def l2_input_vars(self):
        """
        Mandatory property for Level2ProcessorStep children
        :return: list (str)
        """
        return ["surface_type", "elev", "mss"]

    @property
    def l2_output_vars(self):
        """
        Mandatory property for Level2ProcessorStep children
        :return: list (str)
        """
        return ["sla"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["sla"]


class SLARaw(Level2ProcessorStep, SLABaseFunctionality):
    """
    Get the raw SLA.
    This class will compute the sea level anomaly (sla)
    :sectionauthor: D. J. Brockley (UCL)
    """

    def __init__(self, *args, **kwargs):
        """
        Init the class. Options will be passed to parent class
        (pysiral.l2proc.procsteps.Level2ProcessorStep)
        :param args:
        :param kwargs:
        """

        # Init both base classes
        Level2ProcessorStep.__init__(self, *args, **kwargs)
        SLABaseFunctionality.__init__(self)

    def execute_procstep(self, l1b, l2):
        """
        Mandatory Level-2 processor method that will execute the processing step
        and modify the L2 data object in-place.
        This method will extracts raw SLA measurements by computing elev-mss
        at lead (+ ocean) locations.
        :param l1b:
        :param l2:
        :return:
        """

        # Step 1: Get a list of valid SSH tie points
        # This method will return a list of indices for all SSH observations
        # with an optional pre-filtering step
        # filter_max_mss_offset_m = self.cfg.options.get("filter_max_mss_offset_m", None)
        use_ocean_wfm = self.cfg.options.get("use_ocean_wfm", False)
        ssh_tiepoint_indices = self.get_ssh_tiepoints_indices(l2.surface_type, l2.elev, use_ocean_wfm)

        # Verification that there is any ssh tie points
        # -> Will return all NaN sla if not
        if len(ssh_tiepoint_indices) == 0:
            all_nans = np.full(l2.n_records, np.nan)
            l2.sla_raw.set_value(all_nans)
            l2.sla_raw.set_uncertainty(all_nans)
            return np.isnan(l2.sla_raw[:])

        # Step 2: Get sla = elev - mss and its measurement uncertainty
        coefficients = self.cfg.options.get("uncertainty_coefficients", None)
        lrm_misfit_parameter = self.cfg.options.get("uncertainty_lrm_misfit_parameter", None)
        sla_raw, sla_raw_unc = self.sla_from_raw_process(
            l2, ssh_tiepoint_indices, coefficients=coefficients, lrm_misfit_parameter=lrm_misfit_parameter
        )

        # Step 3: Apply sea-ice and land masks
        surface_types = self.cfg.options.get("surface_types_masks", [])
        sla_raw, sla_raw_unc = self.apply_surface_type_masks(sla_raw, sla_raw_unc, l2, surface_types)

        # Step 4: Modify the Level-2 data container with the result in-place
        l2.sla_raw.set_value(sla_raw)
        l2.sla_raw.set_uncertainty(sla_raw_unc)

        # Return the error status
        return np.isnan(l2.sla_raw[:])

    @staticmethod
    def sla_from_raw_process(l2, ssh_tiepoint_indices, coefficients=None, lrm_misfit_parameter=None):
        """
        Compute sea level anomaly and its measurement uncertainty at the ssh tie points
        (see `sla_raw_measurement_uncertainty`). Requires the SAMOSA+ auxiliary parameters
        (samosa_swh, samosa_misfit, samosa_mean_square_slope) in the l2 data object.

        :param l2:
        :param ssh_tiepoint_indices:
        :param coefficients: uncertainty model coefficients (default: SLA_RAW_UNCERTAINTY_COEFFICIENTS)
        :param lrm_misfit_parameter: name of the l2 parameter with the MLE4 misfit for LRM records
        :return:
        """

        # Step 1: Get the observed (noisy) sea surface elevations
        sla_raw = np.full(l2.n_records, np.nan)
        sla_raw[ssh_tiepoint_indices] = l2.elev[ssh_tiepoint_indices] - l2.mss[ssh_tiepoint_indices]

        # Step 2: Measurement uncertainty at the tie points
        sla_unc = np.full(l2.n_records, np.nan)
        sla_unc[ssh_tiepoint_indices], _ = tiepoint_measurement_uncertainty(
            l2, ssh_tiepoint_indices, coefficients=coefficients, lrm_misfit_parameter=lrm_misfit_parameter
        )

        # Return the two parameters
        return sla_raw, sla_unc

    @property
    def l2_input_vars(self):
        """
        Mandatory property for Level2ProcessorStep children
        :return: list (str)
        """
        return ["surface_type", "elev", "mss"]

    @property
    def l2_output_vars(self):
        """
        Mandatory property for Level2ProcessorStep children
        :return: list (str)
        """
        return ["sla"]

    @property
    def error_bit(self):
        return self.error_flag_bit_dict["sla"]


def get_rolling_standard_deviation_elevation_window(
        elevation: npt.NDArray,
        window_size: int,
        minimum_standard_deviation_m: float = 0.0,
        maximum_standard_deviation_m: float = np.inf,
        upper_limit_standard_deviation: float = 1.0,
        lower_limit_standard_deviation: float = 1.0
) -> tp.Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute a rolling window determined by

        upper_limit = rolling_mean + upper_limit_standard_deviation * rolling_standard_devation

        lower_limit = rolling_mean - lower_limit_standard_deviation * rolling_standard_devation

    the window size for the computation of rolling mean and standard devation is specified
    as number of points. The elevation array is assumed to be equidistant.

    :param elevation: Elevation array
    :param window_size: Number of points used to compute rolling mean and standard deviation
    :param minimum_standard_deviation_m: Minimum permissible rolling standard deviation
    :param maximum_standard_deviation_m: Maximum permissible rolling standard deviation
    :param upper_limit_standard_deviation: Multiplier for local standard deviation for upper limit
    :param lower_limit_standard_deviation: Multiplier for local standard deviation for lower limit

    :return: lower limit & upper limit with same dimension as input array.
    """

    x = np.arange(elevation.shape[0])
    data_fraction = min(window_size / float(x.shape[0]), 1.)
    elevation_rolling_mean = sm.nonparametric.lowess(elevation, x, frac=data_fraction, return_sorted=False)

    # Step 3: Compute the elevation standard deviation of trend-corrected elevation profile
    rolling_kwargs = dict(window=window_size, center=True, min_periods=1)
    ts = pd.Series(elevation - elevation_rolling_mean)
    elevation_rolling_sdev = ts.rolling(**rolling_kwargs).std()

    # Step 4: Set minimum value for standard deviation
    elevation_rolling_sdev = np.maximum(elevation_rolling_sdev, minimum_standard_deviation_m)
    elevation_rolling_sdev = np.minimum(elevation_rolling_sdev, maximum_standard_deviation_m)

    # Step 5: Find ssh tie-point indices outside the valid bounds
    upper_limit = elevation_rolling_mean + upper_limit_standard_deviation * elevation_rolling_sdev
    lower_limit = elevation_rolling_mean - lower_limit_standard_deviation * elevation_rolling_sdev

    return lower_limit.values, upper_limit.values
