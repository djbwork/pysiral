# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from typing import Any, Dict

from pysiral.l1data import Level1bData
from pysiral.l1preproc.procitems import L1PProcItem


class CryoTEMPOFES2022bTide(L1PProcItem):
    """
    A class to update tides specifically for the production of Cryo-TEMPO data.
    Here, the tide information is extracted from the FES2022b tide model files with identical
    temporal coverage as CryoSat-2 ICE Level-1 files. Thus, only the correct file has to
    be identified and then mapped to the Level-1 data object.

    NOTES:
        - This Level-1 processor item is purpose built for the Cryo-TEMPO
          production workflow.
        - This Level-1 processor item can only be run at stage `post_source_file`,
          because otherwise the mapping strategy does not work.
    """

    def __init__(self, **cfg: Dict[str, Any]):
        super(CryoTEMPOFES2022bTide, self).__init__(**cfg)

    def apply(self, l1: Level1bData) -> None:
        """
        Replace the tide values in the l1b data object with the FES2022b tide model.
        This is done by reading tide data extracted from a file with the same coverage
        as the

        :param l1: The Level-1 data container

        :return:
        """

        # Ensure that the Level-1 data is only based on one CryoSat-2 file
        if len(l1.source_filepaths) != 1 and l1.info.mission != "cryosat2":
            msg = "CryoTEMPOFES2022bTide can only be applied to Level-1 data objects based on one CryoSat-2 file"

        # Construct
        fes_file_name = self.get_fes_filename(l1.source_filepaths[0])
