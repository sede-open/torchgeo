# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
import random
import re
import warnings
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from geopandas import GeoDataFrame
from rtree.index import Index, Property
from shapely.geometry import box
from torch.utils.data import Sampler
from tqdm import tqdm

from ..datasets import BoundingBox, GeoDataset
from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips


def load_file(path: str | GeoDataFrame) -> GeoDataFrame:
    """Load a file from the given path.

    Parameters:
    path (str or GeoDataFrame): The path to the file or a GeoDataFrame object.

    Returns:
    GeoDataFrame: The loaded file as a GeoDataFrame.

    Raises:
    None

    """
    if isinstance(path, GeoDataFrame):
        return path
    if path.endswith('.feather'):
        print(f'Reading feather file: {path}')
        return gpd.read_feather(path)
    else:
        print(f'Reading shapefile: {path}')
        return gpd.read_file(path)


def _get_regex_groups_as_df(dataset: GeoDataset, hits: list) -> pd.DataFrame:
    """Extracts the regex metadata from a list of hits.

    Args:
        dataset (GeoDataset): The dataset to sample from.
        hits (list): A list of hits.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted file metadata.
    """
    has_filename_regex = hasattr(dataset, 'filename_regex')
    if has_filename_regex:
        filename_regex = re.compile(dataset.filename_regex, re.VERBOSE)
    file_metadata = []
    for hit in hits:
        if has_filename_regex:
            match = re.match(filename_regex, str(hit.object))
            if match:
                meta = match.groupdict()
            else:
                meta = {}
        else:
            meta = {}
        meta.update(
            {
                'minx': hit.bounds[0],
                'maxx': hit.bounds[1],
                'miny': hit.bounds[2],
                'maxy': hit.bounds[3],
                'mint': hit.bounds[4],
                'maxt': hit.bounds[5],
            }
        )
        file_metadata.append(meta)
    return pd.DataFrame(file_metadata)


class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: BoundingBox | None = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)
        else:
            # Only keep hits unique in the spatial dimension if return_as_ts is enabled
            # else keep hits that are unique in both spatial and temporal dimensions
            filter_indices = slice(0, 4) if dataset.return_as_ts else slice(0, 6)

            self.index = Index(interleaved=False, properties=Property(dimension=3))
            if isinstance(roi, list):
                for area in roi:
                    hits = dataset.index.intersection(tuple(area), objects=True)
                    for hit in hits:
                        bbox = BoundingBox(*hit.bounds) & area
                        # Filter hits
                        if tuple(bbox)[filter_indices] not in [
                            tuple(item.bounds[filter_indices])
                            for item in list(
                                self.index.intersection(tuple(area), objects=True)
                            )
                        ]:
                            self.index.insert(hit.id, tuple(bbox), hit.object)
            else:
                hits = dataset.index.intersection(tuple(roi), objects=True)
                for hit in hits:
                    bbox = BoundingBox(*hit.bounds) & roi
                    # Filter hits
                    if tuple(bbox)[filter_indices] not in [
                        tuple(item.bounds[filter_indices])
                        for item in list(self.index.intersection(tuple(roi), objects=True))
                    ]:
                        self.index.insert(hit.id, tuple(bbox), hit.object)

        print(f"Index Size: {self.index.get_size()}")
        self.res = dataset.res
        self.roi = roi
        self.dataset = dataset

    @abc.abstractmethod
    def get_chips(self, *args: Any, **kwargs: Any) -> GeoDataFrame:
        """Determines the way to get the extend of the chips (samples) of the dataset.

        Should return a GeoDataFrame with the extend of the chips with the columns
        geometry, minx, miny, maxx, maxy, mint, maxt, fid. Each row is a chip.
        """

    def filter_chips(
        self,
        filter_by: str | GeoDataFrame,
        predicate: str = 'intersects',
        action: str = 'keep',
    ) -> None:
        """Filter the default set of chips in the sampler down to a specific subset by specifying files 
        supported by geopandas such as shapefiles, geodatabases or feather files.

        Args:
            filter_by: The file or geodataframe for which the geometries will be used during filtering
            predicate: Predicate as used in Geopandas sindex.query_bulk
            action: What to do with the chips that satisfy the condition by the predicacte.
            Can either be "drop" or "keep".
        """
        prefilter_leng = len(self.chips)
        filtering_gdf = load_file(filter_by).to_crs(self.dataset.crs)

        if action == 'keep':
            self.chips = self.chips.iloc[
                list(
                    set(
                        self.chips.sindex.query_bulk(
                            filtering_gdf.geometry, predicate=predicate
                        )[1]
                    )
                )
            ].reset_index(drop=True)
        elif action == 'drop':
            self.chips = self.chips.drop(
                index=list(
                    set(
                        self.chips.sindex.query_bulk(
                            filtering_gdf.geometry, predicate=predicate
                        )[1]
                    )
                )
            ).reset_index(drop=True)

        self.chips.fid = self.chips.index
        print(f'Filter step reduced chips from {prefilter_leng} to {len(self.chips)}')
        assert not self.chips.empty, 'No chips left after filtering!'

    def set_worker_split(self, total_workers: int, worker_num: int) -> None:
        """Splits the chips in n equal parts for the number of workers and keeps the set of
        chips for the specific worker id, convenient if you want to split the chips across
        multiple dataloaders for multi-gpu inference.

        Args:
            total_workers: The total number of parts to split the chips
            worker_num: The id of the worker (which part to keep), starts from 0

        """
        self.chips = np.array_split(self.chips, total_workers)[worker_num]

    def save(self, path: str, driver: str = "Shapefile") -> None:
        """Save the chips as a shapefile or feather file"""
        if path.endswith('.feather'):
            self.chips.to_feather(path)
        else:
            self.chips.to_file(path, driver=driver)

    def save_hits(self, path: str, driver: str = "Shapefile") -> None:
        """Save the hits as a shapefile or feather file"""
        bounds = []
        for hit in self.hits:
            bbox = BoundingBox(*hit.bounds)
            minx, maxx, miny, maxy, mint, maxt  = tuple(bbox)
            bound = {
                'geometry': box(minx, miny, maxx, maxy),
                'minx': minx,
                'miny': miny,
                'maxx': maxx,
                'maxy': maxy,
                'mint': mint,
                'maxt': maxt,
                'hit_id': hit.id
            }
            bounds.append(bound)

        bounds_gdf = GeoDataFrame(bounds, crs=self.dataset.crs)
        if path.endswith('.feather'):
            bounds_gdf.to_feather(path)
        else:
            bounds_gdf.to_file(path, driver=driver)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _, chip in self.chips.iterrows():
            yield BoundingBox(
                chip.minx, chip.maxx, chip.miny, chip.maxy, chip.mint, chip.maxt
            )

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.chips)


class RandomGeoSampler(GeoSampler):
    """Differs from TorchGeo's official RandomGeoSampler in that it can sample SITS data.

    Documentation from TorchGeo:
    Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        num_chips = 0
        self.hits = []
        areas = []
        # use only geospatial bounds for filering in case return_as_ts is set to True
        filter_indices = slice(0,4) if dataset.return_as_ts else slice(0,6)

        if isinstance(self.roi, list):
            for area in self.roi:
                for hit in self.index.intersection(tuple(area), objects=True):
                    # Filter out hits in the index that share the same extent
                    if  hit.bounds[filter_indices] not in [ht.bounds[filter_indices] for ht in self.hits]:
                        bounds = BoundingBox(*hit.bounds)
                        if (
                            bounds.maxx - bounds.minx >= self.size[1]
                            and bounds.maxy - bounds.miny >= self.size[0]
                        ):
                            if bounds.area > 0:
                                rows, cols = tile_to_chips(bounds, self.size)
                                num_chips += rows * cols
                            else:
                                num_chips += 1
                            self.hits.append(hit)
                            areas.append(bounds.area)
        else:
            for hit in self.index.intersection(tuple(self.roi), objects=True):
                # Filter out hits in the index that share the same extent
                if  hit.bounds[filter_indices] not in [ht.bounds[filter_indices] for ht in self.hits]:
                    bounds = BoundingBox(*hit.bounds)
                    if (
                        bounds.maxx - bounds.minx >= self.size[1]
                        and bounds.maxy - bounds.miny >= self.size[0]
                    ):
                        if bounds.area > 0:
                            rows, cols = tile_to_chips(bounds, self.size)
                            num_chips += rows * cols
                        else:
                            num_chips += 1
                        self.hits.append(hit)
                        areas.append(bounds.area)
        if length is not None:
            num_chips = length
        self.length = num_chips

        print(f"Unique geospatial file hits: {len(self.hits)}")

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

        self.chips = self.get_chips(num_samples=num_chips)
    
    def get_chips(self, num_samples) -> GeoDataFrame:
        chips = []
        while len(chips) < num_samples:
            if isinstance(self.roi, list):
                # Choose a random ROI, weighted by area
                idx = torch.multinomial(torch.tensor([roi.area for roi in self.roi], dtype=torch.float), 1)
                roi = self.roi[idx]
            else:
                roi = self.roi

            # Choose a random bounding box within the ROI
            bbox = get_random_bounding_box(roi, self.size, self.res)
            minx, maxx, miny, maxy, mint, maxt = tuple(bbox)

            # Find file hits in the index for chosen bounding box
            hits = [hit for hit in list(self.index.intersection(tuple(roi), objects=True)) if BoundingBox(*hit.bounds).area > 0]

            # if the bounding box has no file hit, dont include it in the samples
            if len(hits) == 0:
                continue
            else:
                # Choose a random hit
                hit = random.choice(hits)

                # in case we are randomly sampling also the temporal dimension, get
                # mint, maxt from the randomly chosen hit
                if self.dataset.return_as_ts:
                    mint = self.dataset.bounds.mint
                    maxt = self.dataset.bounds.maxt
                else:
                    mint = hit.bounds[-2]
                    maxt = hit.bounds[-1]
                
                chip = {
                    'geometry': box(minx, miny, maxx, maxy),
                    'minx': minx,
                    'miny': miny,
                    'maxx': maxx,
                    'maxy': maxy,
                    'mint': mint,
                    'maxt': maxt,
                    'hit_id': hit.id
                }
                chips.append(chip)
        
        print('creating geodataframe... ')
        chips_gdf = GeoDataFrame(chips, crs=self.dataset.crs)
        chips_gdf['fid'] = chips_gdf.index

        print("Number of Unique Hit IDs: {}".format(chips_gdf["hit_id"].unique().shape[0]))
        print("Number of Unique Timestamps: {}".format(chips_gdf["mint"].unique().shape[0]))

        return chips_gdf


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        hits = self.index.intersection(tuple(self.roi), objects=True)
        df_path = _get_regex_groups_as_df(self.dataset, hits)

        # Filter out tiles smaller than the chip size
        self.df_path = df_path[
            (df_path.maxx - df_path.minx >= self.size[1])
            & (df_path.maxy - df_path.miny >= self.size[0])
        ]

        # Filter out hits in the index that share the same extent
        if self.dataset.return_as_ts:
            self.df_path.drop_duplicates(
                subset=['minx', 'maxx', 'miny', 'maxy'], inplace=True
            )
        else:
            self.df_path.drop_duplicates(
                subset=['minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt'], inplace=True
            )

        self.chips = self.get_chips()

    def get_chips(self) -> GeoDataFrame:
        print('generating samples... ')
        optional_keys = set(self.df_path.keys()) - set(
            ['geometry', 'minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt']
        )
        chips = []
        for _, row in tqdm(self.df_path.iterrows(), total=len(self.df_path)):
            bounds = BoundingBox(
                row.minx, row.maxx, row.miny, row.maxy, row.mint, row.maxt
            )
            rows, cols = tile_to_chips(bounds, self.size, self.stride)

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    if self.dataset.return_as_ts:
                        mint = self.dataset.bounds.mint
                        maxt = self.dataset.bounds.maxt
                    else:
                        mint = bounds.mint
                        maxt = bounds.maxt

                    chip = {
                        'geometry': box(minx, miny, maxx, maxy),
                        'minx': minx,
                        'miny': miny,
                        'maxx': maxx,
                        'maxy': maxy,
                        'mint': mint,
                        'maxt': maxt,
                    }
                    for key in optional_keys:
                        if key in row.keys():
                            chip[key] = row[key]

                    chips.append(chip)           

        if chips:
            print('creating geodataframe... ')
            chips_gdf = GeoDataFrame(chips, crs=self.dataset.crs)
            chips_gdf['fid'] = chips_gdf.index

        else:
            warnings.warn('Sampler has no chips, check your inputs')
            chips_gdf = GeoDataFrame()
        return chips_gdf


class PreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.NonGeoDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self, dataset: GeoDataset, roi: BoundingBox | None = None, shuffle: bool = False
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.3

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
        """
        super().__init__(dataset, roi)
        self.shuffle = shuffle

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            self.hits.append(hit)

        self.chips = self.get_chips()

    def get_chips(self) -> GeoDataFrame:
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        chips = []
        for idx in generator(len(self.hits)):
            minx, maxx, miny, maxy, mint, maxt = self.hits[idx].bounds
            chip = {
                'geometry': box(minx, miny, maxx, maxy),
                'minx': minx,
                'miny': miny,
                'maxx': maxx,
                'maxy': maxy,
                'mint': mint,
                'maxt': maxt,
            }
            chips.append(chip)

        print('creating geodataframe... ')
        chips_gdf = GeoDataFrame(chips, crs=self.dataset.crs)
        chips_gdf['fid'] = chips_gdf.index

        return chips_gdf
