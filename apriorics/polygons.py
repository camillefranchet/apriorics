import json
from numbers import Number
from os import PathLike
from typing import Any, Optional

import numpy as np
from nptyping import NDArray
from pathaia.util.types import NDBoolMask
from rasterio import features
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, shape


def get_reduced_coords(
    coords: NDArray[(Any, 2), Number], angle_th: float, distance_th: float
) -> NDArray[(Any, 2), int]:
    r"""
    Given polygon vertices coordinates, deletes those that are too close or that form
    a too small angle.

    Args:
        coords: array of coordinates.
        angle_th: minimum angle (in degrees) formed by 3 consecutives vertices. If the
            angle is too small, the middle vertex will be deleted.
        distance_th: minimum distance between vertices. If 2 consecutive sides of the
            polygon are too small, the middle vertex will be deleted.

    Returns:
        Array of polygon coordinates with small and flat sides pruned.
    """
    vector_rep = np.diff(coords, axis=0)
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(len(vector_rep) - 1):
            if len(coords) - len(points_removed) == 3:
                break
            v01 = vector_rep[i]
            v12 = vector_rep[i + 1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
                continue
            angle = np.arccos(np.dot(v01, v12) / (d01 * d12))
            if angle < angle_th_rad:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
        coords = np.delete(coords, points_removed, axis=0)
        vector_rep = np.diff(coords, axis=0)
    return coords.astype(int)


def reduce_polygon(
    polygon: Polygon, angle_th: float = 0, distance_th: float = 0
) -> Polygon:
    r"""
    Given a :class:`shapely.geometry.Polygon`, delete vertices that create small or
    flat sides on the interior and on the exterior.

    Args:
        polygon: input polygon.
        angle_th: minimum angle (in degrees) formed by 3 consecutives vertices. If the
            angle is too small, the middle vertex will be deleted.
        distance_th: minimum distance between vertices. If 2 consecutive sides of the
            polygon are too small, the middle vertex will be deleted.

    Returns:
        Reduced polygon.
    """
    ext_poly_coords = get_reduced_coords(
        np.array(polygon.exterior.coords[:]), angle_th, distance_th
    )
    interior_coords = [
        get_reduced_coords(np.array(interior.coords[:]), angle_th, distance_th)
        for interior in polygon.interiors
    ]
    return Polygon(ext_poly_coords, interior_coords)


def mask_to_polygons_layer(
    mask: NDBoolMask, angle_th: float = 2, distance_th: float = 3
) -> MultiPolygon:
    """
    Convert mask array into :class:`shapely.geometry.MultiPolygon`.

    Args:
        mask: input mask array.

    Returns:
        :class:`shapely.geometry.MultiPolygon` where polygons are extracted from
        positive areas in the mask.
    """
    all_polygons = []
    for sh, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
        all_polygons.append(
            reduce_polygon(shape(sh), angle_th=angle_th, distance_th=distance_th)
        )

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def hovernet_to_wkt(
    infile: PathLike, outfile: PathLike, slide_height: Optional[int] = None
):
    """
    Take a hovernet json output file and convert it to a wkt file. Optionally convert
    coordinates to cytomine format.

    Args:
        infile: path to input hovernet json file.
        outfile: path to output wkt file.
        slide_height: height of the slide on which hovernet was used. If given,
            coordinates will be converted to cytomine format (`y = slide_height - y`).
    """
    with open(infile, "r") as f:
        hovernet_dict = json.load(f)
    polygons = []
    for k in hovernet_dict["nuc"]:
        polygon = Polygon(hovernet_dict["nuc"][k]["contour"])
        if polygon.is_valid:
            polygons.append(polygon)
    polygons = MultiPolygon(polygons)
    if slide_height is not None:
        polygons = affine_transform(polygons, [1, 0, 0, -1, 0, slide_height])

    with open(outfile, "w") as f:
        f.write(polygons.wkt)
