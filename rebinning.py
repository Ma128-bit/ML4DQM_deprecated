import numpy as np
from collections import defaultdict
import copy


def draw_slice(start_angle, end_angle, image, center, value):
    """
    Given an image, and a center, it draws a pie-like slice
    between angles 'start_angle' and 'end_angle'. The pixels of
    in slice will have the value 'value'.
    """
    start_angle = np.deg2rad(start_angle)
    end_angle = np.deg2rad(end_angle)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # calculate the angle of the pixel
            angle = np.arctan2(y - center[1], x - center[0])
            if angle < 0:
                angle += 2 * np.pi

            # this angle check is for the special case were
            # 360 is crossed
            if start_angle > end_angle:
                if angle >= start_angle or angle < end_angle:
                    image[x, y] = value
            else:
                if angle >= start_angle and angle < end_angle:
                    image[x, y] = value

    return image


def draw_wheels(radius, sub_divisions):
    """
    Given a radius and a number of sub-divisions it draws a all the slices
    of the wheel. The main-slices are 10 degrees wide and the sub-divisions
    determine the number of slices in each of the main-slices.
    """
    img = np.zeros((2 * radius, 2 * radius), dtype=np.uint16)
    center = (radius - 0.5, radius - 0.5)

    start_angles = station_angles(sub_divisions)
    num_slices = len(start_angles) - 1

    # make the value unique for each slice
    for i in range(num_slices):
        value = i + 1
        img = draw_slice(start_angles[i], start_angles[i + 1], img, center, value)
    return img


def station_angles(sub_divisions):
    """
    The main 'true' slices are 10 degrees wide, divisions determines the sub-divisions of these 10 degree slices (e.g. division= 2 means 5 degree slices, division=3 means 3.33 degree slices)
    """
    start_angle = 45
    end_angle = 360 + 45 + 1

    start_angles = np.array([])
    for i in range(start_angle, end_angle, 10):
        part_angles = np.linspace(i, i + 10, sub_divisions, endpoint=False)
        start_angles = np.concatenate((start_angles, part_angles))

    # Remove any angles that are greater than 360+45 and convert to 0-360 range
    start_angles = start_angles[start_angles <= 360 + 45] % 360
    return start_angles


def make_disk(img, radius, value):
    center = (img.shape[0] // 2 - 0.5, img.shape[1] // 2 - 0.5)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if distance < radius:
                img[x, y] = value
    return img


def station3_boundaries():
    """
    Returns the inner radius (hole) and the outer radius of the station 3.
    Determined with the method of 'a occhio'
    """
    return 22.0, 43.5


def station_radii(divisions):
    # basically use station boundaries and make a list of radii
    # from boundaries[0] to boundaries[1] so that the total number of
    # divisions is respected
    inner_radius = station3_boundaries()[0]
    outer_radius = station3_boundaries()[1]

    radii = np.linspace(inner_radius, outer_radius, divisions + 1)
    # floor to the nearest 0.5
    radii = np.floor(radii * 2) / 2
    return radii


def make_concentric_disks(radius, radii):
    """
    Given a radius and a number of divisions, it draws a disk with concentric
    circles with the number of divisions.
    """
    img = np.zeros((2 * radius, 2 * radius), dtype=np.uint16)
    radii = np.sort(radii)[::-1]

    for i, r in enumerate(radii):
        img = make_disk(img, r, i + 257)
    return img


def assign_colors(region_map, num_colors=4):
    neighbors = _get_neighbors(region_map)
    color_assignment = _greedy_coloring(neighbors, num_colors)

    if None in color_assignment.values():
        return assign_colors(region_map, num_colors + 1)

    colored_map = np.vectorize(color_assignment.get)(region_map)
    return colored_map.astype(float) / np.max(colored_map)


def rebin(rebinning, image):
    """
    Example:
    arr1 = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [3, 3, 0]
    ])

    arr2 = np.array([
        [1.3, 2., 4.],
        [2., 6., 8.],
        [7., 6., 3.0]
    ])

    arr3 = [
        [1.3 2.  5. ]
        [2.  5.  7. ]
        [7.  7.  3. ]
    ]
    """
    flat_binning = rebinning.flatten()
    flat_image = image.flatten()

    # Get unique values and inverse indices from flat_binning, excluding 0
    non_zero_indices = flat_binning != 0
    _, inverse_indices = np.unique(flat_binning[non_zero_indices], return_inverse=True)

    # Sum the values in image corresponding to each unique value in binning
    sum_vals = np.bincount(inverse_indices, weights=flat_image[non_zero_indices])

    # Count the occurrences of each unique value in binning
    count_vals = np.bincount(inverse_indices)

    # Calculate the mean for each unique value
    mean_vals = sum_vals / count_vals

    arr3 = np.copy(image)

    # Fill in the calculated mean values where arr1 is not zero
    arr3[rebinning != 0] = mean_vals[inverse_indices]

    return arr3


def _get_neighbors(region_map):
    rows, cols = region_map.shape
    neighbors = defaultdict(set)

    for r in range(rows):
        for c in range(cols):
            region = region_map[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and region_map[nr, nc] != region:
                    neighbors[region].add(region_map[nr, nc])
                    neighbors[region_map[nr, nc]].add(region)

    return neighbors


def _greedy_coloring(neighbors, num_colors):
    color_assignment = {}
    for region, region_neighbors in neighbors.items():
        used_colors = {
            color_assignment[neighbor]
            for neighbor in region_neighbors
            if neighbor in color_assignment
        }
        available_colors = set(range(num_colors)) - used_colors
        color_assignment[region] = min(available_colors) if available_colors else None
    return color_assignment
