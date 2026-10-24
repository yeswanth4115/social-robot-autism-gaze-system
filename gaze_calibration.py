import numpy as np


def gaze_coordinates(features):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (4, 7):
        raise ValueError("Expected gaze features with 4 or 7 columns.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Gaze features must contain only finite values.")

    return np.column_stack(
        (
            np.mean(values[:, [0, 2]], axis=1),
            np.mean(values[:, [1, 3]], axis=1),
        )
    )


def fit_four_point_homography(features, targets):
    source = gaze_coordinates(features)
    target = np.asarray(targets, dtype=float)

    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("Four-point calibration requires exactly four source and target points.")
    if not np.all(np.isfinite(target)):
        raise ValueError("Calibration targets must contain only finite values.")

    edges = np.roll(source, -1, axis=0) - source
    next_edges = np.roll(edges, -1, axis=0)
    cross_products = (
        edges[:, 0] * next_edges[:, 1]
        - edges[:, 1] * next_edges[:, 0]
    )
    if not (
        np.all(cross_products > 1e-8)
        or np.all(cross_products < -1e-8)
    ):
        raise ValueError(
            "The four gaze points must form a non-degenerate convex quadrilateral."
        )

    equations = []
    values = []
    for (x, y), (u, v) in zip(source, target):
        equations.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        equations.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.extend((u, v))

    system = np.asarray(equations, dtype=float)
    if np.linalg.matrix_rank(system) != 8:
        raise ValueError(
            "The four gaze points do not uniquely determine a calibration mapping."
        )

    coefficients = np.linalg.solve(system, np.asarray(values, dtype=float))
    homography = np.append(coefficients, 1.0).reshape(3, 3)
    if not np.all(np.isfinite(homography)) or abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("The four gaze points produced an invalid calibration mapping.")

    predicted = transform_gaze_coordinates(source, homography)
    denominators = np.column_stack((source, np.ones(4))) @ homography[2, :]
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError(
            "The four gaze points produce an unstable calibration mapping."
        )
    if not np.allclose(predicted, target, rtol=1e-6, atol=1e-4):
        raise ValueError("The four-point calibration mapping did not fit its targets.")

    return homography


def transform_gaze_coordinates(coordinates, homography):
    points = np.asarray(coordinates, dtype=float)
    matrix = np.asarray(homography, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected gaze coordinates with 2 columns.")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite 3-by-3 calibration mapping.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Gaze coordinates must contain only finite values.")

    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ matrix.T
    denominators = mapped[:, 2]
    if np.any(np.abs(denominators) < 1e-12):
        raise ValueError("The gaze coordinate maps to an invalid screen position.")

    return mapped[:, :2] / denominators[:, np.newaxis]
