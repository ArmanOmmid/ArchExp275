"""Visualization utilies."""

# You can use other visualization from previous homeworks, like Open3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def show_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([0, 4])
    ax.scatter(points[:, 0], points[:, 2], points[:, 1])

def compare_points(points1, points2, scale=1, translate=[0, 0, 0]):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xt, yt, zt = translate
    ax.set_xlim3d([-2*scale + xt, 2*scale + xt])
    ax.set_ylim3d([-2*scale + yt, 2*scale + yt])
    ax.set_zlim3d([0*scale + zt, 4*scale + zt])
    ax.scatter(points1[:, 0], points1[:, 2], points1[:, 1])
    ax.scatter(points2[:, 0], points2[:, 2], points2[:, 1])


"""Metric and visualization."""
def compute_rre(R_est: np.ndarray, R_gt: np.ndarray):
    """Compute the relative rotation error (geodesic distance of rotation)."""
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)
    # relative rotation error (RRE)
    rre = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt) - 1), -1.0, 1.0))
    return rre


def compute_rte(t_est: np.ndarray, t_gt: np.ndarray):
    assert t_est.shape == (3,), 't_est: expected shape (3,), received shape {}.'.format(t_est.shape)
    assert t_gt.shape == (3,), 't_gt: expected shape (3,), received shape {}.'.format(t_gt.shape)
    # relative translation error (RTE)
    rte = np.linalg.norm(t_est - t_gt)
    return rte


VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(
    image, center, size, rotation, extrinsic, intrinsic, color=(0, 1, 0), thickness=1
):
    """Draw a projected 3D bounding box on the image.

    Args:
        image (np.ndarray): [H, W, 3] array.
        center: [3]
        size: [3]
        rotation (np.ndarray): [3, 3]
        extrinsic (np.ndarray): [4, 4]
        intrinsic (np.ndarray): [3, 3]
        color: [3]
        thickness (int): thickness of lines
    Returns:
        np.ndarray: updated image.
    """
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)

    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image