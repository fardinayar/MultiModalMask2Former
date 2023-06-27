import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import interp2d
import copy
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances_argmin
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def generate_point_cloud(depth_map, camera_matrix, fov, subdivisions, resolution):
    # Extract camera intrinsic parameters from the calibration matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Calculate the number of points per horizontal scanline
    num_points = subdivisions


    # Calculate the vertical and horizontal angles for each point
    vertical_angles = np.linspace(-fov/2, fov/2, num_points)
    horizontal_angles = np.arange(-90, 90, resolution)
    v, h = np.meshgrid(vertical_angles, horizontal_angles)

    depth_map = depth_map.astype('float32')[:,:, None]
    h, w = depth_map.shape[:2]
    d=o3d.geometry.Image(depth_map)
    pt = o3d.geometry.PointCloud.create_from_depth_image(d,
                                                        o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy),
                                                        np.array([[1,0,0,0], [0,-1,0,0], [0, 0, -1, 0], [0,0,0,1]]),
                                                        depth_scale=1,
                                                        stride=1)

    numpy_points = np.array(copy.deepcopy(pt.points))
    num_points = numpy_points[~np.isnan(numpy_points)]
    distances = np.sqrt(numpy_points[:, 0]**2 + numpy_points[:, 1]**2 + numpy_points[:, 2]**2)
    numpy_points = numpy_points[np.logical_and(distances>5, distances<120)]
    # Calculate the horizontal angles for each point
    points_azimuths = np.arctan2(numpy_points[:, 0].copy(), -numpy_points[:, 2].copy()) * 180 / np.pi
    # Calculate the vertical angles for each point
    points_vertical_angles = np.arctan2(numpy_points[:, 1].copy(), -numpy_points[:, 2].copy()) * 180 / np.pi
    digitized_points_azimuths_bins = np.digitize(points_azimuths, horizontal_angles)
    digitized_points_vertical_angles_bins = np.digitize(points_vertical_angles, vertical_angles)
    points_azimuths_error = horizontal_angles[digitized_points_azimuths_bins] - points_azimuths
    points_vertical_angles_error = vertical_angles[digitized_points_vertical_angles_bins] - points_vertical_angles
    points_digitize_error = points_azimuths_error ** 2 + points_vertical_angles_error **2
    df = pd.DataFrame(numpy_points, columns=["x", "y", "z"])
    df['azimuths_bins'] = digitized_points_azimuths_bins
    df['vertical_bins'] = digitized_points_vertical_angles_bins
    df['error'] = points_digitize_error
    d = df.loc[df.groupby(['azimuths_bins', 'vertical_bins']).error.idxmin()].reset_index(drop=True)

    # Calculate the distances from the origin to each point


    return d[['x', 'y', 'z']].values

if __name__ == '__main__':
    import os
    import json
    import tqdm
    import glob
    import open3d as o3d
    import sys
    # Configs for Velodyne 64-HD
    vertical_fov = 26.8 
    subdivisions = 64
    horizontal_resolution = 0.08
    dataset_path = 'datasets/cityscapes'
    
    disparities= glob.glob(os.path.join(dataset_path, 'disparity/*/*/*.png'))
    for disparity_path in tqdm.tqdm(disparities):
        camera_calib_path = disparity_path.replace('disparity', 'camera').replace('png', 'json')
        assert os.path.isfile(camera_calib_path), "Camera file not found"
        with open(camera_calib_path, 'r') as camera_file:
            camera_file = json.loads(camera_file.read())
        
        baseline = camera_file['extrinsic']['baseline']
        fx = camera_file['intrinsic']['fx']
        fy = camera_file['intrinsic']['fy']
        u0 = camera_file['intrinsic']['u0']
        v0 = camera_file['intrinsic']['v0']
            
        depth_map = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)
        depth_map = (depth_map)/256 + 1e-10

        depth_map = (fx * baseline) / depth_map

        calibration_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])  # Example calibration matrix

        lidar= generate_point_cloud(depth_map, calibration_matrix, vertical_fov, subdivisions, horizontal_resolution)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar)
        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()        
        point_cloud_path = disparity_path.replace('disparity', 'pointcloud').replace('png', 'pcd')
        point_cloud_dict = os.path.dirname(point_cloud_path)
        if not os.path.isdir(point_cloud_dict):
            os.makedirs(point_cloud_dict)
        bool = o3d.io.write_point_cloud(point_cloud_path, pcd)
        assert bool, 'Something went wrong saving {}'.format(point_cloud_path)
        print('Saved {}'.format(point_cloud_path))
        sys.stdout.flush()


