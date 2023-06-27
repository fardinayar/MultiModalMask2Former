import cv2
from detectron2 import utils
import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata

def read_depth_cityscapes(path, camera_calib):
    """
    Read cityscapes disparity map from `path` and convert it to depth map\
        according to the `camera_calib`.
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:,:,None] / 256 
    
    baseline = camera_calib['extrinsic']['baseline']
    fx = camera_calib['intrinsic']['fx']
    
    depth[depth > 0] = (fx * baseline) / depth[depth > 0]
    depth[depth > 200] = 200
    mean=4.284412912266262
    std=5.848670987278186
    depth = (depth-mean)/std
    return depth


def lidar_to_depth_image(path, camera_calib):
    """
    Read point cloud from `path` and convert it to depth map\
        according to the `camera_calib`.
    """
    fx, fy, cx, cy = camera_calib['intrinsic']['fx'],\
                    camera_calib['intrinsic']['fy'],\
                    camera_calib['intrinsic']['u0'],\
                    camera_calib['intrinsic']['v0']\
                        
    intrinsic = np.array([[fx, 0 , cx], 
                                [0 , fy, cy],
                                [0 , 0 , 1]])
    
    extrinsic = np.eye(4)[:3,:]
    
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    depth_image = create_depth_image(points, intrinsic.dot(extrinsic), (1024, 2048))
    
    missing_coords = np.argwhere(depth_image == -1)
    valid_coords = np.argwhere(depth_image != -1)

    valid_values = depth_image[valid_coords[:,0], valid_coords[:,1]]
    interpolated_values = griddata(valid_coords, valid_values, missing_coords, method='nearest')
    depth_image[missing_coords[:,0], missing_coords[:,1]] = interpolated_values
    return depth_image
    
def create_depth_image(point_cloud, camera_calibration, image_size):
    depth_image = np.ones(image_size) * -1
    homogeneous_points = np.column_stack((point_cloud, np.ones(len(point_cloud))))

    camera_points = np.dot(camera_calibration, homogeneous_points.T).T

    image_points = camera_points[:, :2] / camera_points[:, 2][:, np.newaxis]
    image_points = np.round(image_points).astype(int)

    valid_points_mask = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_size[1]) & \
                        (image_points[:, 1] >= 0) & (image_points[:, 1] < image_size[0])
    # print(valid_points_mask.sum())
    depth_image[image_points[valid_points_mask, 1], image_points[valid_points_mask, 0]] = \
        homogeneous_points[valid_points_mask, 2]

    return depth_image

if __name__ == '__main__':
    lidar_path = 'datasets/cityscapes/pointcloud/test/berlin/berlin_000000_000019_pointcloud.pcd'
    calib_path = 'datasets/cityscapes/camera/test/berlin/berlin_000000_000019_camera.json'
    with open(calib_path, 'r') as camera_file:
            camera_calib = json.loads(camera_file.read())
            
    depth = lidar_to_depth_image(lidar_path, camera_calib)
    plt.imshow(depth)
    plt.show()