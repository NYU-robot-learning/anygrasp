import os 
import math
from typing import List

import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from socket import ZmqSocket
from utils import get_3d_points, visualize_cloud_geometries
from camera import CameraParameters
from image_processor import OwlVITProcessor, LangSAMProcessor, SamProcessor


class ObjectHandler():
    def __init__(
        self,
        cfgs
    ):
        self.cfgs = cfgs
        self.grasping_model = AnyGrasp(self.cfgs)
        self.grasping_model.load_net()

        self.socket = ZmqSocket(self.cfgs) 

        self.owl_vit = OwlVITProcessor()
        self.sam = SamProcessor()
        self.lang_sam = LangSAMProcessor

    def receive_input(self):
        # Reading color array
        colors = self.socket.recv_array()
        self.socket.send_string("RGB received")

        # Depth data
        depths = self.socket.recv_array()
        self.socket.send_string("depth received")

        # Camera Intrinsics
        fx, fy, cx, cy, head_tilt = self.socket.recv_array()
        self.socket.send_string("intrinsics received")

        # Object query
        self.query = self.socket.recv_string()
        self.socket.send_string("text query received")
        print(f"text - {self.query}")

        # action -> ["pick", "place"]
        self.action = self.socket.recv_string()
        self.socket.send_string("Mode received")
        print(f"mode - {self.action}")
        
        # Camera Parameters
        image = Image.fromarray(colors)
        colors = colors / 255.0
        head_tilt = head_tilt / 100
        self.cam = CameraParameters(fx, fy, cx, cy,
                                   head_tilt,
                                   image,
                                   colors,
                                   depths)


    def manipulate(self):
        """
            Wrapper for grasping and placing
        """

        tries = 1 
        retry = True
        while tries > 0:
            action = self.receive_input()

            # Directory for saving visualisations
            self.save_dir = self.cfgs.environment + "/" + self.query + "/anygrasp/" + self.cfgs.method
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # Just for comparing owl_vit vs lang sam this wont be used for further processing, Can be commented if not required
            bbox =  self.owl_vit.get_bounding_box(self.cam.image, self.query, visualize_box = True,
                                                  save_file=self.cfgs.environment + "/" + \
                                                  self.query + "/anygrasp/" + self.cfgs.method)

            # Object Segmentaion Mask
            seg_mask, bbox = self.lang_sam_segment(self.cam.image, self.query, visualize_box = True,
                                                   save_file = self.cfgs.environment + "/" + self.query + \
                                                               "/anygrasp/" + self.cfgs.method + \
                                                               "/lseg_detection_" + str(tries) + ".jpg"
                                                   )
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
            print(bbox)

            # Center the robot
            if tries == 1:
                self.center_robot(bbox)
                continue
            
            while retry:
                points = get_3d_points()

                if action == "place":
                    self.place()
                else:
                    self.pick()

    def center_robot(self, bbox: List[int]):
        ''' 
            Center the robots base and camera to face the center of the Object Bounding box
        '''

        bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = bbox

        bbox_center = [int((bbox_x_min + bbox_x_max)/2), int((bbox_y_min + bbox_y_max)/2)]
        depth_obj = self.depths[bbox_center[1], bbox_center[0]]
        print(f"{self.query} height and depth: {((bbox_y_max - bbox_y_min) * depth_obj)/self.fy}, {depth_obj}")

        # base movement
        dis = (bbox_center[0] - self.cx)/self.fx * depth_obj
        print(f"d displacement {dis}")

        # camera tilt
        tilt = math.atan((bbox_center[1] - self.cy)/self.fy)
        print(f"y tilt {tilt}")

        self.socket.send_msg([-dis], [-tilt], [0, 0, 1])
        self.socket.send_string("Now you received the base and haed trans, good luck.")

    def pickup(self):
        pass

    def place(
        self,
        seg_mask: np.ndarray,
        points: np.ndarray,
        save_dir: str
    ):
        print("placing mode")
        print(seg_mask)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(seg_mask, plt.gca())
        # # mask_img.save(cfgs.environment + "/" + text + "/anygrasp/" + cfgs.method +  "/placing_segmentation.jpg")
        # # show_box(input_box, plt.gca())
        # plt.axis('off')
        # plt.savefig(cfgs.environment + "/" + text + "/anygrasp/" + cfgs.method +  "/placing_segmentation.jpg")
        # # plt.show()
        # print(points_x.shape)

        points_x, points_y, points_z = points[:, 0], points[:, 1], points[:, 2]
        flat_x, flat_y, flat_z = points_x.reshape(-1), -points_y.reshape(-1), -points_z.reshape(-1)

        # Removing all points whose depth is zero(undetermined)
        zero_depth_seg_mask = (flat_x != 0) * (flat_y != 0) * (flat_z != 0) * seg_mask.reshape(-1)
        flat_x = flat_x[zero_depth_seg_mask]
        flat_y = flat_y[zero_depth_seg_mask]
        flat_z = flat_z[zero_depth_seg_mask]

        print(f"colors shape before and after :{self.cam.colors.shape, self.cam.colors.reshape(-1,3).shape}")
        print(f"seg_mask shape before and after :{seg_mask.shape, seg_mask.reshape(-1).sum()}")
        colors = self.cam.colors.reshape(-1, 3)[zero_depth_seg_mask]
        # seg_mask = seg_mask.reshape(-1)[zero_depth_seg_mask]

        # 3d point cloud in camera orientation
        points1 = np.stack([flat_x, flat_y, flat_z], axis=-1)

        # Rotation matrix for camera tilt
        # head_tilt = 0.45
        cam_to_3d_rot = np.array([[1, 0, 0],
                            [0, math.cos(self.cam.head_tilt), math.sin(self.cam.head_tilt)], 
                            [0, -math.sin(self.cam.head_tilt), math.cos(self.cam.head_tilt)]]) 

        # 3d point cloud with upright camera
        transformed_points = np.dot(points1, cam_to_3d_rot)

        # Removing floor points from point cloud
        floor_mask = (transformed_points[:, 1] > -1.25)
        transformed_points = transformed_points[floor_mask] 
        transformed_x = transformed_points[:, 0]
        transformed_y = transformed_points[:, 1]
        transformed_z = transformed_points[:, 2]
        colors = colors[floor_mask]
        # flattened_seg_mask = seg_mask[floor_mask]

        num_points = len(transformed_x)
        # print(f"num points, colors hsape, seg mask shape - {num_points}, {colors.shape}, {flattened_seg_mask.shape}")
        
        # print(f"flattend mask {flattened_seg_mask.sum()}")
        indices = torch.arange(1, num_points + 1)
        # filtered_indices = indices[flattened_seg_mask]
        print(f"filtereted indices : {indices.shape}")
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(colors)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(transformed_points)
        pcd2.colors = o3d.utility.Vector3dVector(colors)
        
        # coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # coordinate_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # # o3d.visualizer.add_geometry(coordinate_frame)
        # o3d.visualization.draw_geometries([pcd1, coordinate_frame1])
        # o3d.visualization.draw_geometries([pcd2, coordinate_frame2])

        # Looking for hallow objects

        # Mean
        # threshold = 0.13
        # max_x, max_z, min_x, min_z = np.max(transformed_x), np.max(transformed_z), np.min(transformed_x), np.min(transformed_z)
        # avg_x, avg_z = (max_x + min_x) / 2, (max_z + min_z)/2
        # print(f"top surface max min x, z's{max_x}, {max_z}, {min_x}, {min_z}")
        
        # Median
        # px, pz = np.median(transformed_x), np.median(transformed_z)

        # Projected Median
        xz = np.stack([transformed_x*100, transformed_z*100], axis = -1).astype(int)
        unique_xz = np.unique(xz, axis = 0)
        unique_xz_x, unique_xz_z = unique_xz[:, 0], unique_xz[:, 1]
        px, pz = np.median(unique_xz_x)/100.0, np.median(unique_xz_z)/100.0

        x_margin, z_margin = 0.1, 0
        x_mask = ((transformed_x < (px + x_margin)) & (transformed_x > (px - x_margin)))
        y_mask = ((transformed_y < 0) & (transformed_y > -1.1))
        z_mask = ((transformed_z < 0) & (transformed_z > (pz - z_margin)))
        mask = x_mask & y_mask & z_mask
        py = np.max(transformed_y[mask])
        point = np.array([px, py, pz])
        
        geometries = []
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.1, height=0.04)
        cylinder_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        cylinder.rotate(cylinder_rot) 
        cylinder.translate(point)
        cylinder.paint_uniform_color([0, 1, 0])
        geometries.append(cylinder)
        
        # o3d.visualization.draw_geometries([pcd2, coordinate_frame2, *geometries])
        visualize_cloud_geometries(pcd2, geometries, 
                                    save_file = self.save_dir + "/placing.jpg")

        point[1] += 0.1
        transformed_point = cam_to_3d_rot @ point

        print(f"transformed_point: {transformed_point}")
        self.socket.send_msg(np.array(transformed_point, dtype=np.float64), [0], [0, 0, 0])
        self.socket.send_string("Now you received the gripper pose, good luck.")
                