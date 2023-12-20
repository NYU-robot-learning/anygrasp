import os 
import math
import copy

# import torch
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from utils.types import Bbox
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from utils.zmq_socket import ZmqSocket
from utils.utils import get_3d_points, visualize_cloud_geometries
from utils.camera import CameraParameters
from image_processors import OwlVITProcessor, LangSAMProcessor, SamProcessor

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
        self.lang_sam = LangSAMProcessor()

    def receive_input(self):
        # Reading color array
        colors = self.socket.recv_array()
        self.socket.send_data("RGB received")

        # Depth data
        depths = self.socket.recv_array()
        self.socket.send_data("depth received")

        # Camera Intrinsics
        fx, fy, cx, cy, head_tilt = self.socket.recv_array()
        self.socket.send_data("intrinsics received")

        # Object query
        self.query = self.socket.recv_string()
        self.socket.send_data("text query received")
        print(f"text - {self.query}")

        # action -> ["pick", "place"]
        self.action = self.socket.recv_string()
        self.socket.send_data("Mode received")
        print(f"mode - {self.action}")
        print(self.socket.recv_string())

        # head_tilt= -45
        # data_dir = "./example_data/"
        # colors = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_rgb21.png')))
        # image = Image.open(os.path.join(data_dir, 'peiqi_test_rgb21.png'))
        # # colors = colors / 255.0
        # depths = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_depth21.png')))
        # fx, fy, cx, cy, scale = 306, 306, 118, 211, 0.001
        # self.action = "pick"
        # self.query = "yellow bottle"
        # depths = depths * scale
        
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
        while retry and tries <= 11:
            self.receive_input()

            # Directory for saving visualisations
            self.save_dir = self.cfgs.environment + "/" + self.query + "/anygrasp/" + self.cfgs.method
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.cam.image.save(self.save_dir + "/clean_" + str(tries) + ".jpg")

            # Just for comparing owl_vit vs lang sam this wont be used for further processing, Can be commented if not required
            bbox =  self.owl_vit.detect_obj(self.cam.image, self.query, visualize_box = True,
                                                bbox_save_filename=self.cfgs.environment + "/" + \
                                                self.query + "/anygrasp/" + self.cfgs.method + \
                                                "/owl_vit_detection_" + str(tries) + ".jpg")

            # Object Segmentaion Mask
            seg_mask, bbox = self.lang_sam.detect_obj(self.cam.image, self.query, 
                                                      visualize_box = True, visualize_mask = True,
                                                      box_filename = self.cfgs.environment + "/" + self.query + \
                                                            "/anygrasp/" + self.cfgs.method + \
                                                            "/lseg_detection_" + str(tries) + ".jpg",
                                                      mask_filename = self.cfgs.environment + "/" + self.query + \
                                                            "/anygrasp/" + self.cfgs.method + \
                                                            "/lseg_segmentation_" + str(tries) + ".jpg")

            if bbox is None:
                print("Didnt detect the object")
                tries = tries + 1
                print(f"try no: {tries}")
                data_msg = "No Objects detected, Have to try again"
                self.socket.send_data([[0], [0], [0, 0, 2], data_msg])
                # self.socket.send_data("No Objects detected, Have to try again")

            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
            print(bbox)

            # Center the robot
            if tries == 1:
                self.center_robot(bbox)
                tries += 1
                continue
            
            points = get_3d_points(self.cam)

            if self.action == "place":
                retry = not self.place(points, seg_mask)
            else:
                retry = not self.pickup(points, seg_mask, bbox)

            if retry:
                tries = tries + 1
                print(f"try no: {tries}")
                data_msg = "No poses, Have to try again"
                self.socket.send_data([[0], [0], [0, 0, 2], data_msg])
                # self.socket.send_data("No poses, Have to try again")

    def center_robot(self, bbox: Bbox):
        ''' 
            Center the robots base and camera to face the center of the Object Bounding box
        '''

        bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = bbox

        bbox_center = [int((bbox_x_min + bbox_x_max)/2), int((bbox_y_min + bbox_y_max)/2)]
        depth_obj = self.cam.depths[bbox_center[1], bbox_center[0]]
        print(f"{self.query} height and depth: {((bbox_y_max - bbox_y_min) * depth_obj)/self.cam.fy}, {depth_obj}")

        # base movement
        dis = (bbox_center[0] - self.cam.cx)/self.cam.fx * depth_obj
        print(f"d displacement {dis}")

        # camera tilt
        tilt = math.atan((bbox_center[1] - self.cam.cy)/self.cam.fy)
        print(f"y tilt {tilt}")

        data_msg = "Now you received the base and haed trans, good luck."
        self.socket.send_data([[-dis], [-tilt], [0, 0, 1], data_msg])
        # self.socket.send_data("Now you received the base and haed trans, good luck.")

    def place(
        self,
        points: np.ndarray,
        seg_mask: np.ndarray
    ) -> bool:

        points_x, points_y, points_z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
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
        print(f"max and min y {np.min(transformed_y)}, {np.max(transformed_y)}")
        # flattened_seg_mask = seg_mask[floor_mask]

        num_points = len(transformed_x)
        # print(f"num points, colors hsape, seg mask shape - {num_points}, {colors.shape}, {flattened_seg_mask.shape}")
        
        # print(f"flattend mask {flattened_seg_mask.sum()}")
        indices = np.arange(1, num_points + 1)
        # filtered_indices = indices[flattened_seg_mask]
        print(f"filtereted indices : {indices.shape}")
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(colors)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(transformed_points)
        pcd2.colors = o3d.utility.Vector3dVector(colors)

        coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coordinate_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])        
        o3d.visualization.draw_geometries([pcd1, coordinate_frame1])
        o3d.visualization.draw_geometries([pcd2, coordinate_frame2])

        # Projected Median
        xz = np.stack([transformed_x*100, transformed_z*100], axis = -1).astype(int)
        unique_xz = np.unique(xz, axis = 0)
        unique_xz_x, unique_xz_z = unique_xz[:, 0], unique_xz[:, 1]
        px, pz = np.median(unique_xz_x)/100.0, np.median(unique_xz_z)/100.0
        print(f"projected median {px, pz}")

        x_margin, z_margin = 0.1, 0
        x_mask = ((transformed_x < (px + x_margin)) & (transformed_x > (px - x_margin)))
        y_mask = ((transformed_y < 0) & (transformed_y > -1.1))
        z_mask = ((transformed_z < 0) & (transformed_z > (pz - z_margin)))
        print(f"x y a dn z masks {np.sum(x_mask)}, {np.sum(y_mask)}, {np.sum(z_mask)}")
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
        data_msg = "Now you received the gripper pose, good luck."
        self.socket.send_data([np.array(transformed_point, dtype=np.float64), [0], [0, 0, 0], data_msg])
        # self.socket.send_data("Now you received the gripper pose, good luck.")

        return True
    
    def pickup(
        self,
        points: np.ndarray,
        seg_mask: np.ndarray,
        bbox: Bbox,
        crop_flag: bool = False,
    ):
        points_x, points_y, points_z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        
        mask = (points_z > 0) & (points_z < 2)
        points = np.stack([points_x, -points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        print(f"points shape: {points.shape}")
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        colors_m = self.cam.colors[mask].astype(np.float32)
        print(f"colors input shape {colors_m.shape}")
        print(points.min(axis=0), points.max(axis=0))

        # get prediction
        gg, cloud = self.grasping_model.get_grasp(points, colors_m, lims)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return False

        gg = gg.nms().sort_by_score()
        filter_gg = GraspGroup()
        print(len(gg))

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
        W, H = self.cam.image.size
        print(f"image height and width {H}, {W}")
        ref_vec = np.array([0, math.cos(self.cam.head_tilt), -math.sin(self.cam.head_tilt)])
        min_score, max_score = 1, -10
        image = copy.deepcopy(self.cam.image)
        img_drw = ImageDraw.Draw(image)
        img_drw.rectangle([(bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max)], outline="red")
        for g in gg:
            grasp_center = g.translation
            ix, iy = int(((grasp_center[0]*self.cam.fx)/grasp_center[2]) + self.cam.cx), int(((-grasp_center[1]*self.cam.fy)/grasp_center[2]) + self.cam.cy)
            if (ix < 0): 
                ix = 0
            if (iy < 0):
                iy = 0
            if (ix >= W):
                ix = W - 1
            if (iy >= H):
                iy = H - 1
            rotation_matrix = g.rotation_matrix
            cur_vec = rotation_matrix[:, 0]
            angle = math.acos(np.dot(ref_vec, cur_vec)/(np.linalg.norm(cur_vec)))
            # score = g.score + 0.13 - 0.005*((angle + 0.75)**6)
            if not crop_flag:
                score = g.score - 0.1*(angle)**4
            else:
                score = g.score

            # print(ix, iy, seg_mask[iy, ix])    
            if not crop_flag:
                if seg_mask[iy, ix]:
                    img_drw.ellipse([(ix-1, iy-1), (ix+1, iy+1)], fill = "green")
                    print(f"diff angle, tilt,  score - {angle}, {g.score}, {score}")
                    if g.score >= 0.095:
                        g.score = score
                    #score  = g.score
                    min_score = min(min_score, g.score)
                    max_score = max(max_score, g.score)
                    filter_gg.add(g)
                else:
                    img_drw.ellipse([(ix-1, iy-1), (ix+1, iy+1)], fill = "red")
            else:
                g.score = score
                filter_gg.add(g)

            # print(grasp_center, ix, iy, g.depth)
        if (len(filter_gg) == 0):
            print("No grasp poses detected for this object try to move the object a little and try again")
            return False

        image.save(self.cfgs.environment + "/" + self.query + "/anygrasp/" + self.cfgs.method +  "/grasp_projections.jpg")
        filter_gg = filter_gg.nms().sort_by_score()

        if self.cfgs.debug:
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            filter_grippers = filter_gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            for gripper in filter_grippers:
                gripper.transform(trans_mat)
            
            visualize_cloud_geometries(cloud, grippers, visualize = True, 
                    save_file = self.cfgs.environment + "/" + self.query  + "/anygrasp/" + self.cfgs.method +  "/poses.jpg")
            visualize_cloud_geometries(cloud, [filter_grippers[0]], visualize=True, 
                    save_file = self.cfgs.environment + "/" + self.query  + "/anygrasp/" + self.cfgs.method + "/best_pose.jpg")
        
        data_msg = "Now you received the gripper pose, good luck."
        self.socket.send_data([filter_gg[0].translation, filter_gg[0].rotation_matrix, [filter_gg[0].depth, crop_flag, 0], data_msg])
        return True