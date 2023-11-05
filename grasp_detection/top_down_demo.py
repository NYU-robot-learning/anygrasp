import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image, ImageChops, ImageDraw
import zmq
import math, copy

from matplotlib import pyplot as plt
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from home_robot.manipulation.voxel_grasps import VoxelGraspGenerator
from utils.utils import get_bounding_box, show_mask, sam_segment, visualize_cloud_grippers

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
parser.add_argument('--open_communication', action='store_true', help='Use image transferred from the robot')
parser.add_argument('--crop', action='store_true', help='Passing cropped image to anygrasp')
parser.add_argument('--environment', default = '/data/pick_and_place_exps/Sofa', help='Environment name')
parser.add_argument('--method', default = 'usa', help='navigation method name')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

device = "cuda"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Creating a REP socket
if cfgs.open_communication:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    #buf = buffer(msg)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def segment(image, bounding_box):
    global predictor
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords = None,
        point_labels = None,
        box = bounding_box,
        multimask_output = False
    )

    return masks

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def plot_gripper_pro_max(grasp, score=1, color=None):
    '''
    Author: chenxi-wang
    
    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''

    center = grasp[:3, 3]
    R = grasp[:3, :3]
    width = 0.03
    depth = 0.03
    
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def grasp_to_geo(grasp):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=grasp[:3, 3]
    )
    coords = coords.rotate(grasp[:3, :3])
    return coords

def send_msg(a, b, c):
    if cfgs.open_communication:
        print(socket.recv_string())
        send_array(socket, np.array(a))
        print(socket.recv_string())
        send_array(socket, np.array(b))
        print(socket.recv_string())
        send_array(socket, np.array(c))
        print(socket.recv_string())


def demo():
    #anygrasp = AnyGrasp(cfgs)
    #anygrasp.load_net()
    grasp_generator = VoxelGraspGenerator(debug = False)

    # get data
    tries = 1
    crop_flag = cfgs.crop
    flag = True
    while tries > 0:
        if cfgs.open_communication:
            colors = recv_array(socket)
            image = Image.fromarray(colors)
            colors = colors / 255.0
            socket.send_string("RGB received")
            depths = recv_array(socket)
            socket.send_string("depth received")
            fx, fy, cx, cy, head_tilt = recv_array(socket)
            #fx, fy, cx, cy = recv_array(socket)
            head_tilt = head_tilt/100
            ref_vec = np.array([0, math.cos(head_tilt), -math.sin(head_tilt)])
            socket.send_string("intrinsics received")
            camera_pose = recv_array(socket)
            socket.send_string('camera_pose received')
            text = socket.recv_string()
            print(f"text - {text}")
            socket.send_string("text query received")
            mode = socket.recv_string()
            print(f"mode - {mode}")
            socket.send_string("Mode received")
        else:
            data_dir = "./example_data/"
            colors = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_rgb21.png')))
            image = Image.open(os.path.join(data_dir, 'peiqi_test_rgb21.png'))
            colors = colors / 255.0
            depths = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_depth21.png')))
            fx, fy, cx, cy, scale = 306, 306, 118, 211, 0.001
            text = "bottle"
            mode = "pick"
            depths = depths * scale
            head_tilt = 45/100
            ref_vec = np.array([0, math.cos(head_tilt), -math.sin(head_tilt)])
        
        if not os.path.exists(cfgs.environment + "/" + text + "/topdown/"):
            os.makedirs(cfgs.environment + "/" + text + "/topdown/")
        [crop_x_min, crop_y_min, crop_x_max, crop_y_max] = get_bounding_box(image, text, tries,
            save_file=cfgs.environment + "/" + text + "/topdown/" + cfgs.method +  "_topdown_open_owl_vit_bboxes.jpg")
        print(crop_x_min, crop_y_min, crop_x_max, crop_y_max)

        bbox_center = [int((crop_x_min + crop_x_max)/2), int((crop_y_min + crop_y_max)/2)]
        depth_obj = depths[bbox_center[1], bbox_center[0]]
        print(f"{text} height and depth: {((crop_y_max - crop_y_min) * depth_obj)/fy}, {depth_obj}")

        x_dis = (bbox_center[0] - cx)/fx * depth_obj
        print(f"d displacement {x_dis}")
        # pan = math.atan((bbox_center[0] - cx)/fx)
        # print(f"pan {pan}")

        tilt = math.atan((bbox_center[1] - cy)/fy)
        print(f"y tilt {tilt}")

        if(tries == 1):
            send_msg([-x_dis], [-tilt], [0, 0, 1])
            # print(socket.recv_string())
            # send_array(socket, np.array([x_dis]))
            # print(socket.recv_string())
            # send_array(socket, np.array([-tilt]))
            # print(socket.recv_string())
            # send_array(socket, np.array([0, 0, 1]))
            # print(socket.recv_string())
            if cfgs.open_communication:
                socket.send_string("Now you received the base and haed trans, good luck.")
            tries += 1
            # if mode == "place":
            #     crop_flag = True
            continue
        
        while flag:
            # get point cloud
            if crop_flag:
                # x_min, y_min, x_max, y_max = max(crop_x_min - 50, 0), max(crop_y_min - 50, 0), min(crop_x_max+50, 480), min(crop_y_max+20, 640)
                x_min, y_min, x_max, y_max = crop_x_min, crop_y_min, crop_x_max, crop_y_max
                xmap, ymap = np.arange(x_min, x_max+1), np.arange(y_min, y_max+1)
                print(colors.shape, depths.shape)
                depths = depths[y_min:y_max+1, x_min:x_max+1]
                colors = colors[y_min:y_max+1, x_min:x_max+1]
            else:
                xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            print(xmap.shape)
            print(depths.shape)
            print(colors.shape)
            points_z = depths
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z
            print(f"points_x shape - {points_x.shape}")

            print(f"x - [{np.min(points_x)}. {np.max(points_x)}]")
            print(f"y - [{np.min(points_y)}. {np.max(points_y)}]")
            print(f"z - [{np.min(points_z)}. {np.max(points_z)}]")

            if mode == "place":
                print("placing mode")
                masks = segment(np.array(image), np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max]))
                seg_mask = np.array(masks[0])
                print(seg_mask)
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(seg_mask, plt.gca())
                # show_box(input_box, plt.gca())
                plt.axis('off')
                plt.show()
                print(points_x.shape)

                flat_x, flat_y, flat_z = points_x.reshape(-1), -points_y.reshape(-1), -points_z.reshape(-1)

                # Removing all points whose depth is zero(undetermined)
                zero_depth_mask = (flat_x != 0) * (flat_y != 0) * (flat_z != 0)
                flat_x = flat_x[zero_depth_mask]
                flat_y = flat_y[zero_depth_mask]
                flat_z = flat_z[zero_depth_mask]

                print(f"colors shape before and after :{colors.shape, colors.reshape(-1,3).shape}")
                print(f"seg_mask shape before and after :{seg_mask.shape, seg_mask.reshape(-1).sum()}")
                colors = colors.reshape(-1, 3)[zero_depth_mask]
                seg_mask = seg_mask.reshape(-1)[zero_depth_mask]

                # 3d point cloud in camera orientation
                points1 = np.stack([flat_x, flat_y, flat_z], axis=-1)

                # Rotation matrix for camera tilt
                # head_tilt = 0.45
                cam_to_3d_rot = np.array([[1, 0, 0],
                                    [0, math.cos(head_tilt), math.sin(head_tilt)], 
                                    [0, -math.sin(head_tilt), math.cos(head_tilt)]]) 

                # 3d point cloud with upright camera
                transformed_points = np.dot(points1, cam_to_3d_rot)

                # Removing floor points from point cloud
                floor_mask = (transformed_points[:, 1] > -1.1)
                transformed_points = transformed_points[floor_mask] 
                transformed_x = transformed_points[:, 0]
                transformed_y = transformed_points[:, 1]
                transformed_z = transformed_points[:, 2]
                colors = colors[floor_mask]
                flattened_seg_mask = seg_mask[floor_mask]

                num_points = len(transformed_x)
                print(f"num points, colors hsape, seg mask shape - {num_points}, {colors.shape}, {flattened_seg_mask.shape}")
                
                print(f"flattend mask {flattened_seg_mask.sum()}")
                indices = torch.arange(1, num_points + 1)
                filtered_indices = indices[flattened_seg_mask]
                print(f"filtereted indices : {filtered_indices.shape}")

                
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(points1)
                pcd1.colors = o3d.utility.Vector3dVector(colors)
                
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(transformed_points)
                pcd2.colors = o3d.utility.Vector3dVector(colors)
                
                
                coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                coordinate_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                # o3d.visualizer.add_geometry(coordinate_frame)
                o3d.visualization.draw_geometries([pcd1, coordinate_frame1])
                o3d.visualization.draw_geometries([pcd2, coordinate_frame2])

                sampled_indices = np.random.choice(filtered_indices, size=500, replace=False)
                x_margin, y_margin, z_margin = 0.15, 0.02, 0.15
                max_sum = 0
                area_ind = []
                for ind in sampled_indices:
                    point = np.asarray(pcd2.points[ind])
                    x_mask = ((transformed_x < (point[0] + x_margin)) & (transformed_x > (point[0] - x_margin)))
                    y_mask = ((transformed_y < (point[1] + y_margin)) & (transformed_y > (point[1] - y_margin)))
                    z_mask = ((transformed_z < (point[2] + z_margin)) & (transformed_z > (point[2] - z_margin)))

                    curr_mask = x_mask & y_mask & z_mask
                    # print(ind, curr_mask.sum())
                    if (curr_mask.sum() > max_sum):
                        max_ind = ind

                    # print(ind, point, curr_mask.sum())
                    area_ind.append((curr_mask.sum(), ind))
                
                sorted_area_ind = sorted(area_ind, key= lambda x: x[0], reverse=True)
                # print(sorted_area_ind)
                geometries = []
                for i in range(1):
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.1, height=0.04)
                    ind = sorted_area_ind[i][1]
                    point = np.asarray(pcd2.points[ind])
                    print(ind, point, sorted_area_ind[i][0])
                    cylinder_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    cylinder.rotate(cylinder_rot) 
                    cylinder.translate(point)
                    if (i == 0):
                        cylinder.paint_uniform_color([0, 1, 0])
                    else:
                        cylinder.paint_uniform_color([1, 0, 0])
                    geometries.append(cylinder)
                
                o3d.visualization.draw_geometries([pcd2, coordinate_frame2, *geometries])

                geometries = []
                max_ind = sorted_area_ind[0][1]
                point = np.asarray(pcd2.points[max_ind])
                point[1] += 0.2

                transformed_point = cam_to_3d_rot @ point
                print(transformed_point)
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.05, height=0.04)
                cylinder.translate(transformed_point)
                cylinder.rotate(cylinder_rot) 
                cylinder.paint_uniform_color([1, 0, 0])
                geometries.append(cylinder)
                o3d.visualization.draw_geometries([pcd1, coordinate_frame1, *geometries])

                print(f"transformed_point: {transformed_point}")
                send_msg(np.array(transformed_point, dtype=np.float64), [0], [0, 0, 0])
                socket.send_string("Now you received the gripper pose, good luck.")
                # send_msg([2, 1, 2], [0], [0, 0, 0])
                exit()
            else:
                # remove outlier
                mask = segment(np.array(image), np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max]))
                mask = (points_z > 0) & (points_z < 3) & mask
                points = np.stack([points_x, points_y, points_z], axis=-1)
                #points = points[mask].astype(np.float32)
                #print(f"points shape: {points.shape}")
                #xmin = points[:, 0].min()
                #xmax = points[:, 0].max()
                #ymin = points[:, 1].min()
                #ymax = points[:, 1].max()
                #zmin = points[:, 2].min()
                #zmax = points[:, 2].max()
                #lims = [xmin, xmax, ymin, ymax, zmin, zmax]

                #colors_m = colors[mask].astype(np.float32)
                #print(points.min(axis=0), points.max(axis=0))

                # get prediction
                #gg, cloud = anygrasp.get_grasp(points, colors_m, lims)
                points = points.reshape(-1, 3)
                colors = colors.reshape(-1, 3)
                mask = mask.reshape(-1)
                print(points.shape, mask.shape)

                cam_to_3d_rot = np.array([[1, 0, 0],
                                    [0, math.cos(head_tilt), math.sin(head_tilt)], 
                                    [0, -math.sin(head_tilt), math.cos(head_tilt)]]) 

                # 3d point cloud with upright camera
                homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
                # transformed_points_homogeneous = np.dot(camera_pose, homogeneous_points.T).T
                transformed_points_homogeneous = homogeneous_points @ camera_pose.T
                world_points = transformed_points_homogeneous[:, :3]

                #grasps, scores, _ = grasp_generator.get_grasps(points, colors, mask, camera_pose.T)
                world_grasps, scores, _ = grasp_generator.get_grasps(points, colors, mask, camera_pose)
                #grasps = np.transpose(np.transpose(grasps[0], [0, 2, 1]) @ np.linalg.inv(camera_pose).T, [0, 2, 1])
                world_grasps = world_grasps[0]
                world_grasps[:, 3, 3] = 1
                grasps = [np.linalg.inv(camera_pose) @ grasp for grasp in world_grasps]

                print(f"world grasps - {world_grasps[0]}\n\n")
                print(f"grasps - {grasps[0]}\n\n")
                print(f"camera pose - {camera_pose}\n\n")
                print(f"inv camera paose - {np.linalg.inv(camera_pose)}\n\n")

                scores = scores[0]
                print(f"scores - {scores}, grasps - {grasps}")

                if len(grasps) == 0:
                    print('No Grasp detected after collision detection!')
                    send_msg([0], [0], [0, 0, 2])
                    if tries < 13:
                        tries = tries + 1
                        print(f"try no: {tries}")
                        if cfgs.open_communication:
                            socket.send_string("No poses, Have to try again")
                        break
                    else :
                        crop_flag = not crop_flag
                        flag = crop_flag
                    break
                
                # gg = grasps_to_gg(grasps)
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
                cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
                
                world_cloud = o3d.geometry.PointCloud()
                world_cloud.points = o3d.utility.Vector3dVector(world_points.astype(np.float32))
                world_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    #             #     # print(grasp_center, ix, iy, g.depth)
                if (len(grasps) == 0):
                    print("No grasp poses detected for this object try to move the object a little and try again")
                    send_msg([0], [0], [0, 0, 2])
                    if tries < 12:
                        tries = tries + 1
                        print(f"try no: {tries}")
                        socket.send_string("No poses, Have to try again")
                        break
                    else :
                        crop_flag = not crop_flag
                        flag = crop_flag
                else:
                    flag = False
                    tries = -1

    # image.save("./example_data/grasp_projections21.png")

    # filter_gg = filter_gg.nms().sort_by_score()

    # # gg_pick = filter_gg[0:20]
    # print('grasp score:', filter_gg[0].score)
    # print(repr(filter_gg[0]))

    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        # transformed_cloud = copy.deepcopy(cloud)
        cloud.transform(trans_mat)
        # world_cloud.transform(trans_mat)
        grippers = []
        world_grippers = []
        for i in range(len(grasps)):
            grasp = grasps[i]
            world_grasp = world_grasps[i]
            grippers.append(grasp_to_geo(grasp))
            world_grippers.append(grasp_to_geo(world_grasp))
        for i in range(len(grippers)):
            # trans_mat = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
            gripper = grippers[i]
            # world_gripper = world_grippers[i]
            # world_to_camera = np.eye(4)
            # world_to_camera[:3, :3] = cam_to_3d_rot
            # world_gripper.transform(world_to_camera)
            # world_gripper.transform(trans_mat)
            gripper.transform(trans_mat)
            # print(f"world to camera : {world_to_camera}")
            
        # pcd.transform(trans_mat)
        
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry([*grippers, cloud])
        # vis.add_geometry([grippers[0], cloud])
        #coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        if not os.path.exists(cfgs.environment + "/" + text):
            os.makedirs(cfgs.environment + "/" + text + "/topdown/")
        visualize_cloud_grippers(cloud, grippers, visualize = True, 
                save_file = cfgs.environment + "/" + text + "/topdown/" + cfgs.method +  "_topdown_poses.jpg")
        #visualize_cloud_grippers(world_cloud, world_grippers, visualize=True, 
        #        save_file = cfgs.environment + "/" + text + "/" + cfgs.method + "_topdown_world_pose.jpg")
        # o3d.visualizer.add_geometry(coordinate_frame)
        #o3d.visualization.draw_geometries([*grippers, cloud, coordinate_frame])
        #o3d.visualization.draw_geometries([*world_grippers, world_cloud, coordinate_frame])
        # plt.imshow(image)
        # plt.show()

        #o3d.visualization.draw_geometries([*filter_grippers, cloud, coordinate_frame])
        # o3d.visualization.draw_geometries([filter_grippers[0], cloud, coordinate_frame])

    grasp = grasps[np.argmax(scores)]
    print(grasp)
    send_msg(grasp[:3, 3], grasp[:3, :3], [0, 0, 0])
    # send_msg(filter_gg[0].translation, filter_gg[0].rotation_matrix, [filter_gg[0].depth, crop_flag, 0])
    # print(socket.recv_string())
    # send_array(socket, filter_gg[0].translation)
    # print(socket.recv_string())
    # send_array(socket, filter_gg[0].rotation_matrix)
    # print(socket.recv_string())
    # send_array(socket, np.array([filter_gg[0].depth, crop_flag, 0]))
    # print(socket.recv_string())
    # send_array(socket, np.array([crop_flag]))
    # print(socket.recv_string())
    if cfgs.open_communication:
        socket.send_string("Now you received the gripper pose, good luck.")
        
if __name__ == '__main__':
    while True:
        demo()
