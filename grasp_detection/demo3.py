import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import zmq
import math

from matplotlib import pyplot as plt

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from utils.utils import get_bounding_box, show_mask, sam_segment, visualize_cloud_grippers

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--port', type=int, default = 5556, help='port')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
parser.add_argument('--open_communication', action='store_true', help='Use image transferred from the robot')
parser.add_argument('--crop', action='store_true', help='Passing cropped image to anygrasp')
parser.add_argument('--environment', default = '/data/pick_and_place_exps/Sofa2', help='Environment name')
parser.add_argument('--method', default = 'usa', help='navigation method name')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

context = zmq.Context()

# Creating a REP socket
if cfgs.open_communication:
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(cfgs.port))

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


def send_msg(a, b, c):
    print(socket.recv_string())
    send_array(socket, np.array(a))
    print(socket.recv_string())
    send_array(socket, np.array(b))
    print(socket.recv_string())
    send_array(socket, np.array(c))
    print(socket.recv_string())

def gpu_stats():
    num_gpus = torch.cuda.device_count()

    print(f"Number of GPUs available: {num_gpus}")

    # Iterate over available GPUs and print their properties
    mem_info = torch.cuda.mem_get_info()
    for gpu_id in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(gpu_id)
        
        print(f"\nGPU {gpu_id} Properties:")
        print(f"  Name: {gpu_properties.name}")
        print(f"  CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
        print(f"  Total Memory: {gpu_properties.total_memory / (1024 ** 2):.2f} MB \n\n")
        print(f"  Free Memory: {mem_info[gpu_id] / (1024 ** 2):.2f} MB")

def demo():
    #gpu_stats()
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()
    #gpu_stats()

    # get data
    tries = 1
    crop_flag = cfgs.crop
    flag = True
    while tries > 0:
        torch.cuda.empty_cache()
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
        text = socket.recv_string()
        print(f"text - {text}")
        socket.send_string("text query received")
        mode = socket.recv_string()
        print(f"mode - {mode}")
        socket.send_string("Mode received")
        if not os.path.exists(cfgs.environment + "/" + text + "/anygrasp/" + cfgs.method):
            os.makedirs(cfgs.environment + "/" + text + "/anygrasp/" + cfgs.method)
        # data_dir = "./example_data/"
        # colors = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_rgb21.png')))
        # image = Image.open(os.path.join(data_dir, 'peiqi_test_rgb21.png'))
        # colors = colors / 255.0
        # depths = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_depth21.png')))
        # fx, fy, cx, cy, scale = 306, 306, 118, 211, 0.001
        # text = "table"
        # mode = "place"
        
        [crop_x_min, crop_y_min, crop_x_max, crop_y_max] = get_bounding_box(image, text, tries,
                                            save_file=cfgs.environment + "/" + text + "/anygrasp/" + cfgs.method +  "/owl_vit_bboxes.jpg")
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

            print(f"x - [{np.min(points_x)}. {np.max(points_x)}]")
            print(f"y - [{np.min(points_y)}. {np.max(points_y)}]")
            print(f"z - [{np.min(points_z)}. {np.max(points_z)}]")

            if mode == "place":
                print("placing mode")
                masks = sam_segment(np.array(image), np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max]))
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
                print(f"max points z - {np.max(points_z)}")
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

                colors_m = colors[mask].astype(np.float32)
                print(points.min(axis=0), points.max(axis=0))

                # get prediction
                gg, cloud = anygrasp.get_grasp(points, colors_m, lims)

                if len(gg) == 0:
                    print('No Grasp detected after collision detection!')
                    send_msg([0], [0], [0, 0, 2])
                    if tries < 10:
                        tries = tries + 1
                        print(f"try no: {tries}")
                        socket.send_string("No poses, Have to try again")
                        break
                    else :
                        crop_flag = not crop_flag
                        flag = crop_flag
                    break

                gg = gg.nms().sort_by_score()
                filter_gg = GraspGroup()
                # print(gg.scores())


                min_score, max_score = 1, -10
                img_drw = ImageDraw.Draw(image)
                img_drw.rectangle([(crop_x_min, crop_y_min), (crop_x_max, crop_y_max)], outline="red")
                for g in gg:
                    grasp_center = g.translation
                    ix, iy = int(((grasp_center[0]*fx)/grasp_center[2]) + cx), int(((-grasp_center[1]*fy)/grasp_center[2]) + cy)
                    rotation_matrix = g.rotation_matrix
                    cur_vec = rotation_matrix[:, 0]
                    angle = math.acos(np.dot(ref_vec, cur_vec)/(np.linalg.norm(cur_vec)))
                    # score = g.score + 0.13 - 0.005*((angle + 0.75)**6)
                    if not crop_flag:
                        score = g.score - 0.1*(angle)**4
                    else:
                        score = g.score
                    
                    if not crop_flag:
                        if (crop_x_min <= ix) and (ix <= crop_x_max) and (crop_y_min <= iy) and (iy <= crop_y_max):
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
                    send_msg([0], [0], [0, 0, 2])
                    if tries < 10:
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
                
                if flag:
                    del gg
                    del cloud
                    
    # image.save("./example_data/grasp_projections21.png")
    image.save(cfgs.environment + "/" + text + "/anygrasp/" + cfgs.method +  "/grasp_projections.jpg")
    filter_gg = filter_gg.nms().sort_by_score()

    # gg_pick = filter_gg[0:20]
    print('grasp score:', filter_gg[0].score)
    print(repr(filter_gg[0]))

    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        filter_grippers = filter_gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        for gripper in filter_grippers:
            gripper.transform(trans_mat)
        
        visualize_cloud_grippers(cloud, grippers, visualize = True, 
                save_file = cfgs.environment + "/" + text  + "/anygrasp/" + cfgs.method +  "/poses.jpg")
        visualize_cloud_grippers(cloud, [filter_grippers[0]], visualize=True, 
                save_file = cfgs.environment + "/" + text  + "/anygrasp/" + cfgs.method + "/best_pose.jpg")
    
    send_msg(filter_gg[0].translation, filter_gg[0].rotation_matrix, [filter_gg[0].depth, crop_flag, 0])
    socket.send_string("Now you received the gripper pose, good luck.")
        
if __name__ == '__main__':
    while True:
        demo()
