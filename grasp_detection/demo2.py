import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image, ImageChops, ImageDraw
import zmq
import math

from matplotlib import pyplot as plt
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
parser.add_argument('--open_communication', action='store_true', help='Use image transferred from the robot')
parser.add_argument('--crop', action='store_true', help='Passing cropped image to anygrasp')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

context = zmq.Context()

# Creating a REP socket
if cfgs.open_communication:
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")

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

def get_bounding_box(image, text, tries):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    texts = [[text, "A photo of " + text]]  
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    print(image.size[::-1])
    target_sizes = torch.Tensor([image.size[::-1]])
    print(target_sizes)
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)
    print(f"results - {results}")
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    if len(boxes) == 0:
        return None
    max_score = np.max(scores.detach().numpy())
    print(f"max_score: {max_score}")
    max_ind = np.argmax(scores.detach().numpy())
    max_box = boxes.detach().numpy()[max_ind].astype(int)

    #mask_predictor.set_image(image.permute(1, 2, 0).numpy())
    #transformed_boxes = mask_predictor.transform.apply_boxes_torch(max_box.reshape(-1, 4), image.shape[1:])  
    #masks, iou_predictions, low_res_masks = mask_predictor.predict_torch(
    #    point_coords=None,
    #    point_labels=None,
    #    boxes=transformed_boxes,
    #    multimask_output=False
    #)
    # masks = masks[:, 0, :, :]

    img_drw = ImageDraw.Draw(image)
    img_drw.rectangle([(max_box[0], max_box[1]), (max_box[2], max_box[3])], outline="green")
    img_drw.text((max_box[0], max_box[1]), str(round(max_score.item(), 3)), fill="green")

    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        if (score == max_score):
            img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
            img_drw.text((box[0], box[1]), str(round(max_score.item(), 3)), fill="red")
        else:
            img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="white")
    image.save(f"./example_data/bounding_box21_{tries}.png")
    return max_box    

def send_msg(a, b, c):
    print(socket.recv_string())
    send_array(socket, np.array(a))
    print(socket.recv_string())
    send_array(socket, np.array(b))
    print(socket.recv_string())
    send_array(socket, np.array(c))
    print(socket.recv_string())

def demo():
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    tries = 1
    while tries > 0:

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
        
        [crop_x_min, crop_y_min, crop_x_max, crop_y_max] = get_bounding_box(image, text, tries)
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
            continue
    
        crop_flag = cfgs.crop
        flag = True
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


            # remove outlier
            mask = (points_z > 0) & (points_z < 3)
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
                print(f"try no: {tries}")
                if tries < 9:
                    tries = tries + 1
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
                print(f"try no: {tries}")
                if tries < 15:
                    tries = tries + 1
                    socket.send_string("No poses, Have to try again")
                    break
                else :
                    crop_flag = not crop_flag
                    flag = crop_flag
            else:
                flag = False
                tries = -1

    image.save("./example_data/grasp_projections21.png")

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

        colors =[[0.3, 0, 0], [0.6, 0, 0], [1, 0, 0],
                 [0, 0.3, 0], [0, 0.6, 0], [0, 1, 0], 
                 [0, 0, 0.3], [0, 0, 0.6], [0, 0, 1],
                 [0.3, 0.3, 0], [0.6, 0.6, 0], [1, 1, 0],
                 [0.3, 0, 0.3], [0.6, 0, 0.6], [1, 0 ,1],
                 [0, 0.3, 0.3], [0, 0.6, 0.6], [0, 1, 1],
                 [0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [1,1,1]]
        for idx, gripper in enumerate(filter_grippers):
            gripper.transform(trans_mat)
            g = filter_gg[idx]
            if max_score != min_score:
                color_val = (g.score - min_score)/(max_score - min_score)
            else:
                color_val = 1
            color = [color_val, 0, 0]
            print(g.score, color)
            # color = colors[idx]
            gripper.paint_uniform_color(color)
        # pcd.transform(trans_mat)
        
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry([*grippers, cloud])
        # vis.add_geometry([grippers[0], cloud])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # o3d.visualizer.add_geometry(coordinate_frame)
        o3d.visualization.draw_geometries([*grippers, cloud, coordinate_frame])
        # plt.imshow(image)
        # plt.show()

        #o3d.visualization.draw_geometries([*filter_grippers, cloud, coordinate_frame])
        o3d.visualization.draw_geometries([filter_grippers[0], cloud, coordinate_frame])
    
    send_msg(filter_gg[0].translation, filter_gg[0].rotation_matrix, [filter_gg[0].depth, crop_flag, 0])
    # print(socket.recv_string())
    # send_array(socket, filter_gg[0].translation)
    # print(socket.recv_string())
    # send_array(socket, filter_gg[0].rotation_matrix)
    # print(socket.recv_string())
    # send_array(socket, np.array([filter_gg[0].depth, crop_flag, 0]))
    # print(socket.recv_string())
    # send_array(socket, np.array([crop_flag]))
    # print(socket.recv_string())
    socket.send_string("Now you received the gripper pose, good luck.")
        
if __name__ == '__main__':
    while True:
        demo()
