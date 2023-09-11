import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image, ImageChops, ImageDraw
import zmq

from matplotlib import pyplot as plt
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

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
    socket.bind("tcp://*:5555")

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

def get_bounding_box(image):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    texts = [["bottle"]]
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
    max_score = np.max(scores.detach().numpy())
    print(f"max_score: {max_score}")
    max_ind = np.argmax(scores.detach().numpy())
    max_box = boxes.detach().numpy()[max_ind].astype(int)

    # img_drw = ImageDraw.Draw(image)
    # img_drw.rectangle([(max_box[0], max_box[1]), (max_box[2], max_box[3])], outline="green")
    # img_drw.text((max_box[0], max_box[1]), str(round(max_score.item(), 3)), fill="green")

    # for box, score, label in zip(boxes, scores, labels):
    #     box = [int(i) for i in box.tolist()]
    #     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    #     if (score == max_score):
    #         img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
    #         img_drw.text((box[0], box[1]), str(round(max_score.item(), 3)), fill="red")
    #     else:
    #         img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="white")
    # image.save("./example_data/bounding_box21.png")
    return max_box
    

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    
    if cfgs.open_communication:
        colors = recv_array(socket)
        image = Image.fromarray(colors)
        colors = colors / 255.0
        socket.send_string("RGB received")
        depths = recv_array(socket)
        socket.send_string("depth received")
    else:
        colors = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_rgb20.png')))
        image = Image.open(os.path.join(data_dir, 'peiqi_test_rgb20.png'))
        colors = colors / 255.0
        depths = np.array(Image.open(os.path.join(data_dir, 'peiqi_test_depth20.png')))

    # print(colors.shape, colors.size, Image.fromarray(colors *255, mode="RGB").size)
    [crop_x_min, crop_y_min, crop_x_max, crop_y_max] = get_bounding_box(image)
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_x_min - 2, crop_y_min - 2, crop_x_max + 2, crop_y_max + 2
    print(crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    if cfgs.crop:
        colors = colors[crop_y_min: (crop_y_max+1), crop_x_min: (crop_x_max+1)]
        depths = depths[crop_y_min: (crop_y_max+1), crop_x_min: (crop_x_max+1)]

    # exit()
    # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # fx, fy,  cx, cy = 306, 306, 211, 122
    fx, fy, cx, cy = 306, 306, 118, 211
    scale = 1000.0

    # set workspace
    xmin, xmax = -1.19, 1.12
    ymin, ymax = -1.02, 1.35
    zmin, zmax = 0.5, 50.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    if not cfgs.crop:
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    else:
        xmap, ymap = np.arange(crop_x_min, crop_x_max+1), np.arange(crop_y_min, crop_y_max+1)
    xmap, ymap = np.meshgrid(xmap, ymap)
    print(xmap.shape)
    print(depths.shape)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    print(f"x - [{np.min(points_x)}. {np.max(points_x)}]")
    print(f"y - [{np.min(points_y)}. {np.max(points_y)}]")
    print(f"z - [{np.min(points_z)}. {np.max(points_z)}]")

    # remove outlier
    mask = (points_z > 0) & (points_z < 2)
    points = np.stack([points_x, -points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    print(points.shape)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    # get prediction
    gg, cloud = anygrasp.get_grasp(points, colors, lims)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    filter_gg = GraspGroup()
    # print(gg.scores())

    img_drw = ImageDraw.Draw(image)
    img_drw.rectangle([(crop_x_min, crop_y_min), (crop_x_max, crop_y_max)], outline="red")
    for g in gg:
        # delp1 = np.matmul(np.transpose(g.rotation_matrix) , np.array([-g.depth, 0, 0]))
        # delp2 = np.matmul(np.transpose(g.rotation_matrix) , np.array([0, g.depth, 0]))
        # delp3 = np.matmul(np.transpose(g.rotation_matrix) , np.array([0, 0, g.depth]))
        # delp4 = np.matmul(g.rotation_matrix , np.array([-g.depth, 0, 0]))
        # delp5 = np.matmul(g.rotation_matrix , np.array([0, g.depth, 0]))
        # delp6 = np.matmul(g.rotation_matrix , np.array([0, 0, g.depth]))
        grasp_center = g.translation
        ix, iy = int(((grasp_center[0]*fx)/grasp_center[2]) + cx), int(((-grasp_center[1]*fy)/grasp_center[2]) + cy)
        if (crop_x_min <= ix) and (ix <= crop_x_max) and (crop_y_min <= iy) and (iy <= crop_y_max):
            filter_gg.add(g)
            img_drw.ellipse([(ix-1, iy-1), (ix+1, iy+1)], fill = "green")
        else:
            img_drw.ellipse([(ix-1, iy-1), (ix+1, iy+1)], fill = "red")
        # print(grasp_center, ix, iy, g.depth)
    image.save("./example_data/grasp_projections20.png")

    filter_gg = filter_gg.nms().sort_by_score()
    gg_pick = filter_gg[0:20]
    print('grasp score:', gg_pick[0].score)
    print(repr(gg_pick[0]))

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        filter_grippers = filter_gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        for gripper in filter_grippers:
            gripper.transform(trans_mat)
        
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry([*grippers, cloud])
        # vis.add_geometry([grippers[0], cloud])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # o3d.visualizer.add_geometry(coordinate_frame)
        o3d.visualization.draw_geometries([*grippers, cloud, coordinate_frame])
        plt.imshow(image)
        plt.show()

        o3d.visualization.draw_geometries([*filter_grippers, cloud, coordinate_frame])
        o3d.visualization.draw_geometries([filter_grippers[0], cloud, coordinate_frame])
        # vis.add_geometry(pcd2)
        # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 1)
        # vis.run()
    
    if cfgs.open_communication:
        print(socket.recv_string())
        send_array(socket, gg_pick[0].translation)
        print(socket.recv_string())
        send_array(socket, gg_pick[0].rotation_matrix)
        print(socket.recv_string())
        send_array(socket, np.array([gg_pick[0].depth]))
        print(socket.recv_string())
        
if __name__ == '__main__':
    demo('./example_data/')
