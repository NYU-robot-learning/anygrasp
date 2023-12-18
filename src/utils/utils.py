from typing import List, Type

import torch
import numpy as np
from PIL import ImageDraw, Image
import copy
from matplotlib import pyplot as plt
import open3d as o3d


from utils.camera import CameraParameters
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from lang_sam import LangSAM

def get_3d_points(cam: CameraParameters):

    xmap, ymap = np.arange(cam.depths.shape[1]), np.arange(cam.depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    print(xmap.shape)
    print(cam.depths.shape)
    print(cam.colors.shape)
    points_z = cam.depths
    points_x = (xmap - cam.cx) / cam.fx * points_z
    points_y = (ymap - cam.cy) / cam.fy * points_z

    print(f"x - [{np.min(points_x)}. {np.max(points_x)}]")
    print(f"y - [{np.min(points_y)}. {np.max(points_y)}]")
    print(f"z - [{np.min(points_z)}. {np.max(points_z)}]")

    return np.stack((points_x, points_y, points_z), axis=-1)

def get_bounding_box(image, text, tries, save_file):
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

    new_image = copy.deepcopy(image)
    img_drw = ImageDraw.Draw(new_image)
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
    image.save(save_file + "/clean_" + str(tries) + ".jpg")
    new_image.save(save_file + "/owl_vit_detection_" + str(tries) + ".jpg")
    return max_box    

def draw_bounding_box(image, bbox, save_file=None):
    new_image = copy.deepcopy(image)
    img_drw = ImageDraw.Draw(new_image)
    img_drw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="green")

    if save_file is not None:
        new_image.save(save_file)
    # new_image.show()
    # img_drw.text((max_box[0], max_box[1]), str(round(max_score.item(), 3)), fill="green")

def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # return mask_image

def draw_seg_mask(image, seg_mask, save_file=None):
    alpha = np.where(seg_mask > 0, 128, 0).astype(np.uint8)

    image_pil = copy.deepcopy(image)
    alpha_pil = Image.fromarray(alpha)
    image_pil.putalpha(alpha_pil)

    if save_file is not None:
        image_pil.save(save_file)


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
    #   axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()

def lang_sam_segment(image, query):
    model = LangSAM()
    masks, boxes, phrases, logits = model.predict(image, query)
    # print(f"lang sam masks - {masks}")
    # print(f"lang sam boxes - {boxes.shape}")
    return masks, boxes
    
def sam_segment(image, bounding_box):
    """
        Inputs
            image : PIL Image
            bounding_box : np.array of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        Outputs
            masks : N * H * W, N -> No.of masks
    """
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords = None,
        point_labels = None,
        box = bounding_box,
        multimask_output = False
    )

    return masks

def color_grippers(grippers, max_score, min_score):
    """
        grippers    : list of grippers of form graspnetAPI grasps
        max_score   : max score of grippers
        min_score   : min score of grippers

        For debugging purpose - color the grippers according to score
    """

    for idx, gripper in enumerate(grippers):
        g = grippers[idx]
        if max_score != min_score:
            color_val = (g.score - min_score)/(max_score - min_score)
        else:
            color_val = 1
        color = [color_val, 0, 0]
        print(g.score, color)
        gripper.paint_uniform_color(color)

    return grippers

def visualize_cloud_geometries(cloud, geometries, translation = None, rotation = None, visualize = True, save_file = None):
    """
        cloud       : Point cloud of points
        grippers    : list of grippers of form graspnetAPI grasps
        visualise   : To show windows
        save_file   : Visualisation file name
    """

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    if translation is not None:
        coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # print(grippers[0])
        translation[2] = -translation[2]
        coordinate_frame1.translate(translation)
        coordinate_frame1.rotate(rotation)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=visualize)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.add_geometry(cloud)
    if translation is not None:
        visualizer.add_geometry(coordinate_frame1)
    visualizer.poll_events()
    visualizer.update_renderer()

    if save_file is not None:
        ## Controlling the zoom
        view_control = visualizer.get_view_control()
        zoom_scale_factor = 1.4  
        view_control.scale(zoom_scale_factor)

        visualizer.capture_screen_image(save_file, do_render = True)

    if visualize:
        visualizer.add_geometry(coordinate_frame)
        visualizer.run()
    else:
        visualizer.destroy_window()    