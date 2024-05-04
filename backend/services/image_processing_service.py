import argparse
import cv2
import numpy as np

from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall

from ultralytics import YOLOWorld
from ultralytics import SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import torchvision
import numpy as np
import scipy.sparse as sp
import torch
import os
import matplotlib.pyplot as plt
import cv2
import sys

#def convert image coordinate to world coordinate(img, h_w)

# (0,0)----------------(width,0)
#   |                       |
#   |   image coordinate    |
#   |                       |
# (0,height)--------(width,height)


# (0,h_w)-----------------(?,?)
#   |                       |
#   |   world coordinate    |
#   |                       |
# (0,0)-------------------(?,?)


def convert_image_to_world(image, h_w, img_coord_x, img_coord_y):
    height, width, _ = image.shape
    world_x = img_coord_y * (h_w / height)
    world_y = h_w - (img_coord_x * (h_w / height))
    return np.array([world_x, world_y])

#def getValidHolds(image):
#  Use Yolo-world to extract holds' bounding boxes <<ExtractBoundingBox>>
#  Use SAM to extract segments, using bounding boxes as prompts <<SegmentAnything>>
#  For each hold, create a hold object with its location <<getHolds>>
#  return set of masks of holds

def ExtractBoundingBox(image_path, directory):

    # Yolo world with open vocabulary to detect objects' boundng boxes
    
    # Initialize the model with pre-trained weights
    model = YOLOWorld('yolov8s-world')

    # Set the classes to find in image
    model.set_classes(["big red object", "big purple object", "big white object", "big blue object", "big black object", 
               "green object, orange object", "pink object", "orange waved object", "small red object", "small purple object", 
               "small white object, small blue object, small black object, small green object, small orange object, "
               "small pink object",  "small orange waved object",  "small yellow-black object", "purple sloped object",
               "white sloped object", "blue sloped object", "black sloped object", "green sloped object", "orange sloped object"])

    # Run object detection for custom classes on image
    results = model.predict(image_path, max_det=200, iou=0.005, conf=0.008)

    # Save the results
    results_path = os.path.join(directory, "yolo_world_result.jpg")
    results[0].save(results_path)

    img = cv2.imread(image_path)

    # Assuming 'results[0].boxes.xywh' holds the tensor with the bounding box information
    xyxy = results[0].boxes.xyxy

    # Convert tensor to a numpy array
    xyxy_np = xyxy.cpu().numpy()

    # Save the bounding box information to a text file
    # Path to the file
    # file_path = os.path.join(directory, "bbox.txt")
    # with open(file_path, "w") as f:
    #     for bbox in xyxy_np:
    #         f.write(f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n")

    i = 0
    # Example: print each bounding box
    for bbox in xyxy_np:
        # print(f"X: {bbox[0]}, Y: {bbox[1]}, X: {bbox[2]}, Y: {bbox[3]}")
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),3)
        i = i + 1
    images_path = os.path.join(directory, "bbox_result.jpg")
    cv2.imwrite(images_path, img)
    print(f"Total {i} items detected in the image.")
    return xyxy_np

# Function that inputs the output and plots image and mask
def show_output(result_dict,axes=None, save_path=None):
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
    # Plot for each segment area
    for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*0.5)))
    plt.savefig(save_path)  

def SegmentAnything(image_path, directory):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # mask_generator = SamAutomaticMaskGenerator(
    #     model = sam,
    #     points_per_side = 32,
    #     pred_iou_thresh = 0.9,
    #     stability_score_thresh = 0.96,
    #     crop_n_layers = 1,
    #     crop_n_points_downscale_factor= 2,
    #     min_mask_region_area= 100,  
    # )

    # Read the image from the path
    image= cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate segmentation mask
    output_mask = mask_generator.generate(image)
    # print(output_mask)

    _,axes = plt.subplots(1,2, figsize=(16,16))
    axes[0].imshow(image)
    image_path = os.path.join(directory, "sam_result.jpg")
    show_output(output_mask, axes[1], image_path)

    masks = mask_generator.generate(image)
    # store the masks in the directory
    # output_dir = os.path.join(directory, "./masks/")
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # for i in range(len(masks)): 
    #     sp.save_npz(os.path.join(output_dir, (str(i)+".npz")), sp.csr_matrix(masks[i]['segmentation'] ))

    return masks

def find_ymax(mask):
    
    rows, cols = np.where(mask)

    if rows.size > 0:
        # find the top edge of the mask
        top_edge = np.min(rows)
        top_points_indices = cols[rows == top_edge]
        top_points = [(top_edge, col) for col in top_points_indices]
        if(len(top_points)>1):
            # find the middle point
            point = top_points[len(top_points)//2]
        else:
            point = top_points[0]
    else:
        top_points = []
        print("No True values found in the mask.")
        point = None

    return point


def getHolds(image_path, bboxes, masks, directory):
    holds = []
    # find the bounding box of the mask corresponding to the bboxes
    image = cv2.imread(image_path)
    # image = np.ones((image.shape[0], image.shape[1], 3))
    # image = image * 255
    count = 0
    print(image.shape)
    for i, bbox in enumerate(bboxes):
        bbox = [int(b) for b in bbox]
        # print(bbox)
        flag = False
        hold_mask = np.zeros((image.shape[0], image.shape[1]))
        for j in range(len(masks)):
            mask_bbox = masks[j]['bbox']
            m0 = mask_bbox[0]
            m1 = mask_bbox[1]
            m2 = mask_bbox[0] + mask_bbox[2]
            m3 = mask_bbox[1] + mask_bbox[3]
            if (abs(bbox[0] - m0) <= 15 and abs(bbox[1] - m1) <= 15 and abs(bbox[2] - m2) <= 15 and abs(bbox[3] - m3) <= 15):
                # print(bbox[0] - m0, bbox[1] - m1, bbox[2] - m2, bbox[3] - m3)
                # print(f"Found bbox {j}:{mask_bbox} within mask bbox {bbox}")
                image = np.where(masks[j]['segmentation'][:,:,None], [0,0,255], image)
                hold_mask = np.where(masks[j]['segmentation'], 1, hold_mask)
                flag = True
            elif(m0 > bbox[0] and m1 > bbox[1] and m2 < bbox[2] and m3 < bbox[3]):
                image = np.where(masks[j]['segmentation'][:,:,None], [0,0,255], image)
                hold_mask = np.where(masks[j]['segmentation'], 1, hold_mask)
                flag = True
        if(flag == True):
            holds.append(hold_mask)
            count = count + 1
            # print(yMax, hold_mask)

        if(flag == False):
            print(bbox[0], bbox[1], bbox[2], bbox[3])
            # cv2.rectangle(image,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,0),3)
            
    print(f"Total {count} items detected in the image.")
    # plt.imshow(image)
    path = os.path.join(directory, "holds.jpg")
    cv2.imwrite(path, image)

    return holds

def main():
    print(os.getcwd())    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='path/to/image', type=str, default='backend/services/files/example_wall.jpg')
    parser.add_argument('--holds_path', help='path/to/holds', type=str, default='backend/services/files/holds')
    parser.add_argument('--wid', help='wall id', type=int, default=0)
    parser.add_argument('--h_w', help='height of wall', type=int, default=450)
    args = parser.parse_args()
    image_path = args.image_path
    holds_path = args.holds_path
    wid = args.wid
    h_w = args.h_w

    # Directory where the file will be saved
    directory = "backend/services/files"

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(directory):
        os.makedirs(directory)

    image = cv2.imread(image_path)
    print(image.shape)
    print(convert_image_to_world(image, h_w, 989, 0))
    print(convert_image_to_world(image, h_w, 0, 0))
    print(convert_image_to_world(image, h_w, 989, 1319))

    if not os.path.exists(holds_path):

        print("\nExtracting bounding boxes...")
        bbox = ExtractBoundingBox(image_path, directory)

        print("\nSegmenting the image...")
        if(torch.cuda.is_available()):
            masks = SegmentAnything(image_path, directory)
        else:
            print("CUDA is not available, please enable CUDA to run the model")
    
        print("Getting holds...")
        holds = getHolds(image_path, bbox, masks, directory)
        # store the holds in the directory as npz files
        output_dir = os.path.join(directory, "./holds/")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i in range(len(holds)):
            # location = holds[i].location npz file
            sp.save_npz(os.path.join(output_dir, (str(i)+".npz")), sp.csr_matrix(holds[i]))
    
    else:
        print("Holds already exist in the directory.")
        # load the holds from the directory
        holds = []
        for file in os.listdir(holds_path):
            if file.endswith(".npz"):
                holds.append(sp.load_npz(os.path.join(holds_path, file)).toarray())
        print(f"Total {len(holds)} holds loaded from the directory.")

    




if __name__ == '__main__':
    main()
