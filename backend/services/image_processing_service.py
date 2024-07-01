from .hold import Hold
from .wall import Wall

import argparse
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
import csv

holds = []

#def convert image coordinate to world coordinate(img, h_w)

# (0,0)----------------(0,width)
#   |                       |
#   |   image coordinate    |
#   |                       |
# (height,0)--------(height,width)


# (0,h_w)-----------------(w_w,h_w) h_w: height of wall
#   |                       |
#   |   world coordinate    |
#   |                       |
# (0,0)-------------------(w_w,0) w_w: width of wall


def convert_image_to_world(image, h_w, img_coord_x, img_coord_y):
    height, width, _ = image.shape
    w_w = h_w * (width/ height) # width of wall
    world_x = w_w / width * img_coord_y
    world_y = h_w - (h_w / height * img_coord_x)
    return np.array([world_x, world_y])

#def getValidHolds(image):
#  Use Yolo-world to extract holds' bounding boxes <<ExtractBoundingBox>>
#  Use SAM to extract segments, using bounding boxes as prompts <<SegmentAnything>>
#  For each hold, create a hold object with its location <<getHolds>>
#  return set of masks of holds

# Remove overlapping bounding boxes
# https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return boxes[pick]


def ExtractBoundingBox(image_path, directory, wid):

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
    # results_path = os.path.join(directory, "yolo_world_result.jpg")
    # results[0].save(results_path)

    img = cv2.imread(image_path)

    # Assuming 'results[0].boxes.xywh' holds the tensor with the bounding box information
    xyxy = results[0].boxes.xyxy

    # Convert tensor to a numpy array
    xyxy_np = xyxy.cpu().numpy()
    bboxes = non_max_suppression_slow(xyxy_np, 0.3)

    i = 0
    for bbox in bboxes:
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),3)
        i = i + 1

    # images_path = os.path.join(directory, f"{wid}_bbox_result.jpg")
    # cv2.imwrite(images_path, img)
    print(f"Total {i} items detected in the image.")
    return bboxes

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
    # mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=40)

    # Read the image from the path
    image= cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate segmentation mask
    # output_mask = mask_generator.generate(image)
    # print(output_mask)

    # _,axes = plt.subplots(1,2, figsize=(16,16))
    # axes[0].imshow(image)
    # image_path = os.path.join(directory, "sam_result.jpg")
    # show_output(output_mask, axes[1], image_path)

    masks = mask_generator.generate(image)
    # store the masks in the directory
    output_dir = os.path.join(directory, "./masks/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(len(masks)): 
        sp.save_npz(os.path.join(output_dir, (str(i)+".npz")), sp.csr_matrix(masks[i]['segmentation'] ))

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



def getHolds(image_path, bboxes, masks, directory, wid):
    
    # find the bounding box of the mask corresponding to the bboxes
    image = cv2.imread(image_path)
    
    count = 0
    # print(image.shape)
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
            # combine the masks that are within the bounding box
            if (abs(bbox[0] - m0) <= 15 and abs(bbox[1] - m1) <= 15 and abs(bbox[2] - m2) <= 15 and abs(bbox[3] - m3) <= 15):       
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
            
        # if(flag == False):
        #     print(f"False: {bbox[0], bbox[1], bbox[2], bbox[3]}")
            
    print(f"Total {count} items auto-detected in the image.")
    # plt.imshow(image)
    path = os.path.join(directory, f"{wid}_holds.jpg")
    cv2.imwrite(path, image)

    return holds, count



def getHolds_manually(image_path, masks, directory, click_points, count, wid):
    # holds = []
    # find the bounding box of the mask corresponding to the bboxes
    image = cv2.imread(image_path)
    # print(f"Total {len(click_points)} items clicked in the image.")
    
    for point in click_points:
        hold_mask = np.zeros((image.shape[0], image.shape[1]))
        for j in range(len(masks)):
            # check if point is in masks[j]['segmentation']
            hold_mask = np.zeros((image.shape[0], image.shape[1]))
            hold_mask = np.where(masks[j]['segmentation'], 1, hold_mask)
            if(hold_mask[point[1], point[0]] == 1):
                image = np.where(masks[j]['segmentation'][:,:,None], [0,0,255], image)
                holds.append(hold_mask)
                count = count + 1
                break
            
    # print(f"Total {count} items detected in the image.")
    # plt.imshow(image)
    path = os.path.join(directory, f"{wid}_holds.jpg")
    cv2.imwrite(path, image)

    return holds, count



# main() function
# holds_path = f'services/result{wall.id}/holds'
# files_path = f'services/result{wall.id}'
def get_holds_main(wall, image_path, holds_path, files_path):

    wid = wall.id
    h_w = wall.height

    # Directory where the file will be saved
    directory = files_path

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(directory):
        os.makedirs(directory)

    image = cv2.imread(image_path)
    count = 0

    if not os.path.exists(holds_path):

        # print("\nExtracting bounding boxes...")
        # bbox = ExtractBoundingBox(image_path, directory, wid)

        print("\nSegmenting the image...")
        if(torch.cuda.is_available()):
            masks = SegmentAnything(image_path, directory)
        else:
            print("CUDA is not available, please enable CUDA to run the model")
    
        # print("Getting holds...")
        # holds, count = getHolds(image_path, bbox, masks, directory, wid)
        
        ######################################################################################################
        # manually add holds
        # count = 0
        print("Manually adding holds...")
        
        
        # click_points = [(728, 315), (586, 259), (619, 307), (625, 373), (872, 244), (1036, 141), (935, 360), (874, 383), 
        #                 (821, 472), (809, 499), (683, 501), (560, 380), (861, 543), (979, 583), (987, 678), (988, 802), 
        #                 (923, 679), (863, 678), (746, 678), (684, 680), (563, 687), (379, 568), (641, 774), (677, 860), 
        #                 (927, 796), (930, 867), (989, 915), (704, 1069), (456, 980), (327, 632), (152, 632), (217, 706), 
        #                 (334, 686), (334, 810), (335, 861), (216, 803), (159, 740), (39, 758), (103, 864), (162, 934), 
        #                 (53, 990), (171, 1098), (295, 1049), (222, 979), (340, 988), (386, 915), (464, 1042), (399, 1142), 
        #                 (171, 1217), (237, 1262), (213, 1324), (274, 1337), (458, 1212), (505, 1334), (739, 1332), (855, 1401), 
        #                 (907, 1335), (924, 1208), (1000, 1152), (1007, 1062), (867, 965), (1050, 1409)]

        click_points = [(130, 645), (82, 1299), (65, 1631), (115, 1916), (274, 1290), (269, 969), (344, 94), (462, 461),
                        (631, 113), (482, 773), (695, 1083), (453, 1218), (642, 1440), (581, 1657), (951, 1237), (970, 801), 
                        (1374, 130), (1548, 155), (1539, 363), (1666, 711), (1331, 849), (1524, 1127), (1792, 1107), (1737, 1304), 
                        (1518, 1377), (1406, 1251), (1078, 1702), (1049, 1936), (1816, 2227), (1802, 1939), (1801, 1588), (1948, 1463), 
                        (1980, 1720)]


        # image_path = os.path.join(directory, f"{wid}_holds.jpg")
        holds, count =  getHolds_manually(image_path, masks, directory, click_points, count, wid)

        print(f"Total {count} items detected in the image.")
        ######################################################################################################
        
        # # store the holds in the directory as npz files
        output_dir = os.path.join(directory, "./holds/")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i in range(len(holds)):
            # location = holds[i].location npz file
            sp.save_npz(os.path.join(output_dir, (str(i)+".npz")), sp.csr_matrix(holds[i]))


    # load the holds from the directory
    Holds = []
    for i in range(len(os.listdir(holds_path))):
        path = os.path.join(holds_path, f'{i}.npz')
        mask = sp.load_npz(path).toarray()
        # m = mask*255
        # cv2.imwrite(f"{i}.jpg", m)
        ymax = find_ymax(mask)
        # print(ymax)
        ymax_world = convert_image_to_world(image, h_w, ymax[0], ymax[1])
        # print(f"[{ymax_world[0]}, {ymax_world[1]}],")
                
        # assign difficulty base on difficulty table
        difficulty_path = os.path.join(directory, "difficulties.csv")
        # print(i)
        if not os.path.exists(difficulty_path):
            difficulty_right, difficulty_top_right, difficulty_top, difficulty_top_left, difficulty_left, difficulty_bottom_left, difficulty_bottom, difficulty_bottom_right = 1, 1, 1, 1, 1, 1, 1, 1
        else:
            d = []
            with open(difficulty_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if int(row['id']) == i:  
                        # print(row)
                        for key, value in row.items():
                            # print(key, value)
                            if(key != 'id'):
                                d.append(int(value))
            csvfile.close()
            if(len(d) == 8):
                difficulty_right, difficulty_top_right, difficulty_top, difficulty_top_left, difficulty_left, difficulty_bottom_left, difficulty_bottom, difficulty_bottom_right = d
            else:
                difficulty_right, difficulty_top_right, difficulty_top, difficulty_top_left, difficulty_left, difficulty_bottom_left, difficulty_bottom, difficulty_bottom_right = 2, 2, 2, 2, 2, 2, 2, 2
                   
        # holds.append(sp.load_npz(os.path.join(holds_path, file)).toarray())
        difficulty = np.random.randint(1,7,1)[0]
        Holds.append(Hold(wall, path, "blue1", False, [ymax_world[0].round(2), ymax_world[1].round(2)], difficulty, 
                          difficulty_right, difficulty_top_right, difficulty_top, difficulty_top_left, difficulty_left, difficulty_bottom_left, difficulty_bottom, difficulty_bottom_right, 
                          i))
        # Holds.append(Hold(dummy_wall, os.path.join(holds_path, file), "blue1", False,  [ymax_world[0], ymax_world[1]]))
        # print(difficulty_1, difficulty_2, difficulty_3, difficulty_4, difficulty_5, difficulty_6, difficulty_7, difficulty_8)
    print(f"Total {len(Holds)} holds extracted from the image.")

    Holds_dict = {}
    for h in Holds:
        Holds_dict[tuple(h.yMax)] = h.id
        
    return Holds, Holds_dict


# if __name__ == '__main__':
#     get_holds_main()


### hahaha
# from ultralytics import YOLO

# # Print the current working directory to confirm it
# print("Current Working Directory:", os.getcwd())
#
#
# # Path to the image file (assuming you've confirmed the current working directory is the project root)
# image_path = 'climbingWallTestImage.jpg'
#
# # Load and run YOLO model
# model = YOLO('yolov9c-seg.pt')
# result = model(image_path, show=True, save=True)


#def getBodyFromImage(image):
  # 1. Call Mediapipe to extract pose.

def get_holds_from_image(dummy_wall):
    #TODO: this is a placeholder! Change code later!
    # 1. Call SAM to extract segments
    # 2. Label segments as "hold" or "not hold".
    #    Option 2a: Train a classifier model (finding data could be hard)
    #    Option 2b: Use a formula to determine if a segment is a hold
    # 3. For each hold, create a hold object with its location, shape, and color.
    # 4. Return the set of holds.

    # dummy_wall = Wall(id=1, height=400, width=500)
    # return [
    #     Hold(dummy_wall, [50, 100], "blue1", False, [50, 105]),  # Lower left
    #     Hold(dummy_wall, [150, 120], "green1", False, [150, 125]),  # Mid left
    #     Hold(dummy_wall, [250, 150], "yellow1", False, [250, 155]),  # Center wall
    #     Hold(dummy_wall, [350, 180], "green2", False, [350, 185]),  # Mid right
    #     Hold(dummy_wall, [450, 210], "red1", False, [450, 215]),  # Lower right
    #     Hold(dummy_wall, [100, 220], "blue2", False, [100, 225]),  # Mid upper left
    #     Hold(dummy_wall, [200, 250], "red2", True, [200, 255]),  # Upper center
    #     Hold(dummy_wall, [300, 280], "yellow2", True, [300, 285]),  # Mid upper right
    #     Hold(dummy_wall, [400, 310], "red3", True, [400, 315]),  # Upper right
    #     Hold(dummy_wall, [50, 340], "blue3", True, [50, 345]),  # Top left
    #     Hold(dummy_wall, [150, 370], "green3", True, [150, 375]),  # Top mid left
    #     Hold(dummy_wall, [250, 400], "yellow3", True, [250, 405]),  # Top center
    # ]

    return [
        Hold(dummy_wall, [50, 100], "blue1", False, [50, 105], 0.5, 1),  # Lower left
        Hold(dummy_wall, [150, 120], "green1", False, [150, 125], 0.5, 2),  # Mid left
        Hold(dummy_wall, [250, 150], "yellow1", False, [250, 155], 0.5, 3),  # Center wall
        Hold(dummy_wall, [350, 180], "green2", False, [350, 185], 0.5, 4),  # Mid right
        Hold(dummy_wall, [450, 210], "red1", False, [450, 215], 0.5, 5),  # Lower right
        Hold(dummy_wall, [100, 220], "blue2", False, [100, 225], 0.5, 6),  # Mid upper left
        Hold(dummy_wall, [200, 250], "red2", True, [200, 255], 0.5, 7),  # Upper center
        Hold(dummy_wall, [300, 280], "yellow2", True, [300, 285], 0.5, 8),  # Mid upper right
        Hold(dummy_wall, [400, 310], "red3", True, [400, 315], 0.5, 9),  # Upper right
        Hold(dummy_wall, [50, 340], "blue3", True, [50, 345], 0.5, 10),  # Top left
        Hold(dummy_wall, [150, 370], "green3", True, [150, 375], 0.5, 11),  # Top mid left
        Hold(dummy_wall, [250, 400], "yellow3", True, [250, 405], 0.5, 12),  # Top center
    ]

# def generate_dense_holds(wall):
#     holds = []
#     max_reach = 70  # climber's max reach for simplicity
#     vertical_spacing = max_reach * 0.3
#     horizontal_spacing = max_reach * 0.3

#     # Calculate how many holds can fit based on spacing and wall dimensions.
#     num_vertical = int(wall.height / vertical_spacing)
#     num_horizontal = int(wall.width / horizontal_spacing)

#     # Generate holds in a grid-like pattern.
#     for v in range(num_vertical):
#         for h in range(num_horizontal):
#             x = h * horizontal_spacing + (vertical_spacing / 2 if v % 2 == 1 else 0)
#             y = v * vertical_spacing
#             if x <= wall.width and y <= wall.height:
#                 holds.append(Hold(wall, [x, y], "green", False, [x, y + 5]))

#     # Add some final holds at the top of the wall for good measure.
#     for h in range(num_horizontal):
#         x = h * horizontal_spacing + (vertical_spacing / 2 if v % 2 == 1 else 0)
#         y = wall.height
#         if x <= wall.width: holds.append(Hold(wall, [x, y], "green", False, [x, y + 5]))

#     return holds
    