import argparse
import cv2
import numpy as np

from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall

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
#  <Use Yolo-world to extract holds' bounding boxes>
#  <Use SAM to extract segments, using bounding boxes as prompts>
#  <For each hold, create a hold object with its location, shape, and color>
#  return set of holds


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='path/to/image', type=str, default='../example_wall.jpg')
    parser.add_argument('--h_w', help='height of wall', type=int, default=450)
    # parser.add_argument('--classifier', help='classifier', type=str, default='nearest_neighbor')
    # parser.add_argument('--dataset_dir', help='dataset directory', type=str, default='../hw2_data/p1_data/')
    args = parser.parse_args()
    image_path = args.image_path
    h_w = args.h_w

    image = cv2.imread(image_path)
    print(image.shape)
    print(convert_image_to_world(image, h_w, 989, 0))
    print(convert_image_to_world(image, h_w, 0, 0))
    print(convert_image_to_world(image, h_w, 989, 1319))



if __name__ == '__main__':
    main()
