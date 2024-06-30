# from image_processing_service import *
import cv2
import numpy as np
import os
import scipy as sp

def output_route(holds, holds_dict, routes, difficulties, wall_path, directory):
    i = 0
    # print(wall_path)
    for rid in routes:
        print(rid, ": ", routes[rid])
        route = routes[rid]
        i = i+1
        img = cv2.imread(wall_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        gray_img_3channel = gray_img_3channel*0.5
        gray_img_3channel = gray_img_3channel.astype(np.uint8)
        bg = np.zeros(img.shape, np.uint8)
        total_diff = 0
        for hold_yMax in route:
            hold_id = holds_dict[hold_yMax]
            # print(holds[hold_id].yMax)
            mask_path = holds[hold_id].location
            total_diff += holds[hold_id].difficulty
            # print(mask_path)
            mask = sp.sparse.load_npz(os.path.join(mask_path)).toarray()
            # convert mask to 3 channels
            mask = np.stack([mask, mask, mask], axis=-1)
            bg = np.where(mask, img, bg)
        # 0 as graysacle, otherwise as colored
        bg = np.where(bg, img, gray_img_3channel)
        # cv2.imwrite(f'backend/services/files/route{i}.jpg', bg)
        cv2.imwrite(f'{directory}/route{i}.jpg', bg)
        # print("Route found: " + str(route))
        avg_diff = total_diff/len(route)
        avg_diff = avg_diff.round(2)
        print("Difficulty using the old system (one value per hold): " + str(avg_diff))
        print("Difficulty using the new system (grip-angle-dependent):", difficulties[rid])
    

# if __name__ == '__main__':

#     test_hold = get_holds_from_image()
#     for h in test_hold:

#         print(h.wall, h.color, h.end, h.yMax, h.difficulty, h.id)
#     print("-----")
#     holds = get_holds_main()

#     # routes = [[2,5,8,11,15,30,35,40],
#     #           [3,4,5,6,7,8,9,10]]

#     # create a dictionary to map yMax to hold index
#     holds_dict = {}
#     i = 0
#     for hold in holds:

#         print(hold.wall, hold.color, hold.end, hold.yMax, hold.difficulty, hold.id)
#         key = (int(hold.yMax[0]), int(hold.yMax[1]))
#         # holds_dict[hold.yMax] = i
#         holds_dict[key] = i
#         i += 1

#     routes = [[(0, 1150),(218, 252),(0, 955),(496, 1033),(559, 111),(364, 121),(407, 538),(96, 390)],
#               [(158, 746),(69, 270),(218, 252),(277, 256),(119, 235),(0, 955),(811, 361),(668, 1143)],
#               [(306, 569), (559, 111), (4, 440), (229, 451), (95, 527), (219, 1024), (200, 345), (438, 87)]]
    
#     wall = cv2.imread('backend/services/files/example_wall.jpg')
#     output_route(holds, holds_dict, routes, wall)