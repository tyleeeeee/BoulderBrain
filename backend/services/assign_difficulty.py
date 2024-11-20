import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import scipy.sparse as sp


def add_holds_img(image_path, holds_path, holds_img_path):
    image = cv2.imread(image_path)
    for i in range(len(os.listdir(holds_path))):
        dark_gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_gray_img_3channel = cv2.cvtColor(dark_gray_img, cv2.COLOR_GRAY2BGR)
        dark_gray_img_3channel = dark_gray_img_3channel*0.5
        dark_gray_img_3channel = dark_gray_img_3channel.astype(np.uint8)
        path = os.path.join(holds_path, f'{i}.npz')
        hold_img = sp.load_npz(path).toarray()
        # dark_gray_img_3channel = np.where(hold_img[:, :, None], image, dark_gray_img_3channel)
        dark_gray_img_3channel = np.where(hold_img[:, :, None], [0,0,255], dark_gray_img_3channel)
        # hold_img = hold_img.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(holds_img_path, (str(i)+".jpg")), dark_gray_img_3channel)

# def assign_difficulty(image_path, holds_path, holds_img_path, files_path, id):
def assign_difficulty(image_path, holds_path, holds_img_path, files_path):

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(holds_img_path):
        os.makedirs(holds_img_path)
        add_holds_img(image_path, holds_path, holds_img_path)
        print("Hold images added successfully")
    else:
        print("Hold images already exist")
       
    # print(f"You are assigning difficulty for hold{id}\n")
    # d1, d2, d3, d4, d5, d6, d7, d8 = input("Please enter 8 difficulties for difficulty ring (split by space): ").split()
    
    # # add the difficulties to the file
    # csv_path = os.path.join(files_path, 'difficulties.csv')
    # if not os.path.exists(csv_path):
    #     with open(csv_path, 'w') as f:  
    #         f.write('id, d1, d2, d3, d4, d5, d6, d7, d8\n') 

    # with open(csv_path, 'a') as f: 
    #     f.write(f'{id}, {d1}, {d2}, {d3}, {d4}, {d5}, {d6}, {d7}, {d8}\n')


if __name__ == '__main__':

    image_path = 'backend/services/files/IMG_7872.jpg'
    holds_path = f'backend/services/result8/holds'
    holds_img_path = f'backend/services/result8/holds_img'
    files_path = f'backend/services/result8'

    # i = int(input("Enter hold id: "))
    # assign_difficulty(image_path, holds_path, holds_img_path, files_path, i)
    assign_difficulty(image_path, holds_path, holds_img_path, files_path)


# python3 backend/services/assign_difficulty.py