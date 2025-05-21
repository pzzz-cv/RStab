import cv2
import os
import numpy as np
import sys

def video2images(video_path, image_save_dir):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        flag, frame = cap.read()
        if not flag:
            break

        if frame_count >= 0:
            image_path = os.path.join(image_save_dir, '{:05d}'.format(frame_count) + '.png')
            # frame = np.where(frame > 220, 255, 0)  #sam产生的mask需要该操作
            cv2.imwrite(image_path, frame)
            
        # if frame_count >= 0:
        #     for i in range(3):
        #         image_path = os.path.join(image_save_dir, '{:05d}-{:01d}'.format(frame_count, i) + '.jpg')
        #         cv2.imwrite(image_path, frame)

        # if frame_count >= 0:
        #     image_path = os.path.join(image_save_dir, '{:05d}'.format(frame_count) + '.jpg')
        #     cv2.imwrite(image_path, frame)
        #     if frame_count == 8:
        #         for i in range(30):
        #             image_path = os.path.join(image_save_dir, '{:05d}-{:02d}'.format(frame_count, i) + '.jpg')
        #             cv2.imwrite(image_path, frame)
        
        frame_count += 1
    
    cap.release()
    print('Transfor finished!')



if __name__ == '__main__':

    video_path = sys.argv[1]
    save_image_dir = sys.argv[2]
    
    # file_name = 'balloon'

    # pwd = os.getcwd()
    # work_path = os.path.join(pwd, file_name)
    # print('Work path: ', work_path)
    # video_path = os.path.join(work_path, file_name + '.mp4')
    # save_image_dir = os.path.join(work_path, 'images')
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    print('Video path: ', video_path)
    print('Images path: ', save_image_dir)

    video2images(video_path, save_image_dir)
