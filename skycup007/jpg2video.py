import os
import cv2
import glob
import argparse


def jpg2video(data_path, video, images):

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    img_size = (640, 480)
    videoWriter = cv2.VideoWriter(video, fourcc, 30, img_size)

    for image in images:
        image_path = os.path.join(data_path, video[:-4], image)
        image = cv2.imread(image_path)
        print(image_path)
        videoWriter.write(image)
    videoWriter.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/skycup/Documents/skycup007/data/output_data/")
    flags = parser.parse_args()

    folders = os.listdir(flags.data_path)
    for folder in folders:
        images = glob.glob(os.path.join(flags.data_path, folder, "*.bmp"))
        jpg2video(flags.data_path, folder + '.avi', sorted(images))