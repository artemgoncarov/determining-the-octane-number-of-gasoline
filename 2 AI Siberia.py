import cv2
import os
from tqdm import tqdm


def split_video(video_path):

    video = cv2.VideoCapture(video_path)
    counter = 1

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        cv2.imwrite(f"./output_data_raw/{video_path.split('/')[-2]}/{video_path.split('/')[-1].split('.mp4')[0]}_{counter}.jpg", frame)
        counter += 1


data_path = 'input_data_raw/' # путь до входных данных

os.mkdir('./output_data_raw')
os.mkdir('./output_data_raw/92')
os.mkdir('./output_data_raw/95')
os.mkdir('./output_data_raw/98') # создание папок

# разбиваем видео на кадры

for i in ['92', '95', '98']:
    for j in tqdm(os.listdir(data_path + i)):
        if '.mp4' in j:
            split_video(data_path + i + '/' + j)
            print(f'Video {i}/{j} was completed')