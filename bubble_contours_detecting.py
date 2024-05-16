import cv2
import os
from tqdm import tqdm
import pandas as pd


MIN_CONTOUR_AREA = 1500  # adjust this value as needed
MAX_CONTOUR_AREA = 7000
df = pd.DataFrame()

for i in ['92', '95', '98']:
    directory = f'./output_data/{i}' # папка, созданная в задании 3
    if not os.path.exists(f'./validate_images/{i}'): # выходная папка
        os.makedirs(f'./validate_images/{i}')
    for file in tqdm(os.listdir(directory)):
        frame = cv2.imread(directory + '/' + file) # read file
        frame_h, frame_w = frame.shape[:2] # get height and weight
        frame = cv2.flip(frame, 0) # flipping image
        blured_frame = cv2.blur(frame, (7, 7), 0) # bluring
        binary = cv2.inRange(blured_frame, (0, 0, 0), (100, 80, 80)) # binarization
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
        counter = 0
        length = 0
        area = 0
        radius = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA and cv2.contourArea(cnt) < MAX_CONTOUR_AREA: # if True, draw contours and save image
                bbox = cv2.boundingRect(cnt)
                x, y, w, h = bbox
                # Ignore the bounding box if it's at the edges
                if x > 1 and y > 1 and (x + w) < frame_w-1 and (y + h) < frame_h-1:
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                    cv2.imwrite(f'./validate_images/{i}/{file}', frame)
                
                    # for dataframe

                    counter += 1
                    area += cv2.contourArea(cnt)
                    length += cv2.arcLength(cnt, True)
                    radius += w/2

        if counter > 0:
            d = pd.DataFrame({
                'Image Name': file,
                'Area': [area / counter],
                'Radius': [radius / counter],
                'Length': [length / counter],
                'Octane Number': [i]
            }, index=[0])
            df = pd.concat([df, d], ignore_index=True)


df.to_csv('./output_data.csv', index=False)
