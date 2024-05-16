import os
import sys
from PIL import Image
from tqdm import tqdm


def resize_images(input_dir, output_dir, width, height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in tqdm(os.listdir(input_dir)):
        if image_name.endswith('.jpg'):
            input_path = os.path.join(input_dir, image_name)
            output_path = os.path.join(output_dir, image_name)

            with Image.open(input_path) as img:
                resized_img = img.resize((width, height))
                resized_img.save(output_path, 'JPEG')
                # print(f"Image {image_name} was resized and saved to {output_dir}")


if __name__ == '__main__':
    width, height = map(int, sys.argv[1:])
    output_dir = './output_data/92' # выходные данные
    input_dir = './output_data_raw/92' # входные данные
    resize_images(input_dir, output_dir, width, height)
    output_dir = './output_data/95'
    input_dir = './output_data_raw/95'
    resize_images(input_dir, output_dir, width, height)
    output_dir = './output_data/98'
    input_dir = './output_data_raw/98'
    resize_images(input_dir, output_dir, width, height)
