import os
import os.path as osp
import torch
from PIL import Image
from tqdm import tqdm


label_path = "/path/to/yolov5 predicted labels"
image_path = "/path/to/images"
output_path = "/path/to/cropped images"
crop_width = 512
crop_height = 512


def extend_box(cx, cy, width, height):
    assert crop_width <= width and crop_height <= height
    radius_width = crop_width // 2
    radius_height = crop_height // 2

    if cx + radius_width > width:
        x = width - crop_width
    else:
        x = max(0, cx - radius_width)

    if cy + radius_height > height:
        y = height - crop_height
    else:
        y = max(0, cy - radius_height)

    return x, y


def crop(labels, img_id):
    img = Image.open(osp.join(image_path, img_id+".png"))
    width, height = img.size

    os.makedirs(osp.join(output_path, img_id), exist_ok=True)

    cnt = 0
    for cx, cy, w, h in labels:
        cx = int(cx * width)
        cy = int(cy * height)
        w = int(w * width)
        h = int(h * height)
        
        x, y = extend_box(cx, cy, width, height)

        patch = img.crop((x, y, x+crop_width, y+crop_height))

        patch.save(osp.join(output_path, img_id, f"{cnt:05d}.png"))

        cnt += 1


def random_crop(img_id, num_crops=5):
    img = Image.open(osp.join(image_path, img_id+".png"))
    width, height = img.size

    os.makedirs(osp.join(output_path, img_id), exist_ok=True)

    x = torch.randint(0, width - crop_width, (num_crops,)).tolist()
    y = torch.randint(0, height - crop_height, (num_crops,)).tolist()

    for i in range(num_crops):
        patch = img.crop((x[i], y[i], x[i]+crop_width, y[i]+crop_height))

        patch.save(osp.join(output_path, img_id, f"{i:05d}.png"))


def main():
    img_list = os.listdir(image_path)
    img_list = list(filter(lambda x: x.endswith("png"), img_list))
    img_list.sort()
    for img_file in tqdm(img_list):
        img_id = img_file.split(".")[0]
        label_file = img_id+".txt"

        if osp.exists(osp.join(label_path, label_file)):
            with open(osp.join(label_path, label_file), "r") as f:
                labels = f.readlines()
            
            labels = [label.strip().split(' ')[1:-1] for label in labels]
            labels = [list(map(float, label)) for label in labels]

            crop(labels, img_id)
        else:
            # random crop
            random_crop(img_id)


if __name__ == "__main__":
    main()