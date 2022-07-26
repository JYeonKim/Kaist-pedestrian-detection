from unicodedata import category
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def show_gt_image(orininal_image, annotation):
    bboxes = []
    labels = []
    for obj in annotation:
        if obj["category_id"] == -1:
            continue
        bboxes.append(obj["bbox"])
        labels.append(obj["category_id"])
    
    if len(bboxes) == 0:
        return original_image

    labels = [rev_label_map[l] for l in labels]
    
    if labels == ['background']:
    # Just return original image
        return original_image
    
    annotated_image = original_image
    draw = ImageDraw.Draw(orininal_image)
    # pdb.set_trace()
    font = ImageFont.load_default()

    for i in range(len(bboxes)):
        box_location = bboxes[i]
        draw.rectangle(xy=box_location, outline=label_color_map[labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[labels[i]])

        text_size = font.getsize(labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[labels[i]])
        draw.text(xy=text_location, text=labels[i].upper(), fill='white', font=font)
    
    del draw

    return annotated_image

if __name__ == '__main__':
    print("시작8")
    out_path = "/content/drive/MyDrive/kaist_gt_image/gt_image.mp4"
    fps = 5
    frame_array = []
    with open(os.path.join("/content/drive/MyDrive/kaist_output", 'TEST_rgb_images.json'), 'r') as j:
        img_path = json.load(j)   
        for index, img in tqdm(enumerate(img_path)):
            f_name = img.split("/")[-1]
            json_name = f_name.split(".")[0]
            img_split = img.split("/")
            json_path = os.path.join("/content/drive/MyDrive/KAIST_PD", "annotation_json", img_split[6], img_split[7], json_name + ".json")
            with open(os.path.join(json_path), 'r') as j:
                objects = json.load(j)
            original_image = Image.open(img, mode='r')
            original_image = original_image.convert('RGB')
            annotated_image = show_gt_image(original_image, objects["annotation"])
            numpy_image = np.array(annotated_image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            height, width, layers = opencv_image.shape
            size = (width, height)
            frame_array.append(opencv_image)
            annotated_image.save(os.path.join("/content/drive/MyDrive/kaist_gt_image", "gt_" + f_name))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
