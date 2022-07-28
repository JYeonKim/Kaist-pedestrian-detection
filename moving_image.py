from unicodedata import category
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

prediction_json_list = list()

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    #pdb.set_trace()
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])

        text = str(round(det_scores[0][i].cpu().detach().item(), 4))
        draw.text(xy=text_location, text=text, fill='white',
                font=font)
    del draw
    
    return annotated_image

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
        draw.rectangle(xy=box_location, outline='blue')
        draw.rectangle(xy=[l + 1. for l in box_location], outline='blue')
    
    del draw

    return annotated_image

if __name__ == '__main__':
    out_path = "/content/drive/MyDrive/kaist_output/ssd-h_detection.mp4"
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
            annotated_image = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
            annotated_image = show_gt_image(annotated_image, objects["annotation"])
            numpy_image = np.array(annotated_image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            # print(opencv_image)
            # annotated_image.save(os.path.join("/content/drive/MyDrive/kaist_detect_image_1", "detect_" + f_name))
            height, width, layers = opencv_image.shape
            size = (width, height)
            frame_array.append(opencv_image)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
