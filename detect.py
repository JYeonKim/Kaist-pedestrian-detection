from unicodedata import category
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

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

def detect(original_image, min_score, max_overlap, top_k, image_id, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

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
    
    # 필요한 변수들
    # det_boxes
    # det_labels
    # det_scores

    det_labels_not_reverse = det_labels[0].clone().detach()
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    # 에러 발생
    # PIL Issue, OSError: cannot open resource
    # font = ImageFont.truetype("./calibril.ttf", 15)
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
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    create_prediction_json(image_id, det_boxes, det_labels_not_reverse, det_scores)

    return annotated_image

def create_prediction_json(image_id, det_boxes, det_labels, det_scores):

    for i in range(len(det_boxes)):
        bbox = det_boxes[i].cpu().detach().tolist()
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        category_id = det_labels[i].cpu().detach().item()
        score = det_scores[0][i].cpu().detach().item()
        prediction_json_list.append({'image_id': image_id,
                                "category_id": category_id,
                                'bbox': bbox, 
                                'score': score})
    """
    predict_example.json
    {
        "image_id": 0,
        "category_id": 0,
        "bbox": [
            0.0,
            0.0,
            640.0,
            512.0
        ],
        "score": 0.0
    },
    --------------------------
    test object annotation json example
    {
        "image": {
            "width": 640,
            "height": 512,
            "id": 1539,
            "file_name": "I01539"
        },
        "annotation": [
            {
                "bbox": [
                    0,
                    0,
                    0,
                    0
                ],
                "category_id": -1,
                "id": 0,
                "image_id": 1539,
                "is_crowd": 0
            }
        ]
    }
    """


if __name__ == '__main__':
    # file_name = list()
    with open(os.path.join("/content/drive/MyDrive/kaist_output", 'TEST_rgb_images.json'), 'r') as j:
        img_path = json.load(j)   
        for index, img in tqdm(enumerate(img_path)):
            f_name = img.split("/")[-1]
            original_image = Image.open(img, mode='r')
            original_image = original_image.convert('RGB')
            annotated_image = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200, image_id=index)
            # annotated_image.save(os.path.join("/content/drive/MyDrive/kaist_detect_image_1", "detect_" + f_name))
    
    # Save to file(prediction.json)
    with open(os.path.join('/content/drive/MyDrive/kaist_output', 'rgb_prediction_0726.json'), 'w') as j:
        json.dump(prediction_json_list, j, indent=4)
