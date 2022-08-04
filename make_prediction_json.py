from unicodedata import category
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

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

def detect(rgb_original_image, thermal_original_image, min_score, max_overlap, top_k, image_id, suppress=None):
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
    # rgb, thermal 이미지를 받기 위해서 모두 전처리
    original_image = rgb_original_image # original_image를 rgb로 설정해둔다

    # rgb_original_image, thermal_original_image는 단순히 모델에 넣어두는 용도로만 사용 (헷갈림x)
    # 시각화는 original_image로 처리
    rgb_original_image = normalize(to_tensor(resize(rgb_original_image)))
    thermal_original_image = normalize(to_tensor(resize(thermal_original_image)))

    # Move to default device
    rgb_original_image = rgb_original_image.to(device)
    thermal_original_image = thermal_original_image.to(device)

    # Forward prop.
    # model의 rgb, thermal image를 모두 줘야한다.
    predicted_locs, predicted_scores = model(rgb_original_image.unsqueeze(0), thermal_original_image.unsqueeze(0))

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

    det_labels_not_reverse = det_labels[0].clone().detach()
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    create_prediction_json(image_id, det_boxes, det_labels_not_reverse, det_scores)

# prediction.json 생성
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

if __name__ == '__main__':
    with open(os.path.join("/content/drive/MyDrive/kaist_output", 'TEST_rgb_images.json'), 'r') as j:
        img_path = json.load(j) # rgb 이미지 경로를 img_path로 받음 
    with open(os.path.join("/content/drive/MyDrive/kaist_output", 'TEST_lwir_images.json'), 'r') as j:
        thermal_img_path = json.load(j) # thermal 이미지 경로

        for index, img in tqdm(enumerate(img_path)):
            f_name = img.split("/")[-1]
            
            # rgb image
            rgb_original_image = Image.open(img, mode='r')
            rgb_original_image = rgb_original_image.convert('RGB')
            
            # thermal image
            thermal_original_image = Image.open(thermal_img_path[index], mode='r')
            thermal_original_image = thermal_original_image.convert('RGB')
            
            # annotated images
            detect(rgb_original_image, thermal_original_image, min_score=0.2, max_overlap=0.5, top_k=200, image_id=index) # 0.2, 0.5를 바꾸지 말것.

    # Save to file(prediction.json)
    with open(os.path.join('/content/drive/MyDrive/kaist_output', 'ssd-h_crowded_make2.json'), 'w') as j:
        json.dump(prediction_json_list, j, indent=4)
