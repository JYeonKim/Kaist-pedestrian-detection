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

def detect(rgb_original_image, thermal_original_image, annotation, min_score, max_overlap, top_k, image_id, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    
    original_image = rgb_original_image 

    # Transform
    # rgb_image, thermal_image 단순히 모델에 넣어두는 용도로만 사용 (헷갈림x)
    # 시각화는 original_image로 처리
    rgb_image = normalize(to_tensor(resize(rgb_original_image)))
    thermal_image = normalize(to_tensor(resize(thermal_original_image)))

    # Move to default device
    rgb_image = rgb_image.to(device)
    thermal_image = thermal_image.to(device)

    # Forward prop.
    # model의 rgb, thermal image를 모두 줘야한다.
    predicted_locs, predicted_scores = model(rgb_image.unsqueeze(0), thermal_image.unsqueeze(0))

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
    create_prediction_json(image_id, det_boxes, det_labels_not_reverse, det_scores)
    
    if det_labels == ['background']:
        # Just return original image
        return rgb_original_image, thermal_original_image

    annotated_image = original_image
    annotated_image2 = thermal_original_image

    for i in range(0,2):
      if i == 0:
          draw = ImageDraw.Draw(annotated_image)
      else:
          draw = ImageDraw.Draw(annotated_image2)
      font = ImageFont.load_default()

      # Suppress specific classes, if needed
      for i in range(det_boxes.size(0)):
          if suppress is not None:
              if det_labels[i] in suppress:
                  continue
          # Boxes
          box_location = det_boxes[i].tolist()
          draw.rectangle(xy=box_location, outline='red')
          draw.rectangle(xy=[l + 1. for l in box_location], outline='red')

          # Text
          text_size = font.getsize(det_labels[i].upper())
          text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
          textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                              box_location[1]]
          draw.rectangle(xy=textbox_location, fill='red')

          text = str(round(det_scores[0][i].cpu().detach().item(), 4))
          draw.text(xy=text_location, text=text, fill='white',
                  font=font)
      del draw


    return annotated_image, annotated_image2

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

# image에 gt bounding box 추가
def show_gt_image(original_image, annotation):
    bboxes = []
    labels = []
    for obj in annotation:
        if obj["category_id"] == -1:
            continue
        bboxes.append(obj["bbox"])
        labels.append(obj["category_id"])
    
    if len(bboxes) == 0:
        return original_image
    
    annotated_image = original_image
    draw = ImageDraw.Draw(original_image)

    for i in range(len(bboxes)):
        box_location = bboxes[i]
        draw.rectangle(xy=box_location, outline='blue')
        draw.rectangle(xy=[l + 1. for l in box_location], outline='blue')
    del draw

    return annotated_image

if __name__ == '__main__':
    out_path_min = "/content/drive/MyDrive/kaist_output/ssd-h_small2"
    fps = 3
    frame_array = []
    with open(os.path.join("/content/drive/MyDrive/kaist_output", 'TEST_rgb_images.json'), 'r') as j:
        img_path = json.load(j) # rgb 이미지 경로를 img_path로 받음 
    with open(os.path.join("/content/drive/MyDrive/kaist_output", 'TEST_lwir_images.json'), 'r') as j:
        thermal_img_path = json.load(j) # thermal 이미지 경로

        for index, img in tqdm(enumerate(img_path)):
            f_name = img.split("/")[-1]
            
            # annotation json 접근 (gt를 같이 표시하기 위해서)
            json_name = f_name.split(".")[0]
            img_split = img.split("/")
            json_path = os.path.join("/content/drive/MyDrive/KAIST_PD", "annotation_json", img_split[6], img_split[7], json_name + ".json")
            with open(os.path.join(json_path), 'r') as j:
                objects = json.load(j)
            
            # rgb image
            rgb_original_image = Image.open(img_path[index], mode='r')
            rgb_original_image = rgb_original_image.convert('RGB')
            
            # thermal image
            thermal_original_image = Image.open(thermal_img_path[index], mode='r')
            thermal_original_image = thermal_original_image.convert('RGB')

            # annotated images
            annotated_image, thermal_annotated_image = detect(rgb_original_image, thermal_original_image, objects["annotation"], min_score=0.2, max_overlap=0.5, top_k=200, image_id=index) # 0.2, 0.5를 바꾸지 말것.
            annotated_image = show_gt_image(annotated_image, objects["annotation"]) # annotated_image를 받았을 때 이미 gt가 있어야 한다.
            thermal_annotated_image = show_gt_image(thermal_annotated_image, objects["annotation"]) # annotated_image를 받았을 때 이미 gt가 있어야 한다.
            # annotated_image.save('/content/drive/MyDrive/kaist_output/output_try/'+ str(index) + '_' + f_name + '.jpg',"JPEG")
            
            an_size = annotated_image.size
            new_image = Image.new('RGB', (2*an_size[0], an_size[1]), (250,250,250))
            new_image.paste(annotated_image, (0,0))
            new_image.paste(thermal_annotated_image, (an_size[0],0))
            # # an_size (640, 512) # new_image.size # (1280, 512)

            # mp4를 위해서 frame_array에 frame 추가
            numpy_image = np.array(new_image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            
            height, width, layers = opencv_image.shape
            size = (width, height)
            
            frame_array.append(opencv_image)

    # mp4 생성
    out = cv2.VideoWriter(out_path_min+'_all.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

    # Save to file(prediction.json)
    with open(os.path.join('/content/drive/MyDrive/kaist_output', 'ssd-h_small2.json'), 'w') as j:
        json.dump(prediction_json_list, j, indent=4)
