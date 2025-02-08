from utils.conversion_helpers import yolo_to_bbox
from utils.evaluation_helper import ObjectDetectionEvaluator
import os
import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Subset
import random
from tqdm import tqdm

class LicensePlateDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, processor = None, resize = False, resize_shape = (640, 640)):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.processor = processor
        self.resize = resize
        self.resize_shape = resize_shape
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        annotation_path = os.path.join(self.annotations_dir, image_name.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        boxes = yolo_to_bbox(annotation_path, W, H)
        
        # -> gave me worse results than internal scaling
        if self.resize:
            image, boxes = self.resize_and_pad(image, boxes, self.resize_shape) 
        else:
            image = transforms.ToTensor()(image)
        
        if self.processor is not None:
            target = self.__parse_to_processor_targets(idx, image, boxes)
            target["processor"] = self.processor
            target["boxes_xyxy"] = self.processor
            return image, target
         
        # Handle empty bounding boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # No bounding boxes
            labels = torch.zeros(1, dtype=torch.int64)  # No labels
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        return image, target
    
    def __parse_to_processor_targets(self, idx, image, boxes):
        if len(boxes) == 0:
            target = {"image_id": idx, "annotations": []}
        else:
            # Detr uses xywh format
            boxes = np.array(boxes, dtype=np.float32)
            annotations = [
                {"bbox": [x_min, y_min, x_max - x_min, y_max - y_min], 
                 "category_id": 1,
                 "area": (x_max - x_min) * (y_max - y_min),
                 "iscrowd": 0}
                for x_min, y_min, x_max, y_max in boxes
            ]
            target = {
                "image_id": idx,
                "annotations": annotations
            }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return encoding
        
    def resize_and_pad(self, image, boxes, target_size):
        # gave me worse results than internal scaling
        h, w, _ = image.shape
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        
        pad_x = (target_size[0] - new_w) // 2
        pad_y = (target_size[1] - new_h) // 2
        
        padded_image = np.full((target_size[1], target_size[0], 3), 0, dtype=np.uint8)
        padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
        
        adjusted_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            x_min = x_min * scale + pad_x
            y_min = y_min * scale + pad_y
            x_max = x_max * scale + pad_x
            y_max = y_max * scale + pad_y
            adjusted_boxes.append([x_min, y_min, x_max, y_max])
        
        return transforms.ToTensor()(padded_image), adjusted_boxes
    
    def __len__(self):
        return len(self.image_files)
    
    
# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, processor=None):
    model.train()
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    total_loss = 0
    for images, targets in tqdm(data_loader, total=len(data_loader), desc="Processing batches"):
        if processor is not None:
            pixel_values = targets["pixel_values"].to(device)
            pixel_mask = targets["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in targets["labels"]]
            with autocast("cuda", enabled=scaler is not None):
                outputs = model(pixel_values = pixel_values, pixel_mask = pixel_mask, labels = labels)
                losses = outputs.loss
        else:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Enable mixed precision -> helps with GPU Memory
            with autocast("cuda", enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
            
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        total_loss += losses.item()
        
    return total_loss / len(data_loader)


@torch.inference_mode()
def detr_predict(model, img, conf_threshold=0.001, **kwargs):
    model.eval()
    processor = kwargs["processor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoding = processor(img, return_tensors="pt").to(device)
    outputs = model(**encoding)
    
    confidences = outputs.logits.softmax(-1)[..., :-1].max(-1)[0].cpu().numpy()
    boxes = outputs.pred_boxes.cpu().numpy()
    
    valid_indices = confidences > conf_threshold
    confidences = confidences[valid_indices]
    boxes = boxes[valid_indices]
    
    return confidences.tolist(), boxes.tolist()


@torch.inference_mode()
def fasterrcnn_predict(model, img, conf_threshold = 0.001, **kwargs):
    model.eval()
    if not isinstance(img, list):
        img = [img]

    with torch.inference_mode():
        outputs = model(img)

    confidences = []
    boxes = []
    
    for output in outputs:
        if "scores" in output and "boxes" in output:
            scores = output["scores"].cpu().numpy()
            bboxes = output["boxes"].cpu().numpy()
            
            valid_indices = scores > conf_threshold
            scores = scores[valid_indices]
            bboxes = bboxes[valid_indices]
    
            confidences.extend(scores.tolist())
            boxes.extend(bboxes.tolist())
    
    return confidences, boxes


@torch.inference_mode()
def evaluate(model, data_loader, device, prediction_function, processor=None):
    model.eval()
    
    all_imges = []
    all_bboxes = []
    for images, targets in data_loader:
        all_imges.extend([img.to(device) for img in images])
        all_bboxes.extend([target["boxes"].cpu().numpy().tolist() for target in targets])
       
    evaluator = ObjectDetectionEvaluator(model, all_imges, all_bboxes, prediction_function, processor=processor)
    metric_summary = evaluator.get_metric_summary(verbose=False)
    
    return metric_summary


def collate_fn_detr(batch):
    processor = batch[0][1]["processor"]
    pixel_values = [target["pixel_values"].squeeze() for _, target in batch]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [{k: v.to(device) for k, v in target["labels"][0].items()} for _, target in batch]
    new_targets = {}
    new_targets['pixel_values'] = encoding['pixel_values'].to(device)
    new_targets['pixel_mask'] = encoding['pixel_mask'].to(device)
    new_targets['labels'] = labels
    return "dummy", new_targets


def collate_fn(batch):
    return tuple(zip(*batch))


def get_subset(dataset, fraction=0.2, seed=1234):
    subset_size = int(len(dataset) * fraction)
    random.seed(seed)
    indices = random.sample(range(len(dataset)), subset_size)
    return Subset(dataset, indices)