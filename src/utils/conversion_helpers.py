import cv2

def yolo_to_bbox(annotation_path, img_width, img_height):
    bboxes = []
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            # I have only one class and ignore the class id
            _, x_center, y_center, width, height = map(float, parts)  
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            
            bboxes.append([x_min, y_min, x_max, y_max])
    
    return bboxes


def convert_yolo_bboxes(annotation_paths, image_paths):
    all_bboxes = []
    for annotation_path, img_path in zip(annotation_paths, image_paths):
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[0], img.shape[1]
        bboxes = yolo_to_bbox(annotation_path, img_width, img_height)
        all_bboxes.append(bboxes)
        
    return all_bboxes