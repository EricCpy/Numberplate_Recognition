import random
import cv2
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import tensor
from torchmetrics.classification import AveragePrecision
import torch
import easyocr
import time

class ObjectDetectionEvaluator:
    def __init__(self, model, images, true_bboxes, predict_function, default_iou_threshold = 0.5, processor=None):
        self.model = model
        self.images = images
        self.true_bboxes = true_bboxes
        self.predict_function = predict_function
        self.default_iou_threshold = default_iou_threshold
        self.processor = processor
        self.predicted_confidences, self.predicted_labels, self.iou_scores = self.__calculate_model_predictions(default_iou_threshold)
    
    
    @staticmethod
    def iou(box1, box2):
        x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area_box1 + area_box2 - intersection
        return intersection / (union + 1e-7)
    
    
    def __calculate_model_predictions(self, iou_threshold = 0.5):
        # NOTE: The metric functions are not completly correct when you provide a different IoU than the iou_threshold,
        # bc I dont take into account that when the confidence for one box is too low
        # that another predicted box with lower confidence but with same or higher IoU could be true then, I could recalculate for every IoU
        # but this takes a lot of time -> this trade-off is done to optimize the calculations here
        # (there is probably an algorithm to do this more efficient, but thats not the scope of the project and seems like the influence of this error is small)
        all_predicted_confidences, all_predicted_labels, all_iou_scores = [], [], []
        
        for img, ground_truths in tqdm(zip(self.images, self.true_bboxes), total=len(self.images), desc="Processing images"):
            confidences, predicted_boxes = self.predict_function(self.model, img, processor=self.processor)
            sorted_indices = np.argsort(confidences)[::-1]
            confidences = np.array(confidences)[sorted_indices]
            predicted_boxes = np.array(predicted_boxes)[sorted_indices]

            predicted_confidences, predicted_labels, iou_scores = [], [], []
            matched = set()
            for conf, pred_box in zip(confidences, predicted_boxes):
                iou_max, matched_gt = 0, None
                for i, gt_box in enumerate(ground_truths):
                    iou_score = self.iou(pred_box, gt_box)
                    if iou_score > iou_max:
                        iou_max, matched_gt = iou_score, i
                
                is_match = iou_max >= iou_threshold and matched_gt not in matched
                predicted_confidences.append(conf)
                predicted_labels.append(1 if is_match else 0)
                iou_scores.append(iou_max)         
                if is_match:
                    matched.add(matched_gt)
            
            all_predicted_confidences.append(predicted_confidences)
            all_predicted_labels.append(predicted_labels)
            all_iou_scores.append(iou_scores)
            
        return all_predicted_confidences, all_predicted_labels, all_iou_scores
    
    
    def __recalculate_iou_predictions_if_below_default_threshold(self, iou_threshold):
        if iou_threshold < self.default_iou_threshold:
            self.predicted_confidences, self.predicted_labels, self.iou_scores = self.__calculate_model_predictions(self.default_iou_threshold)
    
        
    def calculate_confusion_matrix(self, conf_threshold=0.25, iou_threshold=0.5) -> dict:
        self.__recalculate_iou_predictions_if_below_default_threshold(iou_threshold)
        # TP: predicted: license plate, true: license plate
        # FP: predicted: license plate, true: background
        # FN: predicted: background, true: license plate (license plates that were not predicted)
        # TN: predicted: background, true: background (I dont predict the background, cant say number of true background bounding boxes)
        tp, fp, fn = 0, 0, 0

        for confidences, labels, ious, ground_truths in zip(
            self.predicted_confidences, self.predicted_labels, self.iou_scores, self.true_bboxes
        ):
            tp_img = 0
            for conf, label, iou in zip(confidences, labels, ious):
                if conf >= conf_threshold: # say the model is so unsure that it predicted background
                    # if I dont do label == 1 here then I have the problem that I detect multiple bboxes for the same groundtruth, 
                    # but I only want one and with the highest confidence (confidences are sorted)
                    if iou >= iou_threshold and label == 1: 
                        tp_img += 1
                    else:
                        fp += 1

            tp += tp_img
            fn += len(ground_truths) - tp_img

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        # Accuracy doesnt make sense here bc we are not sure about the TNs
        #accuracy = tp / (tp + fp + fn + 1e-7)

        return {
            "Box(Precision)": precision,
            "Box(Recall)": recall,
            "Confusion Matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": None}
        }
        
    
    def calculate_mAP_scores(self, lower_iou=50, upper_iou=95) -> dict:
        self.__recalculate_iou_predictions_if_below_default_threshold(lower_iou)
        # https://www.v7labs.com/blog/mean-average-precision
        # Blog Describes AP Wrong: https://www.reddit.com/r/computervision/comments/162ss9x/trouble_understanding_map50_metric/
        # mAP = go over different confidences and calculate average precision, take mean of all classes
        # default function calculates mAP50
        unnested_confidences = [conf for sublist in self.predicted_confidences for conf in sublist]
        unnested_labels = [label for sublist in self.predicted_labels for label in sublist]
        unnested_iou_scores = [iou for sublist in self.iou_scores for iou in sublist]

        ap = AveragePrecision(task="binary")
        mAP = []
        for x in range(lower_iou, upper_iou + 5, 5):
            adjusted_labels = [1 if label == 1 and iou >= x / 100 else 0 
                   for label, iou in zip(unnested_labels, unnested_iou_scores)]
            mAP.append(ap(tensor(unnested_confidences), tensor(adjusted_labels)).item())
        
        mAP_low_high = np.mean(mAP) if mAP else 0.0
        return {"mAP": list(zip(range(lower_iou, upper_iou + 5, 5), mAP)), f"mAP{lower_iou}": mAP[0] if mAP else 0.0, f"mAP{lower_iou}-{upper_iou}": mAP_low_high}
    
    
    def get_metric_summary(self, verbose=True) -> str:
        metrics = {**self.calculate_confusion_matrix(), **self.calculate_mAP_scores()}
        if(verbose):
            print("\n".join(f"{key}: {value}" for key, value in metrics.items()))
        return metrics
    
    
    def visualize_confusion_matrix(self, conf_threshold=0.25, iou_threshold=0.5):
        confusion = self.calculate_confusion_matrix(conf_threshold, iou_threshold)
        tp, fp, fn = confusion["Confusion Matrix"]["TP"], confusion["Confusion Matrix"]["FP"], confusion["Confusion Matrix"]["FN"]
        confusion_matrix = np.array([[tp, fn], [fp, 0]])

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted: Plate", "Predicted: Background"],
                    yticklabels=["True: Plate", "True: Background"], cbar=False)

        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix (Confidence Threshold: {conf_threshold} IoU Threshold: {iou_threshold})')
        plt.show()
    
    
    def __visualize_metric_confidence(self, iou_threshold=0.5, metric_type="precision", visualize=True):
        metrics = []
        thresholds = np.linspace(0.0, 1.0, 101)

        for threshold in thresholds:
            confusion = self.calculate_confusion_matrix(threshold)
            tp, fp = confusion["Confusion Matrix"]["TP"], confusion["Confusion Matrix"]["FP"]
            metric = confusion["Box(Precision)"] if metric_type.lower() == "precision" else confusion["Box(Recall)"]
            if metric_type.lower() == "precision" and tp == 0 and fp == 0:
                print(f"Precision calculation not possible at confidence ge: {threshold}")
                missing_thresholds = int((101 - (threshold * 100)))
                metrics.extend([1] * missing_thresholds)
                break
            
            metrics.append(metric)
            
        if visualize == True:
            plt.plot(thresholds, metrics, label=f'{metric_type.title()} at IoU {iou_threshold}')
            plt.xlabel('Confidence Threshold')
            plt.ylabel(f'{metric_type.title()}')
            plt.title(f'{metric_type.title()} vs Confidence Threshold (IoU: {iou_threshold})')
            plt.grid(True)
            plt.show()
            
        return metrics

    
    def visualize_precision_confidence_curve(self, iou_threshold=0.5):
        self.__visualize_metric_confidence(iou_threshold, "precision")
        
    
    def visualize_recall_confidence_curve(self, iou_threshold=0.5):
        self.__visualize_metric_confidence(iou_threshold, "recall")
    
    
    def visualize_precision_recall_curve(self, iou_threshold=0.5, visualize=True):
        precisions = self.__visualize_metric_confidence(iou_threshold, "precision", False)
        recalls = self.__visualize_metric_confidence(iou_threshold, "recall", False)

        if visualize == True:
            plt.plot(recalls, precisions, label=f'Precision-Recall Curve at IoU {iou_threshold}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision vs Recall (IoU: {iou_threshold})')
            plt.grid(True)
            plt.show() 
        
        return precisions, recalls

   
    def visualize_for_examples(self, conf_threshold=0.25, iou_threshold=0.6, seed=1234):
        random.seed(seed)
        sample_indices = random.sample(range(len(self.images)), 4)

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        for axs_idx, idx in enumerate(sample_indices):
            img = self.images[idx]
            ground_truths = self.true_bboxes[idx]
            confidences, predicted_boxes = self.predict_function(self.model, img)
            sorted_indices = np.argsort(confidences)[::-1]
            confidences = np.array(confidences)[sorted_indices]
            predicted_boxes = np.array(predicted_boxes)[sorted_indices]

            # Convert img
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
            else:
                img = cv2.imread(img)
                
            # Draw BBs into img
            for gt in ground_truths:
                x1, y1, x2, y2 = map(int, gt)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for ground truth

            # This code is a straight copy from the function self.__calculate_model_predictions could be refactored
            matched = set()
            for conf, pred_box in zip(confidences, predicted_boxes):
                iou_max, matched_gt = 0, None
                for i, gt_box in enumerate(ground_truths):
                    iou_score = self.iou(pred_box, gt_box)
                    if iou_score > iou_max:
                        iou_max, matched_gt = iou_score, i

                is_match = iou_max >= iou_threshold and matched_gt not in matched
                if conf >= conf_threshold and is_match:
                    x1, y1, x2, y2 = map(int, pred_box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for predictions
                    cv2.putText(img, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

                    if is_match:
                        matched.add(matched_gt)

            axs[axs_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[axs_idx].axis('off')
            axs[axs_idx].set_title(f'Example {idx + 1}')

        plt.tight_layout()
        plt.show()        

    
    def generate_evaluation_video(self, deepsort, video_path, output_path, conf_threshold=0.25):
        def get_color(track_id):
            if track_id not in track_colors:
                track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            return track_colors[track_id]
        
        track_colors = {}
        ocr_change = {}
        cap = cv2.VideoCapture(video_path)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

        processing_time_frames = []
        reader = easyocr.Reader(['en'])

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            confidences, boxes = self.predict_function(self.model, frame, confidence=conf_threshold)
            track_detections = []
            for confidence, bbox in zip(confidences, boxes):
                bbox = list(map(int, bbox))
                # left top width height format for deepsort
                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                track_detections.append((bbox, confidence, None))

            # Tracking
            tracks = deepsort.update_tracks(track_detections, frame=frame)    
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cropped_plate = frame[y1:y2, x1:x2]
                plate_text = "unknown"
                if cropped_plate.shape[0] > 0 and cropped_plate.shape[1] > 0:
                    cropped_plate_gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                    result = reader.readtext(cropped_plate_gray)
                    plate_text = (result[0][-2] if result else "unknown").lower()  
                    if plate_text != "unknown" and (track_id not in ocr_change or (track_id in ocr_change and plate_text not in ocr_change[track_id])):
                        different_numbers_detected = ocr_change.get(track_id, [])
                        different_numbers_detected.append(plate_text)
                        ocr_change[track_id] = different_numbers_detected

                cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f'Plate: {plate_text}', (x1, y1 - 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f'Conf {confidence:.2f}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            processing_time_frames.append(time.time() - start_time)
            out.write(frame)
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Average Processing Time per Frame: {np.mean(processing_time_frames)}")
        print(f"Average Number of Different Plate Prediction per Tracking Target: {sum(len(x) for x in ocr_change.values()) / len(ocr_change)}")