import numpy as np
from torch import tensor
from torchmetrics.classification import AveragePrecision

class ObjectDetectionEvaluator:
    def __init__(self, model, images, true_bboxes, predict_function, default_iou_threshold = 0.5):
        self.model = model
        self.images = images
        self.true_bboxes = true_bboxes
        self.predict_function = predict_function
        self.default_iou_threshold = default_iou_threshold
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
        all_predicted_confidences, all_predicted_labels, all_iou_scores = [], [], []
        
        for img, ground_truths in zip(self.images, self.true_bboxes):
            confidences, predicted_boxes = self.predict_function(self.model, img)
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
        # I think accuracy doesnt make sense here bc we are not sure about the TNs
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
        # TODO visualize the confusion matrix with matplotlib
        # call calculate_confusion_matrix
        
    def visualize_precision_confidence_curve(self, iou_threshold=0.5):
        # TODO
        
    def visualize_recall_confidence_curve(self, iou_threshold=0.5):
        # TODO
        
    def visualize_precision_recall_curve(self, iou_threshold=0.5):
        # TODO
        
    def visualize_precision_recall_curve(self, iou_threshold=0.5):
        # TODO
        
    def visualize_for_examples(self, conf_threshold=0.25, iou_threshold=0.6, seed=1234):
        # TODO with cv2 or matplotlib take 4 images display them in a grid with in green ground truth boundingbox in blue predicted boundingbox
        # display the confidence at the top of each boundingbox and