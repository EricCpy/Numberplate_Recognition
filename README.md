# License Plate Recognition
**Group Members: Eric Jonas**

## Problem
The ability to recognize vehicle license plates offers significant advantages across various domains, such as traffic management, parking automation and law enforcement. This project aims to develop a system capable of detecting and recognizing license plates in images and videos. The system will create bounding boxes around the plates and extract the license plate numbers from designated areas of interest.

Instead of detecting vehicles, the system directly tracks license plates. Tracking algorithms are used to associate the same license plate with a single object across multiple frames.  

Since license plate recognition or tracking systems have been implemented many times before, the goal of the project is not to build the system itself but rather to experiment with different algorithms for constructing it and compare their performance.


## Related Work
Automatic license plate recognition (ALPR) has been explored in several existing systems, such as OpenALPR. These systems have successfully been deployed in various settings, including parking facilities and highways, showcasing their potential for enhancing vehicle monitoring and control. 
There are numerous videos on YouTube and various websites showcasing how to build an Automatic License Plate Recognition (ALPR) system.

However, many of these videos do not discuss the differences between algorithms or compare them. To choose the optimal algorithm, one must first define requirements, for example, whether the system needs to operate in real-time or not. A good starting point for finding suitable object detection models was the [Papers with Code](https://paperswithcode.com/sota/object-detection-on-coco) Benchmark, which provides an overview of the top-performing models over the years. Additionally, the following paper [link](https://www.sciencedirect.com/science/article/pii/S095219762400616X) was particularly helpful in understanding and evaluating different object detection methods, guiding my decision on which approaches to experiment with.  

This led to the goal of testing license plate object detection using multiple models, starting with Faster R-CNNs, comparing them with the latest YOLO model (YOLO v11) and also implementing an detection transformer (DETR).  

Beyond object detection, the project also involves an OCR and an object tracking task. Various research papers explore image preprocessing techniques for achieving optimal OCR results, such as this one specifically for license plates [Comparison of Image Preprocessing Techniques for Vehicle License Plate Recognition Using OCR: Performance and Accuracy Evaluation](https://arxiv.org/abs/2410.13622). However, we chose to stick with EasyOCR and grayscale images, despite this paper indicating a slight decline in performance when using EasyOCR with grayscale images.

For object tracking, **DeepSORT** was chosen over more advanced models. The decision was based on specific trade-offs, such as easy integration from existing libraries. Many state-of-the-art models listed in this [benchmark](https://paperswithcode.com/sota/multi-object-tracking-on-mot17) build upon DeepSORT (like StrongSORT), enhancing it in some aspects rather than replacing it entirely.  

The goal of this project is **not** to build the best system using the best algorithms from every field. Instead, it focuses on implementing a functional system and evaluating various object tracking algorithms.

## Data


## Implementation


# Implementation
## Object Detection
### Edge Detection
### Fasterrcnn
### YOLO
### DETR

## Object Tracking
Deepsort

# Evaluation

## Metrics
- beschreibe die verschiedenen metriken, die ich zu evaluation verwendet habe

## Object Detection Results

Show evaluation of different models.

## Performance 


# Conclusion
show model applied on final video

Notes:
# Training

YOLO Training ~2h over all data, optimized perfectly everything intern
FasterRCNN (mobilenet ~2h, resnet ~4h) over all data, you need to optimize everything yourself with pytorch