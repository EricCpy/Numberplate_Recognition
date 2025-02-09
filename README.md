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
The models were trained using data from the following sources:
- **[Kaggle Large Dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset/code)**
  - **Training Set**: This subset contains 25,500 carefully curated images for model training.
  - **Validation Set**: Comprising 1,000 images, this subset is used for evaluating model performance during the development process.
  - **Test Set**: The test set includes 400 images, which are reserved for final model evaluation after training.

- **[Kaggle Small Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?select=images)**
  - This dataset includes 433 high-quality images.


### Large Dataset
The **large dataset** was created by merging images from various sources, including websites and other datasets, into a single expansive collection. To increase the diversity of the training data, we applied image augmentations, such as snowflakes and random rotations. The images were then organized into three directories: training, validation, and test sets. This dataset follows the YOLO format for bounding box annotations. YOLO annotations store each object's data using the following five parameters:
```
class_id boundingbox_center_x boundingbox_center_y boundingbox_width boundingbox_height
```
- **class_id**: Integer representing the class of the object (e.g., license plate).
- **boundingbox_center_x** and **boundingbox_center_y**: Coordinates of the center of the bounding box, normalized to the image width and height.
- **boundingbox_width** and **boundingbox_height**: The dimensions of the bounding box, also normalized to the image width and height.


### Small Dataset
The **small dataset** consists of 433 high-quality images, with annotations in VOC format. VOC annotations are stored in XML files and include information about image objects, such as class labels, image size, bounding box coordinates.


### Merged Dataset
Both datasets were combined into a unified dataset for training, validation, and testing. To ensure consistency in annotation formats, we converted the VOC annotations from the smaller dataset into YOLO format using the **[YOLO Format Converter](src/data_processing/yolo_format_converter.ipynb)**.

The YOLO format was chosen because it is natively supported by the Ultralytics YOLO library. Additionally, it was easier to use this format to convert to different bounding box formats, which were necessary for some of our PyTorch models. To manage these formats effectively, we developed a **[custom PyTorch DataClass](src/utils/pytorch_helper.py)**.

For improved data management, we renamed the image files using the dataset name and a corresponding index (**[Rename Script](src/data_processing/rename_yolo_dataset_files.ipynb)**), replacing the previous random-character filenames. This benefited our processing workflow and also minimized potential issues related to inconsistent filename formats.

We decided to keep the dataset split from the large dataset:
1. **Training Data** – Used to train the model.
2. **Validation Data** – Used to fine-tune hyperparameters and prevent overfitting by determining the optimal number of training epochs.
3. **Test Data** – Used for final evaluation after training.

While cross-validation could have been an option, we opted not to implement it due to the extensive training times required, which would have made it impractical us.



## Implementation



### Object Detection
#### Edge Detection
#### Fasterrcnn
#### YOLO
#### DETR
### Object Tracking
Deepsort

## Evaluation

### Metrics
- beschreibe die verschiedenen metriken, die ich zu evaluation verwendet habe

### Object Detection Results

Show evaluation of different models.

### Performance 


## Conclusion
show model applied on final video



# Notes:
## Training

YOLO Training ~2h over all data, optimized perfectly everything intern
FasterRCNN (mobilenet ~2h, resnet ~4h) over all data, you need to optimize everything yourself with pytorch