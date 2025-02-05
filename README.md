# Title: License Plate Recognition
**Group Members: Eric Jonas**

## Motivation / Idea
The ability to recognize vehicle license plates offers significant advantages across various domains, such as traffic management, parking automation and law enforcement. This project aims to develop a system capable of detecting and recognizing license plates in images and videos. The system will create bounding boxes around the plates and either crop them into separate images or directly extract the license plate numbers from designated areas of interest.

Our solution will be optimized for real-time operation, processing both static images and dynamic video feeds. It will detect all vehicles within each frame and track their movements across subsequent frames, ensuring consistent identification of vehicles and their corresponding license plates.
Ultimately, our aim is to create a mobile application that allows us to utilize this technology on our smartphone.

## Related Work
Automatic license plate recognition (ALPR) has been explored in several existing systems, such as OpenALPR. These systems have successfully been deployed in various settings, including parking facilities and highways, showcasing their potential for enhancing vehicle monitoring and control. 
There are numerous videos on YouTube and various websites showcasing how to build an Automatic License Plate Recognition (ALPR) system. Our goal is to compare different approaches and evaluate their performance in terms of both accuracy and efficiency.

**Training Datasets**: Utilize one of the following datasets for training the model:
   - [Roboflow License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) -> included partially in 2.
   - [Kaggle other dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset/code) -> used
   - [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?select=images) -> used

## Planned Implementation / Evaluation
The implementation of this project will contain several key components:

1. **Object Detection**: Develop a module to accurately detect license plates within images and videos.
2. **Optical Character Recognition (OCR)**: Integrate an OCR system to read and extract license plate numbers from detected regions.
4. **Android Application Development**: Build a mobile application to deploy the final product for end-users.
5. **Evaluation**: Assess the systems performance using a test dataset and our own video footage, comparing the effectiveness of various implementations, including MobileNet, YOLO and simple edge detection techniques.


# TODO
- [ ] Object detection
  - [x] build license plate detection and car with edge detection as base line
  - [ ] build license plate and car detection with models
    - [x] ultralytics yolo
    - [x] pytorch object detection api mobilenet
    - [ ] optional: mmdetect?

- [x] OCR
  - [x] OCR selected based on the following video https://www.youtube.com/watch?v=00zR9rJnecA
  - [x] if I would like to improve ocr maybe fine tune ocr or retrain ocr model on my license plate data, but didnt find a license plate dataset which contains plate names

- [ ] Object Tracking
  - [ ] use deep sort and at least one other algorithm to keep track of objects
  - [ ] use simple own algorithm where I match and keep track on objects based on OCR and compare to Sota Methods
  - [ ] evaluate and continue with best

- [ ] use IoU for object detection as metric and use track over bounding box as metric
- [ ] apply on own video and evaluate for errors and different approaches
- [ ] write report
- [ ] create poster presentation
- [ ] optional: try to implement SOTA Method from Paper, compare my results to paper results