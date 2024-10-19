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
   - [Roboflow License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
   - [Kaggle Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)
   - [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?select=images)

## Planned Implementation / Evaluation
The implementation of this project will contain several key components:

1. **Object Detection**: Develop a module to accurately detect license plates within images and videos.
2. **Optical Character Recognition (OCR)**: Integrate an OCR system to read and extract license plate numbers from detected regions.
4. **Android Application Development**: Build a mobile application to deploy the final product for end-users.
5. **Evaluation**: Assess the systems performance using a test dataset and our own video footage, comparing the effectiveness of various implementations, including MobileNet, YOLO and simple edge detection techniques.
