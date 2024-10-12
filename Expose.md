# Title: Numberplate Recognition
**Group Members: Eric Jonas**

## Motivation / Idea:
The ability to recognize vehicle number plates in can be beneficial for numerous applications, including traffic monitoring, parking systems and law enforcement.
The goal of this project is to develop a system that identifies number plates in images and videos by creating bounding boxes around them and either cropping the plates as separate images or directly recognizing the plate number from the regions of interest.

The system will be designed to operate in real-time on both images and videos. It will detect all vehicles in each frame and track moving objects across frames, ensuring consistent identification of the same vehicles, including their number plates. Additionally, it will trigger an alert when certain number plates cannot be read due to issues like overlaps or occlusions in the current image.  

## Related Work:
Existing work on automatic license plate recognition has been explored in systems such as OpenALPR and YOLO (You Only Look Once) for object detection. 

These systems are already used in parking lots and could be used to 



## Planned Implementation / Evaluation:
- YOLO
- OCR
