# PPE Detection: YOLOv8 vs Faster R-CNN

## Objective

This concise project aims to delve into the fundamentals of Object Detection within the realm of Computer Vision. The multitude of technologies in this field can be overwhelming, making it challenging to choose a specific path. While tutorials provide a solid starting point, a true grasp of complex technologies, such as neural networks, only comes through hands-on experience. This project serves as a log of my journey, detailing steps taken in contrast to a conventional repository. Feel free to use it as a guide if you're embarking on a similar learning path.


## Tools

<img src = "https://github.com/bccestari/PPE-Detection/blob/main/images/diagram.png" width = 80% height = 50%> 

- **Ultralytics YOLOv8**: Object detection model
- **PyTorch Faster R-CNN-ResNet50**: Object detection model
- **ClearML**: experiment tracking on Colab
- **Weights and Biases**: for experiment tracking on Kaggle Kernel
- **Google Colab**: Model training and inference 
- **Kaggle Kernel**: Model training and inference
- **Google Drive**: Input data, model weights and outputs



## Problem Definition

I began by looking at a real-life problem suitable for object detection models. Upon researching companies in my country (Brazil), I identified Personal Protective Equipment (PPE) Detection as a compelling area of interest. Efficient PPE detection has the potential to save lives and enhance working conditions across various industries. This problem involves multiclass detection, necessitating identification of persons, machinery, vehicles, safety cones, masks, safety vests, hard hats, and gloves.

These classes may exhibit variations depending on the industry and location; for instance, different colored hard hats or various types of masks. The model must detect swiftly and accurately to prevent injuries or fatalities. Typically, the detector works on camera images within workplaces, alerting managers if workers are not using PPE or are in proximity to hazardous objects like heavy machinery.

## Models

The initial step involved finding the most recent studies addressing the PPE Detection problem. I sought to understand common approaches and models. Three notable studies were identified:

1. [PPE detector: a YOLO-based architecture to detect personal protective equipment (PPE) for construction sites](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9299268/) (2022)

   
2. [Personal Protective Equipment Detection: A Deep-Learning-Based Sustainable Approach
](https://www.mdpi.com/2071-1050/15/18/13990) (September 2023)

3. [Artificial Intelligence System for Detecting the Use of Personal Protective Equipment](https://thesai.org/Downloads/Volume14No5/Paper_61-Artificial_Intelligence_System.pdf) (2023)

All three studies predominantly featured the YOLOv5 model. Study 2, which listed various studies and ranked models, reported that Faster R-CNN with a ResNet50 backbone exhibited a superior mAP50 (96%) compared to YOLOv5 (63%) when trained to 20 epochs. Given the disparate datasets and classes used, I decided to explore and compare Faster R-CNN with the most recent YOLOv8 models.

## Data

I started my search for data suitable for training in the YOLO model, a state-of-the-art approach known for its speed and accurate object detection capabilities. I came across the PPE detection dataset for construction site safety on Roboflow's Universe projects at this link: [Construction Site Safety PPE Detection Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety). The dataset can be downloaded in YOLO format or XML, the latter being compatible with the Faster R-CNN model.

Here are the key properties of the dataset:

**Classes:** 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'.


- **Total Images:** 1759
    - **Training Set:** 1563 images
    - **Validation Set:** 114 images
    - **Test Set:** 82 images

**Augmentations:**
- **Flip:** Horizontal
- **Crop:** 0% minimum zoom, 20% maximum zoom
- **Rotation:** Between -12º and +12º
- **Shear:** ± 2º Horizontal, ± 2º Vertical
- **Grayscale:** Applied to 10% of images
- **Hue:** Between -15º and +15º
- **Saturation:** Between -20% and +20%
- **Brightness:** Between -25% and +25%
- **Exposure:** Between -20% and +20%
- **Blur:** Up to 0.5px
- **Cutout:** 6 boxes with 2% size each
- **Mosaic:** Applied

This dataset, coupled with the specified augmentations, provides a comprehensive and diverse set of images for robust training of object detection models.

	

## Validation 

Validation is the most important step in the start of any machine/deep learning project. I maintained the roboflows dataset proportion for train (80%), valid (6,5%) and test (4,5%).

## Metrics

<img src = https://github.com/bccestari/PPE-Detection/blob/main/images/IOU.png width = 70% height = 70%>

The most relevant metrics in Multiclass Object Detection are:

1. **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth bounding boxes, serving as a key indicator of localization accuracy.

2. **mAP (Mean Average Precision**: Measures the overall accuracy and precision of the model across different object classes, providing a comprehensive evaluation.

3. **mAP50 and mAP50-95**: mAP at different IoU thresholds, such as 50% and 50-95%, gives a nuanced understanding of the model's performance across varying levels of object overlap.

4. **Precision and Recall**: Precision indicates the model's ability to correctly identify positive instances, while recall measures its capacity to capture all relevant instances.

5. **Inference Time**: The time taken by the model to process and make predictions on new data, a crucial metric for real-time applications and deployment.

mAP50, Precision, Recall and Inference time are used to compare models in this project. 



## Experiment 1: Choosing the size of YOLOv8

I initiated experiments on Google Colab, comparing different sizes of YOLOv8 pretrained on the COCO dataset: yolov8n (Nano), yolov8s (Small), yolov8m (Medium), yolov8l (Large), and yolov8x (Extra Large).

Tracking YOLO experiments with ClearML, a tool recommended in the YOLO documentation for its ease of setup and clean, intuitive UI.

Initially, I trained models yolov8n and yolovm for 50 epochs. Encountering two issues: 
1. mAP50 plateaued at 0.65.
2. GPU usage was high, and Google Colab has a 12-hour limit. Training yolov8m took approximately 1 hour.

<img src = https://github.com/bccestari/PPE-Detection/blob/main/images/experiment-1-yolov8m-50-epochs-no-finetunning.png>


**Conclusion**:  I need faster iteration and model tuning.

## Experiment 2: Fine Tuning YOLOv8m

Reduced training to 20 epochs, comparing models. Clear correlation observed between model size and improved mAP, precision, and recall. Leveraged hyperparameter research from Kaggle, adopting the CFG class from [Hinepo's notebook](https://www.kaggle.com/code/hinepo/yolov8-finetuning-for-ppe-detection):

<img src = "https://github.com/bccestari/PPE-Detection/blob/main/images/experiment-2-yolov8m-20-epochs-finetunning.png" width = 60% height = 60%>

**Conclusion**: Inference time improved to 17 ms with 2.5x less training (25 minutes).
 

## Experiment 3: YOLOv8l vs YOLOv8x

After the improvements of fine tuning, I started the comparison of yolov8l and yolov8x till 20 epochs. 

**Yolov8l-Finetuning-20-epochs**:

<img src = "https://github.com/bccestari/PPE-Detection/blob/main/images/Yolov8l-Finetuning-20-epochs.png">

**Yolov8x-Finetuning-20-epochs**:

<img src = "https://github.com/bccestari/PPE-Detection/blob/main/images/Yolov8x-Finetuning-20-epochs.png">

**Conclusion**: Very close to mAP50; yolov8l gains in precision, yolov8x in recall.

Continued training for 30 more epochs. At this point, reaching Colab quota limit. If mAP50 remained equal, preference on best recall and inference time  for safety reasons. A higher recall means the model is better at identifying all instances of non-compliance, ensuring that workers without the necessary safety equipment are less likely to go unnoticed.

The results:

**Yolov8l-Finetuning-50-epochs**:

<img src = "https://github.com/bccestari/PPE-Detection/blob/main/images/Experiment-8-Yolov8l.png">

**Yolov8x-Finetuning-50-epochs**:

<img src = "https://github.com/bccestari/PPE-Detection/blob/main/images/Experiment-10-yolov8x.png">






**Conclusion**:  The results closely align, with YOLOv8l exhibiting a 1.3x faster inference time (21 ms compared to 29 ms), slightly better recall, and nearly identical mAP50. Consequently, I have selected YOLOv8l as the final model for detection. The subsequent phase involves fine-tuning YOLOv8l for 300 epochs to assess potential improvements. 


## Experiment 4: Faster R-CNN

Conducted concurrently with Experiment 3. Attempted to implement Faster R-CNN from scratch, and faced challenges to make it work  and, consequently, explored alternative approaches. 

Stumbled upon a remarkable repository from [sovit-123](https://github.com/sovit-123): [fasterrcnn-pytorch-training-pipeline](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline?tab=readme-ov-file#Train-on-Custom-Dataset](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline?tab=readme-ov-file#Train-on-Custom-Dataset) that provide an easy pipeline for Faster R-CNN training using XML forma t data. Used Kaggle T4 x 2 GPUs for training, treating the background as a class.

I encountered challenges when attempting to log experiments with ClearML, so I switched to Weights and Biases. While using WandB in Kaggle, I faced issues as there was no way to input data in the Command-Line Interface (CLI). I resolved this by employing wandb.login() before initiating the training process. I trained the model for 50 epochs due to poor performance observed in the initial 20 epochs:

<img src = https://github.com/bccestari/PPE-Detection/blob/main/images/Faster-RCNN-wandb.png>


**Conclusion**: The model plateaus after 7 epochs (390 steps/epoch), with mAP50 hovering between 0.65 and 0.70 up to 50 epochs. I've abandoned Faster R-CNN, as suggested in Study 2, due to the necessity for extensive fine-tuning on the specific dataset. However, resource constraints make iterative attempts challenging, especially given that the model requires three times more training time compared to YOLO.



## Experiment 5: Yolov8l Fine-tuning till 133 epochs


I pretend to train the model for 300 epochs, but the mAP50 plateaued at 0.84 after 100 epochs. Satisfied with this performance, I decided to stop the training at that point. The model maintained a fast inference time of 17 ms, which I find satisfactory for my detector. I will utilize this model on the test set for making inferences. Below are the results for the model:  


**--------------------------------------------------------------------------------------------------------------------------**

-**Precision (Class B)**: 0.90

-**Recall (Class B)**: 0.78

-**mAP50 (Class B)**: 0.84

-**mAP50-95 (Class B)**: 0.57

-**Inference Time**: 17 ms

-**Time to Train**: 3.5 hours

-**Time per Epoch**: Approximately 95 seconds


**--------------------------------------------------------------------------------------------------------------------------**

**IMAGES AND GRAPHS FROM THE EXPERIMENT:**


<img src = https://github.com/bccestari/PPE-Detection/blob/main/runs/detect/PPE_Detector_yolov8l_finetuning-133_epochs/val_batch1_pred.jpg>
<img src = https://github.com/bccestari/PPE-Detection/blob/main/runs/detect/PPE_Detector_yolov8l_finetuning-133_epochs/results.png>
<img src = https://github.com/bccestari/PPE-Detection/blob/main/runs/detect/PPE_Detector_yolov8l_finetuning-133_epochs/labels.jpg>
<img src = https://github.com/bccestari/PPE-Detection/blob/main/runs/detect/PPE_Detector_yolov8l_finetuning-133_epochs/confusion_matrix_normalized.png>



## Inference

Videos with Inference Yolov

Videos with Inference Faster CNN




## Final thoughts

- How improve data - augmentations 
- How improve Yolo Models
- How improve Faster R-CNN
- How build the detector
