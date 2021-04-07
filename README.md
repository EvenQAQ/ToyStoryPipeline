# ToyStory: AR Tangible Animation
## Code repo for ToyStory Pipeline
We propose Toy Story, an AR Tangible Animation technology based on skeleton detection all by computer vision techniques. In Toy Story, we compute skeleton data using RGB and depth camera (Realsense D435i in this case, but built-on camera on AR headset in the future). With reliable skeleton data, we figure out a new kind of interaction technique, mapping action to skeleton. One idealized scenario is that a professor of medical science can use the toy to control a real skeleton model on his lectures or presentations.
## Usage
### Palm Detection
#### mediapipe

```python
# /Palm_Detection
python palm_detection.py
```

#### YOLOv3

recommend  OPENCV DNN module with CUDA support

```python
# /Palm_Detection/yolo_hand_detection
python demo.py
python demo_webcam.py
```



### Skeleton Data Estimator



