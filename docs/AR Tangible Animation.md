# AR Tangible Animation

## DONE

Paper survey
Paper draft
Basic Implementation of Skeleton Detection

## TO DO

Data collection
User Interface Implementation
Evaluation user study
Paper Writing

## 6.24

Tangible device(toy) -> skeleton -> virtual reaction on device or extensions

animation, selection, interaction

### KinÊtre

**KinÊtre: animating the world with the human body**

Share on

Animate virtual content at runtime from full body input

Input : Kinect scans and reconstructs a real chair -> Kinect a real-time tracked human skeleton

voice command "possessed"

transfer motions of the user into realistic deformations of the mesh

summary : make physical objects live by humanize it using depth camera 



model rigging : 模型装配，为现有模型构建一个吻合的骨骼结构

skeleton deformation：骨骼变形

keyframing：关键格动画



Embedded deformation for shape manipulation

Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera



评价指标：qualitative experiences



### Annexing Reality

**Annexing Reality: Enabling Opportunistic Use of Everyday Objects as Tangible Proxies in Augmented Reality**

AR headset + depth camera

3-DOF scaling

Kinect->

surrounding physical objects -> similar given virtual objects

mismatch: 

​	Simeone, how the mismatch affects user's feeling of immersion and engagement and have found that the greater the degree of mismatch, lower the believability of the experience. 

​	Kwon

- matching  priorities  for  individual  virtual objects

- matching  priorities  for  different primitive  parts  of  each  virtual  object, 

- matching priorities for physical dimensions of each primitive 

    part

    -> Hungarian algorithm 匈牙利算法



### Questions

- devices : AR headset,   extra depth camera?
- different kinds of the objects： skeleton and other
- evaluation indicator
- feedback, maybe haptic feedback

### Meeting

1. 设备
    - AR + Realsense
    - 核心：模拟制作间环境，尽可能
2. 对象类别
3. 评价指标
4. 反馈



## 7.21

Potential Applications:

1. use humanoid toys as tangible input device in AR/VR. (Puppetry)

Benefits: intuitive connection between the controlled models and the input device. 

Benefits: interaction design between multiple virtual models. (A scene) 

2. use humanoid toys as tangible input device in AR/VR. (Controller)

Benefits: higher information capacity of the controller (high degrees of freedom on the humanoid body)

3. enrich the experience of interaction with humanoid toys. (interactive toys)

Benefits: toy body parts become interactive, visual/audio feedback can be added in the interaction.

4. present different virtual objects in VR. (sword, scissors, handgun)

Benefits: provide haptic feedback for different virtual objects.



Hi, everybody. I am Sicheng from Tsinghua University. My mentor is Yuntao Wang and Yukang Yan. Now let me introduce my AR Tangible Animation project.

As you know,there is a growing trend towards AR and VR technologies being adapted to anywhere. Sadly it is not that comfortable when we just use gaze interaction on Hololens ,1st gen I mean.   

So , to enrich the interaction method in AR/ VR environment, as you can see in the slide, we present ToyStory, a system to design skeletal animation in AR (Augmented Reality) with tactile humanoid toys. With ToyStory, designers use familiar humanoid toys as the control device to manipulate pre-loaded virtual models in AR。Let me make a simple example, like, the user grab a Woody toy on his hand, the skeleton detection system will analyze real-time body keypoints data and we will attach the data to a preloaded object in AR, such as a virtual Captain America. As the user make a pose on the real toy, the virtual one can show the same pose. With certain actions, the virtual one may have more  response like say hello or take up weapons.

We use HoloLens 1st gen as an AR Headset, Intel Realsense D435i as a camera.

Using Realsense, we detect the skeleton data of the humanoid toy. With more analysis, we will make filtering and animation work  to make it possible to use the toy as an input device in AR environment. Then we will add some after effects and design more feedbacks. Such kind of technology will enrich AR/VR experience and make more haptic feedback in AR/VR environment.

By now, we have done preliminary paper survey, built a hardware system, finished the pipeline development and some paper draft. I have setup the devices I need in my project and have the realsense adapted to OpenPose project using GPU speedup to do the skeleton detection. Besides, a skeleton display on Hololens works now.

Now the trickest part is to manage a solution to reduce the noises in the skeleton detection stage. In my test, as you can see in the 2 screenshots on the right, the user's body as a background can have non-ignorable influence to the detection . This week we will figure out some method to filter human hand and body and set up a more idealized environment  prepared for the coming user study. More, We will try to do some gesture elicitation and figure out more scenarios and applications relative to our project.  Then we will design and conduct our user study in detail. We will dig much more in this project in the next 2 months.



## 7.29

升级pip之后openpose莫名其妙挂了。重装了电脑，一代模拟器挂了。

解决了连接的问题打通了整个pipeline，可以部署到2代模拟器

加入了手的识别和卡尔曼滤波去噪声

online的效果在引入MRTK视角后需要调整一下，使用深度数据截掉后面一定距离的内容



下周：

整一下网卡的东西，在设备上做一下

整合一下两个filter，handtracking（readme写的挺多，感觉会好一些，后面也可能换）由于tf版本太老现在有很多bug，并且需要精研，可能涉及到人手的手势识别

运动补偿，开源内容，lucaskanade算法

在unity上用mrtk做一些简单的组件，学习一下直接在脚本里加MRTK内容



application：

控制器、puppetry、增强虚拟、增强现实





切换视角功能

四个模块一个一个加，模块化

计算一下抓握点，先放好再计算手 label手的图像和位置

ICML Google handpose webcamera 

MRTK迁移



Application



## 8.17~8.23

### weekly update

Hi, everyone. I am Sicheng Yin, working on AR Animation project with Yukang and Yuntao.

The temporary goal of our project is to optimize the skeleton detection part and make the pipeline more usable and stable, which can bring convenience in the userstudy and evaluation stage. Next is to make application brain storm and conduct user study for gestures and applications.

What we have done is having developed the whole software pipeline, which is, as you can see in the flow diagram, we make skeleton detection work on the toy held by the users to get necessary pose data along with gestures or triggers pre-set by us, then we analyze this data and make things like virtual puppy, avata in online meeting or other given object alive. Besides as a controller for virtuality , we plan to use the pose data to make enhancement for reality, for example. adding some virtual feedback or after effects for some gestures made by user or the toy. Finally we deploy our app to AR or VR devices such as Hololens, which is chosen in our project. 

Recently, we have finished the development work for the whole pipeline as you can see in the screenshot on the top of the slide. We use depth data to get specific frame data in the distance between set values to filter unnecessary frame data. We add hand pose detection to filter the hand skeleton of the user. Then we added optical flow  using Lucas-Kanade algorithm and added Kalman filter to optimize the value jitter. Now we have a well-done skeleton detector ,an almost done 4 elements filter and can reach up to more 10 frames per second. Besides optimize the detector, I have been collecting data of relationship between hand pose and skeleton covered by hand, in order to solve the hand-occlusion problem. We may find some pattern between the location data and the idealized solution will be like, predicting the location of the body part of the toy covered by the user's hand. 

In the next few days, we will focus on the fps increasement and hand occlusion problem. And we may edit a questionnaire about the gestures and the applications and select the most popular applications based on the result.

That's all of my update today. Thank you for listening.

### meeting note

specific location for holding、



Hi, everyone. I am Sicheng Yin, working on AR Animation project with Yukang and Yuntao.

As our project's name, we are expected to build a pipeline which can do animation work in AR using skeleton data of specific object in reality. By now we have come up with four kind of application including control virtual object, gesture recognition of toy, vision enhancement for toy, and virtual avatar or puppetry which is like an advanced ZEPETO

The temporary goal of our project is to optimize the skeleton detection part and make the pipeline more usable and stable, which can bring convenience in the userstudy and evaluation stage. We use OpenPose by CMU and we found that it can only achieve 11 frames per second even if using dual GTX1080Ti and doing no optimization. Using simple optical flow can just make an advancement to 10~13 fps. It can bring a non-ignorable vision delay if we fake a lot more frames. Finally we make it by doing network quantization and sacrificing about 4% accuracy, reached more than 30 fps.

With this optimized skeleton detector, we will do research on hand occlusion, maybe we will base the prediction algorithm on Bayes Model or any other algorithm, depend on the data analysis result. Besides, we discuss in detail about what I call  "holding position suggestion" advice raised by Alex in the last group update. We decide to add a virtual highlight on the toy to suggest the user where to hold. 

We have finished baseline design for the solid applications and we are developing user interface of these application and will conduct user study soon. 

## 9.8

mapping to unity





##  9.19

### FInal Report



##### Slide1



Hello, everyone. Pleasure to be here to introduce my project of AR Tangible Animation, which is lead by Yukang and Yuntao.



##### Slide2



We are living in a world where everything is made up with reality but we are always expect virtuality. This is what AR is made for. However, what makes me disappointed is that by now there is no such a method to measure and control action in AR accurately and efficiently. Even the basic interactions between user and AR Headset are gazing and quite simple gestures like tap and bloom. 



slide 3



#### background& related work 



#### slide 4



**KinEtre: Animating the World with the Human Body** and **Tangible and modular input device for character articulation** show the wild possibility that we can control action of real objects using skeleton. 



slide 5



While **Annexing Reality: Enabling Opportunistic Use of Everyday Objects as Tangible Proxies in Augmented Reality** shows how limited objects at hand can be mapped to unlimited virtual objects.   



slide6



#### Introduction



slide 7



We propose Toy Story, an AR Tangible Animation technology based on skeleton detection all by computer vision techniques. In Toy Story, we compute skeleton data using RGB and depth camera (Realsense D435i in this case, but built-on camera on AR headset in the future). With reliable skeleton data, we figure out a new kind of interaction technique, mapping action to skeleton. One idealized scenario is that a professor of medical science can use the toy to control a real skeleton model on his lectures or presentations. 



slide 8



What we want to do can be seen clearly on the the flow diagram, we make skeleton detection work on the toy held by the users to get necessary pose ata along with gestures or triggers pre-set by us, then we analyze this data and make things like virtual puppy, avatar in online meeting or other given object alive. 



slide9



Besides as a controller for virtuality , we plan to use the pose data to make enhancement for reality, for example. adding some virtual feedback or after effects for some gestures made by user or the toy. 



slide10



Finally we deploy our app to AR or VR devices such as Hololens, which is chosen in our project. 



slide 11



We have already developed the whole pipeline as is showed on the slide.



slide 12



#### work



slide 13



 To make it work,  we firstly do a simple test that humanoid toys or even any simple object like a dog can have pose data inference predicted by AlphaPose and OpenPose, which are two well-knowed multi-person pose estimators. 

As we can see in the chart, compared to AlphaPose, we found that OpenPose has a more stable detecting speed but kind of less accuracy than AlphaPose. Since we expect to do the skeleton detection in real time, we choose OpenPose as our basic skeleton detection tool.



slide 14



but as we can see, there are  4 main  problems in the usual OpenPose: Firstly , it detects both the toy and the user, while the user's skeleton data is useless in this case. Secondly, 7~9 Frames Per Second is far from enough in real time. At least 24 fps is needed if we want it display smoothly on AR headset. Besides, we need to guarantee the delay as short as possible to get a good effect. Thirdly, there are more or less noises that the skeleton can be tremble sometimes which is wrong to the display. What's more, the occlusion of the user's hand is such a big problem that it can  have a huge amount of influence to the detection of the hand hold object.



slide 15



We design a four-element filter to optimize the OpenPose. First, we use depth stream of RealSense to catch the image in certain scale, which is 0.5 meter to 1 meter. Thanks to this filter, most body part of the user can be removed so that the remain skeleton noises are just user's hands. Secondly, with the help of MediaPipe by Google, we are able to draw bounding boxes of users hands and manage to filter it before we get the inference. Thirdly, to avoid trembling of the skeleton, we use kalman filter as a revision of the data stream. Last but not the least, we implement Lucas-Kanade optical flow algorithm, temping to add frames so that it can reach a good level of FPS. 



slide16



But sadly the normal body25 model pre-trained by OpenPose team can have no more than 12 FPS even using GTX 1080Ti. It can only run at 7-9 FPS on my local machine. What's worse, the LK algorithm has the best performance of 18 Hz, which means we can fake no more than 18 frames per second with 7-9 real frames. Even in this case we can just struggle to 24FPS, taking no account of doing such a lot of computing on a single PC.



slide 17



So we make network quantization on the pose model. We sacrifice about 1% Average Precision of the original COCO model and make it simpler.It is well tested that even the skeleton detection estimator has not that much hardware resource to use, it can have a stream of more than 24 FPS, well, it can be more than 33 FPS if we just turn off as more useless softwares as possible.



So the technical problem lies on optimizing the pose estimator to get the idealized performance. In consideration of  the irreplaceability of the filter I mentioned before. Our solution is to make network quantization on the pose model. We sacrifice about 1% Average Precision of the original COCO model and make it simpler. A 3-depthwise convolutional block is made for replacement convolutions with 7x7 kernel size in refinement stage. The new estimator can have a 16-22 FPS performance depending on the hardware occupancy. Using Lucas-Kanade algorithm, we add a frame for each 3 real frames. It is well tested that even the skeleton detection estimator has not that much hardware resource to use, it can have a stream of more than 24 FPS, well, it can be more than 33 FPS if we just turn off as more useless softwares as possible.

 

slide 18



#### todo





By now, we have the well-prepared skeleton detector but we have one more thing to do before the application user study, which is optimizing the hand occlusion problem as we can do as possible.



slide 20



 The temporary solution we have made is that we designed  a study to collecting the skeleton data and hand pose data when the occlusion happened , then we analyze these data and try to find the corresponding between them using machine learning method. So we added a hand pose detection part based on Google MediaPipe and is in the stage of collecting enough data.  We will make inference of the covered part on the toy and show recommended holding location in form of virtual display in AR.



slide 21



The after plan for our project is that we try to brain storm again and make it clear about our possible applications and innovations in  the interaction part. We will develop user interface of the applications and make some qualitative and quantitative study compared to baselines how what we conduct the same application in AR by now.  Then we will do evaluation work and paper writing.

That's all for my presentation today. Thanks for your listening.

