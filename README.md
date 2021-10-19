# opencv_FaceMask_Detection
1. Recognize if a person is wearing or not wearing a mask using AlexeyAB-darknet YOLO.

2. You can train your own weights using yolov4-custom.cfg if you'd like to give this project a try, then put the weight files into backup folder.

3. Dataset is from <a href='https://www.kaggle.com/andrewmvd/face-mask-detection'>Face Mask Detection - Kaggle</a>, and I re-labelled 600 images myself using labelImg.

4. Please copy the yolov4.conv.137 file to train folder.

5. Sample results

  Original image 1: </br>
  <img src="https://github.com/PDooDP/opencv_FaceMask_Detection/blob/master/results/1168.jpg?raw=true" width="600" height="400">

  test with 1000.weights: </br>
  <img src="https://github.com/PDooDP/opencv_FaceMask_Detection/blob/master/results/predictions_1168_1000w.jpg?raw=true" width="600" height="400">

  test with 2000.weights: </br>
  <img src="https://github.com/PDooDP/opencv_FaceMask_Detection/blob/master/results/predictions_1168_2000w.jpg?raw=true" width="600" height="400">
  
  Original image 2: </br>
  <img src="https://github.com/PDooDP/opencv_FaceMask_Detection/blob/master/results/819.jpg?raw=true" width="600" height="400">

  test with 1000.weights: </br>
  <img src="https://github.com/PDooDP/opencv_FaceMask_Detection/blob/master/results/predictions_819_1000w.jpg?raw=true" width="600" height="400">

  test with 2000.weights: </br>
  <img src="https://github.com/PDooDP/opencv_FaceMask_Detection/blob/master/results/predictions_819_2000w.jpg?raw=true" width="600" height="400">
