# Solution of top 2 team in NTO Infochemistry olympiad

## Description of files

`split_video_on_frames.py` - spliting video on frames <br>
`images_resize.py` - image resizing <br>
`bubble_contours_detecting.py` - contours detection <br>
`ML_on_contours.py` - training ML models on contours data(Area, Radius, Length) <br>
`Neural_network_bubbles_classification.py` - training torch classification NN models using photoes

## ML Scores

| Model  | Precision | Recall | F1 |
| ------------- | ------------- | ------------- | ------------- |
| CatBoostClassifier | 0.93 | 0.93 | 0.93 |
| RandomForestClassifier | 0.93 | 0.93 | 0.93 |
| KNeighbours | 0.81 | 0.83 | 0.81 |
| Decision Tree | 0.91 | 0.91 | 0.91 |
| Support Vector Machines | 0.83 | 0.84 | 0.83 |
| Stochastic Gradient Descent | 0.80 | 0.83 | 0.78 |
| RidgeClassifier | 0.83 | 0.85 | 0.82 |
| GradientBoostingClassifier | 0.91 | 0.91 | 0.91 |
| BaggingClassifier | 0.92 | 0.92 | 0.92 |
| AdaClassifier | 0.84 | 0.85 | 0.82 |

# NN Scores

| Model  | Precision | Recall | F1 |
| ------------- | ------------- | ------------- | ------------- |
| vgg11 | 0.949295 | 0.947912 | 0.946926 |
| vgg13 | 0.908240 | 0.899268 | 0.895954 |
| vgg16 | 0.912063 | 0.908308 | 0.905714 |
| vgg 19 | 0.924416 | 0.922944 | 0.922894 |
| resnet152 | 0.966395 | 0.966423 | 0.966098 |
| resnet101 | 0.881366 | 0.875161 | 0.872666 |
| resnet18 | 0.925533 | 0.919070 | 0.915284 |
| resnet18 | 0.925533 | 0.919070 | 0.915284 |
| resnet34 | 0.939654 | 0.934998 | 0.932295 |
