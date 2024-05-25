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
