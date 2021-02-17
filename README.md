# SmileRecognition
Simple and not cudding edge face recognition system that discriminates between happy and sad expressions.

## Downloads
- [Dataset.](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) Extract it in the *dataset* folder and rename the folder *affwild*.
- [Haar classifier.](https://www.kaggle.com/lalitharajesh/haarcascades) Extract it in the *dataset* folder and rename the folder *haar*.
- [Landmark model.](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) Extract it in the *dataset* folder and rename the file *landmarks.dat*.

## Demo
To run the demo extract the archive datasets/processed.tar.xz and then execute from shell the following command
```
$ python -m demo.demo
```

## Warning
To execute the notebook in the examples folder first move it in the root folder.