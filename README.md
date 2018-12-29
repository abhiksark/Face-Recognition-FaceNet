
## Inspiration
* [OpenFace](https://github.com/cmusatyalab/openface)
* I refer to the facenet repository of [davidsandberg](https://github.com/davidsandberg/facenet).
* also, [shanren7](https://github.com/shanren7/real_time_face_recognition) repository was a great help in implementing.

## Dependencies

Install the dependencies first for running the code.
* Tensorflow 1.2.1 - gpu
* Python 3.5
* Same as [requirement.txt](https://github.com/davidsandberg/facenet/blob/master/requirements.txt) in [davidsandberg](https://github.com/davidsandberg/facenet) repository.

## Pre-trained models
* Inception_ResNet_v1 CASIA-WebFace-> [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)
## Face alignment using MTCNN
* You need [det1.npy, det2.npy, and det3.npy](https://github.com/davidsandberg/facenet/tree/master/src/align) in the [davidsandberg](https://github.com/davidsandberg/facenet) repository.
## How to use
* First, we need align face data. So, if you run 'Make_aligndata.py' first, the face data that is aligned in the 'output_dir' folder will be saved.
* Secord, we will cluster the same photos together.
* Third, we need to create our own classifier with the face data we created. <br/>(In the case of me, I had a high recognition rate when I made 30 pictures for each person.)
* Your own classifier is a ~.pkl file that loads the previously mentioned pre-trained model ('[20170511-185253.pb](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)') and embeds the face for each person.All of these can be obtained by running 'Make_classifier.py'.
* Finally, we load our own 'my_classifier.pkl' obtained above and then open the sensor and start recognition.
</br> (Note that, look carefully at the paths of files and folders in all .py)
## Result
<img src="https://raw.githubusercontent.com/abhiksark/pictures_cluster_classify/master/prediciton.jpg" width="60%">

### Directory Strcuture

```bash
├── 20170511-185253
│   ├── 20170511-185253.pb
│   ├── add your model file here 
│   ├── model-20170511-185253.ckpt-80000.data-00000-of-00001
│   ├── model-20170511-185253.ckpt-80000.index
│   └── model-20170511-185253.meta
├── cls
│   └── my_classifier.pkl
├── data
│   ├── det1.npy
│   ├── det2.npy
│   └── det3.npy
├── faces
│   └── aligned photos atomatically generated from raw_photos
├── labelled_faces
│   └── folder cointaining name of that person
├── raw_faces
│   └── Add your group photos here
├── raw_faces_to_aligned_faces.py
├── making_classifier.py
├── ModelManagement.py
├── labeling_faces.py
├── detect_face.py
├── facenet.py
├── classifying_static_image.py
├── clustering_faces.py
├── combine_cluster_folder.py

````

## Development

Want to contribute? Great!
Contact me in LinkedIn!

## License

* <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
	<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>
