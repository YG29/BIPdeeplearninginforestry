# BIP deep learning in forestry project
Order of source files and uses : 
- **pre_process.py** : add all the new features to the tiles
- **train.py** : creat a big image of shape (32, 1024, 1024, 19) for the model to train. I had "cut the cake" in small pieces to compute the final big_tile which is why the code is weird
- **prediction.py** : load the model and make prediction
- **post_process** : from the raw prediction, extract the bboxes, save them in a .csv file and as .png file for viewing the bboxs on images

All my prediction so far are stored in the RF_prediction directory : 
- **.png** : of the final bboxs
- **.csv** : for counting the bboxs
- **/raw_data** : with all the raw predicted images
