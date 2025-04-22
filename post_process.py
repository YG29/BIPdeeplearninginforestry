from ML import *

print("""which model to use : 
			1 : KNN
			2 : Random Forest """)
model = int(input())

if model == 1 : 
	model = 'model/KNN.joblib'
	output_path = 'KNN_predictions'
if model == 2 : 
	model = 'model/RF.joblib'
	output_path = 'RF_predictions'

print("Loading raw predicted images :\n")

predicted_imgs = []
test_imgs = []
test_labels = []

for i in range(6) : 
	pred = np.load(f'{output_path}/raw_pred/raw_pred_{i}.npy')
	img = ski.io.imread(f'tiles/tile_{i+33}.tif')
	mask = ski.io.imread(f'masks/mask_{i+33}.tif')
	predicted_imgs.append(pred)
	test_imgs.append(img)
	test_labels.append(mask)


####################################################################

print("Bounding box calculation...")

all_pred_bbox = []  # contain bboxs list
all_true_bbox = []  # bboxs of the label tiles

all_img_pred = []   # contain a list of image array 
all_img_true = []

for i in range(len(predicted_imgs)) : 
    pred_img = predicted_imgs[i]
    true_img = test_labels[i]
    
    eroded_pred = make_erosion(pred_img) # 1 erosion
    eroded_true = make_erosion(true_img)
    
    all_img_pred.append(eroded_pred)  # optional
    all_img_true.append(eroded_true)
    
    pred_bboxs = extract_bbox(eroded_pred.reshape(1024,1024), 500) 
    true_bboxs = extract_bbox(eroded_true.reshape(1024,1024), 0) # 1000 is the minimum area for an object 
    
    all_pred_bbox.append(pred_bboxs)  # optional
    all_true_bbox.append(true_bboxs)
    
    # display results
    print(f"""Tile {33+i} 
    	predicted tree : {len(pred_bboxs)} 
    	ground truth   : {len(true_bboxs)}""")

print("\n Display and saving of the images...")

visualize_and_save_bboxs(predicted_imgs, all_pred_bbox, 'RF_pred_imgs(labelPredImg-area500).png', visualise=False) # display all 6 images
#visualize_and_save_bboxs(test_imgs, all_true_bbox, 'RF_true_imgs(area-0).png', visualise=False) 

nb_pred_bboxs = [len(bboxs) for bboxs in all_pred_bbox]
nb_true_bboxs = [len(bboxs) for bboxs in all_true_bbox]


print("Calculation of the performance of the model...")

performance = accuracy_bboxs(all_pred_bbox, all_true_bbox) 
results = {
    'nb_true_bboxs': nb_true_bboxs,
    'nb_pred_bboxs': nb_pred_bboxs,
    **performance}
"""

results = {
    'nb_true_bboxs': nb_true_bboxs,
    'nb_pred_bboxs': nb_pred_bboxs}
"""
   
fileName = f'{output_path}/prediction_bboxs_3.csv'
data2csv(results, fileName)

print(f"Results saved as {fileName} !!")


