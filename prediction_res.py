from ML import make_prediction
import numpy as np


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

print("Prediction starting...")

test_imgs, test_labels, predicted_imgs = make_prediction(range(33,39), model)
print('test imgs : ',len(test_imgs), test_imgs[0].shape)
print('test label : ',len(test_labels), test_labels[0].shape)
print('predicted imgs : ',len(predicted_imgs), predicted_imgs[0].shape)

print('Saving finished, saving...')

# saving of the raw predicted imgs
for i in range(len(predicted_imgs)) : 
	pred = predicted_imgs[i]
	np.save(f'{output_path}/raw_pred/raw_pred_{i}.npy', pred)



