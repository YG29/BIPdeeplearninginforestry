from ML import create_big_img, train_model
import numpy as np

"""
nb_img = range(1,33)
input_path = 'masks/mask'
output_path = 'big_imgs/big_label.npy'
forma = '.tif'

print('creation of the big_img...')

big_tile = create_big_img(nb_img, input_path, forma)
print(big_tile.shape)
np.save(output_path, big_tile)

print(f'big_tile SAVED in {output_path}')


print('creation of the BIG TILE')

ranges = ['1-10', '10-20', '20-30', '30-33'] 
BIG_TILE = np.lib.format.open_memmap('BIG_TILE.npy', mode='w+', dtype=np.float32, shape=(32, 1024, 1024, 19))

start = 0
for i in ranges:
    big_path = f'big_imgs/big_tile_{i}.npy'
    big = np.load(big_path)
    end = start + big.shape[0]
    BIG_TILE[start:end] = big
    start = end
	
print('BIG TILE saved')
"""


print("""which model to use : 
			1 : KNN
			2 : Random Forest """)
model = int(input())

if model == 1 : 
	model = 'KNN'
if model == 2 : 
	model = 'RF'

big_tile = np.load('BIG_TILE.npy', mmap_mode='r')
big_label = np.load('big_imgs/big_label.npy')

print("Reshaping in process...")

X_train = big_tile.reshape(-1, big_tile.shape[-1])
y_train = big_label.flatten()

print("Trainig has start... ")

train_model(X_train, y_train, model) # change KNN to RF for random forest

print("Training is DONE !")

