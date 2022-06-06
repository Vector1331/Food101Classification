
from keras.preprocessing import image
from keras import utils
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
food_list = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']


#loading the model i trained and finetuned
my_model = load_model('model_trained.h5')

#function to help in predicting classes of new images loaded from my computer(for now)
def predict_class(model, images, show = True):
  for img in images:
    img = utils.load_img(img, grayscale=False, color_mode='rgb', target_size=(299,299))
    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    pred = model.predict(img)
    index = np.argmax(pred)    #Returns the indices of the maximum values along an axis, In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    food_list.sort()
    pred_value = food_list[index]
    print(pred_value)
#    if show:
#        plt.imshow(img[0])
#        plt.axis('off')
#        plt.title(pred_value)
#        plt.show()

#add the images you want to predict into a list (these are in the WD)
images = []
images.append('a.jpg')
#images.append('b.jpg')
#images.append('c.jpg')


print("AE PREDICTIONS TEST BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)

