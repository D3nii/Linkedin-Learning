from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

# Load the json file that contains the model's structure
f = Path('model_structure.json')
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights('model_weights.h5')

# Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img = image.load_img('cat.png', target_size=(32, 32))

# Convert the image to a numpy array
image_to_test = image.img_to_array(img) / 255

# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_img = np.expand_dims(image_to_test, axis=0)

# Make a prediction using the model
results = model.predict(list_img)

# Since we are only testing one image, we only need to check the first result
result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely = int(np.argmax(result))
class_likelihood = result[most_likely]

# Get the name of the most likely class
class_label = class_labels[most_likely]

# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))