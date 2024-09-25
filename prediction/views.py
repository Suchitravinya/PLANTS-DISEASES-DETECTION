import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image as keras_image
from .recommendations import recommendations  
MODEL_PATH = os.path.join(settings.BASE_DIR, 'prediction/models/trained_model.keras')
model = load_model(MODEL_PATH)

# Model prediction function
def model_prediction(test_image):
    image = keras_image.load_img(test_image, target_size=(128, 128))
    input_arr = keras_image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def disease_recognition(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        image_path = os.path.join(settings.MEDIA_ROOT, file_path)
        result_index = model_prediction(image_path)

        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
            'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        predicted_class = class_name[result_index]
        disease_description = recommendations.get(predicted_class, 'No description available.')

        context = {
            'result': predicted_class,
            'description': disease_description,
            'image_url': default_storage.url(file_path)
        }

        return render(request, 'disease_recognition.html', context)

    return render(request, 'disease_recognition.html')
       
def about(request):
    disease_data = [
        ('Apple', 'Apple Scab', 'Apple scab is a fungal disease that affects the leaves, fruit, and sometimes the young twigs and shoots of apple trees.'),
        ('Apple', 'Black Rot', 'Black rot is a fungal disease that affects apples, causing dark lesions on leaves and fruit.'),
        ('Apple', 'Cedar Apple Rust', 'Cedar apple rust is a fungal disease that affects apple trees and nearby cedar trees, causing yellow spots on leaves and fruit.'),
        ('Apple', 'Healthy', 'No disease detected in the apple.'),
        ('Blueberry', 'Healthy', 'No disease detected in the blueberry.'),
        ('Cherry (including sour)', 'Powdery Mildew', 'Powdery mildew is a fungal disease that affects the leaves, stems, and fruit of cherry trees, causing a white powdery growth.'),
        ('Cherry (including sour)', 'Healthy', 'No disease detected in the cherry.'),
        ('Corn (maize)', 'Cercospora Leaf Spot / Gray Leaf Spot', 'Gray leaf spot is a fungal disease that causes elongated gray lesions on corn leaves.'),
        ('Corn (maize)', 'Common Rust', 'Common rust is a fungal disease that causes reddish-brown pustules on corn leaves.'),
        ('Corn (maize)', 'Northern Leaf Blight', 'Northern leaf blight is a fungal disease that causes cigar-shaped lesions on corn leaves.'),
        ('Corn (maize)', 'Healthy', 'No disease detected in the corn.'),
        ('Grape', 'Black Rot', 'Black rot is a fungal disease that affects grapes, causing dark spots on the leaves and fruit.'),
        ('Grape', 'Esca (Black Measles)', 'Esca, also known as black measles, is a fungal disease that causes dark streaks in the wood and black spots on the fruit.'),
        ('Grape', 'Leaf Blight (Isariopsis Leaf Spot)', 'Leaf blight is a fungal disease that causes irregular brown spots on grape leaves.'),
        ('Grape', 'Healthy', 'No disease detected in the grape.'),
        ('Orange', 'Haunglongbing (Citrus Greening)', 'Citrus greening is a bacterial disease that causes yellowing of leaves and misshapen, bitter fruit.'),
        ('Peach', 'Bacterial Spot', 'Bacterial spot is a disease that causes small, dark, water-soaked spots on the leaves and fruit of peach trees.'),
        ('Peach', 'Healthy', 'No disease detected in the peach.'),
        ('Pepper, bell', 'Bacterial Spot', 'Bacterial spot is a disease that causes small, dark, water-soaked spots on the leaves and fruit of bell peppers.'),
        ('Pepper, bell', 'Healthy', 'No disease detected in the bell pepper.'),
        ('Potato', 'Early Blight', 'Early blight is a fungal disease that causes dark, concentric rings on potato leaves and tubers.'),
        ('Potato', 'Late Blight', 'Late blight is a disease caused by the water mold Phytophthora infestans, notorious for being the cause of the Irish Potato Famine.'),
        ('Potato', 'Healthy', 'No disease detected in the potato.'),
        ('Raspberry', 'Healthy', 'No disease detected in the raspberry.'),
        ('Soybean', 'Healthy', 'No disease detected in the soybean.'),
        ('Squash', 'Powdery Mildew', 'Powdery mildew is a fungal disease that causes a white powdery growth on squash leaves.'),
        ('Strawberry', 'Leaf Scorch', 'Leaf scorch is a fungal disease that causes dark, irregular blotches on strawberry leaves.'),
        ('Strawberry', 'Healthy', 'No disease detected in the strawberry.'),
        ('Tomato', 'Bacterial Spot', 'Bacterial spot is a disease that causes small, water-soaked spots on tomato leaves, stems, and fruit.'),
        ('Tomato', 'Early Blight', 'Early blight is a fungal disease that causes dark, concentric rings on tomato leaves and fruit.'),
        ('Tomato', 'Late Blight', 'Late blight is a disease caused by the water mold Phytophthora infestans, affecting tomato leaves, stems, and fruit.'),
        ('Tomato', 'Leaf Mold', 'Leaf mold is a fungal disease that causes yellow spots on tomato leaves, which turn brown and fuzzy underneath.'),
        ('Tomato', 'Septoria Leaf Spot', 'Septoria leaf spot is a fungal disease that causes small, dark spots on tomato leaves.'),
        ('Tomato', 'Spider Mites / Two-spotted Spider Mite', 'Two-spotted spider mites are tiny pests that cause yellowing and speckling on tomato leaves.'),
        ('Tomato', 'Target Spot', 'Target spot is a fungal disease that causes dark, concentric spots on tomato leaves and fruit.'),
        ('Tomato', 'Tomato Yellow Leaf Curl Virus', 'Tomato yellow leaf curl virus is a viral disease that causes yellowing and curling of tomato leaves.'),
        ('Tomato', 'Tomato Mosaic Virus', 'Tomato mosaic virus is a viral disease that causes mottling and distortion of tomato leaves.'),
        ('Tomato', 'Healthy', 'No disease detected in the tomato.'),
    ]

    context = {
        'disease_data': disease_data,
    }
    
    return render(request, 'about.html', context)
