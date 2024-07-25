import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
#lib for loading bar
from tqdm import tqdm

def extract_data_from_annotation(file_path, attributes= ['object/name','object/difficult']):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = {}

    for attr in attributes:
        elem = root.find(attr)
        if elem is not None:
            data[attr] = elem.text

    return data

def apply_rotation_random(img):
    return img.rotate(random.randrange(-180, 180))

def apply_flip_random(img):
    return img.transpose(random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]))

def apply_zoom_random(img, target_size=(216, 216), zoom_range=(0.8, 1)):
    # Taille actuelle de l'image
    width, height = img.size
    
    # Déterminer le facteur de zoom
    zoom_factor = random.uniform(*zoom_range)
    
    # Calculer les nouvelles dimensions de l'image zoomée
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    # S'assurer que les nouvelles dimensions ne dépassent pas celles de l'image d'origine
    new_width = min(new_width, width)
    new_height = min(new_height, height)
    
    # Choisir aléatoirement les coordonnées du coin supérieur gauche du rectangle de zoom
    x1 = random.randint(0, width - new_width)
    y1 = random.randint(0, height - new_height)
    x2 = x1 + new_width
    y2 = y1 + new_height
    
    # Recadrer l'image zoomée pour revenir à la taille cible
    zoomed_img = img.crop((x1, y1, x2, y2))
    zoomed_img = zoomed_img.resize(target_size, Image.LANCZOS)    
    return zoomed_img

def apply_color_adjustment_random(img,max_colour=8):
    chosen_interval = random.choices([(0, 1), (1, max_colour)], weights=[0.5, 0.5])[0]
    factor =  random.uniform(chosen_interval[0], chosen_interval[1])
    return ImageEnhance.Color(img).enhance(factor)

def apply_constrast_random(img,random=random.randint(1, 25)):
    return ImageEnhance.Sharpness(img).enhance(random)

def apply_brightness_random(img,random=random.uniform(0.1, 0.9)):
    return ImageEnhance.Brightness(img).enhance(random)

def apply_gaussian_blur_random(img,random=random.uniform(0, 2)):
    return img.filter(ImageFilter.GaussianBlur(random))

def apply_equalize_random(img,random=random.uniform(0, 2)):
    return ImageOps.equalize(img)

def weighted_random_choice(n):
    weights = np.linspace(n, 1, n) ** 2  # Crée une distribution avec des poids exponentiels décroissants
    #print(weights)
    weights = weights / weights.sum()
    return random.choices(range(1, n + 1), weights)[0]

custom_image_functions = [
    apply_rotation_random,
    apply_flip_random,
    apply_zoom_random,
    apply_color_adjustment_random,
    apply_constrast_random,
    apply_brightness_random,
    apply_gaussian_blur_random,
    apply_equalize_random,
]

def apply_alteration_random(img):
    #num_functions = random.randint(1, len(custom_image_functions))
    num_functions = weighted_random_choice(len(custom_image_functions)+1)
    modified_image = img.copy()
    for _ in range(num_functions):
        random_function = random.choice(custom_image_functions)
        modified_image = random_function(modified_image)    
    return modified_image

def get_dogs_picture_breed(images_dir= "./Datas/Images", annotations_dir= "./Datas/Annotation", output_size=(224, 224)):
    """
    Parcourt les répertoires d'images et d'annotations pour trouver les fichiers correspondants,
    redimensionne les images & leur applique aléatoirement des transformation + associe les données d'annotation.

    Args:
        images_dir (str): Chemin vers le répertoire contenant les dossiers des images /breed/"imagages .jpg"
        annotations_dir (str): Chemin vers le répertoire contenant les annotations /breed/"annotations"
        output_size (tuple): Taille de sortie pour le redimensionnement des images, par défaut les dimensions 224/224 sont celle de VGG16

    Returns:
        pd.DataFrame: DataFrame contenant les images et les données extraites des annotations.
    """
    data = []

    total_dirs = sum(1 for _ in os.walk(images_dir))

    with tqdm(total=total_dirs, desc="Processing directories") as pbar:
        for root, _, files in os.walk(images_dir):
            pbar.set_postfix(current_directory=root)

            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, images_dir)
                    corresponding_xml_path = os.path.join(annotations_dir, rel_path, file)
                    corresponding_xml_path = corresponding_xml_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

                    if os.path.exists(corresponding_xml_path):
                        resized_image = resize_image(image_path, output_size)
                        data.append((apply_alteration_random(resized_image), extract_data_from_annotation(corresponding_xml_path)["object/name"]))
    
            pbar.update(1)
    return pd.DataFrame(data, columns=['Image', 'Race'])


def resize_image(image_path, output_size):
    """
    Redimensionne l'image à la taille spécifiée.

    Args:
        image_path (str): Chemin vers l'image à redimensionner.
        output_size (tuple): Taille de sortie pour le redimensionnement de l'image.

    Returns:
        Image: Image redimensionnée.
    """
    with Image.open(image_path) as img:
        resized_img = img.resize(output_size)
        return resized_img
