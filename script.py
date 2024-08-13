import os
import shutil
import random
from math import ceil

# Fonction pour créer les répertoires si nécessaire
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Fonction pour répartir les fichiers
def split_files(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(files)
    total = len(files)
    train_end = ceil(total * train_ratio)
    val_end = train_end + ceil(total * val_ratio)
    return files[:train_end], files[train_end:val_end], files[val_end:]

# Chemins des dossiers
source_dir = "./Datas/Images_M"
dest_dir = "./dataset-script-M"

# Création des sous-dossiers de destination
create_dir(dest_dir)
for subset in ['train', 'val', 'test']:
    create_dir(os.path.join(dest_dir, subset))

# Parcours des sous-dossiers dans le dossier source
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        # Extraire le nouveau nom de sous-dossier en enlevant la partie avant le tiret
        new_subdir_name = subdir.split('-')[-1]
        
        # Créer les sous-dossiers correspondants dans train, val et test
        for subset in ['train', 'val', 'test']:
            create_dir(os.path.join(dest_dir, subset, new_subdir_name))
        
        # Obtenir la liste des fichiers dans le sous-dossier
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        
        # Répartir les fichiers selon les ratios spécifiés
        train_files, val_files, test_files = split_files(files)
        
        # Copier les fichiers dans les sous-dossiers correspondants
        for f in train_files:
            shutil.copy(os.path.join(subdir_path, f), os.path.join(dest_dir, 'train', new_subdir_name, f))
        for f in val_files:
            shutil.copy(os.path.join(subdir_path, f), os.path.join(dest_dir, 'val', new_subdir_name, f))
        for f in test_files:
            shutil.copy(os.path.join(subdir_path, f), os.path.join(dest_dir, 'test', new_subdir_name, f))
