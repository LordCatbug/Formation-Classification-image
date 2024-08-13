import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
import pandas as pd

model = YOLO('./runs/classify/train3/weights/best.pt') 

def load_image(image_file):
    try:
        img = Image.open(image_file)
        img.verify()  # Vérifie que le fichier est une image valide
        img = Image.open(image_file)  # Recharge l'image après la vérification
        return img
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image : {e}")
        return None

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image depuis l'URL : {e}")
        return None

def predict_breed(img):
    results = model(img)
    return results

def handle_feedback():
    st.write("")
    st.write("")
    feedback = None    
    col1, col2 = st.columns([1, 1]) 
    with col1:
        if st.button("👍"):
            feedback = "👍"
    with col2:
        if st.button("👎"):
            feedback = "👎"

    if feedback:
        if feedback == "👍":
            st.success("Merci pour votre retour positif !")
        elif feedback == "👎":
            st.error("Pour améliorer nos résultat faites nous parvenir votre photo ainsi que la race du chien correspondante")
    else:
        st.write("Veuillez choisir un feedback en cliquant sur l'un des boutons ci-dessus.")


def display_breed_list(names):
    breed_list = pd.DataFrame(names.values(), columns=['Races'])
    st.write("Liste des races reconnues par le modèle :")
    html = breed_list.to_html(index=False, header=False, classes='custom-table')
    st.markdown(html, unsafe_allow_html=True)

st.title("doggofy v0.1b")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Détection de Race", "Liste des races connus"])

if page == "Détection de Race":
    st.title("Détection de la Race de Chien avec YOLOv8")

    # Option de chargement de l'image
    option = st.selectbox("Choisissez la source de l'image :", ["Fichier local", "URL"])

    if option == "Fichier local":
        print("local")
        image_file = st.file_uploader("Chargez une image", type=["jpg", "jpeg", "png"])
        print(image_file)
        if image_file is not None:
            print(image_file)
            img = load_image(image_file)
            if img:
                st.image(img, caption="Image chargée", use_column_width=True) 
                
                with st.spinner("En cours de prédiction..."):
                    results = predict_breed(img)

                for result in results:
                    probs = result.probs  # Obtenir les probabilités
                    print(probs)
                    listing = probs.top5conf.tolist()
                    for i in range (0,len(probs.top5)) :
                        print(result.names[i])
                        print(listing[i])
                        st.write(f"{result.names[i]} : {100*listing[i]:.2f}%")
                    
                handle_feedback()
    elif option == "URL":
        url = st.text_input("Entrez l'URL de l'image")
        if url:
            try:
                img = load_image_from_url(url)
                st.image(img, caption="Image chargée depuis l'URL", use_column_width=True)
                
                # Faire la prédiction

                with st.spinner("En cours de prédiction..."):
                    results = predict_breed(img)
                # Afficher les résultats
                st.write("Résultats de la prédiction :")
                for result in results:
                    probs = result.probs  # Obtenir les probabilités
                    print(probs)
                    listing = probs.top5conf.tolist()
                    for i in range (0,len(probs.top5)) :
                        print(result.names[i])
                        print(listing[i])
                        st.write(f"{result.names[i]} : {100*listing[i]:.2f}%")

                handle_feedback()

            except Exception as e:
              st.error(f"Erreur lors du chargement de l'image : {e}")
elif page == "Liste des races connus":
    names = model.names
    display_breed_list(names)
    