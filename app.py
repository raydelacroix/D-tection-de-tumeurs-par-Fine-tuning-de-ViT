import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import joblib
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

class MedicalImageClassifier:
    def __init__(self, model_path='./fine_tuned_model'):
        # Charger le modèle et le feature extractor
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        
        # Charger l'encodeur de labels
        self.label_encoder = joblib.load(f'{model_path}/label_encoder.joblib')
        
    def predict(self, image):
        # Prétraiter l'image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
            pred_class_idx = torch.argmax(predictions, dim=1).item()
        
        # Convertir la prédiction en label original
        label = self.label_encoder.inverse_transform([pred_class_idx])[0]
        confidence = predictions[0][pred_class_idx].item() * 100
        
        # Préparer les probabilités pour toutes les classes
        class_probs = {}
        for i, prob in enumerate(predictions[0]):
            class_name = self.label_encoder.inverse_transform([i])[0]
            class_probs[class_name] = prob.item() * 100
        
        return label, confidence, class_probs

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Classification d'Images Médicales",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Style personnalisé avec un fond plus foncé
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #ECF0F1;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #BDC3C7;
        text-align: center;
        margin-bottom: 20px;
    }
    .stApp {
        background-color: #2C3E50;  /* Fond plus foncé */
        color: #ECF0F1;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
    }
    .result-box {
        background-color: #34495E;  /* Fond de la boîte de résultats */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #ECF0F1;
    }
    /* Styles pour les textes */
    body, .stMarkdown, .stTextInput>div>div>input {
        color: #ECF0F1 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ECF0F1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Titre de l'application
    st.markdown('<h1 class="main-title">🩺 Classification d\'Images Médicales</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligence Artificielle pour l\'Analyse d\'Images Médicales</p>', unsafe_allow_html=True)

    # Initialiser le classificateur
    classifier = MedicalImageClassifier()

    # Colonnes pour la mise en page
    col1, col2 = st.columns([2, 1])

    with col1:
        # Téléchargement de l'image
        uploaded_file = st.file_uploader(
            "Télécharger une image médicale", 
            type=["png", "jpg", "jpeg"],
            help="Téléchargez une image médicale pour analyse"
        )

        if uploaded_file is not None:
            # Ouvrir l'image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Afficher l'image
            st.image(image, caption="Image téléchargée", use_column_width=True)

    with col2:
        if uploaded_file is not None:
            # Faire la prédiction
            with st.spinner('Analyse en cours...'):
                prediction, confidence, class_probs = classifier.predict(image)
            
            # Conteneur de résultats
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            # Titre des résultats
            st.subheader("🔬 Résultats de l'Analyse")
            
            # Graphique de confiance
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Niveau de Confiance"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Résultat de la classification
            if confidence > 70:
                st.success(f"✅ Résultat : {prediction}")
            elif confidence > 50:
                st.warning(f"⚠️ Résultat probable : {prediction}")
            else:
                st.error(f"❓ Résultat incertain : {prediction}")
            
            # Graphique des probabilités de classes
            st.subheader("Probabilités par Classe")
            
            # Correction pour créer le DataFrame
            class_prob_df = pd.DataFrame(list(class_probs.items()), columns=['Classe', 'Probabilité'])
            
            fig_prob = px.bar(
                class_prob_df, 
                x='Classe', 
                y='Probabilité',
                title='Probabilités de Classification',
                labels={'Probabilité': 'Probabilité (%)'},
                color='Probabilité',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Section explicative
    st.markdown("## Comment ça fonctionne ?")
    st.markdown("""
    Notre système d'analyse d'images médicales utilise :
    - Un modèle Vision Transformer (ViT) pré-entraîné
    - Une analyse approfondie des caractéristiques visuelles
    - Une classification multi-classes 
    
    📝 **Avertissement important** : 
    - Cette analyse est un support, pas un diagnostic médical
    - Consultez toujours un professionnel de santé
    """)

if __name__ == "__main__":
    main()