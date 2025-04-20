# app_simple.py
# Version simplifiée sans matplotlib pour éviter les erreurs d'importation

import streamlit as st
import pydicom
import os
import numpy as np
from io import BytesIO
import base64
import pandas as pd
import uuid
import datetime
import tempfile
from PIL import Image

# Configuration de la page Streamlit
st.set_page_config(
    page_title="DicomAnonymizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger une image DICOM
def load_dicom(file):
    try:
        dicom_data = pydicom.dcmread(file)
        return dicom_data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier DICOM: {e}")
        return None

# Fonction pour afficher l'image DICOM sans matplotlib
def display_dicom_image(dicom_data):
    if dicom_data is None:
        return None
    
    try:
        # Extraction de l'image
        if hasattr(dicom_data, 'pixel_array'):
            pixel_array = dicom_data.pixel_array
            
            # Normalisation pour affichage
            if pixel_array.dtype != np.uint8:
                img_min = pixel_array.min()
                img_max = pixel_array.max()
                if img_max != img_min:
                    pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            
            # Conversion pour Pillow/Streamlit
            if len(pixel_array.shape) == 2:  # Image en niveaux de gris
                image = Image.fromarray(pixel_array)
            else:  # Image RGB
                image = Image.fromarray(pixel_array[:,:,0])
            
            return image
        else:
            st.warning("Ce fichier DICOM ne contient pas d'image")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'image: {e}")
        return None

# Extraction et affichage des métadonnées pertinentes pour l'anonymisation
def extract_patient_info(dicom_data):
    if dicom_data is None:
        return {}
    
    # Liste des tags contenant potentiellement des informations personnelles
    personal_info_tags = [
        ('PatientName', 'Nom du patient'),
        ('PatientID', 'ID patient'),
        ('PatientBirthDate', 'Date de naissance'),
        ('PatientSex', 'Sexe'),
        ('ReferringPhysicianName', 'Médecin référent'),
        ('InstitutionName', 'Institution'),
        ('StudyDate', 'Date d\'étude'),
        ('AccessionNumber', 'Numéro d\'accession'),
        ('StudyID', 'ID de l\'étude'),
        ('StudyDescription', 'Description de l\'étude')
    ]
    
    patient_info = {}
    for tag, label in personal_info_tags:
        if hasattr(dicom_data, tag):
            value = getattr(dicom_data, tag)
            if value != '':
                patient_info[label] = str(value)
    
    return patient_info

# Fonction d'anonymisation des données DICOM
def anonymize_dicom(dicom_data, options):
    if dicom_data is None:
        return None, {}
    
    # Créer une copie pour l'anonymisation
    anonymized_dicom = dicom_data.copy()
    mapping = {}
    
    # Nouvelles valeurs pour l'anonymisation
    patient_id = str(uuid.uuid4())[:8].upper() if options['anonymize_id'] else anonymized_dicom.PatientID
    
    # Dictionnaire de correspondance
    if hasattr(anonymized_dicom, 'PatientID'):
        original_id = anonymized_dicom.PatientID
        mapping['PatientID'] = {'original': original_id, 'anonymized': patient_id}
    
    # Appliquer les règles d'anonymisation selon les options
    if options['anonymize_name'] and hasattr(anonymized_dicom, 'PatientName'):
        mapping['PatientName'] = {'original': str(anonymized_dicom.PatientName), 'anonymized': 'ANONYMIZED'}
        anonymized_dicom.PatientName = 'ANONYMIZED'
    
    if options['anonymize_id'] and hasattr(anonymized_dicom, 'PatientID'):
        anonymized_dicom.PatientID = patient_id
    
    if options['anonymize_birthdate'] and hasattr(anonymized_dicom, 'PatientBirthDate'):
        mapping['PatientBirthDate'] = {'original': anonymized_dicom.PatientBirthDate, 'anonymized': '19000101'}
        anonymized_dicom.PatientBirthDate = '19000101'
    
    if options['anonymize_physician'] and hasattr(anonymized_dicom, 'ReferringPhysicianName'):
        mapping['ReferringPhysicianName'] = {'original': str(anonymized_dicom.ReferringPhysicianName), 'anonymized': 'ANONYMIZED'}
        anonymized_dicom.ReferringPhysicianName = 'ANONYMIZED'
    
    if options['anonymize_institution'] and hasattr(anonymized_dicom, 'InstitutionName'):
        mapping['InstitutionName'] = {'original': str(anonymized_dicom.InstitutionName), 'anonymized': 'ANONYMIZED'}
        anonymized_dicom.InstitutionName = 'ANONYMIZED'
    
    # Traitement avancé selon le niveau d'anonymisation
    if options['anonymization_level'] == 'Élevé (DICOM Clean)':
        # Liste de tags supplémentaires à anonymiser en mode élevé
        tags_to_anonymize = [
            'AccessionNumber', 'StudyID', 'StudyDescription', 
            'SeriesDescription', 'PerformingPhysicianName', 
            'OperatorsName', 'PatientSex'
        ]
        
        for tag in tags_to_anonymize:
            if hasattr(anonymized_dicom, tag):
                value = getattr(anonymized_dicom, tag)
                if tag == 'PatientSex':  # Garder le sexe pour des raisons médicales
                    continue
                mapping[tag] = {'original': str(value), 'anonymized': 'ANONYMIZED'}
                setattr(anonymized_dicom, tag, 'ANONYMIZED')
    
    # Ajouter une métadonnée indiquant que le fichier a été anonymisé
    anonymized_dicom.add_new([0x0012, 0x0062], 'CS', 'YES')  # De-identification Method
    anonymized_dicom.add_new([0x0012, 0x0063], 'LO', f'Anonymized on {datetime.datetime.now().strftime("%Y-%m-%d")}')  # De-identification Method Code Sequence
    
    return anonymized_dicom, mapping

# Fonction pour sauvegarder le fichier DICOM anonymisé
def save_anonymized_dicom(anonymized_dicom):
    if anonymized_dicom is None:
        return None
    
    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            anonymized_dicom.save_as(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde du fichier anonymisé: {e}")
        return None

# Fonction pour générer un lien de téléchargement
def get_download_link(file_path, file_name):
    if file_path is None:
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Télécharger le fichier DICOM anonymisé</a>'
        return href
    except Exception as e:
        st.error(f"Erreur lors de la création du lien de téléchargement: {e}")
        return None

# Interface utilisateur Streamlit
def main():
    st.title("🏥 DicomAnonymizer")
    st.markdown("Anonymisation de fichiers DICOM pour le partage sécurisé avec des plateformes d'analyse")
    
    # Création de colonnes pour la mise en page
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Charger un fichier DICOM")
        uploaded_file = st.file_uploader("Choisir un fichier DICOM", type=['dcm'])
        
        if uploaded_file is not None:
            # Sauvegarder temporairement le fichier uploadé
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            # Charger le fichier DICOM
            dicom_data = load_dicom(temp_file.name)
            
            if dicom_data is not None:
                st.success("Fichier DICOM chargé avec succès!")
                
                # Afficher l'image
                st.subheader("Image DICOM")
                image = display_dicom_image(dicom_data)
                if image is not None:
                    st.image(image, caption="Image DICOM originale")
                
                # Afficher les informations du patient
                st.subheader("Informations personnelles détectées")
                patient_info = extract_patient_info(dicom_data)
                if patient_info:
                    df = pd.DataFrame(patient_info.items(), columns=['Catégorie', 'Valeur'])
                    st.table(df)
                else:
                    st.info("Aucune information personnelle détectée dans ce fichier DICOM.")
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_file.name)
    
    with col2:
        st.subheader("2. Configuration de l'anonymisation")
        
        # Options d'anonymisation
        anonymize_name = st.checkbox("Supprimer le nom du patient", value=True)
        anonymize_id = st.checkbox("Remplacer l'ID patient par un identifiant aléatoire", value=True)
        anonymize_birthdate = st.checkbox("Supprimer la date de naissance", value=True)
        anonymize_physician = st.checkbox("Anonymiser le médecin référent", value=True)
        anonymize_institution = st.checkbox("Anonymiser l'institution", value=True)
        
        # Niveau d'anonymisation
        anonymization_level = st.radio(
            "Niveau d'anonymisation :",
            ["Standard (DICOM Basic)", "Élevé (DICOM Clean)", "Personnalisé"]
        )
        
        # Options avancées si personnalisé est sélectionné
        advanced_options = {}
        if anonymization_level == "Personnalisé":
            st.subheader("Options avancées")
            advanced_options['keep_clinical'] = st.checkbox("Conserver les métadonnées cliniques", value=True)
            advanced_options['randomize_ids'] = st.checkbox("Générer des IDs cohérents", value=True)
            advanced_options['clean_private_tags'] = st.checkbox("Supprimer les tags privés", value=False)
        
        # Collecte des options
        anonymization_options = {
            'anonymize_name': anonymize_name,
            'anonymize_id': anonymize_id,
            'anonymize_birthdate': anonymize_birthdate,
            'anonymize_physician': anonymize_physician,
            'anonymize_institution': anonymize_institution,
            'anonymization_level': anonymization_level,
            **advanced_options
        }
        
        # Bouton d'anonymisation
        if uploaded_file is not None:
            if st.button("Anonymiser le fichier DICOM"):
                # Recharger le fichier DICOM original
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
                temp_file.write(uploaded_file.getvalue())
                temp_file.close()
                
                dicom_data = load_dicom(temp_file.name)
                
                # Anonymiser
                with st.spinner("Anonymisation en cours..."):
                    anonymized_dicom, mapping = anonymize_dicom(dicom_data, anonymization_options)
                    
                    if anonymized_dicom is not None:
                        # Sauvegarder le fichier anonymisé
                        anonymized_file_path = save_anonymized_dicom(anonymized_dicom)
                        
                        if anonymized_file_path is not None:
                            st.success("Anonymisation réussie!")
                            
                            # Afficher le lien de téléchargement
                            st.markdown(get_download_link(anonymized_file_path, f"anonymized_{uploaded_file.name}"), unsafe_allow_html=True)
                            
                            # Afficher la table de correspondance
                            st.subheader("Table de correspondance (à conserver en sécurité)")
                            if mapping:
                                mapping_data = []
                                for key, values in mapping.items():
                                    mapping_data.append([key, values['original'], values['anonymized']])
                                
                                mapping_df = pd.DataFrame(mapping_data, columns=['Champ', 'Valeur originale', 'Valeur anonymisée'])
                                st.table(mapping_df)
                                
                                # Option pour télécharger la table de correspondance
                                csv = mapping_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="table_correspondance.csv">Télécharger la table de correspondance</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                                st.warning("⚠️ IMPORTANT: Cette table de correspondance doit être conservée de façon sécurisée pour permettre au médecin de faire le lien entre les résultats d'analyse et le patient réel.")
                            else:
                                st.info("Aucune table de correspondance générée.")
                
                # Nettoyer le fichier temporaire
                os.unlink(temp_file.name)

    # Footer
    st.markdown("---")
    st.markdown("Application développée pour l'anonymisation d'images médicales DICOM avant partage avec des plateformes d'analyse.")

if __name__ == "__main__":
    main()