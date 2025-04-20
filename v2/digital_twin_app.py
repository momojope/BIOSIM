import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from io import BytesIO
import json
import uuid
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
# Importer les nouveaux modules
from clinical_data_integration import ClinicalDataIntegrator
from realtime_dashboard import RealtimeDashboard
from anatomical_visualization import anatomical_visualization_tab, AnatomicalVisualization
# Importer les nouveaux modules
from clinical_data_integration import ClinicalDataIntegrator
from realtime_dashboard import RealtimeDashboard
from anatomical_visualization import anatomical_visualization_tab, AnatomicalVisualization
import datetime




class PatientDigitalTwin:
    def __init__(self, params=None):
        """Initialise un jumeau numérique avec des paramètres par défaut ou personnalisés"""
        self.default_params = {
            'age': 50,                    # ans
            'weight': 70,                 # kg
            'sex': 'M',                   # M ou F
            'baseline_glucose': 140,      # mg/dL
            'insulin_sensitivity': 0.5,   # coefficient de sensibilité [0-1]
            'glucose_absorption': 0.02,   # taux d'absorption de glucose
            'insulin_clearance': 0.01,    # taux d'élimination de l'insuline
            'hepatic_glucose': 0.8,       # production hépatique de glucose
            'renal_function': 0.9,        # fonction rénale [0-1]
            'liver_function': 0.9,        # fonction hépatique [0-1]
            'immune_response': 0.9,       # réponse immunitaire [0-1]
            'inflammatory_response': 0.5,  # réponse inflammatoire [0-1]
            'heart_rate': 75,             # battements par minute
            'blood_pressure': 120         # pression systolique mmHg
        }
        
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
            
        # État initial du patient
        self.state = {
            'glucose': self.params['baseline_glucose'],
            'insulin': 15,                # mU/L
            'drug_plasma': 0,             # concentration du médicament dans le plasma
            'drug_tissue': 0,             # concentration du médicament dans les tissus
            'immune_cells': 100,          # niveau relatif de cellules immunitaires
            'inflammation': 10,           # niveau d'inflammation (unités arbitraires)
            'heart_rate': self.params['heart_rate'],
            'blood_pressure': self.params['blood_pressure']
        }
        
        # Historique des simulations
        self.history = {
            'time': [],
            'glucose': [],
            'insulin': [],
            'drug_plasma': [],
            'drug_tissue': [],
            'immune_cells': [],
            'inflammation': [],
            'heart_rate': [],
            'blood_pressure': [],
            'interventions': [],
            'interactions': []  # Nouvelles entrées pour les interactions médicamenteuses
        }
        
        # ID unique pour ce jumeau
        self.id = str(uuid.uuid4())
        
        # Métriques de la simulation
        self.metrics = {}
    
    def pk_pd_model(self, t, y, medications=None, meal=0):
        """
        Modèle PK/PD complet avec composantes métaboliques, immunitaires et inflammatoires
        y[0]: glucose
        y[1]: insuline
        y[2]: concentration du médicament dans le plasma
        y[3]: concentration du médicament dans les tissus
        y[4]: cellules immunitaires
        y[5]: inflammation
        y[6]: fréquence cardiaque
        y[7]: pression artérielle
        """
        glucose, insulin, drug_plasma, drug_tissue, immune_cells, inflammation, heart_rate, blood_pressure = y
        
        # Initialisation des doses et types de médicaments
        drug_doses = {}
        drug_types = {}
        
        # Si des médicaments sont administrés
        if medications and len(medications) > 0:
            for med in medications:
                med_type = med.get('type', 'antidiabetic')
                med_dose = med.get('dose', 0)
                
                if med_type in drug_doses:
                    drug_doses[med_type] += med_dose
                else:
                    drug_doses[med_type] = med_dose
                    
                drug_types[med_type] = True
        
        # Constantes du modèle
        k_glucose_insulin = 0.001 * self.params['insulin_sensitivity']
        k_insulin_secretion = 0.05
        k_drug_absorption = 0.1
        k_drug_distribution = 0.05
        k_drug_elimination = 0.02 * self.params['renal_function'] * self.params['liver_function']
        
        # Effet des médicaments en fonction du type
        k_drug_effect_glucose = 0.0
        k_drug_effect_immune = 0.0
        k_drug_effect_heart = 0.0
        k_drug_effect_bp = 0.0
        
        # Calculer les effets des médicaments
        for drug_type, dose in drug_doses.items():
            if drug_type == 'antidiabetic':
                k_drug_effect_glucose += 0.1 * dose / 10
            elif drug_type == 'antiinflammatory':
                k_drug_effect_immune += 0.05 * dose / 10
            elif drug_type == 'beta_blocker':
                k_drug_effect_heart += 0.08 * dose / 10
                k_drug_effect_bp += 0.05 * dose / 10
            elif drug_type == 'vasodilator':
                k_drug_effect_bp += 0.1 * dose / 10
        
        # Interactions médicamenteuses
        interaction_factor = 1.0
        
        # Vérifier les interactions connues
        if 'antidiabetic' in drug_types and 'beta_blocker' in drug_types:
            # Les beta-bloquants peuvent masquer les symptômes d'hypoglycémie
            interaction_factor = 1.2
            self.history['interactions'].append((t, "Interaction: Les bêta-bloquants peuvent masquer les symptômes d'hypoglycémie"))
        
        if 'antiinflammatory' in drug_types and 'antidiabetic' in drug_types:
            # Les anti-inflammatoires peuvent réduire l'efficacité des antidiabétiques
            k_drug_effect_glucose *= 0.8
            self.history['interactions'].append((t, "Interaction: Les anti-inflammatoires réduisent l'efficacité des antidiabétiques"))
        
        k_immune_inflammation = 0.02 * self.params['immune_response']
        k_inflammation_decay = 0.01
        
        # Facteurs cardiovasculaires
        k_heart_rate_recovery = 0.05  # Retour à la normale
        k_blood_pressure_recovery = 0.02  # Retour à la normale
        
        # Équations du modèle
        
        # Dynamique du glucose
        dglucose_dt = (meal * self.params['glucose_absorption'] + 
                      self.params['hepatic_glucose'] - 
                      k_glucose_insulin * glucose * insulin -
                      k_drug_effect_glucose * drug_tissue * interaction_factor)
        
        # Dynamique de l'insuline
        dinsulin_dt = (k_insulin_secretion * max(0, glucose - 100) - 
                      self.params['insulin_clearance'] * insulin)
        
        # Pharmacocinétique du médicament
        total_drug_dose = sum(drug_doses.values())
        
        ddrug_plasma_dt = (total_drug_dose * k_drug_absorption - 
                          k_drug_distribution * drug_plasma + 
                          k_drug_distribution * 0.2 * drug_tissue -
                          k_drug_elimination * drug_plasma)
        
        ddrug_tissue_dt = (k_drug_distribution * drug_plasma - 
                          k_drug_distribution * 0.2 * drug_tissue)
        
        # Dynamique immunitaire et inflammatoire
        dimmune_cells_dt = (0.01 * (100 - immune_cells) + 
                           0.001 * inflammation - 
                           k_drug_effect_immune * drug_tissue * immune_cells / 100)
        
        dinflammation_dt = ((0.1 * (glucose - 100) / 100 if glucose > 100 else 0) + 
                           (k_immune_inflammation * immune_cells * 0.01) - 
                           (k_inflammation_decay * inflammation) - 
                           (k_drug_effect_immune * drug_tissue * inflammation / 50))
        
        # Dynamique cardiovasculaire
        base_heart_rate = self.params['heart_rate']
        dhr_dt = ((0.1 * (glucose - 70) if glucose < 70 else 0) +  # Hypoglycémie augmente le rythme cardiaque
                 (0.05 * inflammation / 10) -  # L'inflammation affecte le cœur
                 (k_drug_effect_heart * drug_tissue) +  # Effet des bêta-bloquants
                 (k_heart_rate_recovery * (base_heart_rate - heart_rate)))  # Tendance à revenir à la normale
        
        base_bp = self.params['blood_pressure']
        dbp_dt = ((0.2 * inflammation / 10) -  # L'inflammation augmente la pression
                 (k_drug_effect_bp * drug_tissue) +  # Effet des médicaments BP
                 (k_blood_pressure_recovery * (base_bp - blood_pressure)))  # Retour à la normale
        
        return [dglucose_dt, dinsulin_dt, ddrug_plasma_dt, 
                ddrug_tissue_dt, dimmune_cells_dt, dinflammation_dt,
                dhr_dt, dbp_dt]
    
    def simulate(self, duration=24, medications=None, meals=None):
        """
        Simuler l'évolution du patient sur une période donnée avec interventions
        """
        if medications is None:
            medications = []
        if meals is None:
            meals = [(7, 60), (12, 80), (19, 70)]  # Repas par défaut (heure, g de glucides)
        
        # Temps d'évaluation (en heures)
        t_eval = np.linspace(0, duration, 100 * duration)
        
        # État initial complet
        y0 = [
            self.state['glucose'], 
            self.state['insulin'], 
            self.state['drug_plasma'], 
            self.state['drug_tissue'],
            self.state['immune_cells'],
            self.state['inflammation'],
            self.state['heart_rate'],
            self.state['blood_pressure']
        ]
        
        # Réinitialiser l'historique des interactions
        self.history['interactions'] = []
        
        # Fonction d'intervention pour les doses et repas
        def intervention(t, y):
            active_medications = []
            meal_value = 0
            
            # Vérifier si un médicament est administré à ce moment
            for med_time, med_type, med_dose in medications:
                if abs(t - med_time) < 0.1:  # Dans un intervalle de 6 minutes
                    active_medications.append({
                        'type': med_type, 
                        'dose': med_dose
                    })
                    self.history['interventions'].append((t, f"Médicament: {med_type} - {med_dose} mg"))
            
            # Vérifier si un repas est pris à ce moment
            for meal_time, meal_carbs in meals:
                if abs(t - meal_time) < 0.1:  # Dans un intervalle de 6 minutes
                    meal_value += meal_carbs
                    self.history['interventions'].append((t, f"Repas: {meal_carbs} g"))
            
            return self.pk_pd_model(t, y, active_medications, meal_value)
        
        # Résolution des équations différentielles
        solution = solve_ivp(intervention, [0, duration], y0, t_eval=t_eval, method='RK45')
        
        # Mise à jour de l'état du patient et historique
        self.state['glucose'] = solution.y[0][-1]
        self.state['insulin'] = solution.y[1][-1]
        self.state['drug_plasma'] = solution.y[2][-1]
        self.state['drug_tissue'] = solution.y[3][-1]
        self.state['immune_cells'] = solution.y[4][-1]
        self.state['inflammation'] = solution.y[5][-1]
        self.state['heart_rate'] = solution.y[6][-1]
        self.state['blood_pressure'] = solution.y[7][-1]
        
        self.history['time'] = solution.t
        self.history['glucose'] = solution.y[0]
        self.history['insulin'] = solution.y[1]
        self.history['drug_plasma'] = solution.y[2]
        self.history['drug_tissue'] = solution.y[3]
        self.history['immune_cells'] = solution.y[4]
        self.history['inflammation'] = solution.y[5]
        self.history['heart_rate'] = solution.y[6]
        self.history['blood_pressure'] = solution.y[7]
        
        # Calculer les métriques de la simulation
        self.calculate_metrics()
        
        return solution
    
    def calculate_metrics(self):
        """Calcule des métriques utiles à partir des résultats de simulation"""
        if len(self.history['glucose']) == 0:
            return
        
        # Métriques glycémiques
        self.metrics['glucose_mean'] = np.mean(self.history['glucose'])
        self.metrics['glucose_min'] = np.min(self.history['glucose'])
        self.metrics['glucose_max'] = np.max(self.history['glucose'])
        
        # Temps passé en hyperglycémie (>180 mg/dL)
        hyperglycemia = np.sum(np.array(self.history['glucose']) > 180) / len(self.history['glucose']) * 100
        self.metrics['percent_hyperglycemia'] = hyperglycemia
        
        # Temps passé en hypoglycémie (<70 mg/dL)
        hypoglycemia = np.sum(np.array(self.history['glucose']) < 70) / len(self.history['glucose']) * 100
        self.metrics['percent_hypoglycemia'] = hypoglycemia
        
        # Temps dans la plage cible (70-180 mg/dL)
        in_range = np.sum((np.array(self.history['glucose']) >= 70) & 
                          (np.array(self.history['glucose']) <= 180)) / len(self.history['glucose']) * 100
        self.metrics['percent_in_range'] = in_range
        
        # Variabilité glycémique (écart-type)
        self.metrics['glucose_variability'] = np.std(self.history['glucose'])
        
        # Exposition médicamenteuse
        self.metrics['drug_exposure'] = np.trapz(self.history['drug_plasma'], self.history['time'])
        
        # Charge inflammatoire
        self.metrics['inflammation_burden'] = np.trapz(self.history['inflammation'], self.history['time'])
        
        # Stabilité cardiovasculaire (variabilité)
        self.metrics['hr_variability'] = np.std(self.history['heart_rate'])
        self.metrics['bp_variability'] = np.std(self.history['blood_pressure'])
        
        # Score de santé global (0-100, plus élevé = meilleur)
        # Formule simplifiée qui peut être améliorée
        health_score = 100
        
        # Pénalités pour les valeurs hors plage
        health_score -= hyperglycemia * 0.3  # Pénalité pour hyperglycémie
        health_score -= hypoglycemia * 0.5   # Pénalité forte pour hypoglycémie (plus dangereux)
        health_score -= self.metrics['glucose_variability'] * 0.2  # Pénalité pour variabilité
        health_score -= (self.metrics['inflammation_burden'] / 1000) * 10  # Pénalité pour inflammation
        
        # S'assurer que le score reste entre 0 et 100
        health_score = max(0, min(100, health_score))
        self.metrics['health_score'] = health_score
    
    def get_plot_data(self):
        """Retourne les données pour les graphiques"""
        return self.history
    
    def export_results(self):
        """Exporte les résultats sous forme de DataFrame"""
        results = pd.DataFrame({
            'Temps (heures)': self.history['time'],
            'Glycémie (mg/dL)': self.history['glucose'],
            'Insuline (mU/L)': self.history['insulin'],
            'Médicament (plasma)': self.history['drug_plasma'],
            'Médicament (tissus)': self.history['drug_tissue'],
            'Cellules immunitaires': self.history['immune_cells'],
            'Inflammation': self.history['inflammation'],
            'Rythme cardiaque (bpm)': self.history['heart_rate'],
            'Pression artérielle (mmHg)': self.history['blood_pressure']
        })
        return results
    
    def to_json(self):
        """Convertit le jumeau numérique en JSON pour sauvegarde"""
        twin_data = {
            'id': self.id,
            'params': self.params,
            'state': self.state,
            'metrics': self.metrics
        }
        return json.dumps(twin_data)
    
    @classmethod
    def from_json(cls, json_data):
        """Crée un jumeau numérique à partir de données JSON"""
        data = json.loads(json_data)
        twin = cls(data['params'])
        twin.id = data['id']
        twin.state = data['state']
        if 'metrics' in data:
            twin.metrics = data['metrics']
        return twin


# Profils de patients prédéfinis
predefined_profiles = {
    'normal': {
        'name': 'Patient Standard',
        'description': 'Adulte en bonne santé avec des paramètres physiologiques normaux',
        'params': {
            'age': 35,
            'weight': 70,
            'sex': 'M',
            'baseline_glucose': 100,
            'insulin_sensitivity': 0.8,
            'glucose_absorption': 0.02,
            'insulin_clearance': 0.01,
            'hepatic_glucose': 0.7,
            'renal_function': 1.0,
            'liver_function': 1.0,
            'immune_response': 1.0,
            'inflammatory_response': 0.3,
            'heart_rate': 70,
            'blood_pressure': 120
        }
    },
    'diabetic': {
        'name': 'Patient Diabétique',
        'description': 'Diabète de type 2 avec sensibilité à l\'insuline réduite',
        'params': {
            'age': 55,
            'weight': 85,
            'sex': 'M',
            'baseline_glucose': 180,
            'insulin_sensitivity': 0.3,
            'glucose_absorption': 0.025,
            'insulin_clearance': 0.01,
            'hepatic_glucose': 0.9,
            'renal_function': 0.8,
            'liver_function': 0.9,
            'immune_response': 0.8,
            'inflammatory_response': 0.6,
            'heart_rate': 75,
            'blood_pressure': 140
        }
    },
    'elderly': {
        'name': 'Patient Âgé',
        'description': 'Patient âgé avec fonctions physiologiques réduites',
        'params': {
            'age': 78,
            'weight': 65,
            'sex': 'F',
            'baseline_glucose': 130,
            'insulin_sensitivity': 0.6,
            'glucose_absorption': 0.015,
            'insulin_clearance': 0.008,
            'hepatic_glucose': 0.7,
            'renal_function': 0.6,
            'liver_function': 0.7,
            'immune_response': 0.6,
            'inflammatory_response': 0.5,
            'heart_rate': 80,
            'blood_pressure': 150
        }
    },
    'renal': {
        'name': 'Insuffisance Rénale',
        'description': 'Patient avec fonction rénale sévèrement réduite',
        'params': {
            'age': 60,
            'weight': 75,
            'sex': 'M',
            'baseline_glucose': 120,
            'insulin_sensitivity': 0.5,
            'glucose_absorption': 0.02,
            'insulin_clearance': 0.015,
            'hepatic_glucose': 0.8,
            'renal_function': 0.3,
            'liver_function': 0.8,
            'immune_response': 0.7,
            'inflammatory_response': 0.6,
            'heart_rate': 85,
            'blood_pressure': 160
        }
    },
    'inflammatory': {
        'name': 'Maladie Inflammatoire',
        'description': 'Patient avec niveau élevé d\'inflammation chronique',
        'params': {
            'age': 45,
            'weight': 68,
            'sex': 'F',
            'baseline_glucose': 110,
            'insulin_sensitivity': 0.6,
            'glucose_absorption': 0.02,
            'insulin_clearance': 0.01,
            'hepatic_glucose': 0.7,
            'renal_function': 0.9,
            'liver_function': 0.8,
            'immune_response': 1.2,
            'inflammatory_response': 0.9,
            'heart_rate': 78,
            'blood_pressure': 125
        }
    }
}

# Types de médicaments disponibles avec leurs propriétés
medication_types = {
    'antidiabetic': {
        'name': 'Antidiabétique',
        'description': 'Médicament qui réduit la glycémie',
        'primary_effect': 'glycémie',
        'side_effects': ['hypoglycémie', 'prise de poids']
    },
    'antiinflammatory': {
        'name': 'Anti-inflammatoire',
        'description': 'Réduit l\'inflammation et soulage la douleur',
        'primary_effect': 'inflammation',
        'side_effects': ['ulcère gastrique', 'rétention d\'eau']
    },
    'beta_blocker': {
        'name': 'Bêta-bloquant',
        'description': 'Ralentit le rythme cardiaque et réduit la pression artérielle',
        'primary_effect': 'rythme cardiaque',
        'side_effects': ['fatigue', 'masque les symptômes d\'hypoglycémie']
    },
    'vasodilator': {
        'name': 'Vasodilatateur',
        'description': 'Dilate les vaisseaux sanguins et réduit la pression artérielle',
        'primary_effect': 'pression artérielle',
        'side_effects': ['maux de tête', 'vertiges']
    }
}

# Interactions médicamenteuses connues
medication_interactions = {
    ('antidiabetic', 'beta_blocker'): {
        'description': 'Les bêta-bloquants peuvent masquer les symptômes d\'hypoglycémie',
        'severity': 'modérée',
        'recommendation': 'Surveiller attentivement la glycémie'
    },
    ('antidiabetic', 'antiinflammatory'): {
        'description': 'Les anti-inflammatoires peuvent réduire l\'efficacité des antidiabétiques',
        'severity': 'faible',
        'recommendation': 'Ajuster la dose d\'antidiabétique si nécessaire'
    },
    ('beta_blocker', 'vasodilator'): {
        'description': 'Peut provoquer une hypotension excessive',
        'severity': 'élevée',
        'recommendation': 'Éviter cette combinaison ou réduire les doses'
    }
}

def main():
    st.set_page_config(page_title="Jumeau Numérique Clinique", layout="wide")
    
    # Initialisation des variables de session si nécessaire
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False
    
    if 'twin_a' not in st.session_state:
        st.session_state.twin_a = None
    
    if 'twin_b' not in st.session_state:
        st.session_state.twin_b = None
    
    if 'has_results_a' not in st.session_state:
        st.session_state.has_results_a = False
    
    if 'has_results_b' not in st.session_state:
        st.session_state.has_results_b = False
    
    # Initialisation du dashboard en temps réel
    if 'realtime_dashboard' not in st.session_state:
        st.session_state.realtime_dashboard = RealtimeDashboard()
    
    # Initialisation de l'intégrateur de données cliniques
    if 'clinical_data_integrator' not in st.session_state:
        st.session_state.clinical_data_integrator = ClinicalDataIntegrator()
    
    st.title("Digital Twin pour la Simulation du Suivi Clinique")
    
    # Choix du mode: ajout des nouveaux onglets
    mode_tabs = st.tabs(["Mode Simple", "Mode Comparaison", "Visualisation Anatomique", 
                        "Données Cliniques Réelles", "Dashboard Temps Réel"])
    
    with mode_tabs[0]:
        st.session_state.comparison_mode = False
        simple_mode()
    
    with mode_tabs[1]:
        st.session_state.comparison_mode = True
        comparison_mode()
    
    with mode_tabs[2]:
        # Onglet pour la visualisation anatomique
        if st.session_state.has_results_a:
            anatomical_visualization_tab(st.session_state.twin_a)
        else:
            anatomical_visualization_tab()
    
    with mode_tabs[3]:
        # Nouvel onglet pour l'intégration des données cliniques réelles
        clinical_data_mode()
    
    with mode_tabs[4]:
        # Nouvel onglet pour le dashboard en temps réel
        realtime_dashboard_mode()



def simple_mode():
    """Interface pour le mode de simulation simple"""
    st.markdown("""
    Cette application permet de créer un modèle virtuel d'un patient pour tester l'impact des interventions médicales.
    Ajustez les paramètres du patient et les traitements pour observer la réponse simulée.
    """)
    
    # Mise en page en colonnes
    col1, col2 = st.columns([1, 2])
    
    # Colonne des paramètres
    with col1:
        st.header("Paramètres du Patient")
        
        # Profils prédéfinis
        st.subheader("Sélection du profil")
        profile_options = ["Personnalisé"] + [profile['name'] for profile in predefined_profiles.values()]
        selected_profile = st.selectbox("Profil du patient", profile_options)
        
        # Si on a sélectionné un profil prédéfini
        initial_params = {}
        if selected_profile != "Personnalisé":
            # Trouver le profil correspondant
            for profile_key, profile in predefined_profiles.items():
                if profile['name'] == selected_profile:
                    initial_params = profile['params']
                    st.info(profile['description'])
                    break
        
        with st.expander("Paramètres de base", expanded=True):
            age = st.slider("Âge", 18, 90, initial_params.get('age', 50))
            weight = st.slider("Poids (kg)", 40, 150, initial_params.get('weight', 70))
            sex = st.selectbox("Sexe", ["M", "F"], 0 if initial_params.get('sex', 'M') == 'M' else 1)
            
        with st.expander("Paramètres métaboliques", expanded=True):
            baseline_glucose = st.slider("Glycémie initiale (mg/dL)", 70, 300, initial_params.get('baseline_glucose', 140))
            insulin_sensitivity = st.slider("Sensibilité à l'insuline", 0.1, 1.0, initial_params.get('insulin_sensitivity', 0.5), 0.1)
            renal_function = st.slider("Fonction rénale", 0.1, 1.0, initial_params.get('renal_function', 0.9), 0.1)
            liver_function = st.slider("Fonction hépatique", 0.1, 1.0, initial_params.get('liver_function', 0.9), 0.1)
            
        with st.expander("Paramètres immunitaires", expanded=False):
            immune_response = st.slider("Réponse immunitaire", 0.1, 1.0, initial_params.get('immune_response', 0.9), 0.1)
            inflammatory_response = st.slider("Tendance inflammatoire", 0.1, 1.0, initial_params.get('inflammatory_response', 0.5), 0.1)
            
        with st.expander("Paramètres cardiovasculaires", expanded=False):
            heart_rate = st.slider("Fréquence cardiaque (bpm)", 40, 120, initial_params.get('heart_rate', 75))
            blood_pressure = st.slider("Pression artérielle (mmHg)", 90, 180, initial_params.get('blood_pressure', 120))
        
        st.header("Configuration de la Simulation")
        duration = st.slider("Durée de simulation (heures)", 12, 72, 24)
        
        # Configuration des repas
        st.subheader("Repas")
        
        num_meals = st.number_input("Nombre de repas", 0, 5, 3, 1)
        meals = []
        
        for i in range(num_meals):
            col_time, col_carbs = st.columns(2)
            with col_time:
                meal_time = st.number_input(f"Heure du repas {i+1}", 0.0, 24.0, 
                                            float(7 + i*5 if i < 3 else 7 + (i-3)*5), 0.5)
            with col_carbs:
                meal_carbs = st.number_input(f"Glucides (g) repas {i+1}", 0, 200, 70, 5)
            meals.append((meal_time, meal_carbs))
        
        # Configuration des médicaments
        st.subheader("Médicaments")
        
        num_meds = st.number_input("Nombre d'administrations", 0, 5, 2, 1)
        medications = []
        
        # Afficher les types de médicaments disponibles avec description au survol
        med_types = list(medication_types.keys())
        med_names = [medication_types[t]['name'] for t in med_types]
        
        for i in range(num_meds):
            col_time, col_type, col_dose = st.columns(3)
            with col_time:
                med_time = st.number_input(f"Heure médicament {i+1}", 0.0, 24.0, 
                                           float(8 + i*12 if i < 2 else 8 + (i-2)*8), 0.5)
            with col_type:
                med_type_name = st.selectbox(f"Type médicament {i+1}", 
                                          med_names,
                                          0 if i % 2 == 0 else 1)
                # Conversion du nom affiché vers la clé interne
                med_type = med_types[med_names.index(med_type_name)]
            with col_dose:
                med_dose = st.number_input(f"Dose (mg) {i+1}", 0.0, 50.0, 10.0, 2.5)
            medications.append((med_time, med_type, med_dose))
            
            # Afficher la description du médicament
            st.caption(f"**{med_type_name}**: {medication_types[med_type]['description']}")
        
        # Afficher les interactions médicamenteuses potentielles
        if num_meds > 1:
            st.subheader("Interactions médicamenteuses potentielles")
            
            # Collecter les types de médicaments utilisés
            used_med_types = [med[1] for med in medications]
            
            # Vérifier les interactions potentielles
            interactions_found = False
            for pair, interaction in medication_interactions.items():
                if pair[0] in used_med_types and pair[1] in used_med_types:
                    med1_name = medication_types[pair[0]]['name']
                    med2_name = medication_types[pair[1]]['name']
                    
                    # Couleur selon la sévérité
                    severity_color = "orange"
                    if interaction['severity'] == 'élevée':
                        severity_color = "red"
                    elif interaction['severity'] == 'faible':
                        severity_color = "blue"
                    
                    st.markdown(f"""
                    <div style='background-color: rgba(255, 200, 200, 0.3); padding: 10px; border-left: 5px solid {severity_color};'>
                    <b>{med1_name} + {med2_name}</b>: {interaction['description']}<br>
                    <b>Sévérité</b>: {interaction['severity']}<br>
                    <b>Recommandation</b>: {interaction['recommendation']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    interactions_found = True
            
            if not interactions_found:
                st.write("Aucune interaction connue entre les médicaments sélectionnés.")
        
        # Bouton de simulation
        if st.button("Lancer la Simulation"):
            # Création du patient avec les paramètres spécifiés
            patient_params = {
                'age': age,
                'weight': weight,
                'sex': sex,
                'baseline_glucose': baseline_glucose,
                'insulin_sensitivity': insulin_sensitivity,
                'renal_function': renal_function,
                'liver_function': liver_function,
                'immune_response': immune_response,
                'inflammatory_response': inflammatory_response,
                'heart_rate': heart_rate,
                'blood_pressure': blood_pressure
            }
            
            # Création et simulation du jumeau numérique
            twin = PatientDigitalTwin(patient_params)
            # Stocker les médicaments pour la visualisation
            twin.medications = medications
            twin.duration = duration
            twin.simulate(duration=duration, medications=medications, meals=meals)
            
            # Stockage des résultats dans la session
            st.session_state.twin_a = twin
            st.session_state.has_results_a = True
    
    # Colonne des résultats
    with col2:
        if st.session_state.has_results_a:
            twin = st.session_state.twin_a
            plot_data = twin.get_plot_data()
            
            st.header("Résultats de la Simulation")
            
            # Afficher les métriques principales
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                health_score = twin.metrics.get('health_score', 0)
                st.metric("Score de Santé", f"{health_score:.1f}/100")
                # Couleur selon le score
                if health_score > 80:
                    st.markdown("<div style='color:green;font-weight:bold'>Excellent</div>", unsafe_allow_html=True)
                elif health_score > 60:
                    st.markdown("<div style='color:orange;font-weight:bold'>Acceptable</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:red;font-weight:bold'>Préoccupant</div>", unsafe_allow_html=True)
            
            with metrics_cols[1]:
                pct_in_range = twin.metrics.get('percent_in_range', 0)
                st.metric("Glycémie dans la cible", f"{pct_in_range:.1f}%")
            
            with metrics_cols[2]:
                pct_hyper = twin.metrics.get('percent_hyperglycemia', 0)
                st.metric("Hyperglycémie", f"{pct_hyper:.1f}%", delta=None, delta_color="inverse")
            
            with metrics_cols[3]:
                pct_hypo = twin.metrics.get('percent_hypoglycemia', 0)
                st.metric("Hypoglycémie", f"{pct_hypo:.1f}%", delta=None, delta_color="inverse")
            
            # Onglets pour différents graphiques
            tabs = st.tabs(["Glycémie et Insuline", "Médicament", "Cardiovasculaire", "Inflammation", 
                           "Interactions", "Données", "Visualisation Anatomique"])
            
            with tabs[0]:
                # Graphique de la glycémie et insuline
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Glycémie
                ax1.set_xlabel('Temps (heures)')
                ax1.set_ylabel('Glycémie (mg/dL)', color='tab:blue')
                ax1.plot(plot_data['time'], plot_data['glucose'], 'b-', linewidth=2)
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax1.axhline(y=100, color='g', linestyle='--', alpha=0.7)  # Glycémie normale
                ax1.axhline(y=180, color='r', linestyle='--', alpha=0.7)  # Seuil hyperglycémie
                ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7)   # Seuil hypoglycémie
                
                # Zones colorées pour les plages glycémiques
                ax1.fill_between(plot_data['time'], 70, 180, alpha=0.1, color='green', label='Plage cible')
                
                # Insuline sur le second axe Y
                ax2 = ax1.twinx()
                ax2.set_ylabel('Insuline (mU/L)', color='tab:green')
                ax2.plot(plot_data['time'], plot_data['insulin'], 'g-', linewidth=2)
                ax2.tick_params(axis='y', labelcolor='tab:green')
                
                # Annotations pour les repas et médicaments
                for time, label in plot_data['interventions']:
                    if "Repas" in label:
                        ax1.annotate(label, xy=(time, 80), xytext=(time, 60),
                                    arrowprops=dict(facecolor='green', shrink=0.05))
                    elif "Médicament" in label:
                        ax1.annotate(label, xy=(time, 200), xytext=(time, 220),
                                    arrowprops=dict(facecolor='red', shrink=0.05))
                
                plt.title('Évolution de la glycémie et de l\'insuline')
                st.pyplot(fig)
                
                # Afficher les statistiques de glycémie
                st.write("### Statistiques de glycémie")
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("Moyenne", f"{twin.metrics['glucose_mean']:.1f} mg/dL")
                with stats_cols[1]:
                    st.metric("Minimum", f"{twin.metrics['glucose_min']:.1f} mg/dL")
                with stats_cols[2]:
                    st.metric("Maximum", f"{twin.metrics['glucose_max']:.1f} mg/dL")
                with stats_cols[3]:
                    st.metric("Variabilité", f"{twin.metrics['glucose_variability']:.1f}")
            
            with tabs[1]:
                # Graphique du médicament
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(plot_data['time'], plot_data['drug_plasma'], 'r-', linewidth=2, label='Plasma')
                ax.plot(plot_data['time'], plot_data['drug_tissue'], 'b-', linewidth=2, label='Tissus')
                ax.set_xlabel('Temps (heures)')
                ax.set_ylabel('Concentration du médicament')
                
                # Annotations pour les administrations
                for time, label in plot_data['interventions']:
                    if "Médicament" in label:
                        idx = min(int(time*100), len(plot_data['drug_plasma'])-1)
                        try:
                            y_pos = plot_data['drug_plasma'][idx]
                            ax.annotate(label, xy=(time, y_pos), 
                                      xytext=(time, y_pos+1),
                                      arrowprops=dict(facecolor='red', shrink=0.05))
                        except:
                            pass
                
                plt.title('Pharmacocinétique du médicament')
                plt.legend()
                st.pyplot(fig)
                
                # Exposition totale au médicament
                st.metric("Exposition totale au médicament (AUC)", f"{twin.metrics['drug_exposure']:.1f}")
            
            with tabs[2]:
                # Graphique cardiovasculaire
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                ax1.set_xlabel('Temps (heures)')
                ax1.set_ylabel('Fréquence cardiaque (bpm)', color='tab:red')
                ax1.plot(plot_data['time'], plot_data['heart_rate'], 'r-', linewidth=2)
                ax1.tick_params(axis='y', labelcolor='tab:red')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('Pression artérielle (mmHg)', color='tab:blue')
                ax2.plot(plot_data['time'], plot_data['blood_pressure'], 'b-', linewidth=2)
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                
                plt.title('Paramètres cardiovasculaires')
                st.pyplot(fig)
                
                # Statistiques cardiovasculaires
                cv_cols = st.columns(4)
                with cv_cols[0]:
                    st.metric("FC moyenne", f"{np.mean(plot_data['heart_rate']):.1f} bpm")
                with cv_cols[1]:
                    st.metric("Variabilité FC", f"{twin.metrics['hr_variability']:.1f}")
                with cv_cols[2]:
                    st.metric("PA moyenne", f"{np.mean(plot_data['blood_pressure']):.1f} mmHg")
                with cv_cols[3]:
                    st.metric("Variabilité PA", f"{twin.metrics['bp_variability']:.1f}")
            
            with tabs[3]:
                # Graphique de l'inflammation et réponse immunitaire
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                ax1.set_xlabel('Temps (heures)')
                ax1.set_ylabel('Inflammation', color='tab:red')
                ax1.plot(plot_data['time'], plot_data['inflammation'], 'r-', linewidth=2)
                ax1.tick_params(axis='y', labelcolor='tab:red')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('Cellules immunitaires', color='tab:blue')
                ax2.plot(plot_data['time'], plot_data['immune_cells'], 'b-', linewidth=2)
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                
                plt.title('Réponse inflammatoire et immunitaire')
                st.pyplot(fig)
                
                # Charge inflammatoire
                st.metric("Charge inflammatoire totale", f"{twin.metrics['inflammation_burden']:.1f}")
            
            with tabs[4]:
                # Afficher les interactions médicamenteuses détectées pendant la simulation
                st.subheader("Interactions médicamenteuses détectées")
                
                if len(plot_data['interactions']) > 0:
                    for time, interaction in plot_data['interactions']:
                        st.markdown(f"**À {time:.1f} heures**: {interaction}")
                    
                    # Afficher un graphique de ligne temporelle des interactions
                    if len(plot_data['interactions']) > 0:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        
                        # Extraire les temps des interactions
                        interaction_times = [t for t, _ in plot_data['interactions']]
                        
                        # Créer une visualisation simplifiée des interactions
                        ax.eventplot(interaction_times, colors='red', linewidths=2)
                        ax.set_xlabel('Temps (heures)')
                        ax.set_title('Chronologie des interactions médicamenteuses')
                        ax.set_xlim(0, duration)
                        ax.set_yticks([])
                        
                        st.pyplot(fig)
                else:
                    st.write("Aucune interaction médicamenteuse n'a été détectée pendant la simulation.")
                
                # Afficher une matrice d'interaction pour les médicaments utilisés
                used_med_types = list(set([med[1] for med in medications]))
                if len(used_med_types) > 1:
                    st.subheader("Matrice d'interactions des médicaments utilisés")
                    
                    # Créer une matrice pour les médicaments utilisés
                    fig, ax = plt.subplots(figsize=(8, 6))
                    n_meds = len(used_med_types)
                    interaction_matrix = np.zeros((n_meds, n_meds))
                    
                    # Remplir la matrice
                    for i, med1 in enumerate(used_med_types):
                        for j, med2 in enumerate(used_med_types):
                            if i != j:
                                # Vérifier si cette paire a une interaction connue
                                if (med1, med2) in medication_interactions:
                                    severity = medication_interactions[(med1, med2)]['severity']
                                    if severity == 'élevée':
                                        interaction_matrix[i, j] = 3
                                    elif severity == 'modérée':
                                        interaction_matrix[i, j] = 2
                                    else:  # faible
                                        interaction_matrix[i, j] = 1
                                elif (med2, med1) in medication_interactions:
                                    severity = medication_interactions[(med2, med1)]['severity']
                                    if severity == 'élevée':
                                        interaction_matrix[i, j] = 3
                                    elif severity == 'modérée':
                                        interaction_matrix[i, j] = 2
                                    else:  # faible
                                        interaction_matrix[i, j] = 1
                    
                    # Créer la heatmap
                    cmap = LinearSegmentedColormap.from_list('interaction_cmap', 
                                                           ['white', 'yellow', 'orange', 'red'])
                    im = ax.imshow(interaction_matrix, cmap=cmap, vmin=0, vmax=3)
                    
                    # Ajouter étiquettes
                    med_names = [medication_types[t]['name'] for t in used_med_types]
                    ax.set_xticks(np.arange(n_meds))
                    ax.set_yticks(np.arange(n_meds))
                    ax.set_xticklabels(med_names)
                    ax.set_yticklabels(med_names)
                    
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Ajouter les valeurs dans les cellules
                    for i in range(n_meds):
                        for j in range(n_meds):
                            if interaction_matrix[i, j] > 0:
                                severity_text = {1: "Faible", 2: "Modérée", 3: "Élevée"}
                                text = ax.text(j, i, severity_text[interaction_matrix[i, j]],
                                              ha="center", va="center", color="black")
                    
                    ax.set_title("Interactions entre médicaments")
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Légende
                    st.write("""
                    **Sévérité des interactions:**
                    - **Élevée**: Interaction potentiellement dangereuse
                    - **Modérée**: Précautions nécessaires
                    - **Faible**: Surveillance conseillée
                    """)
            
            with tabs[5]:
                # Affichage des données sous forme de tableau
                results_df = twin.export_results()
                st.dataframe(results_df)
                
                # Bouton pour télécharger les résultats en CSV
                buffer = BytesIO()
                results_df.to_csv(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="Télécharger les données (CSV)",
                    data=buffer,
                    file_name="simulation_results.csv",
                    mime="text/csv"
                )
                
                # Résumé des paramètres de simulation
                st.subheader("Résumé des paramètres")
                params_df = pd.DataFrame([twin.params])
                st.dataframe(params_df)
                
                # Bouton pour sauvegarder ce scénario pour comparaison
                if st.button("Sauvegarder ce scénario pour comparaison"):
                    st.session_state.scenario_a = {
                        'twin': twin,
                        'params': patient_params,
                        'medications': medications,
                        'meals': meals,
                        'duration': duration
                    }
                    st.success("Scénario sauvegardé pour comparaison future dans l'onglet 'Mode Comparaison'")
            
            # Nouvel onglet pour la visualisation anatomique
            with tabs[6]:
                # Importer le module de visualisation anatomique
                from anatomical_visualization import AnatomicalVisualization
                
                # Préparer les données pour la visualisation anatomique
                medication_concentrations = {}
                for med_time, med_type, med_dose in medications:
                    if med_type in medication_concentrations:
                        medication_concentrations[med_type] += med_dose
                    else:
                        medication_concentrations[med_type] = med_dose
                
                # Initialiser la visualisation
                viz = AnatomicalVisualization()
                
                viz_type = st.radio("Type de visualisation", 
                    ["2D Statique", "3D Interactive", "Animation Temporelle"],
                    key="viz_type_simple_mode")
                
                # Afficher la visualisation selon le type choisi
                if viz_type == "2D Statique":
                    fig = viz.create_2d_visualization(medication_concentrations)
                    st.pyplot(fig)
                    
                    # Section d'information sur les organes
                    st.subheader("Informations sur les organes")
                    selected_organ = st.selectbox(
                        "Sélectionnez un organe pour plus d'informations",
                        options=list(viz.organs_2d.keys()),
                        format_func=lambda x: viz.organs_2d[x]['name'],
                        key="simple_mode_organ_selector"
                    )
                    
                    # Afficher les informations sur l'organe sélectionné
                    organ_info = viz.display_organ_info(selected_organ)
                    if isinstance(organ_info, dict):
                        st.markdown(f"### {organ_info['title']}")
                        st.markdown(f"**Fonction:** {organ_info['function']}")
                        
                        st.markdown("**Effets des médicaments:**")
                        for med, effect in organ_info['medication_effects'].items():
                            st.markdown(f"- **{med.capitalize()}**: {effect}")
                    else:
                        st.write(organ_info)
                
                elif viz_type == "3D Interactive":
                    fig = viz.create_interactive_3d_visualization(medication_concentrations)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Cliquez et faites glisser pour faire pivoter la visualisation 3D. Survolez les organes pour voir leur nom.")
                
                elif viz_type == "Animation Temporelle":
                    # Créer une animation basée sur les données de la simulation
                    steps = 5  # Nombre d'étapes de l'animation
                    
                    # Calculer les concentrations au fil du temps
                    concentrations_over_time = []
                    for step in range(steps):
                        # Simuler l'évolution des concentrations au fil du temps
                        t_concentrations = {}
                        time_factor = step / (steps - 1)  # 0 au début, 1 à la fin
                        
                        for med_time, med_type, med_dose in medications:
                            if time_factor * duration >= med_time:
                                # Calculer un facteur d'absorption/élimination
                                elapsed = time_factor * duration - med_time
                                absorption_factor = min(1.0, elapsed)
                                elimination_factor = max(0.0, 1.0 - elapsed/12)  # Élimination sur 12h
                                
                                effect = med_dose * absorption_factor * elimination_factor
                                
                                if med_type in t_concentrations:
                                    t_concentrations[med_type] += effect
                                else:
                                    t_concentrations[med_type] = effect
                        
                        concentrations_over_time.append(t_concentrations)
                    
                    # Créer un curseur pour contrôler l'animation
                    time_step = st.slider("Temps", 0, steps-1, 0)
                    
                    # Afficher l'image pour le pas de temps sélectionné
                    if time_step < len(concentrations_over_time):
                        fig = viz.create_2d_visualization(concentrations_over_time[time_step])
                        st.pyplot(fig)
                        
                        # Afficher le temps relatif
                        current_time = time_step / (steps - 1) * duration
                        st.write(f"Temps: {current_time:.1f} heures")
                
                # Information supplémentaire
                with st.expander("À propos de cette visualisation"):
                    st.markdown("""
                    Cette visualisation montre comment différents médicaments affectent les organes du corps. 
                    
                    **Comment l'interpréter:**
                    - Les couleurs indiquent l'intensité de l'effet (blanc: aucun effet, rouge: effet important)
                    - Chaque type de médicament cible des organes spécifiques
                    - La distribution est basée sur les concentrations de médicaments dans les tissus
                    
                    **Limitations:**
                    - Il s'agit d'une simplification pour la visualisation éducative
                    - Les effets réels dépendent de nombreux facteurs individuels
                    """)
def comparison_mode():
    """Interface pour le mode de comparaison de scénarios"""
    st.markdown("""
    Ce mode vous permet de comparer deux scénarios de traitement différents côte à côte.
    Configurez deux jumeaux numériques et observez les différences dans les résultats.
    """)
    
    # Créer deux colonnes pour les configurations
    col1, col2 = st.columns(2)
    
    # Configuration du scénario A
    with col1:
        st.header("Scénario A")
        
        # Vérifier si on a un scénario sauvegardé
        if hasattr(st.session_state, 'scenario_a'):
            st.success("Scénario chargé depuis la sauvegarde")
            # Charger les paramètres du scénario sauvegardé
            scenario_a = st.session_state.scenario_a
            twin_a = scenario_a['twin']
            
            # Afficher les paramètres clés
            st.write(f"**Patient**: {twin_a.params['age']} ans, {twin_a.params['sex']}, {twin_a.params['weight']} kg")
            st.write(f"**Glycémie initiale**: {twin_a.params['baseline_glucose']} mg/dL")
            
            # Afficher les médicaments
            st.write("**Médicaments:**")
            for time, med_type, dose in scenario_a['medications']:
                med_name = medication_types[med_type]['name']
                st.write(f"- {med_name} {dose} mg à {time}h")
            
            # Bouton pour réinitialiser
            if st.button("Réinitialiser Scénario A"):
                if 'scenario_a' in st.session_state:
                    del st.session_state.scenario_a
                st.experimental_rerun()
            
            # Stocker dans la session
            st.session_state.twin_a = twin_a
            st.session_state.has_results_a = True
            
        else:
            st.write("Configurez le scénario A dans l'onglet 'Mode Simple' puis sauvegardez-le.")
            st.session_state.has_results_a = False
    
    # Configuration du scénario B
    with col2:
        st.header("Scénario B")
        
        # Sélection du profil
        profile_options = ["Personnalisé"] + [profile['name'] for profile in predefined_profiles.values()]
        selected_profile = st.selectbox("Profil du patient", profile_options, key="profile_b")
        
        # Si on a sélectionné un profil prédéfini
        initial_params = {}
        if selected_profile != "Personnalisé":
            # Trouver le profil correspondant
            for profile_key, profile in predefined_profiles.items():
                if profile['name'] == selected_profile:
                    initial_params = profile['params']
                    st.info(profile['description'])
                    break
        
        # Paramètres du patient pour le scénario B
        with st.expander("Paramètres du patient", expanded=False):
            age_b = st.slider("Âge", 18, 90, initial_params.get('age', 50), key="age_b")
            weight_b = st.slider("Poids (kg)", 40, 150, initial_params.get('weight', 70), key="weight_b")
            sex_b = st.selectbox("Sexe", ["M", "F"], 0 if initial_params.get('sex', 'M') == 'M' else 1, key="sex_b")
            baseline_glucose_b = st.slider("Glycémie initiale (mg/dL)", 70, 300, 
                                         initial_params.get('baseline_glucose', 140), key="glucose_b")
            insulin_sensitivity_b = st.slider("Sensibilité à l'insuline", 0.1, 1.0, 
                                            initial_params.get('insulin_sensitivity', 0.5), 0.1, key="insulin_b")
            renal_function_b = st.slider("Fonction rénale", 0.1, 1.0, 
                                       initial_params.get('renal_function', 0.9), 0.1, key="renal_b")
        
        # Configuration de la simulation B
        duration_b = st.slider("Durée (heures)", 12, 72, 24, key="duration_b")
        
        # Configuration simplifiée des médicaments pour le scénario B
        st.subheader("Médicaments")
        num_meds_b = st.number_input("Nombre de médicaments", 0, 5, 2, 1, key="num_meds_b")
        medications_b = []
        
        med_types = list(medication_types.keys())
        med_names = [medication_types[t]['name'] for t in med_types]
        
        for i in range(num_meds_b):
            col_time, col_type, col_dose = st.columns(3)
            with col_time:
                med_time = st.number_input(f"Heure {i+1}", 0.0, 24.0, 8.0 + i*4, 0.5, key=f"med_time_b_{i}")
            with col_type:
                med_type_name = st.selectbox(f"Type {i+1}", med_names, i % len(med_names), key=f"med_type_b_{i}")
                med_type = med_types[med_names.index(med_type_name)]
            with col_dose:
                med_dose = st.number_input(f"Dose (mg) {i+1}", 0.0, 50.0, 10.0, 2.5, key=f"med_dose_b_{i}")
            medications_b.append((med_time, med_type, med_dose))
        
        # Repas simplifiés
        meals_b = [(7, 60), (12, 80), (19, 70)]  # Repas par défaut
        
        # Bouton pour simuler le scénario B
        if st.button("Simuler Scénario B"):
            # Créer les paramètres du patient B
            patient_params_b = {
                'age': age_b,
                'weight': weight_b,
                'sex': sex_b,
                'baseline_glucose': baseline_glucose_b,
                'insulin_sensitivity': insulin_sensitivity_b,
                'renal_function': renal_function_b,
                'liver_function': initial_params.get('liver_function', 0.9),
                'immune_response': initial_params.get('immune_response', 0.9),
                'inflammatory_response': initial_params.get('inflammatory_response', 0.5),
                'heart_rate': initial_params.get('heart_rate', 75),
                'blood_pressure': initial_params.get('blood_pressure', 120)
            }
            
            # Créer et simuler le jumeau B
            twin_b = PatientDigitalTwin(patient_params_b)
            twin_b.simulate(duration=duration_b, medications=medications_b, meals=meals_b)
            
            # Stockage dans la session
            st.session_state.twin_b = twin_b
            st.session_state.has_results_b = True
    
    # Affichage des résultats comparatifs
    if st.session_state.has_results_a and st.session_state.has_results_b:
        st.header("Comparaison des Résultats")
        
        twin_a = st.session_state.twin_a
        twin_b = st.session_state.twin_b
        
        # Tableau comparatif des métriques principales
        comparison_df = pd.DataFrame({
            'Métrique': ['Score de santé', 'Glycémie moyenne', 'Temps dans la cible (%)', 
                       'Temps en hyperglycémie (%)', 'Temps en hypoglycémie (%)',
                       'Charge inflammatoire', 'Exposition au médicament'],
            'Scénario A': [
                f"{twin_a.metrics.get('health_score', 0):.1f}",
                f"{twin_a.metrics.get('glucose_mean', 0):.1f}",
                f"{twin_a.metrics.get('percent_in_range', 0):.1f}",
                f"{twin_a.metrics.get('percent_hyperglycemia', 0):.1f}",
                f"{twin_a.metrics.get('percent_hypoglycemia', 0):.1f}",
                f"{twin_a.metrics.get('inflammation_burden', 0):.1f}",
                f"{twin_a.metrics.get('drug_exposure', 0):.1f}"
            ],
            'Scénario B': [
                f"{twin_b.metrics.get('health_score', 0):.1f}",
                f"{twin_b.metrics.get('glucose_mean', 0):.1f}",
                f"{twin_b.metrics.get('percent_in_range', 0):.1f}",
                f"{twin_b.metrics.get('percent_hyperglycemia', 0):.1f}",
                f"{twin_b.metrics.get('percent_hypoglycemia', 0):.1f}",
                f"{twin_b.metrics.get('inflammation_burden', 0):.1f}",
                f"{twin_b.metrics.get('drug_exposure', 0):.1f}"
            ]
        })
        
        st.table(comparison_df)
        
        # Graphiques comparatifs sur des onglets
        compare_tabs = st.tabs(["Glycémie", "Médicament", "Inflammation", "Cardiovasculaire"])
        
        with compare_tabs[0]:
            # Comparaison des glycémies
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Tracer les deux courbes de glycémie
            ax.plot(twin_a.history['time'], twin_a.history['glucose'], 'b-', linewidth=2, label='Scénario A')
            ax.plot(twin_b.history['time'], twin_b.history['glucose'], 'r-', linewidth=2, label='Scénario B')
            
            # Lignes de référence
            ax.axhline(y=100, color='g', linestyle='--', alpha=0.5)  # Glycémie normale
            ax.axhline(y=180, color='orange', linestyle='--', alpha=0.5)  # Seuil hyperglycémie
            ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5)   # Seuil hypoglycémie
            
            # Zone cible
            ax.fill_between([0, max(twin_a.history['time'][-1], twin_b.history['time'][-1])], 
                           70, 180, alpha=0.1, color='green')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Glycémie (mg/dL)')
            ax.set_title('Comparaison des profils glycémiques')
            ax.legend()
            
            st.pyplot(fig)
            
            # Calcul des différences
            glucose_diff = twin_b.metrics['glucose_mean'] - twin_a.metrics['glucose_mean']
            in_range_diff = twin_b.metrics['percent_in_range'] - twin_a.metrics['percent_in_range']
            
            # Afficher les différences significatives
            st.subheader("Différences clés")
            diff_cols = st.columns(3)
            
            with diff_cols[0]:
                st.metric("Différence glycémie moyenne", 
                         f"{glucose_diff:.1f} mg/dL", 
                         delta=f"{glucose_diff:.1f}", 
                         delta_color="inverse")
            
            with diff_cols[1]:
                st.metric("Différence temps en cible", 
                         f"{in_range_diff:.1f}%", 
                         delta=f"{in_range_diff:.1f}%")
            
            with diff_cols[2]:
                glu_var_diff = twin_b.metrics['glucose_variability'] - twin_a.metrics['glucose_variability']
                st.metric("Différence variabilité", 
                         f"{glu_var_diff:.1f}", 
                         delta=f"{glu_var_diff:.1f}", 
                         delta_color="inverse")
        
        with compare_tabs[1]:
            # Comparaison de la pharmacocinétique
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Tracer les courbes de concentration du médicament
            ax.plot(twin_a.history['time'], twin_a.history['drug_plasma'], 'b-', linewidth=2, label='Plasma A')
            ax.plot(twin_a.history['time'], twin_a.history['drug_tissue'], 'b--', linewidth=1.5, label='Tissus A')
            ax.plot(twin_b.history['time'], twin_b.history['drug_plasma'], 'r-', linewidth=2, label='Plasma B')
            ax.plot(twin_b.history['time'], twin_b.history['drug_tissue'], 'r--', linewidth=1.5, label='Tissus B')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Concentration du médicament')
            ax.set_title('Comparaison des profils pharmacocinétiques')
            ax.legend()
            
            st.pyplot(fig)
            
            # Exposition au médicament
            drug_exp_diff = twin_b.metrics['drug_exposure'] - twin_a.metrics['drug_exposure']
            st.metric("Différence d'exposition au médicament (AUC)", 
                     f"{drug_exp_diff:.1f}", 
                     delta=f"{drug_exp_diff:.1f}")
            
            # Recommandations basées sur la pharmacocinétique
            if abs(drug_exp_diff) > twin_a.metrics['drug_exposure'] * 0.2:
                if drug_exp_diff > 0:
                    st.warning("Le scénario B présente une exposition médicamenteuse significativement plus élevée, ce qui pourrait augmenter le risque d'effets indésirables.")
                else:
                    st.warning("Le scénario B présente une exposition médicamenteuse significativement plus basse, ce qui pourrait réduire l'efficacité du traitement.")
        
        with compare_tabs[2]:
            # Comparaison de l'inflammation
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Tracer les courbes d'inflammation
            ax.plot(twin_a.history['time'], twin_a.history['inflammation'], 'b-', linewidth=2, label='Inflammation A')
            ax.plot(twin_a.history['time'], twin_a.history['immune_cells'], 'b--', linewidth=1.5, label='Immunité A')
            ax.plot(twin_b.history['time'], twin_b.history['inflammation'], 'r-', linewidth=2, label='Inflammation B')
            ax.plot(twin_b.history['time'], twin_b.history['immune_cells'], 'r--', linewidth=1.5, label='Immunité B')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Niveau')
            ax.set_title('Comparaison des réponses inflammatoires et immunitaires')
            ax.legend()
            
            st.pyplot(fig)
            
            # Différence de charge inflammatoire
            infl_diff = twin_b.metrics['inflammation_burden'] - twin_a.metrics['inflammation_burden']
            st.metric("Différence de charge inflammatoire", 
                     f"{infl_diff:.1f}", 
                     delta=f"{infl_diff:.1f}", 
                     delta_color="inverse")
            
            # Interprétation de la différence inflammatoire
            if abs(infl_diff) > twin_a.metrics['inflammation_burden'] * 0.15:
                if infl_diff < 0:
                    st.success("Le scénario B présente une réduction significative de la charge inflammatoire, ce qui est généralement bénéfique.")
                else:
                    st.error("Le scénario B présente une augmentation significative de la charge inflammatoire, ce qui pourrait être préoccupant.")
        
        with compare_tabs[3]:
            # Comparaison cardiovasculaire
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Fréquence cardiaque
            ax1.plot(twin_a.history['time'], twin_a.history['heart_rate'], 'b-', linewidth=2, label='Scénario A')
            ax1.plot(twin_b.history['time'], twin_b.history['heart_rate'], 'r-', linewidth=2, label='Scénario B')
            ax1.set_ylabel('Fréquence cardiaque (bpm)')
            ax1.set_title('Comparaison des paramètres cardiovasculaires')
            ax1.legend()
            
            # Pression artérielle
            ax2.plot(twin_a.history['time'], twin_a.history['blood_pressure'], 'b-', linewidth=2, label='Scénario A')
            ax2.plot(twin_b.history['time'], twin_b.history['blood_pressure'], 'r-', linewidth=2, label='Scénario B')
            ax2.set_xlabel('Temps (heures)')
            ax2.set_ylabel('Pression artérielle (mmHg)')
            ax2.legend()
            
            st.pyplot(fig)
            
            # Métriques cardiovasculaires
            hr_diff = np.mean(twin_b.history['heart_rate']) - np.mean(twin_a.history['heart_rate'])
            bp_diff = np.mean(twin_b.history['blood_pressure']) - np.mean(twin_a.history['blood_pressure'])
            
            cv_cols = st.columns(2)
            with cv_cols[0]:
                st.metric("Différence FC moyenne", 
                         f"{hr_diff:.1f} bpm", 
                         delta=f"{hr_diff:.1f}", 
                         delta_color="inverse")
            
            with cv_cols[1]:
                st.metric("Différence PA moyenne", 
                         f"{bp_diff:.1f} mmHg", 
                         delta=f"{bp_diff:.1f}", 
                         delta_color="inverse")
        
        # Conclusion et recommandations
        st.header("Comparaison globale et recommandations")
        
        # Comparer les scores de santé
        health_diff = twin_b.metrics['health_score'] - twin_a.metrics['health_score']
        
        # Créer un DataFrame avec les avantages et inconvénients
        pros_cons = {
            'Critère': [],
            'Avantage': [],
            'Inconvénient': [],
            'Recommandation': []
        }
        
        # Glycémie
        pros_cons['Critère'].append("Contrôle glycémique")
        if twin_b.metrics['percent_in_range'] > twin_a.metrics['percent_in_range']:
            pros_cons['Avantage'].append("Scénario B")
            pros_cons['Inconvénient'].append("Scénario A")
            pros_cons['Recommandation'].append("Le scénario B offre un meilleur temps en cible glycémique")
        else:
            pros_cons['Avantage'].append("Scénario A")
            pros_cons['Inconvénient'].append("Scénario B")
            pros_cons['Recommandation'].append("Le scénario A offre un meilleur temps en cible glycémique")
        
        # Inflammation
        pros_cons['Critère'].append("Inflammation")
        if twin_b.metrics['inflammation_burden'] < twin_a.metrics['inflammation_burden']:
            pros_cons['Avantage'].append("Scénario B")
            pros_cons['Inconvénient'].append("Scénario A")
            pros_cons['Recommandation'].append("Le scénario B réduit davantage l'inflammation")
        else:
            pros_cons['Avantage'].append("Scénario A")
            pros_cons['Inconvénient'].append("Scénario B")
            pros_cons['Recommandation'].append("Le scénario A réduit davantage l'inflammation")
        
        # Exposition médicamenteuse
        pros_cons['Critère'].append("Exposition médicamenteuse")
        if twin_b.metrics['drug_exposure'] < twin_a.metrics['drug_exposure']:
            pros_cons['Avantage'].append("Scénario B")
            pros_cons['Inconvénient'].append("Scénario A")
            pros_cons['Recommandation'].append("Le scénario B utilise moins de médicament pour l'effet obtenu")
        else:
            pros_cons['Avantage'].append("Scénario A")
            pros_cons['Inconvénient'].append("Scénario B")
            pros_cons['Recommandation'].append("Le scénario A utilise moins de médicament pour l'effet obtenu")
        
        # Créer un dataframe pour l'affichage
        pros_cons_df = pd.DataFrame(pros_cons)
        st.table(pros_cons_df)
        
        # Recommendation finale
        st.subheader("Recommandation finale")
        if health_diff > 5:
            st.success(f"Le scénario B est recommandé avec un score de santé supérieur de {health_diff:.1f} points.")
        elif health_diff < -5:
            st.success(f"Le scénario A est recommandé avec un score de santé supérieur de {-health_diff:.1f} points.")
        else:
            st.info("Les deux scénarios présentent des résultats similaires en termes de score de santé global. Le choix peut dépendre d'autres facteurs spécifiques au patient.")

def clinical_data_mode():
    """Interface pour l'intégration et la calibration avec des données cliniques réelles"""
    st.markdown("""
    ## Intégration de données cliniques réelles
    
    Dans cet onglet, vous pouvez importer des données cliniques réelles pour:
    1. Calibrer votre jumeau numérique
    2. Comparer les prédictions du modèle avec les données réelles
    3. Exporter le modèle calibré
    """)
    
    # Initialiser ou récupérer l'intégrateur
    integrator = st.session_state.clinical_data_integrator
    
    # Sélection du jumeau numérique à calibrer
    if st.session_state.has_results_a:
        integrator.twin = st.session_state.twin_a
        st.success(f"Jumeau numérique connecté: ID {integrator.twin.id}")
    else:
        st.warning("Veuillez d'abord configurer un jumeau numérique dans l'onglet 'Mode Simple'")
        return
    
    # Interface divisée en onglets
    data_tabs = st.tabs(["Importation de données", "Calibration", "Comparaison", "Export/Import"])
    
    with data_tabs[0]:
        st.subheader("Importation de données cliniques")
        st.markdown("""
        Importez vos fichiers CSV contenant des données cliniques. Formats acceptés:
        - Glycémie
        - Fréquence cardiaque
        - Pression artérielle
        - Médicaments administrés
        """)
        
        # Importer des données de glycémie
        st.write("### Données de glycémie")
        glucose_file = st.file_uploader("Fichier de glycémie (CSV)", type=['csv'], key='glucose_file')
        if glucose_file:
            glucose_data = integrator.load_csv_data(glucose_file, 'glucose')
            if glucose_data is not None:
                st.success(f"Données de glycémie chargées: {len(glucose_data)} points")
                st.dataframe(glucose_data.head())
                
                # Afficher un aperçu
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(glucose_data['hours'], glucose_data['value'], 'b.-')
                ax.set_xlabel('Temps (heures)')
                ax.set_ylabel('Glycémie (mg/dL)')
                ax.set_title('Aperçu des données de glycémie')
                st.pyplot(fig)
        
        # Importer des données de fréquence cardiaque
        st.write("### Données cardiaques")
        col1, col2 = st.columns(2)
        
        with col1:
            hr_file = st.file_uploader("Fichier de fréquence cardiaque (CSV)", type=['csv'], key='hr_file')
            if hr_file:
                hr_data = integrator.load_csv_data(hr_file, 'heart_rate')
                if hr_data is not None:
                    st.success(f"Données de fréquence cardiaque chargées: {len(hr_data)} points")
        
        with col2:
            bp_file = st.file_uploader("Fichier de pression artérielle (CSV)", type=['csv'], key='bp_file')
            if bp_file:
                bp_data = integrator.load_csv_data(bp_file, 'blood_pressure')
                if bp_data is not None:
                    st.success(f"Données de pression artérielle chargées: {len(bp_data)} points")
        
        # Importer des données sur les médicaments
        st.write("### Données sur les médicaments")
        med_file = st.file_uploader("Fichier des médicaments administrés (CSV)", type=['csv'], key='med_file')
        if med_file:
            meds = integrator.load_medication_data(med_file)
            if meds:
                st.success(f"Données de médicaments chargées: {len(meds)} administrations")
                
                # Afficher un tableau des médicaments
                meds_df = pd.DataFrame(meds, columns=['Heure', 'Type', 'Dose'])
                st.dataframe(meds_df)
    
    with data_tabs[1]:
        st.subheader("Calibration du modèle")
        st.markdown("""
        La calibration permet d'ajuster automatiquement les paramètres du jumeau numérique
        pour correspondre aux données cliniques réelles. Le processus utilise l'optimisation
        pour minimiser l'écart entre les prédictions du modèle et les données mesurées.
        """)
        
        # Vérifier si des données sont disponibles pour la calibration
        has_data = any(len(integrator.clinical_data.get(key, [])) > 0 
                     for key in ['glucose', 'heart_rate', 'blood_pressure', 'medications'])
        
        if not has_data:
            st.warning("Veuillez d'abord importer des données cliniques dans l'onglet 'Importation de données'")
        else:
            # Afficher les paramètres actuels du jumeau
            st.write("### Paramètres actuels du jumeau numérique")
            
            # Sélectionner les paramètres pertinents pour la calibration
            params_to_show = {
                'insulin_sensitivity': "Sensibilité à l'insuline",
                'glucose_absorption': "Absorption du glucose",
                'insulin_clearance': "Clairance de l'insuline",
                'hepatic_glucose': "Production hépatique de glucose",
                'renal_function': "Fonction rénale",
                'liver_function': "Fonction hépatique",
                'immune_response': "Réponse immunitaire",
                'inflammatory_response': "Réponse inflammatoire"
            }
            
            current_params = {display_name: integrator.twin.params.get(param_name, "N/A") 
                             for param_name, display_name in params_to_show.items()}
            
            # Afficher un tableau des paramètres actuels
            params_df = pd.DataFrame(list(current_params.items()), columns=['Paramètre', 'Valeur actuelle'])
            st.dataframe(params_df)
            
            # Bouton pour lancer la calibration
            if st.button("Lancer la calibration automatique"):
                success, calibrated_params_df = integrator.calibrate_model()
                
                if success:
                    st.write("### Résultats de la calibration")
                    st.dataframe(calibrated_params_df.style.format({
                        'Valeur originale': '{:.3f}',
                        'Valeur calibrée': '{:.3f}',
                        'Variation (%)': '{:.1f}'
                    }))
                    
                    # Bouton pour appliquer les paramètres calibrés
                    if st.button("Appliquer les paramètres calibrés au jumeau numérique"):
                        if integrator.apply_calibration():
                            st.success("Paramètres appliqués avec succès")
                            
                            # Simuler avec les nouveaux paramètres
                            medications = integrator.clinical_data.get('medications', [])
                            max_time = 0
                            for key in ['glucose', 'heart_rate', 'blood_pressure']:
                                if key in integrator.clinical_data and 'hours' in integrator.clinical_data[key].columns:
                                    max_time = max(max_time, integrator.clinical_data[key]['hours'].max())
                            
                            duration = max(24, max_time)  # Au moins 24h
                            integrator.twin.simulate(duration=duration, medications=medications)
                            
                            # Mettre à jour le jumeau dans la session
                            st.session_state.twin_a = integrator.twin
                            st.session_state.has_results_a = True
                            
                            # Rafraîchir la page
                            st.experimental_rerun()
                        else:
                            st.error("Erreur lors de l'application des paramètres")
    
    with data_tabs[2]:
        st.subheader("Comparaison modèle vs. données réelles")
        
        # Vérifier si le jumeau a été simulé et si des données sont disponibles
        if not st.session_state.has_results_a:
            st.warning("Veuillez d'abord configurer et simuler un jumeau numérique")
            return
        
        has_data = any(len(integrator.clinical_data.get(key, [])) > 0 
                     for key in ['glucose', 'heart_rate', 'blood_pressure'])
        
        if not has_data:
            st.warning("Veuillez d'abord importer des données cliniques")
            return
        
        # Obtenir la comparaison entre les données réelles et simulées
        comparison, metrics = integrator.compare_real_vs_simulated()
        
        if not comparison:
            st.error("Impossible de comparer les données. Vérifiez que les données réelles et la simulation couvrent la même période.")
            return
        
        # Sélectionner le type de données à comparer
        data_types = [key for key in comparison.keys()]
        if data_types:
            selected_data = st.selectbox("Sélectionner le type de données à comparer", data_types,
                                        format_func=lambda x: integrator.mapping.get(x, x.capitalize()))
            
            # Afficher le graphique de comparaison
            fig = integrator.plot_comparison(selected_data)
            if fig:
                st.pyplot(fig)
            
            # Afficher les métriques dans un tableau
            if selected_data in metrics:
                st.write("### Métriques de comparaison")
                metrics_df = pd.DataFrame({
                    'Métrique': ['RMSE', 'MAE', 'MAPE (%)', 'Corrélation'],
                    'Valeur': [
                        metrics[selected_data]['RMSE'],
                        metrics[selected_data]['MAE'],
                        metrics[selected_data]['MAPE'],
                        metrics[selected_data]['Correlation']
                    ]
                })
                st.table(metrics_df)
                
                # Interprétation des métriques
                st.write("### Interprétation")
                
                if metrics[selected_data]['Correlation'] > 0.8:
                    st.success(f"Forte corrélation ({metrics[selected_data]['Correlation']:.2f}) entre le modèle et les données réelles.")
                elif metrics[selected_data]['Correlation'] > 0.5:
                    st.info(f"Corrélation modérée ({metrics[selected_data]['Correlation']:.2f}) entre le modèle et les données réelles.")
                else:
                    st.warning(f"Faible corrélation ({metrics[selected_data]['Correlation']:.2f}) entre le modèle et les données réelles. Une calibration supplémentaire peut être nécessaire.")
                
                if metrics[selected_data]['MAPE'] < 10:
                    st.success(f"Erreur moyenne de seulement {metrics[selected_data]['MAPE']:.1f}%. Le modèle est précis.")
                elif metrics[selected_data]['MAPE'] < 20:
                    st.info(f"Erreur moyenne de {metrics[selected_data]['MAPE']:.1f}%. Le modèle est relativement précis.")
                else:
                    st.warning(f"Erreur moyenne élevée de {metrics[selected_data]['MAPE']:.1f}%. Considérez une calibration supplémentaire.")
        else:
            st.warning("Aucune donnée disponible pour la comparaison")
    
    with data_tabs[3]:
        st.subheader("Export et Import de modèle calibré")
        
        # Exporter le modèle calibré
        st.write("### Exporter le modèle calibré")
        if st.button("Exporter le modèle calibré (JSON)"):
            model_json = integrator.export_calibrated_model()
            if model_json:
                # Créer un lien de téléchargement
                st.download_button(
                    label="Télécharger le modèle JSON",
                    data=model_json,
                    file_name=f"modele_calibre_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        # Importer un modèle calibré
        st.write("### Importer un modèle calibré")
        model_file = st.file_uploader("Fichier JSON du modèle calibré", type=['json'])
        if model_file:
            model_content = model_file.read().decode('utf-8')
            if integrator.import_calibrated_model(model_content):
                st.success("Modèle calibré importé avec succès")
                
                # Appliquer le modèle importé au jumeau actuel
                if st.button("Appliquer le modèle importé au jumeau numérique"):
                    # Mettre à jour le jumeau dans la session
                    st.session_state.twin_a = integrator.twin
                    st.session_state.has_results_a = True
                    
                    # Rafraîchir la page
                    st.experimental_rerun()
            else:
                st.error("Erreur lors de l'importation du modèle")

def realtime_dashboard_mode():
    """Interface pour le dashboard de suivi en temps réel"""
    st.markdown("""
    ## Dashboard de suivi en temps réel
    
    Visualisez les paramètres critiques du jumeau numérique en temps réel pendant la simulation,
    avec alertes et chronologie interactive des interventions.
    """)
    
    # Initialiser ou récupérer le dashboard
    dashboard = st.session_state.realtime_dashboard
    
    # Vérifier si un jumeau est disponible
    if not st.session_state.has_results_a:
        st.warning("Veuillez d'abord configurer un jumeau numérique dans l'onglet 'Mode Simple'")
        return
    
    # Connecter le jumeau au dashboard
    dashboard.set_twin(st.session_state.twin_a)
    
    # Configuration de la simulation
    st.sidebar.header("Configuration de la simulation")
    
    # Définir la durée de la simulation
    duration = st.sidebar.slider("Durée de simulation (heures)", 4, 48, 12)
    
    # Configuration des repas
    st.sidebar.subheader("Repas")
    num_meals = st.sidebar.number_input("Nombre de repas", 0, 5, 3)
    meals = []
    
    for i in range(num_meals):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            meal_time = st.number_input(f"Heure repas {i+1}", 0.0, float(duration), float(i*4 + 2), 0.5, key=f"rt_meal_time_{i}")
        with col2:
            meal_carbs = st.number_input(f"Glucides (g) {i+1}", 0, 200, 60, 10, key=f"rt_meal_carbs_{i}")
        meals.append((meal_time, meal_carbs))
    
    # Configuration des médicaments
    st.sidebar.subheader("Médicaments")
    num_meds = st.sidebar.number_input("Nombre de médicaments", 0, 5, 2)
    medications = []
    
    med_types = list(medication_types.keys())
    med_names = [medication_types[t]['name'] for t in med_types]
    
    for i in range(num_meds):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            med_time = st.number_input(f"Heure méd. {i+1}", 0.0, float(duration), float(i*6 + 3), 0.5, key=f"rt_med_time_{i}")
            med_type_name = st.selectbox(f"Type méd. {i+1}", med_names, i % len(med_names), key=f"rt_med_type_{i}")
            med_type = med_types[med_names.index(med_type_name)]
        with col2:
            med_dose = st.number_input(f"Dose (mg) {i+1}", 0.0, 50.0, 10.0, 2.5, key=f"rt_med_dose_{i}")
        medications.append((med_time, med_type, med_dose))
    
    # Affichage du dashboard
    st.header("Dashboard de surveillance en temps réel")
    
    # Créer les composants du dashboard
    dashboard_components = dashboard.create_dashboard()
    
    # Boutons de contrôle de la simulation
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if not dashboard.running:
            if st.button("Démarrer la simulation en temps réel"):
                if dashboard.start_simulation(duration, medications, meals):
                    st.session_state.simulation_running = True
                else:
                    st.error("Erreur lors du démarrage de la simulation")
        else:
            st.info("Simulation en cours...")
    
    with col2:
        if dashboard.running:
            if st.button("Arrêter la simulation"):
                dashboard.stop_simulation()
                st.session_state.simulation_running = False
                st.experimental_rerun()
    
    with col3:
        # Afficher la progression
        if dashboard.running:
            progress = dashboard.current_time / duration
            st.progress(min(1.0, progress))
            st.text(f"Progression: {dashboard.current_time:.2f}h / {duration}h ({progress*100:.1f}%)")
    
    # Vérifier s'il y a des mises à jour à récupérer
    if dashboard.running:
        update = dashboard.get_update()
        if update:
            if 'finished' in update:
                st.success("Simulation terminée")
                st.experimental_rerun()
            elif 'error' in update:
                st.error(f"Erreur de simulation: {update['error']}")
                dashboard.stop_simulation()
            else:
                # Mettre à jour le dashboard
                dashboard.update_dashboard(dashboard_components)
    else:
        # Si pas de simulation en cours, afficher le dashboard avec les données actuelles
        dashboard.update_dashboard(dashboard_components)
    
    # Afficher la chronologie interactive des événements
    st.header("Chronologie des événements")
    st.info("Cette visualisation montre l'impact des interventions et les alertes déclenchées pendant la simulation.")
    
    # Afficher uniquement si la simulation a des données
    if len(dashboard.display_data['time']) > 0:
        dashboard.render_timeline_view()

# Représentation anatomique pour la visualisation des effets des médicaments
def body_system_diagram():
    """Crée un diagramme simple du corps montrant les systèmes affectés par les médicaments"""
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Définir les coordonnées des zones du corps
    body_parts = {
        'brain': {'x': 0.5, 'y': 0.9, 'r': 0.1},
        'heart': {'x': 0.5, 'y': 0.7, 'r': 0.08},
        'lungs': {'x': [0.4, 0.6], 'y': [0.7, 0.7], 'r': 0.07},
        'liver': {'x': 0.4, 'y': 0.5, 'r': 0.08},
        'pancreas': {'x': 0.6, 'y': 0.5, 'r': 0.06},
        'kidneys': {'x': [0.35, 0.65], 'y': [0.4, 0.4], 'r': 0.05},
        'intestines': {'x': 0.5, 'y': 0.3, 'r': 0.15},
        'bloodstream': {'x': [0.2, 0.4, 0.6, 0.8], 'y': [0.8, 0.6, 0.6, 0.8], 'type': 'line'}
    }
    
    # Dessiner les parties du corps
    for part, coords in body_parts.items():
        if 'type' in coords and coords['type'] == 'line':
            # Dessiner le système circulatoire
            ax.plot(coords['x'], coords['y'], 'r-', linewidth=3, alpha=0.7)
        elif isinstance(coords['x'], list):
            # Dessiner les organes pairs
            for i in range(len(coords['x'])):
                circle = plt.Circle((coords['x'][i], coords['y'][i]), coords['r'], fill=True, alpha=0.5)
                ax.add_patch(circle)
        else:
            # Dessiner les organes uniques
            circle = plt.Circle((coords['x'], coords['y']), coords['r'], fill=True, alpha=0.5)
            ax.add_patch(circle)
    
    # Étiquettes pour les organes
    ax.text(0.5, 0.95, 'Cerveau', ha='center')
    ax.text(0.5, 0.75, 'Cœur', ha='center')
    ax.text(0.4, 0.65, 'Poumon', ha='center')
    ax.text(0.6, 0.65, 'Poumon', ha='center')
    ax.text(0.4, 0.55, 'Foie', ha='center')
    ax.text(0.6, 0.55, 'Pancréas', ha='center')
    ax.text(0.35, 0.35, 'Rein', ha='center')
    ax.text(0.65, 0.35, 'Rein', ha='center')
    ax.text(0.5, 0.3, 'Intestins', ha='center')
    ax.text(0.7, 0.8, 'Système circulatoire', ha='center', color='red')
    
    # Configurer les axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Systèmes corporels affectés par les médicaments')
    
    return fig

if __name__ == "__main__":
    main()