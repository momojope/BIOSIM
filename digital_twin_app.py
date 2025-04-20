import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from io import BytesIO, StringIO
import json
import uuid
import streamlit.components.v1 as components
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from datetime import datetime
import base64
import os
import sqlite3
from PIL import Image
import hashlib
from pathlib import Path


# Classe pour la gestion des jumeaux numériques patients
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
            'interactions': []  # Entrées pour les interactions médicamenteuses
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
        
        # Stocker les paramètres de simulation pour référence
        self.duration = duration
        self.medications = medications
        self.meals = meals
        
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
            'metrics': self.metrics,
            'medications': self.medications if hasattr(self, 'medications') else [],
            'meals': self.meals if hasattr(self, 'meals') else [],
            'duration': self.duration if hasattr(self, 'duration') else 24
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
        if 'medications' in data:
            twin.medications = data['medications']
        if 'meals' in data:
            twin.meals = data['meals']
        if 'duration' in data:
            twin.duration = data['duration']
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

# Classe de gestion des utilisateurs
class UserManager:
    def __init__(self, db_path='user_database.db'):
        """
        Initialize user management system with SQLite database
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """
        Create necessary database tables if they don't exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
            ''')
            
            # Patients table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT NOT NULL,
                profile_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            ''')
            
            # Simulation results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                patient_id TEXT,
                simulation_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
            ''')
            
            conn.commit()
    
    def _hash_password(self, password, salt=None):
        """
        Hash password with salt
        """
        if salt is None:
            salt = uuid.uuid4().hex
        
        # Use SHA-256 for password hashing
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash, salt
    
    def register_user(self, username, email, password):
        """
        Register a new user
        """
        # Check if username or email already exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check existing users
            cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                return False, "Username or email already exists"
            
            # Generate unique user ID
            user_id = uuid.uuid4().hex
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            try:
                # Insert new user
                cursor.execute('''
                INSERT INTO users 
                (id, username, email, password_hash, salt) 
                VALUES (?, ?, ?, ?, ?)
                ''', (user_id, username, email, password_hash, salt))
                
                conn.commit()
                return True, user_id
            except Exception as e:
                return False, str(e)
    
    def login_user(self, username, password):
        """
        Authenticate user login
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fetch user by username
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if not user:
                return False, "User not found"
            
            # Unpack user data
            user_id, db_username, db_email, db_password_hash, db_salt, *_ = user
            
            # Verify password
            # Verify password
            input_hash, _ = self._hash_password(password, db_salt)
            
            if input_hash == db_password_hash:
                # Update last login
                cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                            (datetime.now(), user_id))
                conn.commit()
                return True, user_id
            
            return False, "Invalid password"
    
    def add_patient(self, user_id, name, profile_data):
        """
        Add a new patient for a user
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Generate unique patient ID
            patient_id = uuid.uuid4().hex
            
            try:
                # Insert patient
                cursor.execute('''
                INSERT INTO patients 
                (id, user_id, name, profile_data) 
                VALUES (?, ?, ?, ?)
                ''', (patient_id, user_id, name, json.dumps(profile_data)))
                
                conn.commit()
                return True, patient_id
            except Exception as e:
                return False, str(e)
    
    def delete_patient(self, user_id, patient_id):
        """
        Delete a patient and all associated simulations
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # First delete all simulations related to this patient
                cursor.execute('''
                DELETE FROM simulations 
                WHERE user_id = ? AND patient_id = ?
                ''', (user_id, patient_id))
                
                # Then delete the patient
                cursor.execute('''
                DELETE FROM patients 
                WHERE id = ? AND user_id = ?
                ''', (patient_id, user_id))
                
                conn.commit()
                
                # Check if any row was affected
                if cursor.rowcount > 0:
                    return True, "Patient deleted successfully"
                else:
                    return False, "Patient not found or not owned by user"
            except Exception as e:
                return False, str(e)
    
    def get_user_patients(self, user_id):
        """
        Retrieve all patients for a specific user
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, profile_data FROM patients WHERE user_id = ?", (user_id,))
            patients = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [{
                'id': p[0], 
                'name': p[1], 
                'profile_data': json.loads(p[2]) if p[2] else {}
            } for p in patients]
    
    def save_simulation(self, user_id, patient_id, simulation_data):
        """
        Save simulation results for a user and patient
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Generate unique simulation ID
            sim_id = uuid.uuid4().hex
            
            try:
                cursor.execute('''
                INSERT INTO simulations 
                (id, user_id, patient_id, simulation_data) 
                VALUES (?, ?, ?, ?)
                ''', (sim_id, user_id, patient_id, json.dumps(simulation_data)))
                
                conn.commit()
                return True, sim_id
            except Exception as e:
                return False, str(e)
    
    def get_user_simulations(self, user_id, patient_id=None):
        """
        Retrieve simulations for a user, optionally filtered by patient
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if patient_id:
                cursor.execute('''
                SELECT id, patient_id, simulation_data, created_at 
                FROM simulations 
                WHERE user_id = ? AND patient_id = ?
                ORDER BY created_at DESC
                ''', (user_id, patient_id))
            else:
                cursor.execute('''
                SELECT id, patient_id, simulation_data, created_at 
                FROM simulations 
                WHERE user_id = ?
                ORDER BY created_at DESC
                ''', (user_id,))
            
            simulations = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [{
                'id': s[0], 
                'patient_id': s[1], 
                'simulation_data': json.loads(s[2]) if s[2] else {},
                'created_at': s[3]
            } for s in simulations]

# Fonctions utilitaires pour l'interface
def get_base64_encoded_image(image_path):
    """
    Convertit une image en base64 pour l'afficher dans Streamlit
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background_image(image_path):
    """
    Définit une image d'arrière-plan pour l'application Streamlit
    """
    base64_image = get_base64_encoded_image(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def set_header_image(image_path, width="100%"):
    """
    Affiche une image d'en-tête
    """
    base64_image = get_base64_encoded_image(image_path)
    header_html = f'<img src="data:image/png;base64,{base64_image}" width="{width}">'
    st.markdown(header_html, unsafe_allow_html=True)

def login_page():
    """
    Page de connexion modernisée
    """
    # Style CSS personnalisé pour la page de connexion
    st.markdown("""
    <style>
    .login-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 30px 40px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        max-width: 500px;
        margin: 40px auto;
    }
    .login-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .login-logo {
        max-width: 120px;
        margin: 0 auto 20px;
        display: block;
    }
    .login-form {
        padding: 10px 0;
    }
    .login-form input {
        margin-bottom: 15px;
    }
    .login-button {
        background-color: #0066cc;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
        font-weight: 600;
        margin-top: 10px;
    }
    .login-tab {
        padding: 20px 0;
    }
    .login-info {
        font-size: 0.9rem;
        color: #6c757d;
        text-align: center;
        margin-top: 20px;
    }
    
    .stApp {
            background: url("static/ekg_bg.gif") center/cover no-repeat fixed;
        }
    </style>
    
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Encodage base64 pour pouvoir afficher l’image dans un tag <img>
        img_path = "biosim.png"          # adapt‑tez si l’image est ailleurs
        encoded = base64.b64encode(Path(img_path).read_bytes()).decode()

        st.markdown(f"""
            <style>
            /* Réduit l'espace vide avant la carte */
            .login-container {{
                margin-top: -9rem;   /* valeur négative ⇦ remonte la carte */
            }}
            .login-header img {{
                width: 160px;
                margin-bottom: 0.2rem;
            }}
            .login-header p {{
                margin-top: 0;
                font-size: 1rem;
                color: #2c3e50;
            }}
        </style>
        <div class="login-container" style="text-align:center;">
            <div class="login-header">
                <img src="data:image/png;base64,{encoded}" alt="BioSim logo" />
                <p>Simulez l’évolution personnalisée de vos patients</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
        # Initialize UserManager
        user_manager = UserManager()
        
        # Tabs for login and registration
        tab1, tab2 = st.tabs(["🔐 Connexion", "📝 Inscription"])
        
        with tab1:
            st.markdown('<div class="login-tab">', unsafe_allow_html=True)
            st.markdown("<h3>Connectez-vous à votre compte</h3>", unsafe_allow_html=True)
            username = st.text_input("Nom d'utilisateur", key="login_username")
            password = st.text_input("Mot de passe", type="password", key="login_password")
            
            if st.button("Se connecter", type="primary", use_container_width=True):
                if username and password:
                    success, result = user_manager.login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = result
                        st.session_state.username = username
                        st.success("Connexion réussie! Redirection...")
                        st.rerun()
                    else:
                        st.error(f"Erreur: {result}")
                else:
                    st.warning("Veuillez remplir tous les champs")
            
            st.markdown('<div class="login-info">Entrez vos identifiants pour accéder à l\'application</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="login-tab">', unsafe_allow_html=True)
            st.markdown("<h3>Créez un nouveau compte</h3>", unsafe_allow_html=True)
            new_username = st.text_input("Nom d'utilisateur", key="register_username")
            email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Mot de passe", type="password", key="register_password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password", key="register_confirm_password")
            
            if st.button("S'inscrire", type="primary", use_container_width=True):
                # Validation
                if not (new_username and email and new_password and confirm_password):
                    st.warning("Veuillez remplir tous les champs")
                elif new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_password) < 8:
                    st.error("Le mot de passe doit contenir au moins 8 caractères")
                else:
                    success, result = user_manager.register_user(new_username, email, new_password)
                    if success:
                        st.success("Inscription réussie! Vous pouvez maintenant vous connecter.")
                        
                        # Basculer automatiquement vers l'onglet de connexion
                        tab1.button("Se connecter", key="register_success")
                    else:
                        st.error(result)
            
            st.markdown("""
            <div class="login-info">
                En créant un compte, vous acceptez nos conditions d'utilisation et notre politique de confidentialité.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fermer le conteneur de connexion

def patient_management_page(user_manager):
    """
    Page de gestion des patients modernisée
    """
    # Titre et description
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Gestion des Patients</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #e6f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <p style="margin: 0; color: #0066cc; font-size: 16px;">
            <strong>👨‍⚕️ Gestion centralisée:</strong> Créez et gérez les profils de vos patients, puis simulez l'impact de différents traitements.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section d'ajout de patient
    with st.expander("➕ Ajouter un nouveau patient", expanded=True):
        st.markdown("<h3 style='color: #2c3e50;'>Créer un profil patient</h3>", unsafe_allow_html=True)
        
        # Formulaire d'ajout
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Nom du patient", placeholder="ex: Esma Aimeur")
            
            # Options de profil prédéfini
            profile_options = ["Personnalisé"] + [profile['name'] for profile in predefined_profiles.values()]
            selected_profile = st.selectbox("Sélection du profil", profile_options)
            
            # Import de fichier patient
            st.markdown("<h4 style='color: #2c3e50; margin-top: 20px;'>📤 Import de données</h4>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Importer des données patient (CSV, JSON)", type=["csv", "json"])
        
        # Get profile parameters
        initial_params = {}
        if selected_profile != "Personnalisé":
            # Find the selected profile
            for profile_key, profile in predefined_profiles.items():
                if profile['name'] == selected_profile:
                    initial_params = profile['params']
                    # Afficher la description du profil
                    with col2:
                        st.markdown(f"""
                        <div style="background-color: #f0f7ff; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0; color: #0066cc;">{selected_profile}</h4>
                            <p style="margin-bottom: 0;">{profile['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    break
        
        # Si un fichier a été téléchargé, extraire les paramètres
        patient_data_from_file = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if not df.empty and len(df) > 0:
                        # Essayer d'extraire les paramètres
                        row = df.iloc[0]
                        patient_data_from_file = {}
                        # Mapper les colonnes du CSV aux paramètres attendus
                        column_mapping = {
                            'age': 'age',
                            'poids': 'weight',
                            'sexe': 'sex',
                            'glycemie': 'baseline_glucose',
                            'sensibilite_insuline': 'insulin_sensitivity',
                            'fonction_renale': 'renal_function'
                        }
                        
                        # Extraire les valeurs disponibles
                        for file_col, param_name in column_mapping.items():
                            if file_col in df.columns:
                                patient_data_from_file[param_name] = row[file_col]
                        
                        st.success(f"Données importées avec succès à partir du fichier CSV: {len(patient_data_from_file)} paramètres trouvés")
                
                elif uploaded_file.name.endswith('.json'):
                    patient_data_from_file = json.load(uploaded_file)
                    st.success(f"Données importées avec succès à partir du fichier JSON: {len(patient_data_from_file)} paramètres trouvés")
                
                # Mettre à jour les paramètres initiaux avec les données du fichier
                if patient_data_from_file:
                    initial_params.update(patient_data_from_file)
            except Exception as e:
                st.error(f"Erreur lors de l'importation du fichier: {str(e)}")
        
        # Paramètres du patient dans la seconde colonne
        with col2:
            if selected_profile == "Personnalisé" and not patient_data_from_file:
                st.markdown("<h4 style='color: #2c3e50;'>Paramètres personnalisés</h4>", unsafe_allow_html=True)
            
            # Organiser les paramètres en onglets
            params_tabs = st.tabs(["📊 Paramètres de base", "🔬 Paramètres avancés"])
            
            with params_tabs[0]:
                age = st.slider("Âge", 18, 90, initial_params.get('age', 50))
                
                sex_options = {"M": "Homme", "F": "Femme"}
                sex = st.radio("Sexe", options=list(sex_options.keys()), 
                              format_func=lambda x: sex_options[x], 
                              horizontal=True,
                              index=0 if initial_params.get('sex', 'M') == 'M' else 1)
                
                weight = st.slider("Poids (kg)", 40, 150, initial_params.get('weight', 70))
                baseline_glucose = st.slider("Glycémie initiale (mg/dL)", 70, 300, initial_params.get('baseline_glucose', 140))
            
            with params_tabs[1]:
                insulin_sensitivity = st.slider("Sensibilité à l'insuline", 0.1, 1.0, initial_params.get('insulin_sensitivity', 0.5), 0.1)
                renal_function = st.slider("Fonction rénale", 0.1, 1.0, initial_params.get('renal_function', 0.9), 0.1)
                liver_function = st.slider("Fonction hépatique", 0.1, 1.0, initial_params.get('liver_function', 0.9), 0.1)
                immune_response = st.slider("Réponse immunitaire", 0.1, 1.0, initial_params.get('immune_response', 0.9), 0.1)
        
        # Préparer les données du profil
        patient_profile = {
            'age': age,
            'weight': weight,
            'sex': sex,
            'baseline_glucose': baseline_glucose,
            'insulin_sensitivity': insulin_sensitivity,
            'renal_function': renal_function,
            'liver_function': liver_function if 'liver_function' in locals() else initial_params.get('liver_function', 0.9),
            'immune_response': immune_response if 'immune_response' in locals() else initial_params.get('immune_response', 0.9),
            'inflammatory_response': initial_params.get('inflammatory_response', 0.5),
            'heart_rate': initial_params.get('heart_rate', 75),
            'blood_pressure': initial_params.get('blood_pressure', 120),
            'profile_type': selected_profile
        }
        
        # Bouton d'ajout avec style amélioré
        if st.button("💾 Enregistrer le Patient", type="primary", use_container_width=True):
            if not patient_name:
                st.error("Veuillez saisir un nom pour le patient")
            else:
                # Ajouter le patient
                success, patient_id = user_manager.add_patient(
                    st.session_state.user_id, 
                    patient_name, 
                    patient_profile
                )
                
                if success:
                    st.success(f"Patient {patient_name} ajouté avec succès!")
                    
                    # Créer un bouton pour aller directement à la simulation
                    if st.button("▶️ Simuler maintenant pour ce patient", type="secondary"):
                        # Charger le patient
                        patient = {
                            'id': patient_id,
                            'name': patient_name,
                            'profile_data': patient_profile
                        }
                        st.session_state.current_patient = patient
                        st.session_state.nav_option = "🩺 Simulation"
                        st.rerun()
                else:
                    st.error(f"Erreur lors de l'ajout du patient : {patient_id}")
    
    # Liste des patients existants
    st.markdown("<h3 style='color: #2c3e50; margin-top: 30px;'>📋 Mes Patients</h3>", unsafe_allow_html=True)
    
    # Récupérer les patients de l'utilisateur
    patients = user_manager.get_user_patients(st.session_state.user_id)
    
    if patients:
        # Afficher les patients dans une grille
        patient_cols = st.columns(3)
        
        for i, patient in enumerate(patients):
            with patient_cols[i % 3]:
                st.markdown(f"""
                <div style="background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="margin-top: 0; color: #2c3e50;">{patient['name']}</h4>
                    <div style="margin-bottom: 10px;">
                        <span style="background-color: #e6f2ff; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; color: #0066cc;">
                            {patient['profile_data'].get('profile_type', 'Personnalisé')}
                        </span>
                    </div>
                    <p style="font-size: 0.9rem;">
                        <strong>Âge:</strong> {patient['profile_data'].get('age', 'N/A')} ans<br>
                        <strong>Sexe:</strong> {patient['profile_data'].get('sex', 'N/A')}<br>
                        <strong>Glycémie:</strong> {patient['profile_data'].get('baseline_glucose', 'N/A')} mg/dL
                    </p>
                    <div style="display: flex; gap: 10px; margin-top: 15px;">
                """, unsafe_allow_html=True)
                
                # Boutons d'action
                sim_col, del_col = st.columns(2)
                
                with sim_col:
                    if st.button(f"🩺 Simuler", key=f"sim_{patient['id']}", use_container_width=True):
                        st.session_state.current_patient = patient
                        st.session_state.nav_option = "🩺 Simulation"
                        st.rerun()
                
                with del_col:
                    if st.button(f"🗑️ Supprimer", key=f"del_{patient['id']}", use_container_width=True):
                        # Demande de confirmation
                        if st.checkbox(f"Confirmer la suppression de {patient['name']}?", key=f"confirm_{patient['id']}"):
                            success, message = user_manager.delete_patient(
                                st.session_state.user_id, 
                                patient['id']
                            )
                            
                            if success:
                                st.success(f"Patient {patient['name']} supprimé avec succès!")
                                # Supprimer de la session si c'est le patient courant
                                if 'current_patient' in st.session_state and st.session_state.current_patient['id'] == patient['id']:
                                    del st.session_state.current_patient
                                st.rerun()
                            else:
                                st.error(f"Erreur: {message}")
                
                st.markdown("""
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Vous n'avez pas encore de patients. Créez-en un à l'aide du formulaire ci-dessus.")

def get_icons():
    """
    Retourne un dictionnaire d'icônes pour les médicaments
    """
    med_icons = {
        'antidiabetic': '🧪',
        'antiinflammatory': '🔥',
        'beta_blocker': '❤️',
        'vasodilator': '🩸'
    }
    return med_icons

def simple_mode(initial_params=None):
    """Interface pour le mode de simulation simple avec sauvegarde automatique pour comparaison"""
    
    # Si initial_params est None, utiliser un dictionnaire vide
    if initial_params is None:
        initial_params = {}
    
    # Mise en page en colonnes
    col1, col2 = st.columns([1, 2])
    
    # Colonne des paramètres
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Paramètres du Patient</h2>', unsafe_allow_html=True)
        
        # Profils prédéfinis avec un style moderne
        st.markdown("#### 👤 Sélection du profil")
        profile_options = ["Personnalisé"] + [profile['name'] for profile in predefined_profiles.values()]
        selected_profile = st.selectbox("", profile_options, 
                                    help="Choisissez un profil prédéfini ou personnalisez les paramètres")
        
        # Si on a sélectionné un profil prédéfini
        initial_params = {}
        if selected_profile != "Personnalisé":
            # Trouver le profil correspondant
            for profile_key, profile in predefined_profiles.items():
                if profile['name'] == selected_profile:
                    initial_params = profile['params']
                    st.markdown(f"""
                    <div class="patient-info">
                        <strong>{selected_profile}</strong><br>
                        {profile['description']}
                    </div>
                    """, unsafe_allow_html=True)
                    break
        
        # Paramètres regroupés dans des tabs pour une navigation plus facile
        param_tabs = st.tabs(["📋 Base", "🧪 Métabolisme", "🛡️ Immunitaire", "❤️ Cardiovasculaire"])
        
        with param_tabs[0]:
            # Paramètres de base avec design amélioré
            col_age, col_weight = st.columns(2)
            with col_age:
                age = st.slider("Âge (années)", 18, 90, initial_params.get('age', 50), 
                            help="Âge du patient en années")
            with col_weight:
                weight = st.slider("Poids (kg)", 40, 150, initial_params.get('weight', 70), 
                            help="Poids du patient en kilogrammes")
            
            sex = st.selectbox("Sexe", ["M", "F"], 0 if initial_params.get('sex', 'M') == 'M' else 1)
            
        with param_tabs[1]:
            # Paramètres métaboliques avec tooltips explicatifs
            baseline_glucose = st.slider("Glycémie initiale (mg/dL)", 70, 300, 
                                initial_params.get('baseline_glucose', 140),
                                help="Niveau de glucose sanguin à jeun")
            
            insulin_sensitivity = st.slider("Sensibilité à l'insuline", 0.1, 1.0, 
                                    initial_params.get('insulin_sensitivity', 0.5), 0.1,
                                    help="Capacité des cellules à répondre à l'insuline (1.0 = sensibilité maximale)")
            
            col_renal, col_liver = st.columns(2)
            with col_renal:
                renal_function = st.slider("Fonction rénale", 0.1, 1.0, 
                                    initial_params.get('renal_function', 0.9), 0.1,
                                    help="Efficacité de la filtration rénale (1.0 = fonction normale)")
            with col_liver:
                liver_function = st.slider("Fonction hépatique", 0.1, 1.0, 
                                    initial_params.get('liver_function', 0.9), 0.1,
                                    help="Capacité du foie à métaboliser les médicaments (1.0 = fonction normale)")
            
        with param_tabs[2]:
            # Paramètres immunitaires
            immune_response = st.slider("Réponse immunitaire", 0.1, 1.0, 
                                    initial_params.get('immune_response', 0.9), 0.1,
                                    help="Efficacité du système immunitaire (1.0 = réponse optimale)")
            
            inflammatory_response = st.slider("Tendance inflammatoire", 0.1, 1.0, 
                                    initial_params.get('inflammatory_response', 0.5), 0.1,
                                    help="Propension à développer une inflammation (1.0 = forte réponse inflammatoire)")
            
        with param_tabs[3]:
            # Paramètres cardiovasculaires
            col_hr, col_bp = st.columns(2)
            with col_hr:
                heart_rate = st.slider("Fréquence cardiaque (bpm)", 40, 120, 
                                    initial_params.get('heart_rate', 75),
                                    help="Battements cardiaques par minute au repos")
            with col_bp:
                blood_pressure = st.slider("Pression artérielle (mmHg)", 90, 180, 
                                    initial_params.get('blood_pressure', 120),
                                    help="Pression artérielle systolique")
        
        st.markdown("</div>", unsafe_allow_html=True)  # Fermer la carte des paramètres patient
        
        # Carte pour la configuration de simulation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">⚙️ Configuration de la Simulation</h2>', unsafe_allow_html=True)
        
        duration = st.slider("Durée de simulation (heures)", 12, 72, 24, 
                        help="Période totale à simuler en heures")
        
        # Onglets pour repas et médicaments
        sim_tabs = st.tabs(["🍽️ Repas", "💊 Médicaments"])
        
        # Configuration des repas
        with sim_tabs[0]:
            st.markdown("#### Configuration des repas")
            
            num_meals = st.number_input("Nombre de repas", 0, 5, 3, 1, 
                        help="Nombre de repas pendant la période de simulation")
            if num_meals > 0:
                st.markdown('<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">'
                        '<small>Les repas influencent la glycémie selon leur teneur en glucides</small></div>', 
                        unsafe_allow_html=True)
            
            meals = []
            for i in range(num_meals):
                with st.container():
                    st.markdown(f"##### Repas {i+1}")
                    col_time, col_carbs = st.columns(2)
                    with col_time:
                        meal_time = st.number_input(f"Heure", 0.0, 24.0, 
                                                float(7 + i*5 if i < 3 else 7 + (i-3)*5), 0.5,
                                                key=f"meal_time_{i}")
                    with col_carbs:
                        meal_carbs = st.number_input(f"Glucides (g)", 0, 200, 70, 5, 
                                                key=f"meal_carbs_{i}",
                                                help="Quantité de glucides dans le repas")
                    meals.append((meal_time, meal_carbs))
                
                if i < num_meals - 1:
                    st.markdown('<hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">', unsafe_allow_html=True)
        
        # Configuration des médicaments
        with sim_tabs[1]:
            st.markdown("#### Administration des médicaments")
            
            num_meds = st.number_input("Nombre d'administrations", 0, 5, 2, 1,
                                    help="Nombre total de prises médicamenteuses")
            medications = []
            
            # Afficher les types de médicaments disponibles avec description au survol
            med_types = list(medication_types.keys())
            med_names = [medication_types[t]['name'] for t in med_types]
            
            # Afficher des icônes pour chaque type de médicament
            med_icons = get_icons()
            
            for i in range(num_meds):
                st.markdown(f"##### Médicament {i+1}")
                
                # Utilisation de colonnes pour une mise en page plus compacte
                col_time, col_type, col_dose = st.columns([1, 2, 1])
                
                with col_time:
                    med_time = st.number_input(f"Heure", 0.0, 24.0, 
                                            float(8 + i*12 if i < 2 else 8 + (i-2)*8), 0.5,
                                            key=f"med_time_{i}")
                
                with col_type:
                    med_type_name = st.selectbox(f"Type", 
                                            med_names,
                                            0 if i % 2 == 0 else 1,
                                            key=f"med_type_{i}")
                    # Conversion du nom affiché vers la clé interne
                    med_type = med_types[med_names.index(med_type_name)]
                    
                    # Obtenir l'icône pour ce type de médicament
                    med_icon = med_icons.get(med_type, '💊')
                
                with col_dose:
                    med_dose = st.number_input(f"Dose (mg)", 0.0, 50.0, 10.0, 2.5,
                                            key=f"med_dose_{i}")
                
                medications.append((med_time, med_type, med_dose))
                
                # Afficher la description du médicament de manière élégante
                st.markdown(f"""
                <div style="background-color: #f0f7ff; padding: 8px 12px; border-radius: 5px; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem;">{med_icon}</span>
                    <strong>{med_type_name}</strong>: {medication_types[med_type]['description']}
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 4px;">
                        <strong>Effet principal</strong>: {medication_types[med_type]['primary_effect']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Ajouter un séparateur entre les médicaments
                if i < num_meds - 1:
                    st.markdown('<hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">', unsafe_allow_html=True)
            
            # Afficher les interactions médicamenteuses potentielles
            if num_meds > 1:
                st.markdown("#### ⚠️ Interactions médicamenteuses potentielles")
                
                # Collecter les types de médicaments utilisés
                used_med_types = [med[1] for med in medications]
                
                # Vérifier les interactions potentielles
                interactions_found = False
                
                # Conteneur pour les alertes d'interaction
                with st.container():
                    for pair, interaction in medication_interactions.items():
                        if pair[0] in used_med_types and pair[1] in used_med_types:
                            med1_name = medication_types[pair[0]]['name']
                            med2_name = medication_types[pair[1]]['name']
                            med1_icon = med_icons.get(pair[0], '💊')
                            med2_icon = med_icons.get(pair[1], '💊')
                            
                            # Couleur et icône selon la sévérité
                            severity_bg = "#fff3cd"
                            severity_border = "#ffc107"
                            severity_icon = "⚠️"
                            
                            if interaction['severity'] == 'élevée':
                                severity_bg = "#f8d7da"
                                severity_border = "#dc3545"
                                severity_icon = "🛑"
                            elif interaction['severity'] == 'faible':
                                severity_bg = "#d1ecf1"
                                severity_border = "#0dcaf0"
                                severity_icon = "ℹ️"
                            
                            st.markdown(f"""
                            <div style='background-color: {severity_bg}; padding: 12px; border-radius: 8px; border-left: 5px solid {severity_border}; margin-bottom: 12px;'>
                                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                                    <span style='font-size: 1.2rem; margin-right: 8px;'>{severity_icon}</span>
                                    <strong style='font-size: 1.1rem;'>{med1_icon} {med1_name} + {med2_icon} {med2_name}</strong>
                                </div>
                                <p style='margin: 4px 0;'>{interaction['description']}</p>
                                <div style='display: flex; justify-content: space-between; margin-top: 8px;'>
                                    <span><strong>Sévérité</strong>: {interaction['severity'].upper()}</span>
                                    <span><strong>Recommandation</strong>: {interaction['recommendation']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            interactions_found = True
                    
                    if not interactions_found:
                        st.markdown("""
                        <div style='background-color: #d4edda; padding: 12px; border-radius: 8px; border-left: 5px solid #28a745;'>
                            <span style='font-size: 1.2rem; margin-right: 8px;'>✅</span>
                            <strong>Aucune interaction connue</strong> entre les médicaments sélectionnés.
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Fermer la carte de configuration
        
        # Bouton de simulation avec style amélioré
        st.markdown('<div class="simulation-button">', unsafe_allow_html=True)
        
        if st.button("▶️ Lancer la Simulation", 
                help="Exécuter la simulation avec les paramètres configurés",
                use_container_width=True,
                type="primary"):
            # Afficher un message de traitement
            with st.spinner("Simulation en cours..."):
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
                twin.simulate(duration=duration, medications=medications, meals=meals)
                
                # Stockage des résultats dans la session
                st.session_state.twin_a = twin
                st.session_state.has_results_a = True
                
                # Stockage automatique pour la comparaison (nouveau)
                st.session_state.scenario_a = {
                    'twin': twin,
                    'params': patient_params,
                    'medications': medications,
                    'meals': meals,
                    'duration': duration,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            # Afficher un message de succès
            st.success("Simulation terminée avec succès! Le scénario est automatiquement sauvegardé pour comparaison.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Colonne des résultats
    with col2:
        if st.session_state.has_results_a:
            twin = st.session_state.twin_a
            plot_data = twin.get_plot_data()
            
            st.markdown('<h2 class="sub-header">Résultats de la Simulation</h2>', unsafe_allow_html=True)
            
            # Afficher les métriques principales dans des cartes modernes
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                health_score = twin.metrics.get('health_score', 0)
                score_color = "#28a745" if health_score > 80 else ("#ffc107" if health_score > 60 else "#dc3545")
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {score_color};">{health_score:.1f}<small>/100</small></div>
                    <div class="metric-label">Score de Santé</div>
                    <div style="font-weight: bold; color: {score_color}; font-size: 0.9rem; margin-top: 0.5rem;">
                        {("Excellent" if health_score > 80 else "Acceptable" if health_score > 60 else "Préoccupant")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[1]:
                pct_in_range = twin.metrics.get('percent_in_range', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{pct_in_range:.1f}<small>%</small></div>
                    <div class="metric-label">Glycémie dans la cible</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[2]:
                pct_hyper = twin.metrics.get('percent_hyperglycemia', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {'#dc3545' if pct_hyper > 30 else '#0066cc'};">
                        {pct_hyper:.1f}<small>%</small>
                    </div>
                    <div class="metric-label">Hyperglycémie</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[3]:
                pct_hypo = twin.metrics.get('percent_hypoglycemia', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {'#dc3545' if pct_hypo > 5 else '#0066cc'};">
                        {pct_hypo:.1f}<small>%</small>
                    </div>
                    <div class="metric-label">Hypoglycémie</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Onglets pour différents graphiques
            tabs = st.tabs([
                "📊 Glycémie et Insuline", 
                "💊 Médicament", 
                "❤️ Cardiovasculaire", 
                "🔥 Inflammation", 
                "⚠️ Interactions", 
                "📋 Données"
            ])
            
            with tabs[0]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Graphique de la glycémie et insuline avec style amélioré
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Fond plus propre
                ax1.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#ffffff')
                
                # Glycémie
                ax1.set_xlabel('Temps (heures)')
                ax1.set_ylabel('Glycémie (mg/dL)', color='#0066cc')
                ax1.plot(plot_data['time'], plot_data['glucose'], color='#0066cc', linewidth=2.5)
                ax1.tick_params(axis='y', labelcolor='#0066cc')
                
                # Lignes de référence avec style amélioré
                ax1.axhline(y=100, color='#28a745', linestyle='--', alpha=0.7, linewidth=1.5)
                ax1.axhline(y=180, color='#dc3545', linestyle='--', alpha=0.7, linewidth=1.5)
                ax1.axhline(y=70, color='#dc3545', linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Zones colorées pour les plages glycémiques
                ax1.fill_between(plot_data['time'], 70, 180, alpha=0.15, color='#28a745', label='Plage cible')
                
                # Insuline sur le second axe Y
                ax2 = ax1.twinx()
                ax2.set_ylabel('Insuline (mU/L)', color='#28a745')
                ax2.plot(plot_data['time'], plot_data['insulin'], color='#28a745', linewidth=2)
                ax2.tick_params(axis='y', labelcolor='#28a745')
                
                # Grille légère pour la lisibilité
                ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
                
                # Annotations pour les repas et médicaments
                for time, label in plot_data['interventions']:
                    if "Repas" in label:
                        # Extraire la quantité de glucides
                        carbs = int(label.split(": ")[1].split(" ")[0])
                        
                        # Adapter la taille du marqueur à la quantité de glucides
                        marker_size = max(50, min(150, carbs * 1.5))
                        
                        # Ajouter un point pour marquer le repas
                        ax1.scatter(time, 60, color='#28a745', s=marker_size, alpha=0.6, zorder=5,
                                marker='^', edgecolors='white')
                    elif "Médicament" in label:
                        # Extraire le type et la dose
                        med_info = label.split(": ")[1]
                        
                        # Ajouter un point pour marquer le médicament
                        ax1.scatter(time, 220, color='#dc3545', s=80, alpha=0.6, zorder=5,
                                marker='s', edgecolors='white')
                
                # Légende personnalisée
                from matplotlib.lines import Line2D
                
                legend_elements = [
                    Line2D([0], [0], color='#0066cc', lw=2, label='Glycémie'),
                    Line2D([0], [0], color='#28a745', lw=2, label='Insuline'),
                    Line2D([0], [0], color='#28a745', linestyle='--', lw=1.5, label='Glycémie normale (100 mg/dL)'),
                    Line2D([0], [0], color='#dc3545', linestyle='--', lw=1.5, label='Seuils critiques'),
                    Line2D([0], [0], marker='^', color='w', label='Repas',
                        markerfacecolor='#28a745', markersize=10),
                    Line2D([0], [0], marker='s', color='w', label='Médicament',
                        markerfacecolor='#dc3545', markersize=8),
                ]
                ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
                
                plt.title('Évolution de la glycémie et de l\'insuline', fontsize=14, fontweight='bold')
                fig.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Afficher les statistiques de glycémie
                st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 1rem;">Statistiques de glycémie</h3>', unsafe_allow_html=True)
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{twin.metrics['glucose_mean']:.1f}</div>
                        <div class="metric-label">Moyenne (mg/dL)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stats_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{twin.metrics['glucose_min']:.1f}</div>
                        <div class="metric-label">Minimum (mg/dL)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stats_cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{twin.metrics['glucose_max']:.1f}</div>
                        <div class="metric-label">Maximum (mg/dL)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with stats_cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{twin.metrics['glucose_variability']:.1f}</div>
                        <div class="metric-label">Variabilité</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tabs[1]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Graphique du médicament avec style amélioré
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#ffffff')
                
                # Tracer les courbes avec des couleurs plus vives
                ax.plot(plot_data['time'], plot_data['drug_plasma'], color='#e63946', 
                    linewidth=2.5, label='Concentration plasmatique')
                ax.plot(plot_data['time'], plot_data['drug_tissue'], color='#457b9d', 
                    linewidth=2.5, label='Concentration tissulaire')
                
                ax.set_xlabel('Temps (heures)')
                ax.set_ylabel('Concentration du médicament')
                
                # Grille légère
                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
                
                # Annotations pour les administrations avec style moderne
                for time, label in plot_data['interventions']:
                    if "Médicament" in label:
                        # Extraire le type et la dose du médicament
                        med_info = label.split(": ")[1]
                        med_type = med_info.split(" - ")[0]
                        med_dose = med_info.split(" - ")[1].split(" ")[0]
                        
                        idx = min(int(time*100/duration), len(plot_data['drug_plasma'])-1)
                        try:
                            y_pos = plot_data['drug_plasma'][idx]
                            
                            # Trouver l'icône pour ce type de médicament
                            med_key = [k for k, v in medication_types.items() if v['name'] == med_type]
                            icon = med_icons.get(med_key[0] if med_key else '', '💊')
                            
                            # Afficher un marqueur et une annotation
                            ax.scatter(time, y_pos, color='#e63946', s=100, zorder=5, alpha=0.8,
                                    marker='o', edgecolors='white')
                            ax.annotate(f"{icon} {med_dose} mg", 
                                    xy=(time, y_pos), 
                                    xytext=(time + 0.5, y_pos + 0.5),
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#e63946", alpha=0.9))
                        except:
                            pass
                
                plt.title('Pharmacocinétique du médicament', fontsize=14, fontweight='bold')
                plt.legend(loc='upper right', framealpha=0.9)
                fig.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Exposition totale au médicament
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <div class="metric-value">{twin.metrics['drug_exposure']:.1f}</div>
                    <div class="metric-label">Exposition totale au médicament (AUC)</div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                        L'aire sous la courbe de concentration plasmatique représente l'exposition totale au médicament.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with tabs[2]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Graphique cardiovasculaire avec style amélioré
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#ffffff')
                
                ax1.set_xlabel('Temps (heures)')
                ax1.set_ylabel('Fréquence cardiaque (bpm)', color='#e63946')
                ax1.plot(plot_data['time'], plot_data['heart_rate'], color='#e63946', linewidth=2.5)
                ax1.tick_params(axis='y', labelcolor='#e63946')
                
                # Plage normale de fréquence cardiaque
                ax1.axhspan(60, 100, color='#e63946', alpha=0.1, label='Plage normale FC')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('Pression artérielle (mmHg)', color='#457b9d')
                ax2.plot(plot_data['time'], plot_data['blood_pressure'], color='#457b9d', linewidth=2.5)
                ax2.tick_params(axis='y', labelcolor='#457b9d')
                
                # Plage normale de pression artérielle
                ax2.axhspan(110, 130, color='#457b9d', alpha=0.1, label='Plage normale PA')
                
                # Grille légère
                ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
                
                # Légende
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
                
                plt.title('Paramètres cardiovasculaires', fontsize=14, fontweight='bold')
                fig.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Statistiques cardiovasculaires
                st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 1rem;">Statistiques cardiovasculaires</h3>', unsafe_allow_html=True)
                cv_cols = st.columns(4)
                with cv_cols[0]:
                    mean_hr = np.mean(plot_data['heart_rate'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{mean_hr:.1f}</div>
                        <div class="metric-label">FC moyenne (bpm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cv_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{twin.metrics['hr_variability']:.1f}</div>
                        <div class="metric-label">Variabilité FC</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cv_cols[2]:
                    mean_bp = np.mean(plot_data['blood_pressure'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{mean_bp:.1f}</div>
                        <div class="metric-label">PA moyenne (mmHg)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cv_cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{twin.metrics['bp_variability']:.1f}</div>
                        <div class="metric-label">Variabilité PA</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tabs[3]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Graphique de l'inflammation et réponse immunitaire avec style amélioré
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#ffffff')
                
                ax1.set_xlabel('Temps (heures)')
                ax1.set_ylabel('Inflammation', color='#ff6b6b')
                ax1.plot(plot_data['time'], plot_data['inflammation'], color='#ff6b6b', linewidth=2.5)
                ax1.tick_params(axis='y', labelcolor='#ff6b6b')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('Cellules immunitaires', color='#4ecdc4')
                ax2.plot(plot_data['time'], plot_data['immune_cells'], color='#4ecdc4', linewidth=2.5)
                ax2.tick_params(axis='y', labelcolor='#4ecdc4')
                
                # Grille légère
                ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
                
                plt.title('Réponse inflammatoire et immunitaire', fontsize=14, fontweight='bold')
                
                # Légende
                legend_elements = [
                    Line2D([0], [0], color='#ff6b6b', lw=2, label='Inflammation'),
                    Line2D([0], [0], color='#4ecdc4', lw=2, label='Cellules immunitaires')
                ]
                ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
                
                fig.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Charge inflammatoire
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <div class="metric-value">{twin.metrics['inflammation_burden']:.1f}</div>
                    <div class="metric-label">Charge inflammatoire totale</div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                        Représente l'exposition cumulée à l'inflammation pendant la période de simulation.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with tabs[4]:
                st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem;">Interactions médicamenteuses détectées</h3>', unsafe_allow_html=True)
                
                if len(plot_data['interactions']) > 0:
                    # Liste des interactions détectées avec style moderne
                    for time, interaction in plot_data['interactions']:
                        st.markdown(f"""
                        <div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #ffc107;">
                            <strong>⚠️ À {time:.1f} heures</strong>: {interaction}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Afficher un graphique de ligne temporelle des interactions
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.set_facecolor('#f8f9fa')
                    fig.patch.set_facecolor('#ffffff')
                    
                    # Extraire les temps des interactions
                    interaction_times = [t for t, _ in plot_data['interactions']]
                    
                    # Créer une visualisation améliorée des interactions
                    ax.eventplot(interaction_times, colors='#ffc107', linewidths=3, linelengths=0.8)
                    
                    # Ajouter des points pour plus de visibilité
                    for t in interaction_times:
                        ax.scatter(t, 0, color='#ffc107', s=80, zorder=5, alpha=0.8,
                                marker='o', edgecolors='white')
                    
                    ax.set_xlabel('Temps (heures)')
                    ax.set_title('Chronologie des interactions médicamenteuses', fontsize=12, fontweight='bold')
                    ax.set_xlim(0, duration)
                    ax.set_yticks([])
                    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; text-align: center; margin: 20px 0;">
                        <span style="font-size: 24px;">✅</span>
                        <p style="margin: 5px 0 0 0; font-weight: 500;">Aucune interaction médicamenteuse n'a été détectée pendant la simulation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Afficher une matrice d'interaction pour les médicaments utilisés
                used_med_types = list(set([med[1] for med in medications]))
                if len(used_med_types) > 1:
                    st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 1.5rem;">Matrice d\'interactions des médicaments utilisés</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    # Créer une matrice pour les médicaments utilisés
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.set_facecolor('#f8f9fa')
                    fig.patch.set_facecolor('#ffffff')
                    
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
                    
                    # Créer la heatmap avec des couleurs plus modernes
                    cmap = LinearSegmentedColormap.from_list('interaction_cmap', 
                                                        ['#ffffff', '#fffacd', '#ffa07a', '#ff6961'])
                    im = ax.imshow(interaction_matrix, cmap=cmap, vmin=0, vmax=3)
                    
                    # Ajouter étiquettes
                    med_names = [medication_types[t]['name'] for t in used_med_types]
                    ax.set_xticks(np.arange(n_meds))
                    ax.set_yticks(np.arange(n_meds))
                    ax.set_xticklabels(med_names)
                    ax.set_yticklabels(med_names)
                    
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Ajouter les valeurs dans les cellules avec un style amélioré
                    for i in range(n_meds):
                        for j in range(n_meds):
                            if interaction_matrix[i, j] > 0:
                                severity_text = {1: "Faible", 2: "Modérée", 3: "Élevée"}
                                text_color = 'black'
                                if interaction_matrix[i, j] == 3:
                                    text_color = 'white'
                                    
                                text = ax.text(j, i, severity_text[interaction_matrix[i, j]],
                                            ha="center", va="center", color=text_color,
                                            fontweight='bold')
                    
                    ax.set_title("Matrice d'interactions entre médicaments", fontsize=12, fontweight='bold')
                    fig.tight_layout()
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Légende
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin-top: 12px;">
                        <p style="font-weight: 500; margin-bottom: 8px;">Sévérité des interactions:</p>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li><strong style="color: #ff6961;">Élevée</strong>: Interaction potentiellement dangereuse</li>
                            <li><strong style="color: #ffa07a;">Modérée</strong>: Précautions nécessaires</li>
                            <li><strong style="color: #fffacd;">Faible</strong>: Surveillance conseillée</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tabs[5]:
                # Affichage des données sous forme de tableau amélioré
                st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem;">Données de simulation détaillées</h3>', unsafe_allow_html=True)
                
                results_df = twin.export_results()
                
                # Styliser le dataframe
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )
                
                # Bouton pour télécharger les résultats en CSV avec style amélioré
                buffer = BytesIO()
                results_df.to_csv(buffer, index=False)
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Télécharger les données (CSV)",
                    data=buffer,
                    file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    help="Télécharger les résultats complets de la simulation au format CSV"
                )
                
                # Résumé des paramètres de simulation en tableau
                st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 1.5rem;">Résumé des paramètres</h3>', unsafe_allow_html=True)
                
                # Créer un DataFrame plus lisible pour les paramètres
                params_dict = {
                    'Paramètre': list(twin.params.keys()),
                    'Valeur': list(twin.params.values())
                }
                params_df = pd.DataFrame(params_dict)
                
                st.dataframe(
                    params_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Bouton pour sauvegarder ce scénario pour comparaison avec style amélioré
                st.markdown('<div style="margin-top: 20px; text-align: center;">', unsafe_allow_html=True)
                
                # Bouton pour sauvegarder la simulation pour le patient actuel
                if 'current_patient' in st.session_state:
                    patient = st.session_state.current_patient
                    if st.button("💾 Sauvegarder cette simulation", type="primary"):
                        user_manager = UserManager()
                        simulation_data = {
                            'twin_data': twin.to_json(),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        success, sim_id = user_manager.save_simulation(
                            st.session_state.user_id,
                            patient['id'],
                            simulation_data
                        )
                        if success:
                            st.success(f"✅ Simulation sauvegardée avec succès pour {patient['name']}!")
                        else:
                            st.error(f"❌ Erreur lors de la sauvegarde: {sim_id}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Message pour guider l'utilisateur lorsqu'il n'y a pas encore de résultats
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; background-color: #f8f9fa; border-radius: 12px; margin-top: 20px;">
                <img src="https://cdn.pixabay.com/photo/2016/10/18/18/19/folder-1750842_960_720.png" style="width: 80px; height: 80px; margin-bottom: 20px;">
                <h3 style="color: #6c757d; font-weight: 500; margin-bottom: 15px;">Pas encore de résultats</h3>
                <p style="color: #6c757d; margin-bottom: 20px;">Configurez les paramètres du patient et les médicaments dans le panneau de gauche, puis lancez la simulation pour voir les résultats apparaître ici.</p>
                <div style="font-size: 50px; color: #dee2e6; margin-bottom: 15px;">←</div>
                <p style="color: #6c757d; font-size: 0.9rem;">Les résultats incluront des graphiques de glycémie, de concentration médicamenteuse, et des métriques de santé.</p>
            </div>
            """, unsafe_allow_html=True)


def comparison_mode():
    """Interface pour le mode de comparaison de scénarios améliorée"""
    
    st.markdown("""
    <div style="background-color: #e6f2ff; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
        <p style="margin: 0; color: #0066cc;">
            <strong>⚖️ Mode comparaison:</strong> Ce mode vous permet de comparer deux scénarios de traitement 
            différents côte à côte pour évaluer leurs impacts respectifs sur le patient.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si on a un scénario A sauvegardé (automatique depuis simple_mode)
    has_scenario_a = 'scenario_a' in st.session_state
    has_scenario_b = 'scenario_b' in st.session_state
    
    # Créer deux colonnes pour les configurations
    col1, col2 = st.columns(2)
    
    # Configuration du scénario A
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Scénario A</h2>', unsafe_allow_html=True)
        
        if has_scenario_a:
            # Afficher un badge de succès
            st.markdown("""
            <div style="background-color: #d4edda; color: #155724; border-radius: 20px; 
                    padding: 5px 15px; display: inline-block; font-weight: 500; margin-bottom: 15px;">
                ✅ Scénario chargé
            </div>
            """, unsafe_allow_html=True)
            
            # Charger les paramètres du scénario sauvegardé
            scenario_a = st.session_state.scenario_a
            twin_a = scenario_a['twin']
            
            # Afficher quand le scénario a été sauvegardé
            if 'timestamp' in scenario_a:
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 10px;">
                    Sauvegardé le: {scenario_a['timestamp']}
                </div>
                """, unsafe_allow_html=True)
            
            # Afficher les paramètres clés du patient avec style amélioré
            st.markdown("""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h4 style="color: #2c3e50; font-size: 1.1rem; margin-top: 0; margin-bottom: 10px;">📋 Paramètres du patient</h4>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                    <div class="intervention-tag">👤 {twin_a.params['age']} ans</div>
                    <div class="intervention-tag">⚧ {twin_a.params['sex']}</div>
                    <div class="intervention-tag">⚖️ {twin_a.params['weight']} kg</div>
                    <div class="intervention-tag">🩸 {twin_a.params['baseline_glucose']} mg/dL</div>
                </div>
                
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    <div class="intervention-tag">💉 Sensibilité insuline: {twin_a.params['insulin_sensitivity']}</div>
                    <div class="intervention-tag">🫀 FC: {twin_a.params['heart_rate']} bpm</div>
                    <div class="intervention-tag">🩸 PA: {twin_a.params['blood_pressure']} mmHg</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher les médicaments avec style amélioré
            st.markdown("""
            <h4 style="color: #2c3e50; font-size: 1.1rem; margin-bottom: 10px;">💊 Médicaments</h4>
            <div style="background-color: #f0f7ff; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
            """, unsafe_allow_html=True)
            
            # Afficher des icônes pour chaque type de médicament
            med_icons = get_icons()
            
            for time, med_type, dose in scenario_a['medications']:
                med_name = medication_types[med_type]['name']
                med_icon = med_icons.get(med_type, '💊')
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">{med_icon}</span>
                    <strong>{med_name}</strong> - {dose} mg à {time}h
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Bouton pour réinitialiser avec style amélioré
            if st.button("🗑️ Réinitialiser Scénario A", 
                    help="Supprimer ce scénario sauvegardé",
                    type="secondary"):
                if 'scenario_a' in st.session_state:
                    del st.session_state.scenario_a
                st.rerun()
            
            # Stocker dans la session
            st.session_state.twin_a = twin_a
            st.session_state.has_results_a = True
            
        else:
            # Message pour guider l'utilisateur avec un bouton pour rediriger vers le mode simple
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 12px;">
                <img src="https://cdn.pixabay.com/photo/2016/10/18/18/19/folder-1750842_960_720.png" style="width: 60px; height: 60px; margin-bottom: 15px;">
                <p style="color: #6c757d;">Configurez d'abord le scénario A dans l'onglet 'Mode Simple'.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⬅️ Aller au Mode Simple", type="primary"):
                # Rediriger vers le mode simple
                st.session_state.mode_tab_index = 0  # Index de l'onglet mode simple
                st.rerun()
            
            st.session_state.has_results_a = False
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Configuration du scénario B
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Scénario B</h2>', unsafe_allow_html=True)
        
        # Si on a déjà un scénario B, l'afficher
        if has_scenario_b:
            st.markdown("""
            <div style="background-color: #d4edda; color: #155724; border-radius: 20px; 
                    padding: 5px 15px; display: inline-block; font-weight: 500; margin-bottom: 15px;">
                ✅ Scénario B chargé
            </div>
            """, unsafe_allow_html=True)
            
            # Charger les paramètres du scénario B
            scenario_b = st.session_state.scenario_b
            twin_b = scenario_b['twin']
            
            # Afficher quand le scénario a été sauvegardé
            if 'timestamp' in scenario_b:
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 10px;">
                    Créé le: {scenario_b['timestamp']}
                </div>
                """, unsafe_allow_html=True)
                
            # Bouton pour réinitialiser le scénario B
            if st.button("🗑️ Réinitialiser Scénario B", help="Supprimer ce scénario"):
                if 'scenario_b' in st.session_state:
                    del st.session_state.scenario_b
                    del st.session_state.twin_b
                    st.session_state.has_results_b = False
                    st.rerun()
                
            # Stocker dans la session
            st.session_state.twin_b = twin_b
            st.session_state.has_results_b = True
            
        else:
            # Sélection du profil avec style amélioré
            st.markdown("#### 👤 Sélection du profil")
            profile_options = ["Personnalisé"] + [profile['name'] for profile in predefined_profiles.values()]
            selected_profile = st.selectbox("", profile_options, key="profile_b", 
                                        help="Choisissez un profil prédéfini ou personnalisez les paramètres")
            
            # Si on a sélectionné un profil prédéfini
            initial_params_b = {}
            if selected_profile != "Personnalisé":
                # Trouver le profil correspondant
                for profile_key, profile in predefined_profiles.items():
                    if profile['name'] == selected_profile:
                        initial_params_b = profile['params']
                        st.markdown(f"""
                        <div class="patient-info">
                            <strong>{selected_profile}</strong><br>
                            {profile['description']}
                        </div>
                        """, unsafe_allow_html=True)
                        break
            
            # Paramètres du patient pour le scénario B dans un expander modernisé
            with st.expander("📋 Paramètres du patient", expanded=False):
                # Utiliser des colonnes pour une mise en page plus compacte
                col_age_b, col_weight_b = st.columns(2)
                with col_age_b:
                    age_b = st.slider("Âge", 18, 90, initial_params_b.get('age', 50), key="age_b")
                with col_weight_b:
                    weight_b = st.slider("Poids (kg)", 40, 150, initial_params_b.get('weight', 70), key="weight_b")
                    
                sex_b = st.selectbox("Sexe", ["M", "F"], 0 if initial_params_b.get('sex', 'M') == 'M' else 1, key="sex_b")
                
                # Paramètres métaboliques
                col_glucose_b, col_insulin_b = st.columns(2)
                with col_glucose_b:
                    baseline_glucose_b = st.slider("Glycémie initiale (mg/dL)", 70, 300, 
                                                initial_params_b.get('baseline_glucose', 140), key="glucose_b")
                with col_insulin_b:
                    insulin_sensitivity_b = st.slider("Sensibilité à l'insuline", 0.1, 1.0, 
                                                    initial_params_b.get('insulin_sensitivity', 0.5), 0.1, key="insulin_b")
                
                # Fonction rénale
                renal_function_b = st.slider("Fonction rénale", 0.1, 1.0, 
                                        initial_params_b.get('renal_function', 0.9), 0.1, key="renal_b")
                
                # Fonction hépatique
                liver_function_b = st.slider("Fonction hépatique", 0.1, 1.0,
                                        initial_params_b.get('liver_function', 0.9), 0.1, key="liver_b")
                
                # Réponse immunitaire
                immune_response_b = st.slider("Réponse immunitaire", 0.1, 1.0,
                                        initial_params_b.get('immune_response', 1.0), 0.1, key="immune_b")
            
            # Configuration de la simulation B
            st.markdown("#### ⚙️ Configuration de la simulation")
            
            duration_b = st.slider("Durée (heures)", 12, 72, 24, key="duration_b",
                                help="Durée de la simulation en heures")
            
            # Configuration simplifiée des médicaments pour le scénario B avec style amélioré
            st.markdown("#### 💊 Médicaments")
            
            num_meds_b = st.number_input("Nombre de médicaments", 0, 5, 2, 1, key="num_meds_b",
                                        help="Nombre de médicaments à administrer")
            
            # Afficher des icônes pour chaque type de médicament
            med_icons = get_icons()
            
            med_types = list(medication_types.keys())
            med_names = [medication_types[t]['name'] for t in med_types]
            
            medications_b = []
            
            for i in range(num_meds_b):
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                <p style="font-weight: 500; margin-bottom: 8px;">Médicament {i+1}</p>
                """, unsafe_allow_html=True)
                
                col_time_b, col_type_b, col_dose_b = st.columns(3)
                with col_time_b:
                    med_time_b = st.number_input(f"Heure", 0.0, 24.0, 8.0 + i*4, 0.5, key=f"med_time_b_{i}")
                with col_type_b:
                    med_type_name_b = st.selectbox(f"Type", med_names, i % len(med_names), key=f"med_type_b_{i}")
                    med_type_b = med_types[med_names.index(med_type_name_b)]
                    med_icon_b = med_icons.get(med_type_b, '💊')
                with col_dose_b:
                    med_dose_b = st.number_input(f"Dose (mg)", 0.0, 50.0, 10.0, 2.5, key=f"med_dose_b_{i}")
                
                medications_b.append((med_time_b, med_type_b, med_dose_b))
                
                # Afficher icône et type de médicament sélectionné
                st.markdown(f"""
                <div style="margin-top: 5px;">
                    <span style="font-size: 1.2rem;">{med_icon_b}</span> <strong>{med_type_name_b}</strong>
                </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Configuration des repas pour le scénario B
            st.markdown("#### 🍽️ Repas")
            
            # Option pour utiliser les mêmes repas que le scénario A
            use_same_meals = True
            if has_scenario_a:
                use_same_meals = st.checkbox("Utiliser les mêmes repas que le scénario A", value=True, key="same_meals")
            
            if use_same_meals and has_scenario_a:
                meals_b = scenario_a['meals']
                # Afficher les repas
                for i, (time, carbs) in enumerate(meals_b):
                    st.markdown(f"""
                    <div style="background-color: #f0f7ff; padding: 8px 12px; border-radius: 5px; margin-bottom: 8px; display: inline-block;">
                        <span style="font-size: 1.2rem;">🍽️</span> <strong>Repas {i+1}</strong>: {carbs}g à {time}h
                    </div>
                    """, unsafe_allow_html=True)
            else:
                num_meals_b = st.number_input("Nombre de repas", 0, 5, 3, 1, key="num_meals_b")
                meals_b = []
                
                for i in range(num_meals_b):
                    meal_cols = st.columns(2)
                    with meal_cols[0]:
                        meal_time_b = st.number_input(f"Heure repas {i+1}", 0.0, 24.0, 
                                                float(7 + i*5 if i < 3 else 7 + (i-3)*5), 0.5,
                                                key=f"meal_time_b_{i}")
                    with meal_cols[1]:
                        meal_carbs_b = st.number_input(f"Glucides (g)", 0, 200, 70, 5, 
                                                    key=f"meal_carbs_b_{i}")
                    meals_b.append((meal_time_b, meal_carbs_b))
            
            # Bouton pour simuler le scénario B avec style amélioré
            if st.button("▶️ Simuler Scénario B", 
                    type="primary",
                    help="Lancer la simulation avec les paramètres configurés",
                    use_container_width=True):
                
                with st.spinner("Simulation en cours..."):
                    # Créer les paramètres du patient B
                    patient_params_b = {
                        'age': age_b,
                        'weight': weight_b,
                        'sex': sex_b,
                        'baseline_glucose': baseline_glucose_b,
                        'insulin_sensitivity': insulin_sensitivity_b,
                        'renal_function': renal_function_b,
                        'liver_function': liver_function_b if 'liver_function_b' in locals() else initial_params_b.get('liver_function', 0.9),
                        'immune_response': immune_response_b if 'immune_response_b' in locals() else initial_params_b.get('immune_response', 0.9),
                        'inflammatory_response': initial_params_b.get('inflammatory_response', 0.5),
                        'heart_rate': initial_params_b.get('heart_rate', 75),
                        'blood_pressure': initial_params_b.get('blood_pressure', 120)
                    }
                    
                    # Créer et simuler le jumeau B
                    twin_b = PatientDigitalTwin(patient_params_b)
                    twin_b.simulate(duration=duration_b, medications=medications_b, meals=meals_b)
                    
                    # Stockage dans la session
                    st.session_state.twin_b = twin_b
                    st.session_state.has_results_b = True
                    
                    # Stockage pour comparaison future
                    st.session_state.scenario_b = {
                        'twin': twin_b,
                        'params': patient_params_b,
                        'medications': medications_b,
                        'meals': meals_b,
                        'duration': duration_b,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                st.success("Simulation du scénario B terminée avec succès!")
                # Force rerun to update the interface
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Affichage des résultats comparatifs
    if hasattr(st.session_state, 'has_results_a') and st.session_state.has_results_a and hasattr(st.session_state, 'has_results_b') and st.session_state.has_results_b:
        st.markdown('<h2 class="sub-header" style="margin-top: 30px;">⚖️ Comparaison des Résultats</h2>', unsafe_allow_html=True)
        
        twin_a = st.session_state.twin_a
        twin_b = st.session_state.twin_b
        
        # Tableau comparatif des métriques principales avec style amélioré
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 0;">Métriques principales</h3>', unsafe_allow_html=True)
        
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
            ],
            'Différence': [
                f"{twin_b.metrics.get('health_score', 0) - twin_a.metrics.get('health_score', 0):.1f}",
                f"{twin_b.metrics.get('glucose_mean', 0) - twin_a.metrics.get('glucose_mean', 0):.1f}",
                f"{twin_b.metrics.get('percent_in_range', 0) - twin_a.metrics.get('percent_in_range', 0):.1f}",
                f"{twin_b.metrics.get('percent_hyperglycemia', 0) - twin_a.metrics.get('percent_hyperglycemia', 0):.1f}",
                f"{twin_b.metrics.get('percent_hypoglycemia', 0) - twin_a.metrics.get('percent_hypoglycemia', 0):.1f}",
                f"{twin_b.metrics.get('inflammation_burden', 0) - twin_a.metrics.get('inflammation_burden', 0):.1f}",
                f"{twin_b.metrics.get('drug_exposure', 0) - twin_a.metrics.get('drug_exposure', 0):.1f}"
            ]
        })
        
        # Afficher le tableau avec un style modernisé
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Métrique": st.column_config.TextColumn("Métrique"),
                "Scénario A": st.column_config.TextColumn("Scénario A"),
                "Scénario B": st.column_config.TextColumn("Scénario B"),
                "Différence": st.column_config.TextColumn("Différence (B - A)")
            }
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Graphiques comparatifs sur des onglets avec style amélioré
        st.markdown('<div class="tabs-container">', unsafe_allow_html=True)
        compare_tabs = st.tabs([
            "📊 Glycémie", 
            "💊 Médicament", 
            "🔥 Inflammation", 
            "❤️ Cardiovasculaire"
        ])
        
        with compare_tabs[0]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Comparaison des glycémies avec style amélioré
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            
            # Tracer les deux courbes de glycémie avec couleurs plus expressives
            ax.plot(twin_a.history['time'], twin_a.history['glucose'], 
                color='#4361ee', linewidth=2.5, label='Scénario A')
            ax.plot(twin_b.history['time'], twin_b.history['glucose'], 
                color='#e63946', linewidth=2.5, label='Scénario B')
            
            # Lignes de référence
            ax.axhline(y=100, color='#28a745', linestyle='--', alpha=0.5, linewidth=1.5)  # Glycémie normale
            ax.axhline(y=180, color='#dc3545', linestyle='--', alpha=0.5, linewidth=1.5)  # Seuil hyperglycémie
            ax.axhline(y=70, color='#dc3545', linestyle='--', alpha=0.5, linewidth=1.5)   # Seuil hypoglycémie
            
            # Zone cible avec transparence
            ax.fill_between([0, max(twin_a.history['time'][-1], twin_b.history['time'][-1])], 
                        70, 180, alpha=0.1, color='#28a745')
            
            # Grille légère pour meilleure lisibilité
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Glycémie (mg/dL)')
            ax.set_title('Comparaison des profils glycémiques', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', framealpha=0.9)
            
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calcul des différences
            glucose_diff = twin_b.metrics['glucose_mean'] - twin_a.metrics['glucose_mean']
            in_range_diff = twin_b.metrics['percent_in_range'] - twin_a.metrics['percent_in_range']
            glu_var_diff = twin_b.metrics['glucose_variability'] - twin_a.metrics['glucose_variability']
            
            # Afficher les différences significatives avec style modernisé
            st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 1rem; margin-bottom: 1rem;">Différences clés</h3>', unsafe_allow_html=True)
            
            diff_cols = st.columns(3)
            
            with diff_cols[0]:
                # Déterminer la couleur en fonction de la direction du changement
                # Pour la glycémie, une diminution est généralement positive
                diff_color = "#28a745" if glucose_diff < 0 else "#dc3545" if glucose_diff > 0 else "#6c757d"
                diff_icon = "⬇️" if glucose_diff < 0 else "⬆️" if glucose_diff > 0 else "↔️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence glycémie moyenne</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {glucose_diff:.1f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">mg/dL</div>
                </div>
                """, unsafe_allow_html=True)
            
            with diff_cols[1]:
                # Pour le temps en cible, une augmentation est positive
                diff_color = "#28a745" if in_range_diff > 0 else "#dc3545" if in_range_diff < 0 else "#6c757d"
                diff_icon = "⬆️" if in_range_diff > 0 else "⬇️" if in_range_diff < 0 else "↔️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence temps en cible</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {in_range_diff:.1f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">pourcentage</div>
                </div>
                """, unsafe_allow_html=True)
            
            with diff_cols[2]:
                # Pour la variabilité, une diminution est positive
                diff_color = "#28a745" if glu_var_diff < 0 else "#dc3545" if glu_var_diff > 0 else "#6c757d"
                diff_icon = "⬇️" if glu_var_diff < 0 else "⬆️" if glu_var_diff > 0 else "↔️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence variabilité</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {glu_var_diff:.1f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">unités</div>
                </div>
                """, unsafe_allow_html=True)
        
        with compare_tabs[1]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Comparaison de la pharmacocinétique avec style amélioré
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            
            # Tracer les courbes de concentration du médicament
            ax.plot(twin_a.history['time'], twin_a.history['drug_plasma'], 
                color='#4361ee', linewidth=2.5, label='Plasma A')
            ax.plot(twin_a.history['time'], twin_a.history['drug_tissue'], 
                color='#4361ee', linestyle='--', linewidth=1.8, label='Tissus A')
            ax.plot(twin_b.history['time'], twin_b.history['drug_plasma'], 
                color='#e63946', linewidth=2.5, label='Plasma B')
            ax.plot(twin_b.history['time'], twin_b.history['drug_tissue'], 
                color='#e63946', linestyle='--', linewidth=1.8, label='Tissus B')
            
            # Grille légère
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Concentration du médicament')
            ax.set_title('Comparaison des profils pharmacocinétiques', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', framealpha=0.9)
            
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Exposition au médicament
            drug_exp_diff = twin_b.metrics['drug_exposure'] - twin_a.metrics['drug_exposure']
            diff_percent = (drug_exp_diff / twin_a.metrics['drug_exposure']) * 100 if twin_a.metrics['drug_exposure'] > 0 else 0
            
            # Déterminer le style en fonction de la différence
            diff_color = "#6c757d"  # Neutre par défaut
            diff_icon = "↔️"
            diff_text = "Différence non significative dans l'exposition médicamenteuse."
            diff_style = "background-color: #f8f9fa; border-color: #6c757d;"
            
            if abs(diff_percent) > 20:  # Différence significative > 20%
                if drug_exp_diff > 0:
                    diff_color = "#dc3545"  # Rouge pour une exposition plus élevée
                    diff_icon = "⬆️"
                    diff_text = "Le scénario B présente une exposition médicamenteuse significativement plus élevée, ce qui pourrait augmenter le risque d'effets indésirables."
                    diff_style = "background-color: #f8d7da; border-color: #dc3545;"
                else:
                    diff_color = "#0dcaf0"  # Bleu pour une exposition plus basse
                    diff_icon = "⬇️"
                    diff_text = "Le scénario B présente une exposition médicamenteuse significativement plus basse, ce qui pourrait réduire l'efficacité du traitement."
                    diff_style = "background-color: #d1ecf1; border-color: #0dcaf0;"
            
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence d'exposition au médicament</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {drug_exp_diff:.1f} ({diff_percent:.1f}%)</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">Unités d'aire sous la courbe</div>
                </div>
                
                <div style="{diff_style} padding: 15px; border-radius: 8px; border-left: 5px solid {diff_color}; margin-top: 15px;">
                    <strong style="color: {diff_color};">{diff_icon} Interprétation:</strong> {diff_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with compare_tabs[2]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Comparaison de l'inflammation avec style amélioré
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            
            # Tracer les courbes d'inflammation
            ax.plot(twin_a.history['time'], twin_a.history['inflammation'], 
                color='#ff6b6b', linewidth=2.5, label='Inflammation A')
            ax.plot(twin_a.history['time'], twin_a.history['immune_cells'], 
                color='#4ecdc4', linewidth=2.5, label='Immunité A')
            ax.plot(twin_b.history['time'], twin_b.history['inflammation'], 
                color='#ff9e7d', linewidth=2.5, label='Inflammation B')
            ax.plot(twin_b.history['time'], twin_b.history['immune_cells'], 
                color='#83e8e1', linewidth=2.5, label='Immunité B')
            
            # Grille légère
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Niveau')
            ax.set_title('Comparaison des réponses inflammatoires et immunitaires', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', framealpha=0.9)
            
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Différence de charge inflammatoire
            infl_diff = twin_b.metrics['inflammation_burden'] - twin_a.metrics['inflammation_burden']
            infl_diff_percent = (infl_diff / twin_a.metrics['inflammation_burden']) * 100 if twin_a.metrics['inflammation_burden'] > 0 else 0
            
            # Déterminer le style en fonction de la différence
            diff_color = "#6c757d"  # Neutre par défaut
            diff_icon = "↔️"
            diff_text = "Différence non significative dans la charge inflammatoire."
            diff_style = "background-color: #f8f9fa; border-color: #6c757d;"
            
            if abs(infl_diff_percent) > 15:  # Différence significative > 15%
                if infl_diff < 0:
                    diff_color = "#28a745"  # Vert pour une inflammation réduite
                    diff_icon = "⬇️"
                    diff_text = "Le scénario B présente une réduction significative de la charge inflammatoire, ce qui est généralement bénéfique."
                    diff_style = "background-color: #d4edda; border-color: #28a745;"
                else:
                    diff_color = "#dc3545"  # Rouge pour une inflammation accrue
                    diff_icon = "⬆️"
                    diff_text = "Le scénario B présente une augmentation significative de la charge inflammatoire, ce qui pourrait être préoccupant."
                    diff_style = "background-color: #f8d7da; border-color: #dc3545;"
            
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence de charge inflammatoire</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {infl_diff:.1f} ({infl_diff_percent:.1f}%)</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">Unités</div>
                </div>
                
                <div style="{diff_style} padding: 15px; border-radius: 8px; border-left: 5px solid {diff_color}; margin-top: 15px;">
                    <strong style="color: {diff_color};">{diff_icon} Interprétation:</strong> {diff_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with compare_tabs[3]:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Comparaison cardiovasculaire avec style amélioré
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.set_facecolor('#f8f9fa')
            ax2.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            
            # Fréquence cardiaque
            ax1.plot(twin_a.history['time'], twin_a.history['heart_rate'], 
                    color='#4361ee', linewidth=2.5, label='Scénario A')
            ax1.plot(twin_b.history['time'], twin_b.history['heart_rate'], 
                    color='#e63946', linewidth=2.5, label='Scénario B')
            ax1.set_ylabel('Fréquence cardiaque (bpm)')
            ax1.set_title('Comparaison des paramètres cardiovasculaires', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', framealpha=0.9)
            
            # Plage normale de fréquence cardiaque
            ax1.axhspan(60, 100, color='#6c757d', alpha=0.1)
            
            # Grille légère
            ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Pression artérielle
            ax2.plot(twin_a.history['time'], twin_a.history['blood_pressure'], 
                    color='#4361ee', linewidth=2.5, label='Scénario A')
            ax2.plot(twin_b.history['time'], twin_b.history['blood_pressure'], 
                    color='#e63946', linewidth=2.5, label='Scénario B')
            ax2.set_xlabel('Temps (heures)')
            ax2.set_ylabel('Pression artérielle (mmHg)')
            ax2.legend(loc='upper right', framealpha=0.9)
            
            # Plage normale de pression artérielle
            ax2.axhspan(110, 130, color='#6c757d', alpha=0.1)
            
            # Grille légère
            ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Métriques cardiovasculaires
            hr_diff = np.mean(twin_b.history['heart_rate']) - np.mean(twin_a.history['heart_rate'])
            bp_diff = np.mean(twin_b.history['blood_pressure']) - np.mean(twin_a.history['blood_pressure'])
            
            cv_cols = st.columns(2)
            with cv_cols[0]:
                # Déterminer couleur en fonction de la différence
                diff_color = "#6c757d"  # Neutre par défaut
                diff_icon = "↔️"
                
                if abs(hr_diff) > 10:  # Différence significative > 10 bpm
                    if hr_diff < 0:
                        diff_color = "#28a745"  # Vert pour FC réduite (généralement positif)
                        diff_icon = "⬇️"
                    else:
                        diff_color = "#ffc107"  # Jaune pour FC augmentée (à surveiller)
                        diff_icon = "⬆️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence FC moyenne</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {hr_diff:.1f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">bpm</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cv_cols[1]:
                # Déterminer couleur en fonction de la différence
                diff_color = "#6c757d"  # Neutre par défaut
                diff_icon = "↔️"
                
                if abs(bp_diff) > 10:  # Différence significative > 10 mmHg
                    if bp_diff < 0:
                        diff_color = "#28a745"  # Vert pour PA réduite (généralement positif)
                        diff_icon = "⬇️"
                    else:
                        diff_color = "#ffc107"  # Jaune pour PA augmentée (à surveiller)
                        diff_icon = "⬆️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6c757d;">Différence PA moyenne</div>
                    <div class="metric-value" style="color: {diff_color};">{diff_icon} {bp_diff:.1f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">mmHg</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fermer le conteneur d'onglets
        
        # Conclusion et recommandations avec style amélioré
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header" style="margin-top: 0;">🔍 Analyse globale et recommandations</h2>', unsafe_allow_html=True)
        
        # Comparer les scores de santé pour déterminer le meilleur scénario
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
        
        # Transformer en DataFrame et l'afficher avec un style moderne
        pros_cons_df = pd.DataFrame(pros_cons)
        
        # Utiliser un dataframe stylé
        st.dataframe(
            pros_cons_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Critère": st.column_config.TextColumn("Critère d'évaluation"),
                "Avantage": st.column_config.TextColumn("Avantage"),
                "Inconvénient": st.column_config.TextColumn("Inconvénient"),
                "Recommandation": st.column_config.TextColumn("Recommandation clinique")
            }
        )
        
        # Recommendation finale avec un style visuel adapté à la conclusion
        st.markdown('<h3 style="color: #2c3e50; font-size: 1.3rem; margin-top: 1.5rem;">Recommandation finale</h3>', unsafe_allow_html=True)
        
        if health_diff > 5:
            st.markdown(f"""
            <div style="background-color: #d4edda; border-radius: 8px; padding: 15px; border-left: 5px solid #28a745;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 2rem; margin-right: 15px;">✅</span>
                    <div>
                        <strong style="font-size: 1.1rem;">Le scénario B est recommandé</strong>
                        <p style="margin: 5px 0 0 0;">Score de santé supérieur de {health_diff:.1f} points par rapport au scénario A.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif health_diff < -5:
            st.markdown(f"""
            <div style="background-color: #d4edda; border-radius: 8px; padding: 15px; border-left: 5px solid #28a745;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 2rem; margin-right: 15px;">✅</span>
                    <div>
                        <strong style="font-size: 1.1rem;">Le scénario A est recommandé</strong>
                        <p style="margin: 5px 0 0 0;">Score de santé supérieur de {-health_diff:.1f} points par rapport au scénario B.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #e2e3e5; border-radius: 8px; padding: 15px; border-left: 5px solid #6c757d;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 2rem; margin-right: 15px;">⚖️</span>
                    <div>
                        <strong style="font-size: 1.1rem;">Les deux scénarios présentent des résultats similaires</strong>
                        <p style="margin: 5px 0 0 0;">La différence de score de santé est de seulement {abs(health_diff):.1f} points. Le choix peut dépendre d'autres facteurs spécifiques au patient.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Fermer la carte de conclusion
        
        # Bouton pour sauvegarder la comparaison
        if 'current_patient' in st.session_state:
            patient = st.session_state.current_patient
            if st.button("💾 Sauvegarder cette comparaison", type="primary"):
                user_manager = UserManager()
                comparison_data = {
                    'twin_a_data': twin_a.to_json(),
                    'twin_b_data': twin_b.to_json(),
                    'comparison_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'health_diff': health_diff,
                    'recommendation': "Scénario B" if health_diff > 5 else ("Scénario A" if health_diff < -5 else "Indéterminé")
                }
                success, comp_id = user_manager.save_simulation(
                    st.session_state.user_id,
                    patient['id'],
                    comparison_data
                )
                if success:
                    st.success(f"✅ Comparaison sauvegardée avec succès pour {patient['name']}!")
                else:
                    st.error(f"❌ Erreur lors de la sauvegarde: {comp_id}")
    
    elif hasattr(st.session_state, 'has_results_a') and st.session_state.has_results_a:
        # Message guidant l'utilisateur quand seul le scénario A est disponible
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background-color: #f8f9fa; border-radius: 12px; margin-top: 30px;">
            <img src="https://cdn.pixabay.com/photo/2017/01/31/23/42/balance-2028258_960_720.png" style="width: 80px; height: 80px; margin-bottom: 20px;">
            <h3 style="color: #6c757d; font-weight: 500; margin-bottom: 15px;">Scénario B nécessaire pour la comparaison</h3>
            <p style="color: #6c757d; margin-bottom: 20px;">Le scénario A est prêt! Maintenant, configurez et simulez le scénario B pour voir une comparaison complète entre les deux approches.</p>
            <div style="font-size: 50px; color: #dee2e6; margin-bottom: 15px;">→</div>
            <p style="color: #6c757d; font-size: 0.9rem;">La comparaison vous montrera les différences en termes de contrôle glycémique, d'inflammation et d'autres paramètres importants.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Message quand aucun scénario n'est disponible
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background-color: #f8f9fa; border-radius: 12px; margin-top: 30px;">
            <img src="https://cdn.pixabay.com/photo/2016/10/18/18/19/folder-1750842_960_720.png" style="width: 80px; height: 80px; margin-bottom: 20px;">
            <h3 style="color: #6c757d; font-weight: 500; margin-bottom: 15px;">Configuration nécessaire</h3>
            <p style="color: #6c757d; margin-bottom: 20px;">Pour utiliser le mode comparaison, commencez par configurer et sauvegarder le scénario A dans l'onglet "Mode Simple".</p>
            <p style="color: #6c757d; font-size: 0.9rem;">Une fois le scénario A sauvegardé, vous pourrez configurer le scénario B et comparer les résultats.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⬅️ Aller au Mode Simple", type="primary"):
            # Rediriger vers le mode simple
            st.session_state.mode_tab_index = 0  # Index de l'onglet mode simple
            st.rerun()


def anatomical_visualization_tab(twin=None):
    """
    Onglet de visualisation anatomique des effets sur différents organes
    Accepte optionnellement un jumeau numérique pour visualiser ses données
    """
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Visualisation Anatomique</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e6f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <p style="margin: 0; color: #0066cc; font-size: 16px;">
            <strong>🧠 Visualisation interactive:</strong> Cette section vous permet de visualiser l'impact des traitements sur les différents systèmes et organes du patient.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Système/Organe à visualiser
    systems = {
        "cardio": "Système cardiovasculaire",
        "pancreas": "Pancréas et métabolisme",
        "renal": "Système rénal",
        "liver": "Foie et système hépatique",
        "immune": "Système immunitaire"
    }
    
    # Sélectionner le système à visualiser
    selected_system = st.selectbox(
        "Sélectionnez un système à visualiser",
        options=list(systems.keys()),
        format_func=lambda x: systems[x]
    )
    
    # Afficher un message si aucun jumeau numérique n'est disponible
    if twin is None:
        st.info("Aucune simulation active. Effectuez d'abord une simulation pour visualiser les effets sur les organes.")
        
        # Utiliser un placeholder pour montrer le type de visualisations disponibles
        st.markdown("<h3 style='color: #2c3e50;'>Aperçu des visualisations disponibles</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center;'>
                <img src="https://cdn.pixabay.com/photo/2013/07/12/17/22/heart-152377_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Système cardiovasculaire</h4>
                <p>Visualisez les impacts sur le cœur et les vaisseaux sanguins</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px;'>
                <img src="https://cdn.pixabay.com/photo/2017/01/31/22/32/kidneys-2027366_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Système rénal</h4>
                <p>Examinez les effets sur les reins et la filtration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center;'>
                <img src="https://cdn.pixabay.com/photo/2021/10/07/09/27/pancreas-6688196_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Pancréas et métabolisme</h4>
                <p>Visualisez la production d'insuline et le métabolisme du glucose</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px;'>
                <img src="https://cdn.pixabay.com/photo/2021/03/02/22/20/white-blood-cell-6064098_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Système immunitaire</h4>
                <p>Observez les réponses inflammatoires et immunitaires</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton pour aller à la simulation
        if st.button("▶️ Aller à la simulation", type="primary"):
            st.session_state.mode_tab_index = 0  # Index de l'onglet simulation
            st.rerun()
        
        return
    
    # Si un jumeau est disponible, afficher les visualisations réelles
    st.markdown(f"<h2 style='color: #2c3e50;'>Visualisation du {systems[selected_system]}</h2>", unsafe_allow_html=True)
    
    # Préparer les données de la simulation
    time_data = twin.history['time']
    
    # Définir les graphiques selon le système sélectionné
    if selected_system == "cardio":
        # Système cardiovasculaire
        st.markdown("<h3 style='color: #2c3e50;'>Impact sur le système cardiovasculaire</h3>", unsafe_allow_html=True)
        
        # Créer une visualisation de base du cœur et de ses paramètres
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique du rythme cardiaque
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(time_data, twin.history['heart_rate'], color='#e63946', linewidth=2.5)
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Fréquence cardiaque (bpm)')
            ax.set_title('Évolution du rythme cardiaque')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Zone de rythme normal
            ax.axhspan(60, 100, alpha=0.2, color='green', label='Zone normale')
            
            # Annotations pour les médicaments
            for time, label in twin.history['interventions']:
                if "Médicament" in label and "beta_blocker" in label:
                    ax.axvline(x=time, color='blue', linestyle='--', alpha=0.5)
                    ax.annotate('β-bloquant', xy=(time, max(twin.history['heart_rate'])),
                            xytext=(time, max(twin.history['heart_rate']) + 5),
                            arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                            horizontalalignment='center')
            
            st.pyplot(fig)
            
            # Metrics
            hr_mean = np.mean(twin.history['heart_rate'])
            hr_var = np.std(twin.history['heart_rate'])
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="FC moyenne", 
                    value=f"{hr_mean:.1f} bpm",
                    delta=f"{hr_mean - 75:.1f}" if hr_mean != 75 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Variabilité FC", 
                    value=f"{hr_var:.1f}",
                    delta=f"{hr_var - 5:.1f}" if hr_var != 5 else None,
                    delta_color="inverse"
                )
        
        with col2:
            # Graphique de la pression artérielle
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(time_data, twin.history['blood_pressure'], color='#457b9d', linewidth=2.5)
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Pression artérielle (mmHg)')
            ax.set_title('Évolution de la pression artérielle')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Zone de pression normale
            ax.axhspan(110, 130, alpha=0.2, color='green', label='Zone normale')
            
            # Annotations pour les médicaments
            for time, label in twin.history['interventions']:
                if "Médicament" in label and ("vasodilator" in label or "beta_blocker" in label):
                    ax.axvline(x=time, color='purple' if "vasodilator" in label else 'blue', 
                             linestyle='--', alpha=0.5)
                    med_type = "Vasodilatateur" if "vasodilator" in label else "β-bloquant"
                    ax.annotate(med_type, xy=(time, min(twin.history['blood_pressure'])),
                             xytext=(time, min(twin.history['blood_pressure']) - 5),
                             arrowprops=dict(facecolor='purple' if "vasodilator" in label else 'blue', 
                                             shrink=0.05, width=1.5, headwidth=8),
                             horizontalalignment='center')
            
            st.pyplot(fig)
            
            # Metrics
            bp_mean = np.mean(twin.history['blood_pressure'])
            bp_var = np.std(twin.history['blood_pressure'])
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="PA moyenne", 
                    value=f"{bp_mean:.1f} mmHg",
                    delta=f"{bp_mean - 120:.1f}" if bp_mean != 120 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Variabilité PA", 
                    value=f"{bp_var:.1f}",
                    delta=f"{bp_var - 8:.1f}" if bp_var != 8 else None,
                    delta_color="inverse"
                )
        
        # Visualisation anatomique schématique
        st.markdown("<h3 style='color: #2c3e50;'>Schéma interactif du cœur</h3>", unsafe_allow_html=True)
        
        # Créer une visualisation SVG simple du cœur
        heart_impact = calculate_organ_impact(twin, "heart")
        heart_color = get_impact_color(heart_impact)
        
        heart_svg = f"""
        <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
            
            <!-- Heart outline -->
            <path d="M300,120 C350,80 450,80 450,180 C450,300 300,380 300,380 C300,380 150,300 150,180 C150,80 250,80 300,120 Z" 
                fill="{heart_color}" stroke="#333" stroke-width="2" />
            
            <!-- Label -->
            <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                Impact sur le cœur: {heart_impact:.1f}/10
            </text>
            
            <!-- Aorte -->
            <path d="M300,120 C300,100 280,80 250,80 Q220,80 220,50" 
                fill="none" stroke="#cc0000" stroke-width="10" />
            
            <!-- Artère pulmonaire -->
            <path d="M300,120 C300,100 320,80 350,80 Q380,80 380,50" 
                fill="none" stroke="#0044cc" stroke-width="10" />
            
            <!-- Veines pulmonaires -->
            <path d="M260,160 C220,160 200,120 180,120" 
                fill="none" stroke="#0066cc" stroke-width="8" />
            <path d="M340,160 C380,160 400,120 420,120" 
                fill="none" stroke="#0066cc" stroke-width="8" />
            
            <!-- Veine cave -->
            <path d="M300,360 C300,390 270,410 230,410" 
                fill="none" stroke="#0044cc" stroke-width="12" />
        </svg>
        """
        
        st.markdown(heart_svg, unsafe_allow_html=True)
    
    elif selected_system == "pancreas":
        # Système pancréatique et métabolisme
        st.markdown("<h3 style='color: #2c3e50;'>Métabolisme du glucose et fonction pancréatique</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique glucose-insuline
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            # Glucose
            ax1.set_xlabel('Temps (heures)')
            ax1.set_ylabel('Glycémie (mg/dL)', color='#0066cc')
            ax1.plot(time_data, twin.history['glucose'], color='#0066cc', linewidth=2.5)
            ax1.tick_params(axis='y', labelcolor='#0066cc')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Zone cible
            ax1.axhspan(70, 180, alpha=0.1, color='green', label='Zone cible')
            ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7)
            
            # Insuline sur l'axe Y secondaire
            ax2 = ax1.twinx()
            ax2.set_ylabel('Insuline (mU/L)', color='#28a745')
            ax2.plot(time_data, twin.history['insulin'], color='#28a745', linewidth=2)
            ax2.tick_params(axis='y', labelcolor='#28a745')
            
            # Annotations pour les repas
            for time, label in twin.history['interventions']:
                if "Repas" in label:
                    # Extraire la quantité de glucides
                    carbs = int(label.split(": ")[1].split(" ")[0])
                    marker_size = max(50, min(150, carbs * 1.5))
                    
                    # Marquer les repas
                    ax1.scatter(time, 50, color='#f4a261', s=marker_size, alpha=0.7, zorder=5,
                              marker='^', edgecolors='white')
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Metrics
            glucose_mean = twin.metrics.get('glucose_mean', 0)
            in_range = twin.metrics.get('percent_in_range', 0)
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Glycémie moyenne", 
                    value=f"{glucose_mean:.1f} mg/dL",
                    delta=f"{glucose_mean - 100:.1f}" if glucose_mean != 100 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Temps en cible", 
                    value=f"{in_range:.1f}%",
                    delta=f"{in_range - 75:.1f}" if in_range != 75 else None,
                    delta_color="normal"
                )
        
        with col2:
            # Visualisation de l'utilisation du glucose
            # Créons un graphique montrant l'utilisation du glucose par les tissus
            
            # Impact des médicaments antidiabétiques
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Nous allons estimer l'absorption du glucose en fonction des données
            # Utilisons la variation de glycémie après les repas comme indicateur
            glucose_absorption = []
            baseline = twin.params['baseline_glucose']
            
            for i in range(1, len(time_data)):
                # Si la glycémie augmente, c'est l'apport des repas
                # Si elle diminue, c'est l'effet de l'insuline et des médicaments
                if twin.history['glucose'][i] > twin.history['glucose'][i-1]:
                    absorption = 0
                else:
                    # Calculer l'absorption relative
                    absorption = (twin.history['glucose'][i-1] - twin.history['glucose'][i]) * twin.history['insulin'][i] / 100
                    
                glucose_absorption.append(max(0, absorption))
            
            # Ajouter une valeur initiale
            glucose_absorption.insert(0, 0)
            
            # Tracer l'absorption du glucose
            ax.plot(time_data, glucose_absorption, color='#9c6644', linewidth=2.5, label="Absorption du glucose")
            
            # Tracer l'insuline active pour montrer sa corrélation
            insulin_active = np.array(twin.history['insulin']) * np.array(twin.history['drug_tissue']) / 20
            ax.plot(time_data, insulin_active, color='#28a745', linestyle='--', alpha=0.7, label="Insuline active")
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Utilisation relative du glucose')
            ax.set_title('Absorption et utilisation du glucose')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)
            
            # Métriques calculées
            insulin_effect = np.mean(insulin_active) * twin.params['insulin_sensitivity']
            drug_effect = np.mean(twin.history['drug_tissue']) * 0.5
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Effet de l'insuline", 
                    value=f"{insulin_effect:.2f}",
                    delta=f"{insulin_effect - 0.4:.2f}" if insulin_effect != 0.4 else None,
                    delta_color="normal"
                )
            with metric_cols[1]:
                st.metric(
                    label="Effet médicamenteux", 
                    value=f"{drug_effect:.2f}",
                    delta=f"{drug_effect - 0.3:.2f}" if drug_effect != 0.3 else None,
                    delta_color="normal"
                )
        
        # Visualisation schématique du pancréas et du métabolisme du glucose
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation du pancréas et du métabolisme</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le pancréas
        pancreas_impact = calculate_organ_impact(twin, "pancreas")
        pancreas_color = get_impact_color(pancreas_impact)
        
        # Schéma SVG du pancréas et du métabolisme du glucose
        pancreas_svg = f"""
        <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
            
            <!-- Estomac -->
            <ellipse cx="200" cy="150" rx="70" ry="50" fill="#f4a261" stroke="#333" stroke-width="2" />
            <text x="200" y="155" font-family="Arial" font-size="14" text-anchor="middle">Estomac</text>
            
            <!-- Pancréas -->
            <path d="M250,200 C300,180 350,190 400,200 C420,205 430,220 420,240 C400,270 350,280 300,260 C270,250 240,220 250,200 Z" 
                fill="{pancreas_color}" stroke="#333" stroke-width="2" />
            <text x="340" y="230" font-family="Arial" font-size="14" text-anchor="middle">Pancréas</text>
            
            <!-- Îlots de Langerhans -->
            <circle cx="320" cy="220" r="8" fill="#28a745" stroke="#333" stroke-width="1" />
            <circle cx="350" cy="230" r="8" fill="#28a745" stroke="#333" stroke-width="1" />
            <circle cx="380" cy="225" r="8" fill="#28a745" stroke="#333" stroke-width="1" />
            
            <!-- Intestin -->
            <path d="M200,200 C180,220 190,240 170,260 C150,280 160,300 180,310 C200,320 220,310 240,320 C260,330 290,320 310,330 C330,340 360,330 380,340" 
                fill="none" stroke="#cc6b49" stroke-width="15" />
            
            <!-- Foie -->
            <path d="M100,230 C150,200 200,220 230,270 C210,310 150,320 100,290 C80,270 80,250 100,230 Z" 
                fill="#a55233" stroke="#333" stroke-width="2" />
            <text x="150" y="260" font-family="Arial" font-size="14" text-anchor="middle">Foie</text>
            
            <!-- Cellules musculaires -->
            <rect x="450" y="300" width="100" height="60" rx="10" ry="10" fill="#d8bfd8" stroke="#333" stroke-width="2" />
            <text x="500" y="330" font-family="Arial" font-size="14" text-anchor="middle">Muscles</text>
            
            <!-- Cellules adipeuses -->
            <circle cx="480" cy="150" r="50" fill="#ffef99" stroke="#333" stroke-width="2" />
            <text x="480" y="155" font-family="Arial" font-size="14" text-anchor="middle">Tissu adipeux</text>
            
            <!-- Glucose sanguin -->
            <circle cx="300" cy="150" r="15" fill="#0066cc" stroke="#333" stroke-width="1" />
            <text x="300" y="155" font-family="Arial" font-size="10" text-anchor="middle" fill="white">Glucose</text>
            
            <!-- Insuline -->
            <circle cx="350" cy="180" r="10" fill="#28a745" stroke="#333" stroke-width="1" />
            <text x="350" y="183" font-family="Arial" font-size="8" text-anchor="middle" fill="white">Insuline</text>
            
            <!-- Flèches de circulation -->
            <!-- Estomac -> sang -->
            <path d="M240,130 Q270,100 290,140" stroke="#f4a261" stroke-width="3" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Pancréas -> sang (insuline) -->
            <path d="M330,200 Q320,170 350,170" stroke="#28a745" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Sang -> muscles (glucose) -->
            <path d="M320,160 Q380,200 450,320" stroke="#0066cc" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Sang -> tissu adipeux (glucose) -->
            <path d="M320,140 Q350,110 430,150" stroke="#0066cc" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Sang -> foie (glucose) -->
            <path d="M280,160 Q250,200 200,240" stroke="#0066cc" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Définition de la flèche -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs>
            
            <!-- Légende -->
            <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                Impact sur le pancréas: {pancreas_impact:.1f}/10
            </text>
        </svg>
        """
        
        st.markdown(pancreas_svg, unsafe_allow_html=True)
        
        # Informations sur l'impact des médicaments
        med_cols = st.columns(2)
        with med_cols[0]:
            st.markdown("""
            <div style="background-color: #f0f7ff; border-radius: 8px; padding: 15px;">
                <h4 style="margin-top: 0; color: #0066cc;">Impact des médicaments antidiabétiques</h4>
                <p>Les médicaments antidiabétiques agissent en:</p>
                <ul>
                    <li>Augmentant la sensibilité à l'insuline</li>
                    <li>Réduisant la production hépatique de glucose</li>
                    <li>Ralentissant l'absorption intestinale de glucose</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with med_cols[1]:
            # Calculer l'efficacité médicamenteuse
            if hasattr(twin, 'medications') and twin.medications:
                antidiabetic_meds = [med for med in twin.medications if med[1] == 'antidiabetic']
                if antidiabetic_meds:
                    efficacy = min(10, max(0, 10 - abs(glucose_mean - 110) / 10))
                    efficacy_color = "green" if efficacy > 7 else ("orange" if efficacy > 4 else "red")
                    
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; border-radius: 8px; padding: 15px;">
                        <h4 style="margin-top: 0; color: #0066cc;">Efficacité du traitement</h4>
                        <div style="text-align: center; padding: 10px;">
                            <div style="font-size: 24px; font-weight: bold; color: {efficacy_color};">{efficacy:.1f}/10</div>
                            <div style="font-size: 14px; color: #666;">Score d'efficacité du traitement</div>
                        </div>
                        <p>Le traitement {"est efficace" if efficacy > 7 else ("a une efficacité modérée" if efficacy > 4 else "n'est pas optimal")} pour maintenir la glycémie dans les valeurs cibles.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Aucun médicament antidiabétique n'a été administré dans cette simulation.")
            else:
                st.info("Aucun médicament n'a été administré dans cette simulation.")
    
    elif selected_system == "renal":
        # Système rénal
        st.markdown("<h3 style='color: #2c3e50;'>Fonction rénale et élimination des médicaments</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique de concentration du médicament
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(time_data, twin.history['drug_plasma'], color='#e63946', 
                   linewidth=2.5, label='Concentration plasmatique')
            
            # Calculer l'élimination rénale
            renal_elimination = []
            for i in range(len(time_data)):
                # L'élimination rénale est proportionnelle à la concentration plasmatique
                # et à la fonction rénale
                elimination = twin.history['drug_plasma'][i] * twin.params['renal_function'] * 0.02
                renal_elimination.append(elimination)
            
            ax.plot(time_data, renal_elimination, color='#457b9d', 
                   linewidth=2, label='Élimination rénale')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Concentration')
            ax.set_title('Élimination rénale des médicaments')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques rénales
            total_elimination = np.trapz(renal_elimination, time_data)
            drug_exposure = twin.metrics.get('drug_exposure', 0)
            elimination_percent = (total_elimination / max(drug_exposure, 0.001)) * 100
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Fonction rénale", 
                    value=f"{twin.params['renal_function']:.2f}",
                    delta=None
                )
            with metric_cols[1]:
                st.metric(
                    label="Élimination rénale", 
                    value=f"{elimination_percent:.1f}%",
                    delta=f"{elimination_percent - 50:.1f}" if elimination_percent != 50 else None,
                    delta_color="normal"
                )
        
        with col2:
            # Graphique de la filtration glomérulaire estimée
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Calculer la filtration glomérulaire en fonction de la fonction rénale
            # et des autres paramètres
            base_gfr = 90 * twin.params['renal_function']  # mL/min/1.73m2
            
            # La filtration est affectée par la pression artérielle et l'inflammation
            gfr = []
            for i in range(len(time_data)):
                # Ajustement par la pression artérielle (haute pression = diminution de la GFR)
                bp_effect = 1 - max(0, min(0.3, (twin.history['blood_pressure'][i] - 120) / 200))
                
                # Ajustement par l'inflammation (inflammation = diminution de la GFR)
                inflam_effect = 1 - max(0, min(0.3, twin.history['inflammation'][i] / 100))
                
                # GFR calculée
                current_gfr = base_gfr * bp_effect * inflam_effect
                gfr.append(current_gfr)
            
            ax.plot(time_data, gfr, color='#4ecdc4', linewidth=2.5)
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('DFG estimé (mL/min/1.73m²)')
            ax.set_title('Débit de Filtration Glomérulaire Estimé')
            
            # Zones de classification de la fonction rénale
            ax.axhspan(90, 120, alpha=0.1, color='green', label='Normale')
            ax.axhspan(60, 90, alpha=0.1, color='#ADFF2F', label='Légèrement diminuée')
            ax.axhspan(30, 60, alpha=0.1, color='yellow', label='Modérément diminuée')
            ax.axhspan(15, 30, alpha=0.1, color='orange', label='Sévèrement diminuée')
            ax.axhspan(0, 15, alpha=0.1, color='red', label='Insuffisance rénale')
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques de la fonction rénale
            mean_gfr = np.mean(gfr)
            gfr_category = "Normale" if mean_gfr >= 90 else (
                "Légèrement diminuée" if mean_gfr >= 60 else (
                "Modérément diminuée" if mean_gfr >= 30 else (
                "Sévèrement diminuée" if mean_gfr >= 15 else "Insuffisance rénale"
                )
                )
            )
            
            # Couleur selon la catégorie
            cat_color = "green" if mean_gfr >= 90 else (
                "#ADFF2F" if mean_gfr >= 60 else (
                "yellow" if mean_gfr >= 30 else (
                "orange" if mean_gfr >= 15 else "red"
                )
                )
            )
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0; color: #2c3e50;">Classification de la fonction rénale</h4>
                <div style="font-size: 20px; font-weight: bold; color: {cat_color}; margin: 10px 0;">
                    {gfr_category}
                </div>
                <div style="font-size: 16px;">
                    DFG moyen: <strong>{mean_gfr:.1f} mL/min/1.73m²</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualisation schématique du rein
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation anatomique du rein</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le rein
        kidney_impact = calculate_organ_impact(twin, "kidney")
        kidney_color = get_impact_color(kidney_impact)
        
        # Schéma SVG du rein et de la filtration
        kidney_svg = f"""
        <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
            
            <!-- Anatomie du rein -->
            <ellipse cx="300" cy="200" rx="120" ry="160" fill="{kidney_color}" stroke="#333" stroke-width="2" />
            <ellipse cx="300" cy="170" rx="80" ry="110" fill="#ffe4e1" stroke="#333" stroke-width="1" />
            <path d="M300,80 C340,100 350,150 350,200 C350,250 340,300 300,320 C260,300 250,250 250,200 C250,150 260,100 300,80 Z" 
                fill="#f8d7da" stroke="#333" stroke-width="1" />
            
            <!-- Uretère -->
            <path d="M300,360 C300,380 310,400 320,420" stroke="#333" stroke-width="8" fill="none" />
            
            <!-- Artère rénale -->
            <path d="M180,200 C220,180 240,200 260,200" stroke="#cc0000" stroke-width="8" fill="none" />
            <text x="210" y="185" font-family="Arial" font-size="12" text-anchor="middle">Artère rénale</text>
            
            <!-- Veine rénale -->
            <path d="M180,220 C220,240 240,220 260,220" stroke="#0044cc" stroke-width="8" fill="none" />
            <text x="210" y="245" font-family="Arial" font-size="12" text-anchor="middle">Veine rénale</text>
            
            <!-- Néphrons (unités de filtration) -->
            <circle cx="270" cy="150" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            <circle cx="310" cy="130" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            <circle cx="340" cy="170" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            <circle cx="320" cy="210" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            <circle cx="280" cy="190" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            <circle cx="290" cy="230" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            <circle cx="330" cy="250" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
            
            <!-- Glomérules (filtration) -->
            <circle cx="445" cy="170" r="40" fill="#f8f9fa" stroke="#333" stroke-width="1" />
            <circle cx="445" cy="170" r="25" fill="#ffe4e1" stroke="#333" stroke-width="1" />
            <path d="M420,150 Q445,130 470,150" stroke="#cc0000" stroke-width="3" fill="none" />
            <path d="M420,190 Q445,210 470,190" stroke="#0044cc" stroke-width="3" fill="none" />
            <text x="445" y="240" font-family="Arial" font-size="12" text-anchor="middle">Glomérule (filtration)</text>
            
            <!-- Légende -->
            <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                Impact sur les reins: {kidney_impact:.1f}/10
            </text>
        </svg>
        """
        
        st.markdown(kidney_svg, unsafe_allow_html=True)
        
    elif selected_system == "liver":
        # Système hépatique
        st.markdown("<h3 style='color: #2c3e50;'>Fonction hépatique et métabolisme des médicaments</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique du métabolisme hépatique du médicament
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Concentration du médicament
            ax.plot(time_data, twin.history['drug_plasma'], color='#e63946', 
                   linewidth=2.5, label='Concentration plasmatique')
            
            # Calculer le métabolisme hépatique
            hepatic_metabolism = []
            for i in range(len(time_data)):
                # Le métabolisme hépatique est proportionnel à la concentration plasmatique
                # et à la fonction hépatique
                metabolism = twin.history['drug_plasma'][i] * twin.params['liver_function'] * 0.03
                hepatic_metabolism.append(metabolism)
            
            ax.plot(time_data, hepatic_metabolism, color='#a55233', 
                   linewidth=2, label='Métabolisme hépatique')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Concentration')
            ax.set_title('Métabolisme hépatique des médicaments')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques hépatiques
            total_metabolism = np.trapz(hepatic_metabolism, time_data)
            drug_exposure = twin.metrics.get('drug_exposure', 0)
            metabolism_percent = (total_metabolism / max(drug_exposure, 0.001)) * 100
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Fonction hépatique", 
                    value=f"{twin.params['liver_function']:.2f}",
                    delta=None
                )
            with metric_cols[1]:
                st.metric(
                    label="Métabolisme hépatique", 
                    value=f"{metabolism_percent:.1f}%",
                    delta=f"{metabolism_percent - 50:.1f}" if metabolism_percent != 50 else None,
                    delta_color="normal"
                )
        
        with col2:
            # Graphique de la production hépatique de glucose
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Calculer la production hépatique de glucose
            # Elle est élevée quand la glycémie est basse, et réduite quand la glycémie est élevée
            # ou quand l'insuline est élevée
            hepatic_glucose_production = []
            for i in range(len(time_data)):
                # Production de base modulée par la glycémie et l'insuline
                base_production = twin.params['hepatic_glucose']
                glucose_effect = max(0, min(1, 1 - (twin.history['glucose'][i] - 70) / 100))
                insulin_effect = max(0, min(1, 1 - twin.history['insulin'][i] / 30))
                
                # Production calculée
                production = base_production * glucose_effect * insulin_effect
                hepatic_glucose_production.append(production)
            
            ax.plot(time_data, hepatic_glucose_production, color='#a55233', linewidth=2.5)
            
            # Tracer la glycémie pour référence
            ax2 = ax.twinx()
            ax2.plot(time_data, twin.history['glucose'], color='#0066cc', linewidth=1.5, alpha=0.5)
            ax2.set_ylabel('Glycémie (mg/dL)', color='#0066cc')
            ax2.tick_params(axis='y', labelcolor='#0066cc')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Production hépatique de glucose')
            ax.set_title('Production hépatique de glucose')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques de la fonction hépatique
            mean_production = np.mean(hepatic_glucose_production)
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0; color: #2c3e50;">Production hépatique de glucose</h4>
                <div style="font-size: 20px; font-weight: bold; margin: 10px 0;">
                    {mean_production:.2f}
                </div>
                <div style="font-size: 16px;">
                    {"Production élevée" if mean_production > 0.7 else ("Production normale" if mean_production > 0.4 else "Production faible")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualisation schématique du foie
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation anatomique du foie</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le foie
        liver_impact = calculate_organ_impact(twin, "liver")
        liver_color = get_impact_color(liver_impact)
        
        # Schéma SVG du foie et de ses fonctions
        liver_svg = f"""
        <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
            
            <!-- Anatomie du foie -->
            <path d="M180,150 C240,120 320,130 380,180 C420,220 430,280 400,320 C350,370 280,350 220,330 C160,310 140,270 150,220 C160,180 180,150 180,150 Z" 
                fill="{liver_color}" stroke="#333" stroke-width="2" />
            
            <!-- Vésicule biliaire -->
            <ellipse cx="280" cy="310" rx="25" ry="20" fill="#9acd32" stroke="#333" stroke-width="1" />
            <text x="280" y="315" font-family="Arial" font-size="10" text-anchor="middle">Vésicule</text>
            
            <!-- Veine porte -->
            <path d="M130,230 C160,230 180,240 200,250" stroke="#0044cc" stroke-width="10" fill="none" />
            <text x="150" y="220" font-family="Arial" font-size="12" text-anchor="middle">Veine porte</text>
            
            <!-- Artère hépatique -->
            <path d="M130,200 C160,200 180,220 200,230" stroke="#cc0000" stroke-width="6" fill="none" />
            <text x="150" y="190" font-family="Arial" font-size="12" text-anchor="middle">Artère hépatique</text>
            
            <!-- Veine cave -->
            <path d="M320,130 C320,100 330,80 350,60" stroke="#0044cc" stroke-width="12" fill="none" />
            <text x="350" y="90" font-family="Arial" font-size="12" text-anchor="middle">Veine cave</text>
            
            <!-- Flux de bile -->
            <path d="M330,280 Q300,300 280,290" stroke="#9acd32" stroke-width="3" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Cellules hépatiques (hépatocytes) -->
            <circle cx="250" cy="200" r="40" fill="#f8d7da" stroke="#333" stroke-width="1" />
            <circle cx="250" cy="200" r="30" fill="#faf3dd" stroke="#333" stroke-width="1" />
            <text x="250" cy="200" font-family="Arial" font-size="12" text-anchor="middle">Hépatocytes</text>
            
            <!-- Médicament -->
            <circle cx="230" cy="180" r="8" fill="#e63946" stroke="#333" stroke-width="1" />
            <text x="230" cy="180" font-family="Arial" font-size="8" text-anchor="middle" fill="white">Med</text>
            
            <!-- Glucose -->
            <circle cx="270" cy="190" r="8" fill="#0066cc" stroke="#333" stroke-width="1" />
            <text x="270" cy="190" font-family="Arial" font-size="8" text-anchor="middle" fill="white">Glu</text>
            
            <!-- Détail du métabolisme -->
            <rect x="400" y="140" width="150" height="200" rx="10" ry="10" fill="white" stroke="#333" stroke-width="1" />
            <text x="475" y="160" font-family="Arial" font-size="14" text-anchor="middle">Métabolisme hépatique</text>
            
            <!-- Phases du métabolisme -->
            <text x="420" y="190" font-family="Arial" font-size="12" text-anchor="left">Phase I: Oxydation</text>
            <rect x="420" y="200" width="110" r="5" height="10" fill="#f4a261" />
            
            <text x="420" y="230" font-family="Arial" font-size="12" text-anchor="left">Phase II: Conjugaison</text>
            <rect x="420" y="240" width="${min(110, 110 * twin.params['liver_function'])}" r="5" height="10" fill="#2a9d8f" />
            
            <text x="420" y="270" font-family="Arial" font-size="12" text-anchor="left">Excrétion biliaire</text>
            <rect x="420" y="280" width="${min(110, 110 * twin.params['liver_function'] * 0.9)}" r="5" height="10" fill="#9acd32" />
            
            <!-- Définition de la flèche -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs>
            
            <!-- Légende -->
            <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                Impact sur le foie: {liver_impact:.1f}/10
            </text>
        </svg>
        """
        
        st.markdown(liver_svg, unsafe_allow_html=True)
    
    elif selected_system == "immune":
        # Système immunitaire et inflammation
        st.markdown("<h3 style='color: #2c3e50;'>Réponse immunitaire et inflammation</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique de l'inflammation et des cellules immunitaires
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(time_data, twin.history['inflammation'], color='#ff6b6b', 
                   linewidth=2.5, label='Inflammation')
            ax.plot(time_data, twin.history['immune_cells'], color='#4ecdc4', 
                   linewidth=2.5, label='Cellules immunitaires')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Niveau')
            ax.set_title('Réponse inflammatoire et immunitaire')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Annotations pour les médicaments anti-inflammatoires
            for time, label in twin.history['interventions']:
                if "Médicament" in label and "antiinflammatory" in label:
                    ax.axvline(x=time, color='green', linestyle='--', alpha=0.5)
                    ax.annotate('Anti-inflammatoire', xy=(time, max(twin.history['inflammation'])),
                             xytext=(time, max(twin.history['inflammation']) + 5),
                             arrowprops=dict(facecolor='green', shrink=0.05),
                             horizontalalignment='center')
            
            st.pyplot(fig)
            
            # Métriques d'inflammation
            inflammation_burden = twin.metrics.get('inflammation_burden', 0)
            inflammation_relative = inflammation_burden / (twin.params['inflammatory_response'] * 100)
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Charge inflammatoire", 
                    value=f"{inflammation_burden:.1f}",
                    delta=f"{inflammation_burden - 300:.1f}" if inflammation_burden != 300 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Réponse immunitaire", 
                    value=f"{twin.params['immune_response']:.2f}",
                    delta=None
                )
        
        with col2:
            # Graphique de l'effet des médicaments anti-inflammatoires
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Trouver les administrations de médicaments anti-inflammatoires
            antiinflam_times = []
            for time, label in twin.history['interventions']:
                if "Médicament" in label and "antiinflammatory" in label:
                    antiinflam_times.append(time)
            
            # Calculer l'effet direct des médicaments sur l'inflammation
            drug_effect = []
            for i in range(len(time_data)):
                # L'effet est proportionnel à la concentration du médicament
                # et inversement proportionnel au niveau d'inflammation
                if twin.history['drug_tissue'][i] > 0:
                    effect = twin.history['drug_tissue'][i] * twin.params['immune_response'] * 0.1
                else:
                    effect = 0
                drug_effect.append(effect)
            
            ax.plot(time_data, drug_effect, color='#2a9d8f', linewidth=2.5, label='Effet anti-inflammatoire')
            
            # Visualiser aussi le traçage de la concentration du médicament
            ax2 = ax.twinx()
            ax2.plot(time_data, twin.history['drug_plasma'], color='#e63946', linestyle='--', linewidth=1.5, 
                    alpha=0.7, label='Concentration médicament')
            ax2.set_ylabel('Concentration', color='#e63946')
            ax2.tick_params(axis='y', labelcolor='#e63946')
            
            # Combinaison des légendes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Effet anti-inflammatoire')
            ax.set_title('Effet des médicaments anti-inflammatoires')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Marquer les points d'administration
            for t in antiinflam_times:
                ax.axvline(x=t, color='green', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            
            # Efficacité du traitement anti-inflammatoire
            if antiinflam_times:
                # Calculer la réduction d'inflammation
                # Comparer l'inflammation réelle à celle qui serait sans traitement
                theoretical_inflammation = twin.params['inflammatory_response'] * 100
                actual_inflammation = np.mean(twin.history['inflammation'])
                inflammation_reduction = (theoretical_inflammation - actual_inflammation) / theoretical_inflammation * 100
                
                # Limiter entre 0 et 100%
                inflammation_reduction = max(0, min(100, inflammation_reduction))
                
                efficacy_color = "green" if inflammation_reduction > 30 else ("orange" if inflammation_reduction > 10 else "red")
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                    <h4 style="margin-top: 0; color: #2c3e50;">Efficacité anti-inflammatoire</h4>
                    <div style="font-size: 24px; font-weight: bold; color: {efficacy_color};">
                        {inflammation_reduction:.1f}%
                    </div>
                    <div style="font-size: 14px; color: #666;">
                        Réduction de l'inflammation
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Aucun médicament anti-inflammatoire n'a été administré dans cette simulation.")
        
        # Visualisation schématique du système immunitaire
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation du système immunitaire</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le système immunitaire
        immune_impact = calculate_organ_impact(twin, "immune")
        immune_color = get_impact_color(immune_impact)
        
        # Schéma SVG du système immunitaire
        immune_svg = f"""
        <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
            
            <!-- Vaisseaux sanguins -->
            <path d="M100,225 C150,200 200,230 250,225 C300,220 350,240 400,225 C450,210 500,230 550,225" 
                stroke="#cc0000" stroke-width="15" fill="none" />
            
            <!-- Cellules immunitaires -->
            <!-- Neutrophile -->
            <circle cx="150" cy="225" r="20" fill="#f8f9fa" stroke="#333" stroke-width="2" />
            <circle cx="150" cy="225" r="15" fill="{immune_color}" stroke="#333" stroke-width="1" />
            <text x="150" y="225" font-family="Arial" font-size="10" text-anchor="middle">N</text>
            
            <!-- Macrophage -->
            <circle cx="200" cy="225" r="25" fill="#f8f9fa" stroke="#333" stroke-width="2" />
            <circle cx="200" cy="225" r="20" fill="{immune_color}" stroke="#333" stroke-width="1" />
            <text x="200" y="225" font-family="Arial" font-size="10" text-anchor="middle">M</text>
            
            <!-- Lymphocyte T -->
            <circle cx="300" cy="225" r="18" fill="#f8f9fa" stroke="#333" stroke-width="2" />
            <circle cx="300" cy="225" r="14" fill="{immune_color}" stroke="#333" stroke-width="1" />
            <text x="300" y="225" font-family="Arial" font-size="10" text-anchor="middle">T</text>
            
            <!-- Lymphocyte B -->
            <circle cx="350" cy="225" r="18" fill="#f8f9fa" stroke="#333" stroke-width="2" />
            <circle cx="350" cy="225" r="14" fill="{immune_color}" stroke="#333" stroke-width="1" />
            <text x="350" y="225" font-family="Arial" font-size="10" text-anchor="middle">B</text>
            
            <!-- Zone inflammation -->
            <ellipse cx="450" cy="250" rx="80" ry="60" fill="#ff6b6b" fill-opacity="0.3" stroke="#ff6b6b" stroke-width="2" />
            <text x="450" y="250" font-family="Arial" font-size="14" text-anchor="middle">Zone d'inflammation</text>
            
            <!-- Médiation inflammatoire -->
            <path d="M400,225 Q420,260 450,250" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,3" />
            <path d="M350,225 Q400,280 450,250" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,3" />
            
            <!-- Ganglions lymphatiques -->
            <ellipse cx="250" cy="150" rx="40" ry="25" fill="#d8f3dc" stroke="#333" stroke-width="2" />
            <text x="250" y="155" font-family="Arial" font-size="12" text-anchor="middle">Ganglion lymphatique</text>
            
            <!-- Rate -->
            <ellipse cx="400" cy="120" rx="50" ry="35" fill="#d8f3dc" stroke="#333" stroke-width="2" />
            <text x="400" y="125" font-family="Arial" font-size="12" text-anchor="middle">Rate</text>
            
            <!-- Cytokines -->
            <circle cx="420" cy="240" r="8" fill="#ff9e7d" stroke="#333" stroke-width="1" />
            <text x="420" y="240" font-family="Arial" font-size="8" text-anchor="middle">IL</text>
            
            <circle cx="440" cy="270" r="8" fill="#ff9e7d" stroke="#333" stroke-width="1" />
            <text x="440" y="270" font-family="Arial" font-size="8" text-anchor="middle">TNF</text>
            
            <circle cx="470" cy="260" r="8" fill="#ff9e7d" stroke="#333" stroke-width="1" />
            <text x="470" y="260" font-family="Arial" font-size="8" text-anchor="middle">IL</text>
            
            <!-- Médicament anti-inflammatoire -->
            <circle cx="500" cy="300" r="20" fill="#2a9d8f" stroke="#333" stroke-width="2" />
            <text x="500" y="300" font-family="Arial" font-size="10" text-anchor="middle" fill="white">Anti-inf</text>
            
            <!-- Flèche d'effet -->
            <path d="M490,285 Q480,270 470,270" stroke="#2a9d8f" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
            
            <!-- Définition de la flèche -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs>
            
            <!-- Légende -->
            <rect x="160" y="320" width="280" height="100" rx="10" ry="10" fill="white" stroke="#333" stroke-width="1" />
            <text x="300" y="340" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">
                État du système immunitaire
            </text>
            
            <text x="180" y="370" font-family="Arial" font-size="14" text-anchor="left">
                • Fonction immunitaire: {twin.params['immune_response']:.1f}
            </text>
            <text x="180" y="395" font-family="Arial" font-size="14" text-anchor="left">
                • Charge inflammatoire: {twin.metrics.get('inflammation_burden', 0):.1f}
            </text>
        </svg>
        """
        
        st.markdown(immune_svg, unsafe_allow_html=True)


def anatomical_visualization_tab(twin=None):
    """
    Onglet de visualisation anatomique des effets sur différents organes
    Accepte optionnellement un jumeau numérique pour visualiser ses données
    """
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Visualisation Anatomique</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e6f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <p style="margin: 0; color: #0066cc; font-size: 16px;">
            <strong>🧠 Visualisation interactive:</strong> Cette section vous permet de visualiser l'impact des traitements sur les différents systèmes et organes du patient.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Système/Organe à visualiser
    systems = {
        "cardio": "Système cardiovasculaire",
        "pancreas": "Pancréas et métabolisme",
        "renal": "Système rénal",
        "liver": "Foie et système hépatique",
        "immune": "Système immunitaire"
    }
    
    # Sélectionner le système à visualiser
    selected_system = st.selectbox(
        "Sélectionnez un système à visualiser",
        options=list(systems.keys()),
        format_func=lambda x: systems[x]
    )
    
    # Afficher un message si aucun jumeau numérique n'est disponible
    if twin is None:
        st.info("Aucune simulation active. Effectuez d'abord une simulation pour visualiser les effets sur les organes.")
        
        # Utiliser un placeholder pour montrer le type de visualisations disponibles
        st.markdown("<h3 style='color: #2c3e50;'>Aperçu des visualisations disponibles</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center;'>
                <img src="https://cdn.pixabay.com/photo/2013/07/12/17/22/heart-152377_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Système cardiovasculaire</h4>
                <p>Visualisez les impacts sur le cœur et les vaisseaux sanguins</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px;'>
                <img src="https://cdn.pixabay.com/photo/2017/01/31/22/32/kidneys-2027366_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Système rénal</h4>
                <p>Examinez les effets sur les reins et la filtration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center;'>
                <img src="https://cdn.pixabay.com/photo/2021/10/07/09/27/pancreas-6688196_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Pancréas et métabolisme</h4>
                <p>Visualisez la production d'insuline et le métabolisme du glucose</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px;'>
                <img src="https://cdn.pixabay.com/photo/2021/03/02/22/20/white-blood-cell-6064098_960_720.png" style="height: 100px; margin-bottom: 15px;">
                <h4>Système immunitaire</h4>
                <p>Observez les réponses inflammatoires et immunitaires</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton pour aller à la simulation
        if st.button("▶️ Aller à la simulation", type="primary"):
            st.session_state.mode_tab_index = 0  # Index de l'onglet simulation
            st.rerun()
        
        return
    
    # Si un jumeau est disponible, afficher les visualisations réelles
    st.markdown(f"<h2 style='color: #2c3e50;'>Visualisation du {systems[selected_system]}</h2>", unsafe_allow_html=True)
    
    # Préparer les données de la simulation
    time_data = twin.history['time']
    
    # Définir les graphiques selon le système sélectionné
    if selected_system == "cardio":
        # Système cardiovasculaire
        st.markdown("<h3 style='color: #2c3e50;'>Impact sur le système cardiovasculaire</h3>", unsafe_allow_html=True)
        
        # Créer une visualisation de base du cœur et de ses paramètres
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique du rythme cardiaque
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(time_data, twin.history['heart_rate'], color='#e63946', linewidth=2.5)
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Fréquence cardiaque (bpm)')
            ax.set_title('Évolution du rythme cardiaque')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Zone de rythme normal
            ax.axhspan(60, 100, alpha=0.2, color='green', label='Zone normale')
            
            # Annotations pour les médicaments
            for time, label in twin.history['interventions']:
                if "Médicament" in label and "beta_blocker" in label:
                    ax.axvline(x=time, color='blue', linestyle='--', alpha=0.5)
                    ax.annotate('β-bloquant', xy=(time, max(twin.history['heart_rate'])),
                            xytext=(time, max(twin.history['heart_rate']) + 5),
                            arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                            horizontalalignment='center')
            
            st.pyplot(fig)
            
            # Metrics
            hr_mean = np.mean(twin.history['heart_rate'])
            hr_var = np.std(twin.history['heart_rate'])
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="FC moyenne", 
                    value=f"{hr_mean:.1f} bpm",
                    delta=f"{hr_mean - 75:.1f}" if hr_mean != 75 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Variabilité FC", 
                    value=f"{hr_var:.1f}",
                    delta=f"{hr_var - 5:.1f}" if hr_var != 5 else None,
                    delta_color="inverse"
                )
        
        with col2:
            # Graphique de la pression artérielle
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(time_data, twin.history['blood_pressure'], color='#457b9d', linewidth=2.5)
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Pression artérielle (mmHg)')
            ax.set_title('Évolution de la pression artérielle')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Zone de pression normale
            ax.axhspan(110, 130, alpha=0.2, color='green', label='Zone normale')
            
            # Annotations pour les médicaments
            for time, label in twin.history['interventions']:
                if "Médicament" in label and ("vasodilator" in label or "beta_blocker" in label):
                    ax.axvline(x=time, color='purple' if "vasodilator" in label else 'blue', 
                             linestyle='--', alpha=0.5)
                    med_type = "Vasodilatateur" if "vasodilator" in label else "β-bloquant"
                    ax.annotate(med_type, xy=(time, min(twin.history['blood_pressure'])),
                             xytext=(time, min(twin.history['blood_pressure']) - 5),
                             arrowprops=dict(facecolor='purple' if "vasodilator" in label else 'blue', 
                                             shrink=0.05, width=1.5, headwidth=8),
                             horizontalalignment='center')
            
            st.pyplot(fig)
            
            # Metrics
            bp_mean = np.mean(twin.history['blood_pressure'])
            bp_var = np.std(twin.history['blood_pressure'])
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="PA moyenne", 
                    value=f"{bp_mean:.1f} mmHg",
                    delta=f"{bp_mean - 120:.1f}" if bp_mean != 120 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Variabilité PA", 
                    value=f"{bp_var:.1f}",
                    delta=f"{bp_var - 8:.1f}" if bp_var != 8 else None,
                    delta_color="inverse"
                )
        
        # Visualisation anatomique schématique
        st.markdown("<h3 style='color: #2c3e50;'>Schéma interactif du cœur</h3>", unsafe_allow_html=True)
        
        # Créer une visualisation SVG simple du cœur
        heart_impact = calculate_organ_impact(twin, "heart")
        heart_color = get_impact_color(heart_impact)
        
        # Utiliser components.html au lieu de st.markdown pour le SVG
        heart_svg_html = f"""
        <div style="display: flex; justify-content: center;">
            <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
                <!-- Background -->
                <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
                
                <!-- Heart outline -->
                <path d="M300,120 C350,80 450,80 450,180 C450,300 300,380 300,380 C300,380 150,300 150,180 C150,80 250,80 300,120 Z" 
                    fill="{heart_color}" stroke="#333" stroke-width="2" />
                
                <!-- Label -->
                <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                    Impact sur le cœur: {heart_impact:.1f}/10
                </text>
                
                <!-- Aorte -->
                <path d="M300,120 C300,100 280,80 250,80 Q220,80 220,50" 
                    fill="none" stroke="#cc0000" stroke-width="10" />
                
                <!-- Artère pulmonaire -->
                <path d="M300,120 C300,100 320,80 350,80 Q380,80 380,50" 
                    fill="none" stroke="#0044cc" stroke-width="10" />
                
                <!-- Veines pulmonaires -->
                <path d="M260,160 C220,160 200,120 180,120" 
                    fill="none" stroke="#0066cc" stroke-width="8" />
                <path d="M340,160 C380,160 400,120 420,120" 
                    fill="none" stroke="#0066cc" stroke-width="8" />
                
                <!-- Veine cave -->
                <path d="M300,360 C300,390 270,410 230,410" 
                    fill="none" stroke="#0044cc" stroke-width="12" />
            </svg>
        </div>
        """
        
        components.html(heart_svg_html, height=450)
    
    elif selected_system == "pancreas":
        # Système pancréatique et métabolisme
        st.markdown("<h3 style='color: #2c3e50;'>Métabolisme du glucose et fonction pancréatique</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique glucose-insuline
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            # Glucose
            ax1.set_xlabel('Temps (heures)')
            ax1.set_ylabel('Glycémie (mg/dL)', color='#0066cc')
            ax1.plot(time_data, twin.history['glucose'], color='#0066cc', linewidth=2.5)
            ax1.tick_params(axis='y', labelcolor='#0066cc')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Zone cible
            ax1.axhspan(70, 180, alpha=0.1, color='green', label='Zone cible')
            ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7)
            
            # Insuline sur l'axe Y secondaire
            ax2 = ax1.twinx()
            ax2.set_ylabel('Insuline (mU/L)', color='#28a745')
            ax2.plot(time_data, twin.history['insulin'], color='#28a745', linewidth=2)
            ax2.tick_params(axis='y', labelcolor='#28a745')
            
            # Annotations pour les repas
            for time, label in twin.history['interventions']:
                if "Repas" in label:
                    # Extraire la quantité de glucides
                    carbs = int(label.split(": ")[1].split(" ")[0])
                    marker_size = max(50, min(150, carbs * 1.5))
                    
                    # Marquer les repas
                    ax1.scatter(time, 50, color='#f4a261', s=marker_size, alpha=0.7, zorder=5,
                              marker='^', edgecolors='white')
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Metrics
            glucose_mean = twin.metrics.get('glucose_mean', 0)
            in_range = twin.metrics.get('percent_in_range', 0)
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Glycémie moyenne", 
                    value=f"{glucose_mean:.1f} mg/dL",
                    delta=f"{glucose_mean - 100:.1f}" if glucose_mean != 100 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Temps en cible", 
                    value=f"{in_range:.1f}%",
                    delta=f"{in_range - 75:.1f}" if in_range != 75 else None,
                    delta_color="normal"
                )
        
        with col2:
            # Visualisation de l'utilisation du glucose
            # Créons un graphique montrant l'utilisation du glucose par les tissus
            
            # Impact des médicaments antidiabétiques
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Nous allons estimer l'absorption du glucose en fonction des données
            # Utilisons la variation de glycémie après les repas comme indicateur
            glucose_absorption = []
            baseline = twin.params['baseline_glucose']
            
            for i in range(1, len(time_data)):
                # Si la glycémie augmente, c'est l'apport des repas
                # Si elle diminue, c'est l'effet de l'insuline et des médicaments
                if twin.history['glucose'][i] > twin.history['glucose'][i-1]:
                    absorption = 0
                else:
                    # Calculer l'absorption relative
                    absorption = (twin.history['glucose'][i-1] - twin.history['glucose'][i]) * twin.history['insulin'][i] / 100
                    
                glucose_absorption.append(max(0, absorption))
            
            # Ajouter une valeur initiale
            glucose_absorption.insert(0, 0)
            
            # Tracer l'absorption du glucose
            ax.plot(time_data, glucose_absorption, color='#9c6644', linewidth=2.5, label="Absorption du glucose")
            
            # Tracer l'insuline active pour montrer sa corrélation
            insulin_active = np.array(twin.history['insulin']) * np.array(twin.history['drug_tissue']) / 20
            ax.plot(time_data, insulin_active, color='#28a745', linestyle='--', alpha=0.7, label="Insuline active")
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Utilisation relative du glucose')
            ax.set_title('Absorption et utilisation du glucose')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)
            
            # Métriques calculées
            insulin_effect = np.mean(insulin_active) * twin.params['insulin_sensitivity']
            drug_effect = np.mean(twin.history['drug_tissue']) * 0.5
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Effet de l'insuline", 
                    value=f"{insulin_effect:.2f}",
                    delta=f"{insulin_effect - 0.4:.2f}" if insulin_effect != 0.4 else None,
                    delta_color="normal"
                )
            with metric_cols[1]:
                st.metric(
                    label="Effet médicamenteux", 
                    value=f"{drug_effect:.2f}",
                    delta=f"{drug_effect - 0.3:.2f}" if drug_effect != 0.3 else None,
                    delta_color="normal"
                )
        
        # Visualisation schématique du pancréas et du métabolisme du glucose
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation du pancréas et du métabolisme</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le pancréas
        pancreas_impact = calculate_organ_impact(twin, "pancreas")
        pancreas_color = get_impact_color(pancreas_impact)
        
        # Schéma SVG du pancréas et du métabolisme du glucose
        pancreas_svg_html = f"""
        <div style="display: flex; justify-content: center;">
            <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
                <!-- Background -->
                <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
                
                <!-- Estomac -->
                <ellipse cx="200" cy="150" rx="70" ry="50" fill="#f4a261" stroke="#333" stroke-width="2" />
                <text x="200" y="155" font-family="Arial" font-size="14" text-anchor="middle">Estomac</text>
                
                <!-- Pancréas -->
                <path d="M250,200 C300,180 350,190 400,200 C420,205 430,220 420,240 C400,270 350,280 300,260 C270,250 240,220 250,200 Z" 
                    fill="{pancreas_color}" stroke="#333" stroke-width="2" />
                <text x="340" y="230" font-family="Arial" font-size="14" text-anchor="middle">Pancréas</text>
                
                <!-- Îlots de Langerhans -->
                <circle cx="320" cy="220" r="8" fill="#28a745" stroke="#333" stroke-width="1" />
                <circle cx="350" cy="230" r="8" fill="#28a745" stroke="#333" stroke-width="1" />
                <circle cx="380" cy="225" r="8" fill="#28a745" stroke="#333" stroke-width="1" />
                
                <!-- Intestin -->
                <path d="M200,200 C180,220 190,240 170,260 C150,280 160,300 180,310 C200,320 220,310 240,320 C260,330 290,320 310,330 C330,340 360,330 380,340" 
                    fill="none" stroke="#cc6b49" stroke-width="15" />
                
                <!-- Foie -->
                <path d="M100,230 C150,200 200,220 230,270 C210,310 150,320 100,290 C80,270 80,250 100,230 Z" 
                    fill="#a55233" stroke="#333" stroke-width="2" />
                <text x="150" y="260" font-family="Arial" font-size="14" text-anchor="middle">Foie</text>
                
                <!-- Cellules musculaires -->
                <rect x="450" y="300" width="100" height="60" rx="10" ry="10" fill="#d8bfd8" stroke="#333" stroke-width="2" />
                <text x="500" y="330" font-family="Arial" font-size="14" text-anchor="middle">Muscles</text>
                
                <!-- Cellules adipeuses -->
                <circle cx="480" cy="150" r="50" fill="#ffef99" stroke="#333" stroke-width="2" />
                <text x="480" y="155" font-family="Arial" font-size="14" text-anchor="middle">Tissu adipeux</text>
                
                <!-- Glucose sanguin -->
                <circle cx="300" cy="150" r="15" fill="#0066cc" stroke="#333" stroke-width="1" />
                <text x="300" y="155" font-family="Arial" font-size="10" text-anchor="middle" fill="white">Glucose</text>
                
                <!-- Insuline -->
                <circle cx="350" cy="180" r="10" fill="#28a745" stroke="#333" stroke-width="1" />
                <text x="350" y="183" font-family="Arial" font-size="8" text-anchor="middle" fill="white">Insuline</text>
                
                <!-- Flèches de circulation -->
                <!-- Estomac -> sang -->
                <path d="M240,130 Q270,100 290,140" stroke="#f4a261" stroke-width="3" fill="none" />
                
                <!-- Pancréas -> sang (insuline) -->
                <path d="M330,200 Q320,170 350,170" stroke="#28a745" stroke-width="2" fill="none" />
                
                <!-- Sang -> muscles (glucose) -->
                <path d="M320,160 Q380,200 450,320" stroke="#0066cc" stroke-width="2" fill="none" />
                
                <!-- Sang -> tissu adipeux (glucose) -->
                <path d="M320,140 Q350,110 430,150" stroke="#0066cc" stroke-width="2" fill="none" />
                
                <!-- Sang -> foie (glucose) -->
                <path d="M280,160 Q250,200 200,240" stroke="#0066cc" stroke-width="2" fill="none" />
                
                <!-- Légende -->
                <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                    Impact sur le pancréas: {pancreas_impact:.1f}/10
                </text>
            </svg>
        </div>
        """
        
        components.html(pancreas_svg_html, height=450)
        
        # Informations sur l'impact des médicaments
        med_cols = st.columns(2)
        with med_cols[0]:
            st.markdown("""
            <div style="background-color: #f0f7ff; border-radius: 8px; padding: 15px;">
                <h4 style="margin-top: 0; color: #0066cc;">Impact des médicaments antidiabétiques</h4>
                <p>Les médicaments antidiabétiques agissent en:</p>
                <ul>
                    <li>Augmentant la sensibilité à l'insuline</li>
                    <li>Réduisant la production hépatique de glucose</li>
                    <li>Ralentissant l'absorption intestinale de glucose</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with med_cols[1]:
            # Calculer l'efficacité médicamenteuse
            if hasattr(twin, 'medications') and twin.medications:
                antidiabetic_meds = [med for med in twin.medications if med[1] == 'antidiabetic']
                if antidiabetic_meds:
                    efficacy = min(10, max(0, 10 - abs(glucose_mean - 110) / 10))
                    efficacy_color = "green" if efficacy > 7 else ("orange" if efficacy > 4 else "red")
                    
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; border-radius: 8px; padding: 15px;">
                        <h4 style="margin-top: 0; color: #0066cc;">Efficacité du traitement</h4>
                        <div style="text-align: center; padding: 10px;">
                            <div style="font-size: 24px; font-weight: bold; color: {efficacy_color};">{efficacy:.1f}/10</div>
                            <div style="font-size: 14px; color: #666;">Score d'efficacité du traitement</div>
                        </div>
                        <p>Le traitement {"est efficace" if efficacy > 7 else ("a une efficacité modérée" if efficacy > 4 else "n'est pas optimal")} pour maintenir la glycémie dans les valeurs cibles.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Aucun médicament antidiabétique n'a été administré dans cette simulation.")
            else:
                st.info("Aucun médicament n'a été administré dans cette simulation.")
    
    elif selected_system == "renal":
        # Système rénal
        st.markdown("<h3 style='color: #2c3e50;'>Fonction rénale et élimination des médicaments</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique de concentration du médicament
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(time_data, twin.history['drug_plasma'], color='#e63946', 
                   linewidth=2.5, label='Concentration plasmatique')
            
            # Calculer l'élimination rénale
            renal_elimination = []
            for i in range(len(time_data)):
                # L'élimination rénale est proportionnelle à la concentration plasmatique
                # et à la fonction rénale
                elimination = twin.history['drug_plasma'][i] * twin.params['renal_function'] * 0.02
                renal_elimination.append(elimination)
            
            ax.plot(time_data, renal_elimination, color='#457b9d', 
                   linewidth=2, label='Élimination rénale')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Concentration')
            ax.set_title('Élimination rénale des médicaments')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques rénales
            total_elimination = np.trapz(renal_elimination, time_data)
            drug_exposure = twin.metrics.get('drug_exposure', 0)
            elimination_percent = (total_elimination / max(drug_exposure, 0.001)) * 100
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Fonction rénale", 
                    value=f"{twin.params['renal_function']:.2f}",
                    delta=None
                )
            with metric_cols[1]:
                st.metric(
                    label="Élimination rénale", 
                    value=f"{elimination_percent:.1f}%",
                    delta=f"{elimination_percent - 50:.1f}" if elimination_percent != 50 else None,
                    delta_color="normal"
                )
        
        with col2:
            # Graphique de la filtration glomérulaire estimée
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Calculer la filtration glomérulaire en fonction de la fonction rénale
            # et des autres paramètres
            base_gfr = 90 * twin.params['renal_function']  # mL/min/1.73m2
            
            # La filtration est affectée par la pression artérielle et l'inflammation
            gfr = []
            for i in range(len(time_data)):
                # Ajustement par la pression artérielle (haute pression = diminution de la GFR)
                bp_effect = 1 - max(0, min(0.3, (twin.history['blood_pressure'][i] - 120) / 200))
                
                # Ajustement par l'inflammation (inflammation = diminution de la GFR)
                inflam_effect = 1 - max(0, min(0.3, twin.history['inflammation'][i] / 100))
                
                # GFR calculée
                current_gfr = base_gfr * bp_effect * inflam_effect
                gfr.append(current_gfr)
            
            ax.plot(time_data, gfr, color='#4ecdc4', linewidth=2.5)
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('DFG estimé (mL/min/1.73m²)')
            ax.set_title('Débit de Filtration Glomérulaire Estimé')
            
            # Zones de classification de la fonction rénale
            ax.axhspan(90, 120, alpha=0.1, color='green', label='Normale')
            ax.axhspan(60, 90, alpha=0.1, color='#ADFF2F', label='Légèrement diminuée')
            ax.axhspan(30, 60, alpha=0.1, color='yellow', label='Modérément diminuée')
            ax.axhspan(15, 30, alpha=0.1, color='orange', label='Sévèrement diminuée')
            ax.axhspan(0, 15, alpha=0.1, color='red', label='Insuffisance rénale')
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques de la fonction rénale
            mean_gfr = np.mean(gfr)
            gfr_category = "Normale" if mean_gfr >= 90 else (
                "Légèrement diminuée" if mean_gfr >= 60 else (
                "Modérément diminuée" if mean_gfr >= 30 else (
                "Sévèrement diminuée" if mean_gfr >= 15 else "Insuffisance rénale"
                )
                )
            )
            
            # Couleur selon la catégorie
            cat_color = "green" if mean_gfr >= 90 else (
                "#ADFF2F" if mean_gfr >= 60 else (
                "yellow" if mean_gfr >= 30 else (
                "orange" if mean_gfr >= 15 else "red"
                )
                )
            )
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0; color: #2c3e50;">Classification de la fonction rénale</h4>
                <div style="font-size: 20px; font-weight: bold; color: {cat_color}; margin: 10px 0;">
                    {gfr_category}
                </div>
                <div style="font-size: 16px;">
                    DFG moyen: <strong>{mean_gfr:.1f} mL/min/1.73m²</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualisation schématique du rein
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation anatomique du rein</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le rein
        kidney_impact = calculate_organ_impact(twin, "kidney")
        kidney_color = get_impact_color(kidney_impact)
        
        # Schéma SVG du rein et de la filtration
        kidney_svg_html = f"""
        <div style="display: flex; justify-content: center;">
            <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
                <!-- Background -->
                <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
                
                <!-- Anatomie du rein -->
                <ellipse cx="300" cy="200" rx="120" ry="160" fill="{kidney_color}" stroke="#333" stroke-width="2" />
                <ellipse cx="300" cy="170" rx="80" ry="110" fill="#ffe4e1" stroke="#333" stroke-width="1" />
                <path d="M300,80 C340,100 350,150 350,200 C350,250 340,300 300,320 C260,300 250,250 250,200 C250,150 260,100 300,80 Z" 
                    fill="#f8d7da" stroke="#333" stroke-width="1" />
                
                <!-- Uretère -->
                <path d="M300,360 C300,380 310,400 320,420" stroke="#333" stroke-width="8" fill="none" />
                
                <!-- Artère rénale -->
                <path d="M180,200 C220,180 240,200 260,200" stroke="#cc0000" stroke-width="8" fill="none" />
                <text x="210" y="185" font-family="Arial" font-size="12" text-anchor="middle">Artère rénale</text>
                
                <!-- Veine rénale -->
                <path d="M180,220 C220,240 240,220 260,220" stroke="#0044cc" stroke-width="8" fill="none" />
                <text x="210" y="245" font-family="Arial" font-size="12" text-anchor="middle">Veine rénale</text>
                
                <!-- Néphrons (unités de filtration) -->
                <circle cx="270" cy="150" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                <circle cx="310" cy="130" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                <circle cx="340" cy="170" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                <circle cx="320" cy="210" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                <circle cx="280" cy="190" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                <circle cx="290" cy="230" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                <circle cx="330" cy="250" r="10" fill="#e6f7ff" stroke="#333" stroke-width="1" />
                
                <!-- Glomérules (filtration) -->
                <circle cx="445" cy="170" r="40" fill="#f8f9fa" stroke="#333" stroke-width="1" />
                <circle cx="445" cy="170" r="25" fill="#ffe4e1" stroke="#333" stroke-width="1" />
                <path d="M420,150 Q445,130 470,150" stroke="#cc0000" stroke-width="3" fill="none" />
                <path d="M420,190 Q445,210 470,190" stroke="#0044cc" stroke-width="3" fill="none" />
                <text x="445" y="240" font-family="Arial" font-size="12" text-anchor="middle">Glomérule (filtration)</text>
                
                <!-- Légende -->
                <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                    Impact sur les reins: {kidney_impact:.1f}/10
                </text>
            </svg>
        </div>
        """
        
        components.html(kidney_svg_html, height=450)
        
    elif selected_system == "liver":
        # Système hépatique
        st.markdown("<h3 style='color: #2c3e50;'>Fonction hépatique et métabolisme des médicaments</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique du métabolisme hépatique du médicament
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Concentration du médicament
            ax.plot(time_data, twin.history['drug_plasma'], color='#e63946', 
                   linewidth=2.5, label='Concentration plasmatique')
            
            # Calculer le métabolisme hépatique
            hepatic_metabolism = []
            for i in range(len(time_data)):
                # Le métabolisme hépatique est proportionnel à la concentration plasmatique
                # et à la fonction hépatique
                metabolism = twin.history['drug_plasma'][i] * twin.params['liver_function'] * 0.03
                hepatic_metabolism.append(metabolism)
            
            ax.plot(time_data, hepatic_metabolism, color='#a55233', 
                   linewidth=2, label='Métabolisme hépatique')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Concentration')
            ax.set_title('Métabolisme hépatique des médicaments')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques hépatiques
            total_metabolism = np.trapz(hepatic_metabolism, time_data)
            drug_exposure = twin.metrics.get('drug_exposure', 0)
            metabolism_percent = (total_metabolism / max(drug_exposure, 0.001)) * 100
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Fonction hépatique", 
                    value=f"{twin.params['liver_function']:.2f}",
                    delta=None
                )
            with metric_cols[1]:
                st.metric(
                    label="Métabolisme hépatique", 
                    value=f"{metabolism_percent:.1f}%",
                    delta=f"{metabolism_percent - 50:.1f}" if metabolism_percent != 50 else None,
                    delta_color="normal"
                )
        
        with col2:
            # Graphique de la production hépatique de glucose
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Calculer la production hépatique de glucose
            # Elle est élevée quand la glycémie est basse, et réduite quand la glycémie est élevée
            # ou quand l'insuline est élevée
            hepatic_glucose_production = []
            for i in range(len(time_data)):
                # Production de base modulée par la glycémie et l'insuline
                base_production = twin.params['hepatic_glucose']
                glucose_effect = max(0, min(1, 1 - (twin.history['glucose'][i] - 70) / 100))
                insulin_effect = max(0, min(1, 1 - twin.history['insulin'][i] / 30))
                
                # Production calculée
                production = base_production * glucose_effect * insulin_effect
                hepatic_glucose_production.append(production)
            
            ax.plot(time_data, hepatic_glucose_production, color='#a55233', linewidth=2.5)
            
            # Tracer la glycémie pour référence
            ax2 = ax.twinx()
            ax2.plot(time_data, twin.history['glucose'], color='#0066cc', linewidth=1.5, alpha=0.5)
            ax2.set_ylabel('Glycémie (mg/dL)', color='#0066cc')
            ax2.tick_params(axis='y', labelcolor='#0066cc')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Production hépatique de glucose')
            ax.set_title('Production hépatique de glucose')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Métriques de la fonction hépatique
            mean_production = np.mean(hepatic_glucose_production)
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0; color: #2c3e50;">Production hépatique de glucose</h4>
                <div style="font-size: 20px; font-weight: bold; margin: 10px 0;">
                    {mean_production:.2f}
                </div>
                <div style="font-size: 16px;">
                    {"Production élevée" if mean_production > 0.7 else ("Production normale" if mean_production > 0.4 else "Production faible")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualisation schématique du foie
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation anatomique du foie</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le foie
        liver_impact = calculate_organ_impact(twin, "liver")
        liver_color = get_impact_color(liver_impact)
        
        # Schéma SVG du foie et de ses fonctions
        liver_svg_html = f"""
        <div style="display: flex; justify-content: center;">
            <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
                <!-- Background -->
                <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
                
                <!-- Anatomie du foie -->
                <path d="M180,150 C240,120 320,130 380,180 C420,220 430,280 400,320 C350,370 280,350 220,330 C160,310 140,270 150,220 C160,180 180,150 180,150 Z" 
                    fill="{liver_color}" stroke="#333" stroke-width="2" />
                
                <!-- Vésicule biliaire -->
                <ellipse cx="280" cy="310" rx="25" ry="20" fill="#9acd32" stroke="#333" stroke-width="1" />
                <text x="280" y="315" font-family="Arial" font-size="10" text-anchor="middle">Vésicule</text>
                
                <!-- Veine porte -->
                <path d="M130,230 C160,230 180,240 200,250" stroke="#0044cc" stroke-width="10" fill="none" />
                <text x="150" y="220" font-family="Arial" font-size="12" text-anchor="middle">Veine porte</text>
                
                <!-- Artère hépatique -->
                <path d="M130,200 C160,200 180,220 200,230" stroke="#cc0000" stroke-width="6" fill="none" />
                <text x="150" y="190" font-family="Arial" font-size="12" text-anchor="middle">Artère hépatique</text>
                
                <!-- Veine cave -->
                <path d="M320,130 C320,100 330,80 350,60" stroke="#0044cc" stroke-width="12" fill="none" />
                <text x="350" y="90" font-family="Arial" font-size="12" text-anchor="middle">Veine cave</text>
                
                <!-- Flux de bile -->
                <path d="M330,280 Q300,300 280,290" stroke="#9acd32" stroke-width="3" fill="none" />
                
                <!-- Cellules hépatiques (hépatocytes) -->
                <circle cx="250" cy="200" r="40" fill="#f8d7da" stroke="#333" stroke-width="1" />
                <circle cx="250" cy="200" r="30" fill="#faf3dd" stroke="#333" stroke-width="1" />
                <text x="250" y="200" font-family="Arial" font-size="12" text-anchor="middle">Hépatocytes</text>
                
                <!-- Médicament -->
                <circle cx="230" cy="180" r="8" fill="#e63946" stroke="#333" stroke-width="1" />
                <text x="230" y="180" font-family="Arial" font-size="8" text-anchor="middle" fill="white">Med</text>
                
                <!-- Glucose -->
                <circle cx="270" cy="190" r="8" fill="#0066cc" stroke="#333" stroke-width="1" />
                <text x="270" y="190" font-family="Arial" font-size="8" text-anchor="middle" fill="white">Glu</text>
                
                <!-- Détail du métabolisme -->
                <rect x="400" y="140" width="150" height="200" rx="10" ry="10" fill="white" stroke="#333" stroke-width="1" />
                <text x="475" y="160" font-family="Arial" font-size="14" text-anchor="middle">Métabolisme hépatique</text>
                
                <!-- Phases du métabolisme -->
                <text x="420" y="190" font-family="Arial" font-size="12" text-anchor="left">Phase I: Oxydation</text>
                <rect x="420" y="200" width="110" height="10" rx="5" fill="#f4a261" />
                
                <text x="420" y="230" font-family="Arial" font-size="12" text-anchor="left">Phase II: Conjugaison</text>
                <rect x="420" y="240" width="{min(110, 110 * twin.params['liver_function'])}" height="10" rx="5" fill="#2a9d8f" />
                
                <text x="420" y="270" font-family="Arial" font-size="12" text-anchor="left">Excrétion biliaire</text>
                <rect x="420" y="280" width="{min(110, 110 * twin.params['liver_function'] * 0.9)}" height="10" rx="5" fill="#9acd32" />
                
                <!-- Légende -->
                <text x="300" y="420" font-family="Arial" font-size="16" text-anchor="middle">
                    Impact sur le foie: {liver_impact:.1f}/10
                </text>
            </svg>
        </div>
        """
        
        components.html(liver_svg_html, height=450)
    
    elif selected_system == "immune":
        # Système immunitaire et inflammation
        st.markdown("<h3 style='color: #2c3e50;'>Réponse immunitaire et inflammation</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Graphique de l'inflammation et des cellules immunitaires
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(time_data, twin.history['inflammation'], color='#ff6b6b', 
                   linewidth=2.5, label='Inflammation')
            ax.plot(time_data, twin.history['immune_cells'], color='#4ecdc4', 
                   linewidth=2.5, label='Cellules immunitaires')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Niveau')
            ax.set_title('Réponse inflammatoire et immunitaire')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Annotations pour les médicaments anti-inflammatoires
            for time, label in twin.history['interventions']:
                if "Médicament" in label and "antiinflammatory" in label:
                    ax.axvline(x=time, color='green', linestyle='--', alpha=0.5)
                    ax.annotate('Anti-inflammatoire', xy=(time, max(twin.history['inflammation'])),
                             xytext=(time, max(twin.history['inflammation']) + 5),
                             arrowprops=dict(facecolor='green', shrink=0.05),
                             horizontalalignment='center')
            
            st.pyplot(fig)
            
            # Métriques d'inflammation
            inflammation_burden = twin.metrics.get('inflammation_burden', 0)
            inflammation_relative = inflammation_burden / (twin.params['inflammatory_response'] * 100)
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label="Charge inflammatoire", 
                    value=f"{inflammation_burden:.1f}",
                    delta=f"{inflammation_burden - 300:.1f}" if inflammation_burden != 300 else None,
                    delta_color="inverse"
                )
            with metric_cols[1]:
                st.metric(
                    label="Réponse immunitaire", 
                    value=f"{twin.params['immune_response']:.2f}",
                    delta=None
                )
        
        with col2:
            # Graphique de l'effet des médicaments anti-inflammatoires
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Trouver les administrations de médicaments anti-inflammatoires
            antiinflam_times = []
            for time, label in twin.history['interventions']:
                if "Médicament" in label and "antiinflammatory" in label:
                    antiinflam_times.append(time)
            
            # Calculer l'effet direct des médicaments sur l'inflammation
            drug_effect = []
            for i in range(len(time_data)):
                # L'effet est proportionnel à la concentration du médicament
                # et inversement proportionnel au niveau d'inflammation
                if twin.history['drug_tissue'][i] > 0:
                    effect = twin.history['drug_tissue'][i] * twin.params['immune_response'] * 0.1
                else:
                    effect = 0
                drug_effect.append(effect)
            
            ax.plot(time_data, drug_effect, color='#2a9d8f', linewidth=2.5, label='Effet anti-inflammatoire')
            
            # Visualiser aussi le traçage de la concentration du médicament
            ax2 = ax.twinx()
            ax2.plot(time_data, twin.history['drug_plasma'], color='#e63946', linestyle='--', linewidth=1.5, 
                    alpha=0.7, label='Concentration médicament')
            ax2.set_ylabel('Concentration', color='#e63946')
            ax2.tick_params(axis='y', labelcolor='#e63946')
            
            # Combinaison des légendes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax.set_xlabel('Temps (heures)')
            ax.set_ylabel('Effet anti-inflammatoire')
            ax.set_title('Effet des médicaments anti-inflammatoires')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Marquer les points d'administration
            for t in antiinflam_times:
                ax.axvline(x=t, color='green', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            
            # Efficacité du traitement anti-inflammatoire
            if antiinflam_times:
                # Calculer la réduction d'inflammation
                # Comparer l'inflammation réelle à celle qui serait sans traitement
                theoretical_inflammation = twin.params['inflammatory_response'] * 100
                actual_inflammation = np.mean(twin.history['inflammation'])
                inflammation_reduction = (theoretical_inflammation - actual_inflammation) / theoretical_inflammation * 100
                
                # Limiter entre 0 et 100%
                inflammation_reduction = max(0, min(100, inflammation_reduction))
                
                efficacy_color = "green" if inflammation_reduction > 30 else ("orange" if inflammation_reduction > 10 else "red")
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                    <h4 style="margin-top: 0; color: #2c3e50;">Efficacité anti-inflammatoire</h4>
                    <div style="font-size: 24px; font-weight: bold; color: {efficacy_color};">
                        {inflammation_reduction:.1f}%
                    </div>
                    <div style="font-size: 14px; color: #666;">
                        Réduction de l'inflammation
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Aucun médicament anti-inflammatoire n'a été administré dans cette simulation.")
        
        # Visualisation schématique du système immunitaire
        st.markdown("<h3 style='color: #2c3e50;'>Visualisation du système immunitaire</h3>", unsafe_allow_html=True)
        
        # Calculer l'impact sur le système immunitaire
        immune_impact = calculate_organ_impact(twin, "immune")
        immune_color = get_impact_color(immune_impact)
        
        # Schéma SVG du système immunitaire
        immune_svg_html = f"""
        <div style="display: flex; justify-content: center;">
            <svg width="600" height="450" xmlns="http://www.w3.org/2000/svg">
                <!-- Background -->
                <rect width="100%" height="100%" fill="#f8f9fa" rx="10" ry="10" />
                
                <!-- Vaisseaux sanguins -->
                <path d="M100,225 C150,200 200,230 250,225 C300,220 350,240 400,225 C450,210 500,230 550,225" 
                    stroke="#cc0000" stroke-width="15" fill="none" />
                
                <!-- Cellules immunitaires -->
                <!-- Neutrophile -->
                <circle cx="150" cy="225" r="20" fill="#f8f9fa" stroke="#333" stroke-width="2" />
                <circle cx="150" cy="225" r="15" fill="{immune_color}" stroke="#333" stroke-width="1" />
                <text x="150" y="225" font-family="Arial" font-size="10" text-anchor="middle">N</text>
                
                <!-- Macrophage -->
                <circle cx="200" cy="225" r="25" fill="#f8f9fa" stroke="#333" stroke-width="2" />
                <circle cx="200" cy="225" r="20" fill="{immune_color}" stroke="#333" stroke-width="1" />
                <text x="200" y="225" font-family="Arial" font-size="10" text-anchor="middle">M</text>
                
                <!-- Lymphocyte T -->
                <circle cx="300" cy="225" r="18" fill="#f8f9fa" stroke="#333" stroke-width="2" />
                <circle cx="300" cy="225" r="14" fill="{immune_color}" stroke="#333" stroke-width="1" />
                <text x="300" y="225" font-family="Arial" font-size="10" text-anchor="middle">T</text>
                
                <!-- Lymphocyte B -->
                <circle cx="350" cy="225" r="18" fill="#f8f9fa" stroke="#333" stroke-width="2" />
                <circle cx="350" cy="225" r="14" fill="{immune_color}" stroke="#333" stroke-width="1" />
                <text x="350" y="225" font-family="Arial" font-size="10" text-anchor="middle">B</text>
                
                <!-- Zone inflammation -->
                <ellipse cx="450" cy="250" rx="80" ry="60" fill="#ff6b6b" fill-opacity="0.3" stroke="#ff6b6b" stroke-width="2" />
                <text x="450" y="250" font-family="Arial" font-size="14" text-anchor="middle">Zone d'inflammation</text>
                
                <!-- Médiation inflammatoire -->
                <path d="M400,225 Q420,260 450,250" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,3" />
                <path d="M350,225 Q400,280 450,250" stroke="#ff6b6b" stroke-width="2" fill="none" stroke-dasharray="5,3" />
                
                <!-- Ganglions lymphatiques -->
                <ellipse cx="250" cy="150" rx="40" ry="25" fill="#d8f3dc" stroke="#333" stroke-width="2" />
                <text x="250" y="155" font-family="Arial" font-size="12" text-anchor="middle">Ganglion lymphatique</text>
                
                <!-- Rate -->
                <ellipse cx="400" cy="120" rx="50" ry="35" fill="#d8f3dc" stroke="#333" stroke-width="2" />
                <text x="400" y="125" font-family="Arial" font-size="12" text-anchor="middle">Rate</text>
                
                <!-- Cytokines -->
                <circle cx="420" cy="240" r="8" fill="#ff9e7d" stroke="#333" stroke-width="1" />
                <text x="420" y="240" font-family="Arial" font-size="8" text-anchor="middle">IL</text>
                
                <circle cx="440" cy="270" r="8" fill="#ff9e7d" stroke="#333" stroke-width="1" />
                <text x="440" y="270" font-family="Arial" font-size="8" text-anchor="middle">TNF</text>
                
                <circle cx="470" cy="260" r="8" fill="#ff9e7d" stroke="#333" stroke-width="1" />
                <text x="470" y="260" font-family="Arial" font-size="8" text-anchor="middle">IL</text>
                
                <!-- Médicament anti-inflammatoire -->
                <circle cx="500" cy="300" r="20" fill="#2a9d8f" stroke="#333" stroke-width="2" />
                <text x="500" y="300" font-family="Arial" font-size="10" text-anchor="middle" fill="white">Anti-inf</text>
                
                <!-- Flèche d'effet -->
                <path d="M490,285 Q480,270 470,270" stroke="#2a9d8f" stroke-width="2" fill="none" />
                
                <!-- Légende -->
                <rect x="160" y="320" width="280" height="100" rx="10" ry="10" fill="white" stroke="#333" stroke-width="1" />
                <text x="300" y="340" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">
                    État du système immunitaire
                </text>
                
                <text x="180" y="370" font-family="Arial" font-size="14" text-anchor="left">
                    • Fonction immunitaire: {twin.params['immune_response']:.1f}
                </text>
                <text x="180" y="395" font-family="Arial" font-size="14" text-anchor="left">
                    • Charge inflammatoire: {twin.metrics.get('inflammation_burden', 0):.1f}
                </text>
            </svg>
        </div>
        """
        
        components.html(immune_svg_html, height=450)


def calculate_organ_impact(twin, organ_type):
    """
    Calcule l'impact sur un organe spécifique en fonction des paramètres du patient
    et de l'historique de la simulation. Échelle de 0 à 10 (0 = aucun impact, 10 = impact maximal)
    """
    if organ_type == "heart":
        # Impact cardiovasculaire basé sur la variabilité cardiaque et l'inflammation
        hr_var = twin.metrics.get('hr_variability', 0)
        bp_var = twin.metrics.get('bp_variability', 0)
        inflammation = np.mean(twin.history['inflammation'])
        
        # Calcul normalisé pour obtenir une échelle de 0 à 10
        hr_factor = min(10, max(0, hr_var / 3))
        bp_factor = min(10, max(0, bp_var / 5))
        inflammation_factor = min(10, max(0, inflammation / 20))
        
        # Impact combiné
        impact = (hr_factor * 0.3 + bp_factor * 0.3 + inflammation_factor * 0.4)
        return impact
    
    elif organ_type == "pancreas":
        # Impact sur le pancréas basé sur la glycémie et la variabilité
        glucose_mean = twin.metrics.get('glucose_mean', 0)
        glucose_var = twin.metrics.get('glucose_variability', 0)
        
        # Facteurs normalisés
        high_glucose_factor = min(10, max(0, (glucose_mean - 100) / 15))
        var_factor = min(10, max(0, glucose_var / 10))
        
        # Impact combiné
        impact = (high_glucose_factor * 0.7 + var_factor * 0.3)
        return impact
    
    elif organ_type == "kidney":
        # Impact sur les reins basé sur la fonction rénale, médicaments et inflammation
        renal_function = twin.params.get('renal_function', 1.0)
        drug_exposure = twin.metrics.get('drug_exposure', 0)
        inflammation = np.mean(twin.history['inflammation'])
        
        # Facteurs normalisés
        renal_factor = min(10, max(0, (1 - renal_function) * 10))
        drug_factor = min(10, max(0, drug_exposure / 100))
        inflammation_factor = min(10, max(0, inflammation / 20))
        
        # Impact combiné
        impact = (renal_factor * 0.5 + drug_factor * 0.3 + inflammation_factor * 0.2)
        return impact
    
    elif organ_type == "liver":
        # Impact sur le foie basé sur la fonction hépatique, médicaments et inflammation
        liver_function = twin.params.get('liver_function', 1.0)
        drug_exposure = twin.metrics.get('drug_exposure', 0)
        drug_tissue = np.mean(twin.history['drug_tissue'])
        
        # Facteurs normalisés
        liver_factor = min(10, max(0, (1 - liver_function) * 10))
        drug_factor = min(10, max(0, drug_exposure / 100))
        tissue_factor = min(10, max(0, drug_tissue / 10))
        
        # Impact combiné
        impact = (liver_factor * 0.4 + drug_factor * 0.3 + tissue_factor * 0.3)
        return impact
    
    elif organ_type == "immune":
        # Impact sur le système immunitaire basé sur l'inflammation et la réponse immunitaire
        inflammation = np.mean(twin.history['inflammation'])
        immune_response = twin.params.get('immune_response', 1.0)
        inf_burden = twin.metrics.get('inflammation_burden', 0)
        
        # Facteurs normalisés
        inflammation_factor = min(10, max(0, inflammation / 20))
        response_factor = min(10, max(0, (immune_response - 0.5) * 10))
        burden_factor = min(10, max(0, inf_burden / 300))
        
        # Impact combiné
        impact = (inflammation_factor * 0.4 + response_factor * 0.3 + burden_factor * 0.3)
        return impact
    
    else:
        # Par défaut, retourner un impact moyen
        return 5.0


def get_impact_color(impact_level):
    """
    Retourne une couleur RGB basée sur le niveau d'impact (échelle 0-10)
    0 = vert (sain), 10 = rouge (très affecté)
    """
    # Normaliser l'impact entre 0 et 1
    normalized = max(0, min(1, impact_level / 10))
    
    # Calcul de la couleur RGB
    if normalized < 0.5:
        # Vert à jaune (pour impact faible à modéré)
        r = int(255 * (normalized * 2))
        g = 200
        b = int(100 * (1 - normalized * 2))
    else:
        # Jaune à rouge (pour impact modéré à élevé)
        r = 255
        g = int(200 * (1 - (normalized - 0.5) * 2))
        b = 0
    
    # Retourner la couleur au format hexadécimal
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    """
    Fonction principale pour l'application Streamlit modernisée
    """
    # Configuration de la page
    st.set_page_config(
        page_title="BIOSIM",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Charger et définir le CSS personnalisé pour moderniser l'interface
    st.markdown("""
    <style>
    /* Styles généraux */
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0066cc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .interaction-alert {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    .patient-info {
        padding: 0.5rem 1rem;
        background-color: #e9f7fe;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .tabs-container {
        margin-top: 1rem;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f2ff;
        font-weight: 600;
    }
    .med-icon {
        font-size: 1.2rem;
        margin-right: 0.2rem;
    }
    .simulation-button {
        text-align: center;
        margin: 20px 0;
    }
    .chart-container {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .intervention-tag {
        display: inline-block;
        padding: 3px 8px;
        background-color: #e6f2ff;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 8px;
        margin-bottom: 8px;
        color: #0066cc;
    }
    
    /* Style pour l'en-tête de l'application */
    .app-header {
        background: linear-gradient(90deg, #12436d 0%, #0066cc 100%);
        padding: 1.5rem;
        color: white;
        border-radius: 0 0 10px 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Style pour la barre latérale */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #e6f2ff;
    }
    
    /* Style pour les boutons */
    .primary-button {
        background-color: #0066cc;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .primary-button:hover {
        background-color: #0052a3;
    }
    .secondary-button {
        background-color: #6c757d;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .secondary-button:hover {
        background-color: #5a6268;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Vérifier si l'utilisateur est connecté
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Afficher la page de connexion si l'utilisateur n'est pas connecté
    if not st.session_state.logged_in:
        login_page()
    else:
        # Initialiser UserManager pour l'utilisateur connecté
        user_manager = UserManager()
        
        # En-tête de l'application avec bannière
        st.markdown("""
        <div class="app-header">
            <h1 class="app-title">🩺 BIOSIM</h1>
            <p class="app-subtitle">Simulez et visualisez l'évolution personnalisée de vos patients</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Barre de navigation principale dans la sidebar avec l'option de déconnexion
        st.sidebar.markdown("""
        <div class="sidebar-header">
            <h2>Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        nav_option = st.sidebar.radio(
            "",
            ["👥 Gestion des patients", "🩺 Simulation clinique", "📈 Historique des simulations"]
        )
        
        # Informations utilisateur dans la sidebar
        st.sidebar.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <p style="margin: 0; color: #2c3e50;">
                <strong>👤 Utilisateur:</strong> {st.session_state.username}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton de déconnexion
        if st.sidebar.button("🚪 Déconnexion", type="primary"):
            st.session_state.logged_in = False
            if 'current_patient' in st.session_state:
                del st.session_state.current_patient
            st.rerun()
        
        # Initialisation des variables de session si nécessaire
        if 'mode_tab_index' not in st.session_state:
            st.session_state.mode_tab_index = 0  # Index de l'onglet par défaut
            
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
            
        # Afficher les pages en fonction de la navigation
        if nav_option == "👥 Gestion des patients":
            patient_management_page(user_manager)
        
        elif nav_option == "🩺 Simulation clinique":
            # Vérifier si un patient a été sélectionné
            if 'current_patient' in st.session_state:
                patient = st.session_state.current_patient
                st.markdown(f"<h2 style='color: #2c3e50;'>🩺 Simulation pour {patient['name']}</h2>", unsafe_allow_html=True)
                
                # Description du patient
                st.markdown(f"""
                <div style="background-color: #e6f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center;">
                        <div style="font-size: 2.5rem; margin-right: 15px;">👤</div>
                        <div>
                            <div style="font-size: 1.2rem; font-weight: 600; color: #0066cc;">{patient['name']}</div>
                            <div style="color: #4682B4;">
                                {patient['profile_data'].get('age', 'N/A')} ans • 
                                {patient['profile_data'].get('sex', 'N/A')} • 
                                {patient['profile_data'].get('weight', 'N/A')} kg • 
                                Profil: {patient['profile_data'].get('profile_type', 'Personnalisé')}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Onglets de navigation modernisés
                mode_tabs = st.tabs([
                    "💊 Simulation simple", 
                    "⚖️ Mode Comparaison",
                    "🔬 Visualisation Anatomique"
                ])
                
                # Mettre à jour l'index de l'onglet si nécessaire
                if 'mode_tab_index' in st.session_state:
                    # mode_tabs[st.session_state.mode_tab_index].tab()
                    # Réinitialiser le changement d'onglet
                    st.session_state.mode_tab_index = 0

                with mode_tabs[0]:
                    simple_mode(initial_params=patient['profile_data'])

                with mode_tabs[1]:
                    st.session_state.comparison_mode = True
                    comparison_mode()

                with mode_tabs[2]:
                    # Visualisation anatomique
                    anatomical_visualization_tab(st.session_state.twin_a if st.session_state.has_results_a else None)
                
            else:
                st.info("Veuillez sélectionner un patient dans la section 'Gestion des patients' pour commencer une simulation.")
                
                # Ajouter un bouton pour rediriger vers la gestion des patients
                if st.button("🔙 Aller à la gestion des patients", type="primary"):
                    st.session_state.nav_option = "👥 Gestion des patients"
                    st.rerun()
                
                # Afficher un aperçu des fonctionnalités
                st.markdown("<h3 style='color: #2c3e50;'>Fonctionnalités de simulation disponibles</h3>", unsafe_allow_html=True)
                
                feature_cols = st.columns(3)
                
                with feature_cols[0]:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 250px;">
                        <div style="font-size: 2rem; color: #0066cc; margin-bottom: 10px;">💊</div>
                        <h4 style="color: #2c3e50;">Simulation médicamenteuse</h4>
                        <p>Simulez l'impact de différents médicaments et leurs interactions sur les paramètres physiologiques du patient.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with feature_cols[1]:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 250px;">
                        <div style="font-size: 2rem; color: #0066cc; margin-bottom: 10px;">⚖️</div>
                        <h4 style="color: #2c3e50;">Comparaison de traitements</h4>
                        <p>Comparez différentes approches thérapeutiques côte à côte pour identifier la stratégie optimale pour votre patient.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with feature_cols[2]:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 250px;">
                        <div style="font-size: 2rem; color: #0066cc; margin-bottom: 10px;">🔬</div>
                        <h4 style="color: #2c3e50;">Visualisation anatomique</h4>
                        <p>Visualisez les effets des médicaments sur les différents organes et systèmes physiologiques du patient.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif nav_option == "📈 Historique des simulations":
            st.markdown("<h2 style='color: #2c3e50;'>📈 Historique des simulations</h2>", unsafe_allow_html=True)
            
            # Vérifier si on a un patient sélectionné
            if 'current_patient' in st.session_state:
                patient = st.session_state.current_patient
                
                # Récupérer l'historique des simulations pour ce patient
                user_manager = UserManager()
                simulations = user_manager.get_user_simulations(
                    st.session_state.user_id,
                    patient['id']
                )
                
                if simulations:
                    st.markdown(f"""
                    <div style="background-color: #e6f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                        <p style="margin: 0; color: #0066cc;">
                            <strong>📋 Simulations pour {patient['name']}:</strong> {len(simulations)} simulations enregistrées
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, sim in enumerate(simulations):
                        # Extraire les données de la simulation
                        sim_data = sim['simulation_data']
                        created_at = sim['created_at']
                        
                        # Vérifier si c'est une simulation simple ou une comparaison
                        is_comparison = False
                        if 'twin_a_data' in sim_data and 'twin_b_data' in sim_data:
                            is_comparison = True
                            sim_title = f"Comparaison de traitements ({sim_data.get('comparison_timestamp', created_at)})"
                            twin_a_data = json.loads(sim_data['twin_a_data'])
                            twin_b_data = json.loads(sim_data['twin_b_data'])
                            health_diff = sim_data.get('health_diff', 0)
                            recommendation = sim_data.get('recommendation', 'Non déterminé')
                        else:
                            sim_title = f"Simulation ({sim_data.get('timestamp', created_at)})"
                            twin_data = json.loads(sim_data.get('twin_data', '{}'))
                        
                        # Créer un expander pour chaque simulation
                        with st.expander(sim_title):
                            # Affichage différent selon le type de simulation
                            if is_comparison:
                                # Afficher un résumé de la comparaison
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"### Scénario A")
                                    try:
                                        twin_a_metrics = twin_a_data.get('metrics', {})
                                        st.markdown(f"""
                                        - **Score de santé**: {twin_a_metrics.get('health_score', 'N/A'):.1f}/100
                                        - **Glycémie moyenne**: {twin_a_metrics.get('glucose_mean', 'N/A'):.1f} mg/dL
                                        - **Temps en cible**: {twin_a_metrics.get('percent_in_range', 'N/A'):.1f}%
                                        """)
                                    except:
                                        st.error("Erreur lors de l'affichage des données du scénario A")
                                
                                with col2:
                                    st.markdown(f"### Scénario B")
                                    try:
                                        twin_b_metrics = twin_b_data.get('metrics', {})
                                        st.markdown(f"""
                                        - **Score de santé**: {twin_b_metrics.get('health_score', 'N/A'):.1f}/100
                                        - **Glycémie moyenne**: {twin_b_metrics.get('glucose_mean', 'N/A'):.1f} mg/dL
                                        - **Temps en cible**: {twin_b_metrics.get('percent_in_range', 'N/A'):.1f}%
                                        """)
                                    except:
                                        st.error("Erreur lors de l'affichage des données du scénario B")
                                
                                # Afficher la recommandation
                                rec_color = "green" if recommendation == "Scénario B" else ("green" if recommendation == "Scénario A" else "#6c757d")
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 15px; text-align: center;">
                                    <h4 style="margin-top: 0; color: #2c3e50;">Recommandation</h4>
                                    <div style="font-size: 18px; font-weight: bold; color: {rec_color};">
                                        {recommendation}
                                    </div>
                                    <div style="font-size: 14px; color: #666; margin-top: 5px;">
                                        Différence de score: {abs(health_diff):.1f} points
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Bouton pour recharger cette comparaison
                                if st.button(f"🔄 Recharger cette comparaison", key=f"reload_comp_{i}"):
                                    try:
                                        # Recharger les jumeaux numériques
                                        twin_a = PatientDigitalTwin.from_json(sim_data['twin_a_data'])
                                        twin_b = PatientDigitalTwin.from_json(sim_data['twin_b_data'])
                                        
                                        # Stocker dans la session
                                        st.session_state.twin_a = twin_a
                                        st.session_state.twin_b = twin_b
                                        st.session_state.has_results_a = True
                                        st.session_state.has_results_b = True
                                        
                                        # Stocker les scénarios
                                        st.session_state.scenario_a = {
                                            'twin': twin_a,
                                            'timestamp': sim_data.get('comparison_timestamp', created_at)
                                        }
                                        
                                        st.session_state.scenario_b = {
                                            'twin': twin_b,
                                            'timestamp': sim_data.get('comparison_timestamp', created_at)
                                        }
                                        
                                        # Rediriger vers la page de comparaison
                                        st.session_state.nav_option = "🩺 Simulation clinique"
                                        st.session_state.mode_tab_index = 1  # Onglet comparaison
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Erreur lors du chargement de la comparaison: {str(e)}")
                            else:
                                # Afficher un résumé de la simulation simple
                                try:
                                    metrics = twin_data.get('metrics', {})
                                    
                                    # Colonnes pour les métriques principales
                                    metric_cols = st.columns(4)
                                    
                                    with metric_cols[0]:
                                        health_score = metrics.get('health_score', 0)
                                        score_color = "#28a745" if health_score > 80 else ("#ffc107" if health_score > 60 else "#dc3545")
                                        
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value" style="color: {score_color};">{health_score:.1f}<small>/100</small></div>
                                            <div class="metric-label">Score de Santé</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with metric_cols[1]:
                                        glucose_mean = metrics.get('glucose_mean', 0)
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{glucose_mean:.1f}</div>
                                            <div class="metric-label">Glycémie moyenne</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with metric_cols[2]:
                                        pct_in_range = metrics.get('percent_in_range', 0)
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{pct_in_range:.1f}<small>%</small></div>
                                            <div class="metric-label">Temps en cible</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with metric_cols[3]:
                                        inflammation = metrics.get('inflammation_burden', 0)
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{inflammation:.1f}</div>
                                            <div class="metric-label">Charge inflammatoire</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Bouton pour recharger cette simulation
                                    if st.button(f"🔄 Recharger cette simulation", key=f"reload_sim_{i}"):
                                        try:
                                            # Recharger le jumeau numérique
                                            twin = PatientDigitalTwin.from_json(sim_data['twin_data'])
                                            
                                            # Stocker dans la session
                                            st.session_state.twin_a = twin
                                            st.session_state.has_results_a = True
                                            
                                            # Stocker le scénario
                                            st.session_state.scenario_a = {
                                                'twin': twin,
                                                'timestamp': sim_data.get('timestamp', created_at)
                                            }
                                            
                                            # Rediriger vers la page de simulation
                                            st.session_state.nav_option = "🩺 Simulation clinique"
                                            st.session_state.mode_tab_index = 0  # Onglet simulation simple
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Erreur lors du chargement de la simulation: {str(e)}")
                                
                                except Exception as e:
                                    st.error(f"Erreur lors de l'affichage des résultats: {str(e)}")
                else:
                    st.info(f"Aucune simulation n'a été sauvegardée pour {patient['name']}. Réalisez des simulations et sauvegardez-les pour les retrouver ici.")
            else:
                # Message si aucun patient n'est sélectionné
                st.info("Veuillez sélectionner un patient pour voir son historique de simulations.")
                
                # Ajouter un bouton pour rediriger vers la gestion des patients
                if st.button("🔙 Aller à la gestion des patients", type="primary"):
                    st.session_state.nav_option = "👥 Gestion des patients"
                    st.rerun()
                
                # Afficher un résumé des fonctionnalités de l'historique
                st.markdown("""
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-top: 20px;">
                    <h3 style="color: #2c3e50; margin-top: 0;">📊 Fonctionnalités de l'historique</h3>
                    <ul>
                        <li><strong>Accès aux simulations passées</strong> - Consultez les résultats des simulations précédentes</li>
                        <li><strong>Réutilisation des scénarios</strong> - Rechargez des simulations antérieures pour les modifier</li>
                        <li><strong>Suivi de l'évolution</strong> - Observez les changements dans les métriques du patient au fil du temps</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Pied de page avec date et heure
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        st.markdown(f"BIOSIM - Mohamed_DIOP & Saliou_GUEYE © {datetime.now().year} | Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Lancement de l'application si le script est exécuté directement
if __name__ == "__main__":
    main()