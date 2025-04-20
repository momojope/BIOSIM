"""
Module de gestion des utilisateurs pour l'application de Jumeau Numérique Clinique
"""

import streamlit as st
import sqlite3
import hashlib
import uuid
from datetime import datetime
import json
from shared_data import predefined_profiles

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
                ''', (user_id, patient_id))
            else:
                cursor.execute('''
                SELECT id, patient_id, simulation_data, created_at 
                FROM simulations 
                WHERE user_id = ?
                ''', (user_id,))
            
            simulations = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [{
                'id': s[0], 
                'patient_id': s[1], 
                'simulation_data': json.loads(s[2]) if s[2] else {},
                'created_at': s[3]
            } for s in simulations]

def login_page():
    """
    Streamlit login page
    """
    st.title("📋 Jumeau Numérique Clinique - Connexion")
    
    # Initialize UserManager
    user_manager = UserManager()
    
    # Tabs for login and registration
    tab1, tab2 = st.tabs(["🔐 Connexion", "📝 Inscription"])
    
    with tab1:
        st.markdown("### Connectez-vous")
        username = st.text_input("Nom d'utilisateur", key="login_username")
        password = st.text_input("Mot de passe", type="password", key="login_password")
        
        if st.button("Se connecter"):
            success, result = user_manager.login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_id = result
                st.session_state.username = username
                st.rerun()
            else:
                st.error(result)
    
    with tab2:
        st.markdown("### Créez un compte")
        new_username = st.text_input("Nom d'utilisateur", key="register_username")
        email = st.text_input("Email", key="register_email")
        new_password = st.text_input("Mot de passe", type="password", key="register_password")
        confirm_password = st.text_input("Confirmer le mot de passe", type="password", key="register_confirm_password")
        
        if st.button("S'inscrire"):
            # Validation
            if new_password != confirm_password:
                st.error("Les mots de passe ne correspondent pas")
            elif len(new_password) < 8:
                st.error("Le mot de passe doit contenir au moins 8 caractères")
            else:
                success, result = user_manager.register_user(new_username, email, new_password)
                if success:
                    st.success("Inscription réussie! Vous pouvez maintenant vous connecter.")
                else:
                    st.error(result)

def patient_management_page(user_manager, predefined_profiles=None):
    """
    Streamlit page for patient management
    """
    st.title(f"👥 Gestion des Patients - {st.session_state.username}")
    
    # Add new patient section
    st.markdown("## 🆕 Ajouter un Patient")
    
    # Use predefined patient profiles from the main script
    profile_options = ["Personnalisé"] + [profile['name'] for profile in predefined_profiles.values()]
    
    # Patient name
    patient_name = st.text_input("Nom du patient")
    
    # Select patient profile
    selected_profile = st.selectbox("Profil du patient", profile_options)
    
    if predefined_profiles is None:
        predefined_profiles = {}
    # Get profile parameters
    initial_params = {}
    if selected_profile != "Personnalisé":
        # Find the selected profile
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
    
    # Patient parameters inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Âge", 18, 90, initial_params.get('age', 50))
        weight = st.slider("Poids (kg)", 40, 150, initial_params.get('weight', 70))
    
    with col2:
        sex = st.selectbox("Sexe", ["M", "F"], 0 if initial_params.get('sex', 'M') == 'M' else 1)
        baseline_glucose = st.slider("Glycémie initiale (mg/dL)", 70, 300, 
                                   initial_params.get('baseline_glucose', 140))
    
    # Additional metabolic parameters
    col3, col4 = st.columns(2)
    with col3:
        insulin_sensitivity = st.slider("Sensibilité à l'insuline", 0.1, 1.0, 
                                      initial_params.get('insulin_sensitivity', 0.5), 0.1)
    with col4:
        renal_function = st.slider("Fonction rénale", 0.1, 1.0, 
                                 initial_params.get('renal_function', 0.9), 0.1)
    
    # Prepare patient profile data
    patient_profile = {
        'age': age,
        'weight': weight,
        'sex': sex,
        'baseline_glucose': baseline_glucose,
        'insulin_sensitivity': insulin_sensitivity,
        'renal_function': renal_function,
        'profile_type': selected_profile
    }
    
    # Add patient button
    if st.button("💾 Enregistrer le Patient"):
        if not patient_name:
            st.error("Veuillez saisir un nom pour le patient")
        else:
            # Add patient using UserManager
            success, patient_id = user_manager.add_patient(
                st.session_state.user_id, 
                patient_name, 
                patient_profile
            )
            
            if success:
                st.success(f"Patient {patient_name} ajouté avec succès!")
                # Redirection automatique vers la simulation
                if 'nav_option' in st.session_state:
                    st.session_state.nav_option = "🩺 Simulation"
                    # Charger le patient
                    patient = {
                        'id': patient_id,
                        'name': patient_name,
                        'profile_data': patient_profile
                    }
                    st.session_state.current_patient = patient
                    st.rerun()
            else:
                st.error(f"Erreur lors de l'ajout du patient : {patient_id}")
    
    # Display existing patients
    st.markdown("## 📋 Mes Patients")
    
    # Fetch and display user's patients
    patients = user_manager.get_user_patients(st.session_state.user_id)
    
    if patients:
        for patient in patients:
            with st.expander(f"👤 {patient['name']} - {patient['profile_data'].get('profile_type', 'Personnalisé')}"):
                # Display patient details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Âge:** {patient['profile_data']['age']} ans")
                    st.markdown(f"**Sexe:** {patient['profile_data']['sex']}")
                    st.markdown(f"**Poids:** {patient['profile_data']['weight']} kg")
                
                with col2:
                    st.markdown(f"**Glycémie initiale:** {patient['profile_data']['baseline_glucose']} mg/dL")
                    st.markdown(f"**Sensibilité insuline:** {patient['profile_data']['insulin_sensitivity']}")
                    st.markdown(f"**Fonction rénale:** {patient['profile_data']['renal_function']}")
                
                # Actions row with buttons
                action_col1, action_col2 = st.columns(2)
                
                # Option to simulate for this patient
                with action_col1:
                    if st.button(f"🩺 Simuler pour {patient['name']}", key=f"simulate_{patient['id']}"):
                        # Update session state and redirect
                        st.session_state.current_patient = patient
                        st.session_state.nav_option = "🩺 Simulation"
                        st.rerun()
                
                # Button to delete the patient
                with action_col2:
                    if st.button(f"🗑️ Supprimer {patient['name']}", key=f"delete_{patient['id']}", 
                                type="secondary", help="Supprimer ce patient et toutes ses simulations"):
                        # Confirmation dialog
                        confirmation = st.checkbox(f"Confirmer la suppression de {patient['name']}?", 
                                                key=f"confirm_delete_{patient['id']}")
                        if confirmation:
                            # Delete patient
                            success, message = user_manager.delete_patient(
                                st.session_state.user_id, 
                                patient['id']
                            )
                            
                            if success:
                                st.success(f"Patient {patient['name']} supprimé avec succès!")
                                # Check if this patient was the current patient
                                if 'current_patient' in st.session_state and st.session_state.current_patient['id'] == patient['id']:
                                    del st.session_state.current_patient
                                st.rerun()
                            else:
                                st.error(f"Erreur lors de la suppression : {message}")
    else:
        st.info("Vous n'avez pas encore de patients. Créez-en un à l'aide du formulaire ci-dessus.")