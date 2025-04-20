import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st
from io import StringIO
import matplotlib.pyplot as plt
import datetime
import json

class ClinicalDataIntegrator:
    """
    Module pour importer, traiter et calibrer le modèle à partir de données cliniques réelles.
    """
    def __init__(self, twin=None):
        """Initialise l'intégrateur avec un jumeau numérique facultatif"""
        self.twin = twin
        self.clinical_data = {}
        self.calibrated_params = {}
        self.mapping = {
            'glucose': 'Glycémie (mg/dL)',
            'insulin': 'Insuline (mU/L)',
            'heart_rate': 'Fréquence cardiaque (bpm)',
            'blood_pressure': 'Pression artérielle (mmHg)',
            'inflammation': 'Marqueurs d\'inflammation',
            'medications': 'Médicaments administrés'
        }
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.comparison_metrics = {}
    
    def load_csv_data(self, file, data_type):
        """
        Charge les données d'un fichier CSV et les prétraite
        
        Parameters:
        -----------
        file : UploadedFile
            Fichier CSV téléchargé via Streamlit
        data_type : str
            Type de données ('glucose', 'insulin', etc.)
            
        Returns:
        --------
        DataFrame : données traitées
        """
        try:
            # Lire le contenu du fichier
            content = StringIO(file.getvalue().decode('utf-8'))
            
            # Essayer différents parsers pour s'adapter aux formats courants
            try:
                df = pd.read_csv(content, parse_dates=True)
            except:
                # Retour au début du fichier et essayer avec un autre séparateur
                content.seek(0)
                df = pd.read_csv(content, sep=';', parse_dates=True)
            
            # Identifier les colonnes de date/heure et de valeur
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'heure' in col.lower()]
            
            # S'il n'y a pas de colonne de date explicite, essayer d'utiliser la première colonne
            if not date_cols and pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                date_cols = [df.columns[0]]
                
            if not date_cols:
                # Dernier recours: supposer que la première colonne est une date
                try:
                    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                    date_cols = [df.columns[0]]
                except:
                    raise ValueError("Impossible de trouver une colonne de date/heure dans le fichier")
            
            # Identifier les colonnes de valeurs (numériques)
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
            value_cols = [col for col in numeric_cols if col not in date_cols]
            
            if not value_cols:
                raise ValueError("Aucune colonne numérique trouvée pour les valeurs")
            
            # Créer un nouveau DataFrame avec seulement les colonnes pertinentes
            processed_df = pd.DataFrame()
            processed_df['datetime'] = pd.to_datetime(df[date_cols[0]])
            
            # Sélectionner la colonne de valeur appropriée ou permettre à l'utilisateur de choisir
            if len(value_cols) == 1:
                value_col = value_cols[0]
            else:
                # Dans un contexte réel, on pourrait demander à l'utilisateur de choisir
                # Pour l'instant, prendre la première colonne numérique
                value_col = value_cols[0]
            
            processed_df['value'] = df[value_col]
            
            # Trier par date
            processed_df = processed_df.sort_values('datetime')
            
            # Ajouter une colonne 'hours' depuis le début des données
            start_time = processed_df['datetime'].min()
            processed_df['hours'] = (processed_df['datetime'] - start_time).dt.total_seconds() / 3600
            
            # Stocker les données traitées
            self.clinical_data[data_type] = processed_df
            
            return processed_df
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier CSV: {str(e)}")
            return None
    
    def load_medication_data(self, file):
        """
        Charge les données sur les médicaments administrés
        
        Parameters:
        -----------
        file : UploadedFile
            Fichier CSV contenant les données sur les médicaments
            
        Returns:
        --------
        list : liste des administrations de médicaments (heure, type, dose)
        """
        try:
            content = StringIO(file.getvalue().decode('utf-8'))
            df = pd.read_csv(content)
            
            # Vérifier les colonnes nécessaires
            required_cols = ['time', 'type', 'dose']
            col_mapping = {}
            
            for req_col in required_cols:
                # Chercher des colonnes correspondantes
                matches = [col for col in df.columns if req_col.lower() in col.lower()]
                if matches:
                    col_mapping[req_col] = matches[0]
                else:
                    # Si pas de correspondance exacte, essayer des synonymes
                    if req_col == 'time':
                        alt_matches = [col for col in df.columns if 'heure' in col.lower() or 'date' in col.lower()]
                    elif req_col == 'type':
                        alt_matches = [col for col in df.columns if 'med' in col.lower() or 'drug' in col.lower() or 'médicament' in col.lower()]
                    elif req_col == 'dose':
                        alt_matches = [col for col in df.columns if 'dosage' in col.lower() or 'quantité' in col.lower() or 'mg' in col.lower()]
                    else:
                        alt_matches = []
                    
                    if alt_matches:
                        col_mapping[req_col] = alt_matches[0]
                    else:
                        raise ValueError(f"Colonne requise non trouvée: {req_col}")
            
            # Créer la liste d'administrations
            medications = []
            for _, row in df.iterrows():
                med_time = float(row[col_mapping['time']])
                med_type = str(row[col_mapping['type']])
                med_dose = float(row[col_mapping['dose']])
                medications.append((med_time, med_type, med_dose))
            
            self.clinical_data['medications'] = medications
            return medications
            
        except Exception as e:
            st.error(f"Erreur lors du chargement des données de médicaments: {str(e)}")
            return []
    
    def calibrate_model(self):
        """
        Calibre les paramètres du modèle pour correspondre aux données cliniques
        
        Returns:
        --------
        bool, DataFrame : Succès de la calibration et tableau des paramètres
        """
        if not self.twin or not self.clinical_data:
            st.error("Aucun jumeau numérique ou données cliniques disponibles pour la calibration")
            return False, None
        
        # Paramètres à calibrer et leurs limites
        param_bounds = {
            'insulin_sensitivity': (0.1, 1.0),
            'glucose_absorption': (0.01, 0.05),
            'insulin_clearance': (0.005, 0.02),
            'hepatic_glucose': (0.5, 1.0),
            'renal_function': (0.3, 1.0),
            'liver_function': (0.3, 1.0),
            'immune_response': (0.3, 1.5),
            'inflammatory_response': (0.1, 1.0)
        }
        
        # Paramètres initiaux (les valeurs actuelles du jumeau)
        initial_params = {param: self.twin.params.get(param, (bounds[0] + bounds[1])/2) 
                         for param, bounds in param_bounds.items()}
        
        # Données cliniques à utiliser pour la calibration
        calibration_data = {}
        for data_type in ['glucose', 'insulin', 'heart_rate', 'blood_pressure']:
            if data_type in self.clinical_data:
                calibration_data[data_type] = self.clinical_data[data_type]
        
        if not calibration_data:
            st.warning("Aucune donnée disponible pour la calibration")
            return False, None
        
        # Fonction objectif à minimiser
        def objective_function(params_array):
            # Convertir l'array en dictionnaire
            params_dict = {}
            for i, param_name in enumerate(param_bounds.keys()):
                params_dict[param_name] = params_array[i]
            
            # Appliquer temporairement ces paramètres au jumeau
            original_params = self.twin.params.copy()
            self.twin.params.update(params_dict)
            
            # Simuler avec ces paramètres
            max_duration = 0
            for data in calibration_data.values():
                if 'hours' in data.columns:
                    max_duration = max(max_duration, data['hours'].max())
            
            if max_duration == 0:
                max_duration = 24  # Valeur par défaut
                
            medications = self.clinical_data.get('medications', [])
            
            # Simuler avec les mêmes médicaments et durée que les données cliniques
            self.twin.simulate(duration=max_duration, medications=medications)
            
            # Calculer l'erreur entre les données simulées et réelles
            total_error = 0
            for data_type, data in calibration_data.items():
                if data_type in self.twin.history and len(self.twin.history[data_type]) > 0:
                    # Interpoler les valeurs simulées aux mêmes points temporels que les données réelles
                    sim_times = self.twin.history['time']
                    sim_values = self.twin.history[data_type]
                    
                    for _, row in data.iterrows():
                        # Trouver la valeur simulée la plus proche en temps
                        hour = row['hours']
                        closest_idx = np.abs(np.array(sim_times) - hour).argmin()
                        sim_value = sim_values[closest_idx]
                        real_value = row['value']
                        
                        # Ajouter à l'erreur quadratique
                        error = ((sim_value - real_value) / real_value) ** 2
                        total_error += error
            
            # Restaurer les paramètres originaux
            self.twin.params = original_params
            
            return total_error
        
        # Convertir les paramètres initiaux et les limites en arrays pour l'optimisation
        x0 = np.array([initial_params[param] for param in param_bounds.keys()])
        bounds = [param_bounds[param] for param in param_bounds.keys()]
        
        # Exécuter l'optimisation
        try:
            with st.spinner("Calibration du modèle en cours..."):
                result = minimize(
                    objective_function, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100}
                )
            
            # Stocker les paramètres calibrés
            self.calibrated_params = {
                param: result.x[i] for i, param in enumerate(param_bounds.keys())
            }
            
            # Afficher les résultats
            st.success("Calibration réussie!")
            
            # Créer un tableau comparatif des paramètres
            params_df = pd.DataFrame({
                'Paramètre': list(param_bounds.keys()),
                'Valeur originale': [initial_params[p] for p in param_bounds.keys()],
                'Valeur calibrée': [self.calibrated_params[p] for p in param_bounds.keys()],
                'Variation (%)': [((self.calibrated_params[p] - initial_params[p]) / initial_params[p] * 100) 
                                 for p in param_bounds.keys()]
            })
            
            return True, params_df
            
        except Exception as e:
            st.error(f"Erreur lors de la calibration: {str(e)}")
            return False, None
    
    def apply_calibration(self):
        """
        Applique les paramètres calibrés au jumeau numérique
        
        Returns:
        --------
        bool : Succès de l'application
        """
        if not self.twin or not self.calibrated_params:
            return False
        
        # Appliquer les paramètres calibrés
        self.twin.params.update(self.calibrated_params)
        return True
    
    def compare_real_vs_simulated(self):
        """
        Compare les données réelles avec les simulations du modèle
        
        Returns:
        --------
        dict, dict : données de comparaison et métriques
        """
        if not self.twin or not self.clinical_data:
            return {}, {}
        
        comparison = {}
        metrics = {}
        
        for data_type, data in self.clinical_data.items():
            if data_type != 'medications' and data_type in self.twin.history:
                # Préparer les données pour la comparaison
                real_data = data
                sim_times = np.array(self.twin.history['time'])
                sim_values = np.array(self.twin.history[data_type])
                
                # Interpoler les valeurs simulées aux mêmes points temporels
                interpolated_sim = []
                for _, row in real_data.iterrows():
                    hour = row['hours']
                    
                    # Trouver l'index le plus proche dans les temps simulés
                    if hour <= sim_times.max():
                        closest_idx = np.abs(sim_times - hour).argmin()
                        sim_value = sim_values[closest_idx]
                    else:
                        # Si le temps réel dépasse la simulation, utiliser la dernière valeur
                        sim_value = sim_values[-1]
                    
                    interpolated_sim.append(sim_value)
                
                # Calculer les métriques
                real_values = real_data['value'].values
                rmse = np.sqrt(np.mean((real_values - interpolated_sim) ** 2))
                mae = np.mean(np.abs(real_values - interpolated_sim))
                mape = np.mean(np.abs((real_values - interpolated_sim) / real_values)) * 100
                
                # Coefficient de corrélation
                correlation = np.corrcoef(real_values, interpolated_sim)[0, 1]
                
                metrics[data_type] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'Correlation': correlation
                }
                
                # Préparer les données pour le graphique
                comparison[data_type] = {
                    'real_time': real_data['hours'].values,
                    'real_values': real_values,
                    'sim_values': interpolated_sim
                }
        
        self.comparison_metrics = metrics
        return comparison, metrics
    
    def plot_comparison(self, data_type):
        """
        Génère un graphique comparant données réelles et simulées
        
        Parameters:
        -----------
        data_type : str
            Type de données à comparer ('glucose', 'insulin', etc.)
            
        Returns:
        --------
        fig : matplotlib figure
        """
        if data_type not in self.clinical_data or data_type not in self.twin.history:
            return None
        
        # Obtenir les données de comparaison
        comparison, _ = self.compare_real_vs_simulated()
        
        if data_type not in comparison:
            return None
        
        comp_data = comparison[data_type]
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracer les données réelles
        ax.scatter(comp_data['real_time'], comp_data['real_values'], 
                  color='blue', alpha=0.6, label='Données réelles')
        
        # Tracer les données simulées
        ax.plot(comp_data['real_time'], comp_data['sim_values'], 
               color='red', linewidth=2, label='Simulation')
        
        # Ajouter les légendes et labels
        ax.set_xlabel('Temps (heures)')
        display_name = self.mapping.get(data_type, data_type.capitalize())
        ax.set_ylabel(display_name)
        ax.set_title(f'Comparaison entre données réelles et simulation: {display_name}')
        ax.legend()
        
        # Ajouter les métriques au graphique
        if data_type in self.comparison_metrics:
            metrics = self.comparison_metrics[data_type]
            metrics_text = (f"RMSE: {metrics['RMSE']:.2f}\n"
                           f"MAE: {metrics['MAE']:.2f}\n"
                           f"MAPE: {metrics['MAPE']:.2f}%\n"
                           f"Corrélation: {metrics['Correlation']:.2f}")
            
            # Ajouter la boîte de texte pour les métriques
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig
    
    def export_calibrated_model(self):
        """
        Exporte le modèle calibré au format JSON
        
        Returns:
        --------
        str : JSON du modèle calibré
        """
        if not self.twin:
            return None
        
        # Créer un dictionnaire avec les données du modèle
        export_data = {
            'model_id': self.twin.id,
            'calibration_date': datetime.datetime.now().isoformat(),
            'params': self.twin.params,
            'calibration_metrics': self.comparison_metrics
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_calibrated_model(self, json_data):
        """
        Importe un modèle calibré à partir de JSON
        
        Parameters:
        -----------
        json_data : str
            Données JSON du modèle calibré
            
        Returns:
        --------
        bool : Succès de l'importation
        """
        try:
            data = json.loads(json_data)
            
            if 'params' not in data:
                return False
            
            # Appliquer les paramètres au jumeau
            if self.twin:
                self.twin.params.update(data['params'])
                self.calibrated_params = data['params']
                
                if 'calibration_metrics' in data:
                    self.comparison_metrics = data['calibration_metrics']
                
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Erreur lors de l'importation du modèle: {str(e)}")
            return False