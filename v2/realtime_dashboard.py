import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
import queue
import datetime

class RealtimeDashboard:
    """
    Module pour la visualisation en temps réel des paramètres du jumeau numérique
    pendant la simulation, avec alertes et chronologie interactive.
    """
    
    def __init__(self, twin=None):
        """Initialise le dashboard avec un jumeau numérique facultatif"""
        self.twin = twin
        self.running = False
        self.update_interval = 0.5  # secondes entre mises à jour
        self.simulation_queue = queue.Queue()
        self.simulation_thread = None
        self.current_time = 0
        
        # Seuils d'alerte par défaut
        self.alert_thresholds = {
            'glucose': {'low': 70, 'high': 180, 'unit': 'mg/dL', 'name': 'Glycémie'},
            'insulin': {'low': 5, 'high': 30, 'unit': 'mU/L', 'name': 'Insuline'},
            'heart_rate': {'low': 50, 'high': 100, 'unit': 'bpm', 'name': 'Fréquence cardiaque'},
            'blood_pressure': {'low': 90, 'high': 140, 'unit': 'mmHg', 'name': 'Pression artérielle'},
            'inflammation': {'low': 0, 'high': 15, 'unit': '', 'name': 'Inflammation'},
            'drug_plasma': {'low': 0, 'high': 15, 'unit': '', 'name': 'Médicament (plasma)'}
        }
        
        # Historique des alertes
        self.alerts_history = []
        
        # Historique des interventions
        self.interventions_history = []
        
        # Stockage des données pour l'affichage
        self.display_data = {
            'time': [],
            'glucose': [],
            'insulin': [],
            'heart_rate': [],
            'blood_pressure': [],
            'inflammation': [],
            'drug_plasma': []
        }
    
    def set_twin(self, twin):
        """Définit le jumeau numérique à surveiller"""
        self.twin = twin
    
    def update_alert_thresholds(self, param_name, low=None, high=None):
        """Met à jour les seuils d'alerte pour un paramètre spécifique"""
        if param_name in self.alert_thresholds:
            if low is not None:
                self.alert_thresholds[param_name]['low'] = low
            if high is not None:
                self.alert_thresholds[param_name]['high'] = high
            return True
        return False
    
    def check_alerts(self, param_name, value, time):
        """
        Vérifie si un paramètre dépasse les seuils d'alerte
        
        Returns:
        --------
        alert_type : None, 'low' ou 'high'
        """
        if param_name not in self.alert_thresholds:
            return None
        
        thresholds = self.alert_thresholds[param_name]
        
        if value < thresholds['low']:
            # Ajouter à l'historique des alertes
            self.alerts_history.append({
                'time': time,
                'param': param_name,
                'value': value,
                'threshold': thresholds['low'],
                'type': 'low',
                'message': f"{thresholds['name']} bas: {value:.1f} {thresholds['unit']} (seuil: {thresholds['low']} {thresholds['unit']})"
            })
            return 'low'
        elif value > thresholds['high']:
            # Ajouter à l'historique des alertes
            self.alerts_history.append({
                'time': time,
                'param': param_name,
                'value': value,
                'threshold': thresholds['high'],
                'type': 'high',
                'message': f"{thresholds['name']} élevé: {value:.1f} {thresholds['unit']} (seuil: {thresholds['high']} {thresholds['unit']})"
            })
            return 'high'
        
        return None
    
    def run_simulation_thread(self, duration, medications, meals):
        """
        Exécute la simulation dans un thread séparé et envoie les 
        mises à jour à la file d'attente
        """
        try:
            # Réinitialiser l'historique de la simulation
            self.twin.history = {
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
                'interactions': []
            }
            
            # Calculer le nombre total d'étapes pour la simulation
            total_steps = int(duration * (1 / self.update_interval))
            step_size = duration / total_steps
            
            # État initial
            y = [
                self.twin.state['glucose'],
                self.twin.state['insulin'],
                self.twin.state['drug_plasma'],
                self.twin.state['drug_tissue'],
                self.twin.state['immune_cells'],
                self.twin.state['inflammation'],
                self.twin.state['heart_rate'],
                self.twin.state['blood_pressure']
            ]
            
            # Simulation pas à pas
            for step in range(total_steps + 1):
                if not self.running:
                    break
                
                t = step * step_size
                self.current_time = t
                
                # Vérifier les interventions (médicaments et repas) à ce moment
                active_medications = []
                meal_value = 0
                
                # Vérifier si un médicament est administré à ce moment
                for med_time, med_type, med_dose in medications:
                    if abs(t - med_time) < 0.1:  # Dans un intervalle de 6 minutes
                        active_medications.append({
                            'type': med_type,
                            'dose': med_dose
                        })
                        self.twin.history['interventions'].append((t, f"Médicament: {med_type} - {med_dose} mg"))
                        
                        # Ajouter à l'historique des interventions pour le dashboard
                        self.interventions_history.append({
                            'time': t,
                            'type': 'medication',
                            'details': f"{med_type} - {med_dose} mg",
                            'impact': 'En cours d\'évaluation...'
                        })
                
                # Vérifier si un repas est pris à ce moment
                for meal_time, meal_carbs in meals:
                    if abs(t - meal_time) < 0.1:  # Dans un intervalle de 6 minutes
                        meal_value += meal_carbs
                        self.twin.history['interventions'].append((t, f"Repas: {meal_carbs} g"))
                        
                        # Ajouter à l'historique des interventions pour le dashboard
                        self.interventions_history.append({
                            'time': t,
                            'type': 'meal',
                            'details': f"{meal_carbs} g de glucides",
                            'impact': 'En cours d\'évaluation...'
                        })
                
                # Calculer les dérivées avec le modèle
                dy = self.twin.pk_pd_model(t, y, active_medications, meal_value)
                
                # Mise à jour de l'état avec la méthode d'Euler
                y = [y[i] + dy[i] * step_size for i in range(len(y))]
                
                # Enregistrer les résultats
                self.twin.history['time'].append(t)
                self.twin.history['glucose'].append(y[0])
                self.twin.history['insulin'].append(y[1])
                self.twin.history['drug_plasma'].append(y[2])
                self.twin.history['drug_tissue'].append(y[3])
                self.twin.history['immune_cells'].append(y[4])
                self.twin.history['inflammation'].append(y[5])
                self.twin.history['heart_rate'].append(y[6])
                self.twin.history['blood_pressure'].append(y[7])
                
                # Vérifier les alertes pour chaque paramètre
                for param_idx, param_name in enumerate(['glucose', 'insulin', 'drug_plasma', 'drug_tissue', 
                                                      'immune_cells', 'inflammation', 'heart_rate', 'blood_pressure']):
                    if param_name in self.alert_thresholds:
                        self.check_alerts(param_name, y[param_idx], t)
                
                # Mise à jour des données d'affichage
                self.display_data['time'].append(t)
                self.display_data['glucose'].append(y[0])
                self.display_data['insulin'].append(y[1])
                self.display_data['heart_rate'].append(y[6])
                self.display_data['blood_pressure'].append(y[7])
                self.display_data['inflammation'].append(y[5])
                self.display_data['drug_plasma'].append(y[2])
                
                # Envoyer les données mises à jour à la file d'attente
                self.simulation_queue.put({
                    'time': t,
                    'state': y.copy(),
                    'progress': step / total_steps
                })
                
                # Attendre l'intervalle de mise à jour
                time.sleep(self.update_interval)
            
            # Mise à jour de l'état final du jumeau
            self.twin.state['glucose'] = y[0]
            self.twin.state['insulin'] = y[1]
            self.twin.state['drug_plasma'] = y[2]
            self.twin.state['drug_tissue'] = y[3]
            self.twin.state['immune_cells'] = y[4]
            self.twin.state['inflammation'] = y[5]
            self.twin.state['heart_rate'] = y[6]
            self.twin.state['blood_pressure'] = y[7]
            
            # Calculer les métriques de la simulation
            self.twin.calculate_metrics()
            
            # Mise à jour des impacts des interventions
            for intervention in self.interventions_history:
                # Évaluer l'impact des interventions une fois la simulation terminée
                if intervention['type'] == 'medication':
                    med_type = intervention['details'].split(' - ')[0]
                    time_point = intervention['time']
                    
                    # Trouver l'index de temps le plus proche
                    time_idx = min(range(len(self.twin.history['time'])), 
                                  key=lambda i: abs(self.twin.history['time'][i] - time_point))
                    
                    # Évaluer l'effet 1h après l'intervention
                    effect_idx = min(time_idx + int(1 / step_size), len(self.twin.history['time']) - 1)
                    
                    if med_type in ['antidiabetic']:
                        glucose_change = self.twin.history['glucose'][effect_idx] - self.twin.history['glucose'][time_idx]
                        impact = f"Changement de glycémie: {glucose_change:.1f} mg/dL"
                    elif med_type in ['antiinflammatory']:
                        inflam_change = self.twin.history['inflammation'][effect_idx] - self.twin.history['inflammation'][time_idx]
                        impact = f"Changement d'inflammation: {inflam_change:.1f}"
                    elif med_type in ['beta_blocker', 'vasodilator']:
                        hr_change = self.twin.history['heart_rate'][effect_idx] - self.twin.history['heart_rate'][time_idx]
                        bp_change = self.twin.history['blood_pressure'][effect_idx] - self.twin.history['blood_pressure'][time_idx]
                        impact = f"FC: {hr_change:.1f} bpm, PA: {bp_change:.1f} mmHg"
                    else:
                        impact = "Impact indéterminé"
                    
                    intervention['impact'] = impact
                
                elif intervention['type'] == 'meal':
                    time_point = intervention['time']
                    
                    # Trouver l'index de temps le plus proche
                    time_idx = min(range(len(self.twin.history['time'])), 
                                  key=lambda i: abs(self.twin.history['time'][i] - time_point))
                    
                    # Évaluer l'effet 2h après le repas
                    effect_idx = min(time_idx + int(2 / step_size), len(self.twin.history['time']) - 1)
                    
                    glucose_change = self.twin.history['glucose'][effect_idx] - self.twin.history['glucose'][time_idx]
                    impact = f"Pic glycémique: {glucose_change:.1f} mg/dL"
                    intervention['impact'] = impact
            
            # Envoyer un signal de fin à la file d'attente
            self.simulation_queue.put({'finished': True})
            
        except Exception as e:
            # Gérer les erreurs
            self.simulation_queue.put({'error': str(e)})
        
        finally:
            self.running = False
    
    def start_simulation(self, duration, medications, meals):
        """
        Démarre la simulation en temps réel
        
        Parameters:
        -----------
        duration : float
            Durée de la simulation en heures
        medications : list
            Liste des médicaments (heure, type, dose)
        meals : list
            Liste des repas (heure, glucides)
            
        Returns:
        --------
        bool : Succès du démarrage
        """
        if not self.twin:
            return False
        
        if self.running:
            self.stop_simulation()
        
        # Réinitialiser les données d'affichage
        self.display_data = {
            'time': [],
            'glucose': [],
            'insulin': [],
            'heart_rate': [],
            'blood_pressure': [],
            'inflammation': [],
            'drug_plasma': []
        }
        
        # Réinitialiser l'historique des alertes et interventions
        self.alerts_history = []
        self.interventions_history = []
        
        # Démarrer le thread de simulation
        self.running = True
        self.simulation_thread = threading.Thread(
            target=self.run_simulation_thread,
            args=(duration, medications, meals)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return True
    
    def stop_simulation(self):
        """Arrête la simulation en cours"""
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        
        # Vider la file d'attente
        while not self.simulation_queue.empty():
            try:
                self.simulation_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_update(self):
        """
        Récupère la mise à jour la plus récente de la simulation
        
        Returns:
        --------
        dict : Mise à jour de la simulation ou None si pas de mise à jour
        """
        try:
            update = self.simulation_queue.get_nowait()
            return update
        except queue.Empty:
            return None
    
    def create_dashboard(self):
        """
        Crée les composants du dashboard pour Streamlit
        """
        if not self.twin:
            st.error("Aucun jumeau numérique connecté au dashboard")
            return
        
        # Créer une mise en page en 3 colonnes pour les métriques principales
        st.subheader("Paramètres vitaux en temps réel")
        
        # Conteneur pour les métriques
        metrics_container = st.container()
        
        # Conteneur pour les graphiques
        charts_container = st.container()
        
        # Section pour les alertes
        alerts_container = st.container()
        
        # Section pour la chronologie des interventions
        timeline_container = st.container()
        
        # Configuration des seuils d'alerte
        with st.expander("Configuration des seuils d'alerte"):
            st.subheader("Seuils d'alerte")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Glycémie")
                glucose_low = st.number_input("Seuil bas (mg/dL)", 
                                           value=self.alert_thresholds['glucose']['low'],
                                           min_value=50, max_value=90, step=5,
                                           key="glucose_low")
                glucose_high = st.number_input("Seuil haut (mg/dL)", 
                                            value=self.alert_thresholds['glucose']['high'],
                                            min_value=140, max_value=250, step=5,
                                            key="glucose_high")
                self.update_alert_thresholds('glucose', glucose_low, glucose_high)
            
            with col2:
                st.write("Fréquence cardiaque")
                hr_low = st.number_input("Seuil bas (bpm)", 
                                       value=self.alert_thresholds['heart_rate']['low'],
                                       min_value=40, max_value=60, step=5,
                                       key="hr_low")
                hr_high = st.number_input("Seuil haut (bpm)", 
                                        value=self.alert_thresholds['heart_rate']['high'],
                                        min_value=80, max_value=140, step=5,
                                        key="hr_high")
                self.update_alert_thresholds('heart_rate', hr_low, hr_high)
            
            with col3:
                st.write("Pression artérielle")
                bp_low = st.number_input("Seuil bas (mmHg)", 
                                       value=self.alert_thresholds['blood_pressure']['low'],
                                       min_value=70, max_value=100, step=5,
                                       key="bp_low")
                bp_high = st.number_input("Seuil haut (mmHg)", 
                                        value=self.alert_thresholds['blood_pressure']['high'],
                                        min_value=120, max_value=180, step=5,
                                        key="bp_high")
                self.update_alert_thresholds('blood_pressure', bp_low, bp_high)
        
        return {
            'metrics': metrics_container,
            'charts': charts_container,
            'alerts': alerts_container,
            'timeline': timeline_container
        }
    
    def update_dashboard(self, dashboard_components):
        """
        Met à jour les composants du dashboard avec les dernières données
        
        Parameters:
        -----------
        dashboard_components : dict
            Composants du dashboard créés par create_dashboard()
        """
        # Mettre à jour les métriques
        with dashboard_components['metrics']:
            if len(self.display_data['time']) > 0:
                # Obtenir les dernières valeurs
                last_idx = -1
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    glucose_value = self.display_data['glucose'][last_idx]
                    glucose_delta = 0
                    if len(self.display_data['glucose']) > 1:
                        glucose_delta = glucose_value - self.display_data['glucose'][-2]
                    
                    # Définir la couleur en fonction des seuils
                    color = "normal"
                    if glucose_value < self.alert_thresholds['glucose']['low']:
                        color = "inverse"  # Rouge pour bas
                    elif glucose_value > self.alert_thresholds['glucose']['high']:
                        color = "inverse"  # Rouge pour haut
                    
                    st.metric("Glycémie", f"{glucose_value:.1f} mg/dL", 
                             delta=f"{glucose_delta:.1f}", delta_color=color)
                
                with col2:
                    hr_value = self.display_data['heart_rate'][last_idx]
                    hr_delta = 0
                    if len(self.display_data['heart_rate']) > 1:
                        hr_delta = hr_value - self.display_data['heart_rate'][-2]
                    
                    color = "normal"
                    if hr_value < self.alert_thresholds['heart_rate']['low'] or hr_value > self.alert_thresholds['heart_rate']['high']:
                        color = "inverse"
                    
                    st.metric("Fréquence cardiaque", f"{hr_value:.1f} bpm", 
                             delta=f"{hr_delta:.1f}", delta_color=color)
                
                with col3:
                    bp_value = self.display_data['blood_pressure'][last_idx]
                    bp_delta = 0
                    if len(self.display_data['blood_pressure']) > 1:
                        bp_delta = bp_value - self.display_data['blood_pressure'][-2]
                    
                    color = "normal"
                    if bp_value < self.alert_thresholds['blood_pressure']['low'] or bp_value > self.alert_thresholds['blood_pressure']['high']:
                        color = "inverse"
                    
                    st.metric("Pression artérielle", f"{bp_value:.1f} mmHg", 
                             delta=f"{bp_delta:.1f}", delta_color=color)
                
                with col4:
                    infl_value = self.display_data['inflammation'][last_idx]
                    infl_delta = 0
                    if len(self.display_data['inflammation']) > 1:
                        infl_delta = infl_value - self.display_data['inflammation'][-2]
                    
                    color = "normal"
                    if infl_value > self.alert_thresholds['inflammation']['high']:
                        color = "inverse"
                    
                    st.metric("Inflammation", f"{infl_value:.1f}", 
                             delta=f"{infl_delta:.1f}", delta_color=color)
                
                # Afficher l'heure de simulation
                st.text(f"Temps de simulation: {self.current_time:.2f} heures")
        
        # Mettre à jour les graphiques
        with dashboard_components['charts']:
            if len(self.display_data['time']) > 0:
                # Créer un graphique interactif avec Plotly
                fig = make_subplots(rows=2, cols=2, 
                                   subplot_titles=("Glycémie", "Paramètres Cardiovasculaires", 
                                                  "Inflammation", "Concentration Médicamenteuse"))
                
                # Tracer la glycémie avec les seuils
                fig.add_trace(
                    go.Scatter(x=self.display_data['time'], y=self.display_data['glucose'], 
                              name="Glycémie", line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Ajouter les seuils glycémiques
                fig.add_shape(
                    type="line",
                    x0=min(self.display_data['time']),
                    y0=self.alert_thresholds['glucose']['high'],
                    x1=max(self.display_data['time']) if len(self.display_data['time']) > 0 else 1,
                    y1=self.alert_thresholds['glucose']['high'],
                    line=dict(color="red", width=1, dash="dash"),
                    row=1, col=1
                )
                
                fig.add_shape(
                    type="line",
                    x0=min(self.display_data['time']),
                    y0=self.alert_thresholds['glucose']['low'],
                    x1=max(self.display_data['time']) if len(self.display_data['time']) > 0 else 1,
                    y1=self.alert_thresholds['glucose']['low'],
                    line=dict(color="red", width=1, dash="dash"),
                    row=1, col=1
                )
                
                # Tracer les paramètres cardiovasculaires
                fig.add_trace(
                    go.Scatter(x=self.display_data['time'], y=self.display_data['heart_rate'], 
                              name="Fréquence cardiaque", line=dict(color='red')),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=self.display_data['time'], y=self.display_data['blood_pressure'], 
                              name="Pression artérielle", line=dict(color='purple')),
                    row=1, col=2
                )
                
                # Inflammation
                fig.add_trace(
                    go.Scatter(x=self.display_data['time'], y=self.display_data['inflammation'], 
                              name="Inflammation", line=dict(color='orange')),
                    row=2, col=1
                )
                
                # Concentration médicamenteuse
                fig.add_trace(
                    go.Scatter(x=self.display_data['time'], y=self.display_data['drug_plasma'], 
                              name="Médicament (plasma)", line=dict(color='green')),
                    row=2, col=2
                )
                
                # Ajouter les interventions (médicaments et repas) comme des annotations
                for intervention in self.interventions_history:
                    if intervention['time'] <= self.current_time:
                        marker_symbol = "triangle-up" if intervention['type'] == 'medication' else "circle"
                        marker_color = "red" if intervention['type'] == 'medication' else "green"
                        
                        # Déterminer dans quel sous-graphique placer le marqueur
                        row, col = 1, 1  # Par défaut dans le graphique de glycémie
                        if intervention['type'] == 'medication' and 'beta_blocker' in intervention['details'] or 'vasodilator' in intervention['details']:
                            row, col = 1, 2  # Cardiovasculaire
                        elif intervention['type'] == 'medication' and 'antiinflammatory' in intervention['details']:
                            row, col = 2, 1  # Inflammation
                            
                        # Ajouter un marqueur pour l'intervention
                        if intervention['type'] == 'medication':
                            fig.add_trace(
                                go.Scatter(
                                    x=[intervention['time']],
                                    y=[100 if row == 1 and col == 1 else 70],  # Valeur arbitraire pour la visibilité
                                    mode="markers",
                                    marker=dict(symbol=marker_symbol, size=10, color=marker_color),
                                    name=intervention['details'],
                                    text=intervention['details'],
                                    hoverinfo="text"
                                ),
                                row=row, col=col
                            )
                        elif intervention['type'] == 'meal':
                            fig.add_trace(
                                go.Scatter(
                                    x=[intervention['time']],
                                    y=[80],  # Valeur arbitraire pour la visibilité
                                    mode="markers",
                                    marker=dict(symbol=marker_symbol, size=10, color=marker_color),
                                    name=f"Repas: {intervention['details']}",
                                    text=f"Repas: {intervention['details']}",
                                    hoverinfo="text"
                                ),
                                row=1, col=1
                            )
                
                # Configurer la mise en page
                fig.update_layout(
                    height=600,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    showlegend=True
                )
                
                # Configurer les axes
                fig.update_xaxes(title_text="Temps (heures)")
                fig.update_yaxes(title_text="mg/dL", row=1, col=1)
                fig.update_yaxes(title_text="Valeur", row=1, col=2)
                fig.update_yaxes(title_text="Niveau", row=2, col=1)
                fig.update_yaxes(title_text="Concentration", row=2, col=2)
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
        
        # Mettre à jour les alertes
        with dashboard_components['alerts']:
            # Afficher les 5 dernières alertes
            if len(self.alerts_history) > 0:
                st.subheader("Dernières alertes")
                
                # Créer un DataFrame pour les alertes
                alerts_data = []
                for alert in reversed(self.alerts_history[-10:]):
                    alerts_data.append({
                        'Temps': f"{alert['time']:.2f}h",
                        'Paramètre': self.alert_thresholds[alert['param']]['name'],
                        'Valeur': f"{alert['value']:.1f} {self.alert_thresholds[alert['param']]['unit']}",
                        'Type': "Bas" if alert['type'] == 'low' else "Élevé",
                        'Message': alert['message']
                    })
                
                if alerts_data:
                    alerts_df = pd.DataFrame(alerts_data)
                    
                    # Appliquer un style aux alertes
                    def highlight_alerts(row):
                        if row['Type'] == 'Bas':
                            return ['background-color: rgba(255, 150, 150, 0.3)'] * len(row)
                        elif row['Type'] == 'Élevé':
                            return ['background-color: rgba(255, 200, 150, 0.3)'] * len(row)
                        return [''] * len(row)
                    
                    # Afficher le tableau avec style
                    st.dataframe(alerts_df.style.apply(highlight_alerts, axis=1))
                else:
                    st.info("Aucune alerte à afficher")
            else:
                st.info("Aucune alerte détectée")
        
        # Mettre à jour la chronologie des interventions
        with dashboard_components['timeline']:
            if len(self.interventions_history) > 0:
                st.subheader("Chronologie des interventions")
                
                # Créer un DataFrame pour les interventions
                interventions_data = []
                for intervention in self.interventions_history:
                    if intervention['time'] <= self.current_time:
                        interventions_data.append({
                            'Temps': f"{intervention['time']:.2f}h",
                            'Type': "Médicament" if intervention['type'] == 'medication' else "Repas",
                            'Détails': intervention['details'],
                            'Impact': intervention['impact']
                        })
                
                if interventions_data:
                    interventions_df = pd.DataFrame(interventions_data)
                    
                    # Appliquer un style aux interventions
                    def highlight_interventions(row):
                        if row['Type'] == 'Médicament':
                            return ['background-color: rgba(150, 200, 255, 0.3)'] * len(row)
                        else:
                            return ['background-color: rgba(150, 255, 150, 0.3)'] * len(row)
                    
                    # Afficher le tableau avec style
                    st.dataframe(interventions_df.style.apply(highlight_interventions, axis=1))
                else:
                    st.info("Aucune intervention à afficher")
            else:
                st.info("Aucune intervention enregistrée")
    
    def render_timeline_view(self):
        """
        Affiche une vue de chronologie interactive des événements de la simulation
        """
        if not self.twin or len(self.twin.history['time']) == 0:
            st.info("Aucune donnée de simulation disponible pour la chronologie")
            return
        
        st.subheader("Chronologie interactive de la simulation")
        
        # Combiner les interventions et les alertes sur une même chronologie
        timeline_events = []
        
        # Ajouter les interventions
        for intervention in self.interventions_history:
            event_type = "Médicament" if intervention['type'] == 'medication' else "Repas"
            timeline_events.append({
                'time': intervention['time'],
                'type': event_type,
                'details': intervention['details'],
                'impact': intervention['impact'],
                'category': 'intervention'
            })
        
        # Ajouter les alertes
        for alert in self.alerts_history:
            timeline_events.append({
                'time': alert['time'],
                'type': 'Alerte',
                'details': f"{self.alert_thresholds[alert['param']]['name']} {alert['type']}",
                'impact': alert['message'],
                'category': 'alert'
            })
        
        # Trier les événements par temps
        timeline_events.sort(key=lambda x: x['time'])
        
        # Créer un graphique chronologique avec Plotly
        fig = go.Figure()
        
        # Ajouter une ligne pour représenter la glycémie
        fig.add_trace(go.Scatter(
            x=self.twin.history['time'],
            y=self.twin.history['glucose'],
            mode='lines',
            name='Glycémie',
            line=dict(color='blue', width=2)
        ))
        
        # Ajouter des marqueurs pour les interventions
        med_times = [event['time'] for event in timeline_events if event['type'] == 'Médicament']
        med_values = [self.twin.history['glucose'][min(range(len(self.twin.history['time'])), 
                                                     key=lambda i: abs(self.twin.history['time'][i] - t))] 
                     for t in med_times]
        
        if med_times:
            fig.add_trace(go.Scatter(
                x=med_times,
                y=med_values,
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='triangle-up',
                    color='red',
                    line=dict(width=1, color='red')
                ),
                name='Médicaments',
                text=[event['details'] for event in timeline_events if event['type'] == 'Médicament'],
                hoverinfo='text'
            ))
        
        # Ajouter des marqueurs pour les repas
        meal_times = [event['time'] for event in timeline_events if event['type'] == 'Repas']
        meal_values = [self.twin.history['glucose'][min(range(len(self.twin.history['time'])), 
                                                      key=lambda i: abs(self.twin.history['time'][i] - t))]
                      for t in meal_times]
        
        if meal_times:
            fig.add_trace(go.Scatter(
                x=meal_times,
                y=meal_values,
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='circle',
                    color='green',
                    line=dict(width=1, color='green')
                ),
                name='Repas',
                text=[event['details'] for event in timeline_events if event['type'] == 'Repas'],
                hoverinfo='text'
            ))
        
        # Ajouter des marqueurs pour les alertes
        alert_times = [event['time'] for event in timeline_events if event['type'] == 'Alerte']
        alert_values = [self.twin.history['glucose'][min(range(len(self.twin.history['time'])), 
                                                       key=lambda i: abs(self.twin.history['time'][i] - t))]
                       for t in alert_times]
        
        if alert_times:
            fig.add_trace(go.Scatter(
                x=alert_times,
                y=alert_values,
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='x',
                    color='orange',
                    line=dict(width=1, color='orange')
                ),
                name='Alertes',
                text=[event['impact'] for event in timeline_events if event['type'] == 'Alerte'],
                hoverinfo='text'
            ))
        
        # Configurer la mise en page
        fig.update_layout(
            title="Chronologie des événements et impact sur la glycémie",
            xaxis_title="Temps (heures)",
            yaxis_title="Glycémie (mg/dL)",
            height=500,
            hovermode='closest',
            showlegend=True
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les détails des événements
        st.subheader("Détails des événements")
        
        # Créer un tableau d'événements
        if timeline_events:
            # Créer un DataFrame pour faciliter l'affichage
            events_df = pd.DataFrame([{
                'Temps': f"{event['time']:.2f}h",
                'Type': event['type'],
                'Détails': event['details'],
                'Impact': event['impact']
            } for event in timeline_events])
            
            # Définir une fonction de style conditionnel
            def highlight_event_type(row):
                if row['Type'] == 'Médicament':
                    return ['background-color: rgba(150, 200, 255, 0.3)'] * len(row)
                elif row['Type'] == 'Repas':
                    return ['background-color: rgba(150, 255, 150, 0.3)'] * len(row)
                else:  # Alerte
                    return ['background-color: rgba(255, 200, 150, 0.3)'] * len(row)
            
            # Afficher le tableau avec style
            st.dataframe(events_df.style.apply(highlight_event_type, axis=1))
        else:
            st.info("Aucun événement enregistré pendant la simulation")