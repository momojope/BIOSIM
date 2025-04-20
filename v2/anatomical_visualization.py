import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

class AnatomicalVisualization:
    def __init__(self):
        """Initialise la visualisation anatomique avec les coordonnées des organes"""
        # Définition des organes et leurs coordonnées en 2D
        self.organs_2d = {
            'brain': {'x': 0.5, 'y': 0.9, 'r': 0.1, 'name': 'Cerveau'},
            'heart': {'x': 0.5, 'y': 0.7, 'r': 0.08, 'name': 'Cœur'},
            'left_lung': {'x': 0.4, 'y': 0.7, 'r': 0.07, 'name': 'Poumon gauche'},
            'right_lung': {'x': 0.6, 'y': 0.7, 'r': 0.07, 'name': 'Poumon droit'},
            'liver': {'x': 0.4, 'y': 0.5, 'r': 0.08, 'name': 'Foie'},
            'pancreas': {'x': 0.6, 'y': 0.5, 'r': 0.06, 'name': 'Pancréas'},
            'left_kidney': {'x': 0.35, 'y': 0.4, 'r': 0.05, 'name': 'Rein gauche'},
            'right_kidney': {'x': 0.65, 'y': 0.4, 'r': 0.05, 'name': 'Rein droit'},
            'intestines': {'x': 0.5, 'y': 0.3, 'r': 0.15, 'name': 'Intestins'},
            'stomach': {'x': 0.5, 'y': 0.5, 'r': 0.07, 'name': 'Estomac'}
        }
        
        # Définir les organes cibles pour chaque type de médicament
        self.medication_targets = {
            'antidiabetic': ['pancreas', 'liver', 'intestines'],
            'antiinflammatory': ['brain', 'intestines', 'heart', 'left_lung', 'right_lung'],
            'beta_blocker': ['heart', 'brain'],
            'vasodilator': ['heart', 'left_kidney', 'right_kidney']
        }
        
        # Définir une palette de couleurs pour l'intensité d'effet
        self.effect_cmap = LinearSegmentedColormap.from_list(
            'effect_cmap', ['#ffffff', '#ffcc00', '#ff6600', '#ff0000'])
        
        # Définir les coordonnées des vaisseaux sanguins pour le flux sanguin
        self.blood_vessels = {
            'main': {'x': [0.5, 0.5, 0.5, 0.5], 'y': [0.9, 0.7, 0.5, 0.3]},
            'to_lungs': {'x': [0.5, 0.4, 0.6, 0.5], 'y': [0.7, 0.7, 0.7, 0.7]},
            'to_liver': {'x': [0.5, 0.4], 'y': [0.5, 0.5]},
            'to_kidneys': {'x': [0.5, 0.35, 0.5, 0.65], 'y': [0.4, 0.4, 0.4, 0.4]}
        }
    
    def create_2d_visualization(self, medication_concentrations=None):
        """
        Crée une visualisation 2D anatomique montrant les organes et les effets des médicaments
        
        Parameters:
        medication_concentrations (dict): Un dictionnaire contenant les concentrations 
                                         des médicaments par type
        """
        if medication_concentrations is None:
            medication_concentrations = {}
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Dessiner le contour du corps
        body_contour_x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.65, 0.65, 0.7, 0.65, 0.5, 0.35, 0.3, 0.35, 0.35, 0.3]
        body_contour_y = [0.95, 1.0, 0.99, 1.0, 0.95, 0.8, 0.6, 0.3, 0.1, 0.05, 0.02, 0.05, 0.1, 0.3, 0.6, 0.8]
        ax.plot(body_contour_x, body_contour_y, 'k-', linewidth=2, alpha=0.8)
        
        # Calculer les effets des médicaments sur chaque organe
        organ_effects = self._calculate_organ_effects(medication_concentrations)
        
        # Dessiner les vaisseaux sanguins
        for vessel_name, coords in self.blood_vessels.items():
            ax.plot(coords['x'], coords['y'], 'r-', linewidth=2, alpha=0.7)
        
        # Dessiner les organes avec les effets des médicaments
        for organ_id, organ in self.organs_2d.items():
            effect = organ_effects.get(organ_id, 0)
            color = self.effect_cmap(effect)
            
            circle = plt.Circle((organ['x'], organ['y']), organ['r'], 
                               fill=True, alpha=0.8, color=color)
            ax.add_patch(circle)
            
            # Ajouter une bordure noire pour mieux voir l'organe
            border = plt.Circle((organ['x'], organ['y']), organ['r'], 
                               fill=False, alpha=0.8, color='black')
            ax.add_patch(border)
            
            # Ajouter le nom de l'organe
            ax.text(organ['x'], organ['y'], organ['name'], 
                   ha='center', va='center', fontsize=8)
        
        # Configurer les axes
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0, 1.05)
        ax.axis('off')
        ax.set_title('Effet des médicaments sur les organes')
        
        # Créer une légende pour l'intensité des effets
        legend_elements = [
            mpatches.Patch(color=self.effect_cmap(0), label='Aucun effet'),
            mpatches.Patch(color=self.effect_cmap(0.33), label='Effet faible'),
            mpatches.Patch(color=self.effect_cmap(0.66), label='Effet modéré'),
            mpatches.Patch(color=self.effect_cmap(1.0), label='Effet important')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
    
    def create_interactive_3d_visualization(self, medication_concentrations=None):
        """
        Crée une visualisation 3D interactive des organes avec Plotly
        
        Parameters:
        medication_concentrations (dict): Un dictionnaire contenant les concentrations 
                                         des médicaments par type
        """
        if medication_concentrations is None:
            medication_concentrations = {}
        
        # Calculer les effets des médicaments sur chaque organe
        organ_effects = self._calculate_organ_effects(medication_concentrations)
        
        # Créer une figure Plotly 3D
        fig = go.Figure()
        
        # Définir les coordonnées en 3D (ajouter une dimension z)
        z_positions = {
            'brain': 0.9,
            'heart': 0.2,
            'left_lung': 0.3,
            'right_lung': 0.3,
            'liver': 0.1,
            'pancreas': 0,
            'left_kidney': -0.1,
            'right_kidney': -0.1,
            'intestines': -0.3,
            'stomach': 0
        }
        
        # Ajouter des sphères pour chaque organe
        for organ_id, organ in self.organs_2d.items():
            effect = organ_effects.get(organ_id, 0)
            
            # Générer la couleur RGB en fonction de l'effet
            r, g, b, _ = self.effect_cmap(effect)
            color = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
            
            # Créer une sphère pour l'organe
            fig.add_trace(go.Scatter3d(
                x=[organ['x']],
                y=[organ['y']],
                z=[z_positions[organ_id]],
                mode='markers',
                marker=dict(
                    size=organ['r'] * 30,  # Ajuster la taille pour la visualisation 3D
                    color=color,
                    opacity=0.8
                ),
                text=[organ['name']],
                name=organ['name'],
                hoverinfo='text'
            ))
        
        # Ajouter des lignes pour les vaisseaux sanguins principaux
        for vessel_name, coords in self.blood_vessels.items():
            # Ajouter la dimension z pour chaque point
            z_coords = []
            for x, y in zip(coords['x'], coords['y']):
                # Estimer la coordonnée z en fonction de y (hauteur)
                z_coords.append((y - 0.5) * 0.5)
            
            fig.add_trace(go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=z_coords,
                mode='lines',
                line=dict(color='red', width=5),
                opacity=0.7,
                showlegend=False
            ))
        
        # Configurer la scène 3D
        fig.update_layout(
            title='Visualisation 3D des effets des médicaments sur les organes',
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=1)
            ),
            legend=dict(x=0, y=0),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Désactiver les grilles et les axes
        fig.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False
        )
        
        return fig
    
    def _calculate_organ_effects(self, medication_concentrations):
        """
        Calcule l'effet des médicaments sur chaque organe
        
        Parameters:
        medication_concentrations (dict): Un dictionnaire contenant les concentrations 
                                         des médicaments par type
        
        Returns:
        dict: Un dictionnaire des effets normalisés (0-1) par organe
        """
        # Initialiser les effets à zéro pour chaque organe
        organ_effects = {organ_id: 0 for organ_id in self.organs_2d}
        
        if not medication_concentrations:
            return organ_effects
        
        # Calculer l'effet pour chaque organe
        for med_type, concentration in medication_concentrations.items():
            if med_type in self.medication_targets:
                target_organs = self.medication_targets[med_type]
                for organ in target_organs:
                    # L'effet est proportionnel à la concentration
                    if organ in organ_effects:
                        organ_effects[organ] += concentration * 0.01
        
        # Normaliser les effets entre 0 et 1
        max_effect = max(organ_effects.values())
        if max_effect > 0:
            for organ in organ_effects:
                organ_effects[organ] = min(1.0, organ_effects[organ] / max_effect)
        
        return organ_effects
    
    def display_organ_info(self, organ_id):
        """
        Affiche des informations détaillées sur un organe
        
        Parameters:
        organ_id (str): Identifiant de l'organe
        """
        if organ_id not in self.organs_2d:
            return "Information non disponible"
        
        organ_info = {
            'brain': {
                'title': 'Cerveau',
                'function': 'Centre de contrôle du système nerveux. Régule la température corporelle, la respiration, les mouvements et traite les informations sensorielles.',
                'medication_effects': {
                    'beta_blocker': 'Réduit l\'anxiété et peut avoir un effet calmant.',
                    'antiinflammatory': 'Réduit l\'inflammation cérébrale, peut soulager les maux de tête.'
                }
            },
            'heart': {
                'title': 'Cœur',
                'function': 'Pompe le sang à travers le corps, fournissant de l\'oxygène et des nutriments aux tissus.',
                'medication_effects': {
                    'beta_blocker': 'Ralentit le rythme cardiaque, réduit la pression artérielle et la demande en oxygène.',
                    'vasodilator': 'Dilate les vaisseaux sanguins, réduit la pression artérielle et la charge de travail cardiaque.',
                    'antiinflammatory': 'Peut réduire l\'inflammation du muscle cardiaque mais présente des risques cardiovasculaires à long terme.'
                }
            },
            'left_lung': {
                'title': 'Poumon gauche',
                'function': 'Échange d\'oxygène et de dioxyde de carbone avec le sang.',
                'medication_effects': {
                    'antiinflammatory': 'Réduit l\'inflammation des voies respiratoires, utile dans l\'asthme et la BPCO.'
                }
            },
            'right_lung': {
                'title': 'Poumon droit',
                'function': 'Échange d\'oxygène et de dioxyde de carbone avec le sang.',
                'medication_effects': {
                    'antiinflammatory': 'Réduit l\'inflammation des voies respiratoires, utile dans l\'asthme et la BPCO.'
                }
            },
            'liver': {
                'title': 'Foie',
                'function': 'Métabolisme des nutriments, détoxification, production de protéines et stockage du glycogène.',
                'medication_effects': {
                    'antidiabetic': 'Réduit la production hépatique de glucose et peut améliorer la sensibilité à l\'insuline.'
                }
            },
            'pancreas': {
                'title': 'Pancréas',
                'function': 'Production d\'insuline et d\'enzymes digestives.',
                'medication_effects': {
                    'antidiabetic': 'Stimule la production d\'insuline ou améliore son efficacité.'
                }
            },
            'left_kidney': {
                'title': 'Rein gauche',
                'function': 'Filtration du sang, élimination des déchets et régulation des fluides corporels.',
                'medication_effects': {
                    'vasodilator': 'Améliore le flux sanguin rénal, peut améliorer la fonction rénale.'
                }
            },
            'right_kidney': {
                'title': 'Rein droit',
                'function': 'Filtration du sang, élimination des déchets et régulation des fluides corporels.',
                'medication_effects': {
                    'vasodilator': 'Améliore le flux sanguin rénal, peut améliorer la fonction rénale.'
                }
            },
            'intestines': {
                'title': 'Intestins',
                'function': 'Digestion et absorption des nutriments, hébergement du microbiome intestinal.',
                'medication_effects': {
                    'antidiabetic': 'Certains ralentissent l\'absorption du glucose, d\'autres modifient le microbiome intestinal.',
                    'antiinflammatory': 'Réduit l\'inflammation intestinale, peut causer des irritations gastriques.'
                }
            },
            'stomach': {
                'title': 'Estomac',
                'function': 'Stockage temporaire de la nourriture, sécrétion d\'acide et d\'enzymes pour commencer la digestion.',
                'medication_effects': {
                    'antiinflammatory': 'Peut irriter la muqueuse gastrique et augmenter le risque d\'ulcères.'
                }
            }
        }
        
        return organ_info.get(organ_id, "Information non disponible")
    
    def create_animation_frames(self, medication_concentrations_over_time):
        """
        Crée une série d'images pour l'animation de la distribution des médicaments
        
        Parameters:
        medication_concentrations_over_time (list): Liste de dictionnaires de concentrations
                                                   à différents moments
        
        Returns:
        list: Liste de figures matplotlib pour l'animation
        """
        frames = []
        
        for concentrations in medication_concentrations_over_time:
            fig = self.create_2d_visualization(concentrations)
            frames.append(fig)
        
        return frames


# Intégration dans l'interface Streamlit principale
def anatomical_visualization_tab(twin=None):
    """
    Crée un onglet de visualisation anatomique dans l'interface Streamlit
    
    Parameters:
    twin (PatientDigitalTwin): Le jumeau numérique du patient
    """
    st.header("Visualisation Anatomique des Effets des Médicaments")
    
    # Initialiser la visualisation
    viz = AnatomicalVisualization()
    
    # Types de visualisation disponibles
    viz_type = st.radio("Type de visualisation", ["2D Statique", "3D Interactive"], key="viz_type_anatom_tab")
    
    # Extraire les données du jumeau numérique si disponible
    medication_concentrations = {}
    if twin is not None and hasattr(twin, 'history') and len(twin.history['drug_plasma']) > 0:
        # Calculer la concentration moyenne pour chaque médicament
        for med_time, med_type, med_dose in twin.medications:
            if med_type in medication_concentrations:
                medication_concentrations[med_type] += med_dose
            else:
                medication_concentrations[med_type] = med_dose
    else:
        # Démonstration avec des valeurs par défaut
        st.info("Aucune donnée de simulation disponible. Affichage d'une démonstration.")
        medication_concentrations = {
            'antidiabetic': 10,
            'antiinflammatory': 5,
            'beta_blocker': 8,
            'vasodilator': 3
        }
    
    # Panneau interactif pour ajuster manuellement les concentrations
    st.sidebar.header("Ajustement des Médicaments")
    
    adjusted_concentrations = {}
    for med_type, initial_conc in medication_concentrations.items():
        adjusted_concentrations[med_type] = st.sidebar.slider(
            f"{med_type.capitalize()}", 
            0.0, 20.0, float(initial_conc), 0.5
        )
    
    # Afficher la visualisation sélectionnée
    if viz_type == "2D Statique":
        fig = viz.create_2d_visualization(adjusted_concentrations)
        st.pyplot(fig)
        
        # Section d'information sur les organes
        selected_organ = st.selectbox(
            "Sélectionnez un organe pour plus d'informations",
            options=list(viz.organs_2d.keys()),
            format_func=lambda x: viz.organs_2d[x]['name'],
            key="anatom_tab_organ_selector"
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
        fig = viz.create_interactive_3d_visualization(adjusted_concentrations)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Cliquez et faites glisser pour faire pivoter la visualisation 3D. Survolez les organes pour voir leur nom.")
        
    elif viz_type == "Animation Temporelle":
        st.subheader("Évolution temporelle des effets des médicaments")
        
        if twin is not None and hasattr(twin, 'history') and len(twin.history['drug_plasma']) > 0:
            # Créer une animation basée sur les données de simulation
            steps = 5  # Nombre d'étapes pour l'animation
            time_points = np.linspace(0, len(twin.history['drug_plasma'])-1, steps).astype(int)
            
            # Extraire les concentrations aux points temporels sélectionnés
            concentrations_over_time = []
            
            for t_idx in time_points:
                # Récupérer les concentrations à ce moment
                t_concentrations = {}
                for med_time, med_type, med_dose in twin.medications:
                    if t_idx / len(twin.history['drug_plasma']) * twin.duration >= med_time:
                        plasma_conc = twin.history['drug_plasma'][t_idx]
                        if med_type in t_concentrations:
                            t_concentrations[med_type] += plasma_conc * med_dose / 10
                        else:
                            t_concentrations[med_type] = plasma_conc * med_dose / 10
                
                concentrations_over_time.append(t_concentrations)
        else:
            # Créer une animation de démonstration
            steps = 5
            base_concentrations = {
                'antidiabetic': 10,
                'antiinflammatory': 5,
                'beta_blocker': 8,
                'vasodilator': 3
            }
            
            concentrations_over_time = []
            for i in range(steps):
                # Simuler l'évolution des concentrations au fil du temps
                time_factor = i / (steps - 1)  # 0 au début, 1 à la fin
                t_concentrations = {}
                
                for med_type, base_conc in base_concentrations.items():
                    # Simuler une courbe d'absorption puis d'élimination
                    if i < steps / 2:
                        t_concentrations[med_type] = base_conc * (2 * time_factor)
                    else:
                        t_concentrations[med_type] = base_conc * (2 - 2 * time_factor)
                
                concentrations_over_time.append(t_concentrations)
        
        # Créer un curseur pour contrôler l'animation
        time_step = st.slider("Temps", 0, steps-1, 0)
        
        # Afficher l'image pour le pas de temps sélectionné
        if time_step < len(concentrations_over_time):
            fig = viz.create_2d_visualization(concentrations_over_time[time_step])
            st.pyplot(fig)
            
            # Afficher le temps relatif
            if twin is not None and hasattr(twin, 'duration'):
                current_time = time_step / (steps - 1) * twin.duration
                st.write(f"Temps: {current_time:.1f} heures")
            else:
                st.write(f"Étape temporelle: {time_step + 1}/{steps}")
    
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


# Si exécuté directement, afficher une démonstration de la visualisation
if __name__ == "__main__":
    st.set_page_config(page_title="Visualisation Anatomique", layout="wide")
    anatomical_visualization_tab()