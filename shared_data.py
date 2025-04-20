# shared_data.py
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

# Définitions de médicaments et interactions
# (vous pouvez également déplacer ces données ici si nécessaire)