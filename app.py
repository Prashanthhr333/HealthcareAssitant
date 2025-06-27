from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from difflib import get_close_matches
import numpy as np

app = Flask(__name__)

# Expanded dataset with more diseases
disease_data = {
    'symptoms': [
        'fever headache body ache fatigue muscle pain chills',
        'cough fever difficulty breathing shortness of breath fatigue chest pain',
        'runny nose sneezing congestion sore throat cough mild fever',
        'stomach pain nausea vomiting diarrhea fever cramps',
        'rash itching redness swelling hives skin irritation',
        'severe headache sensitivity to light nausea vomiting aura',
        'joint pain swelling stiffness fatigue morning stiffness',
        'chest pain shortness of breath sweating nausea arm pain',
        'frequent urination excessive thirst fatigue blurred vision hunger',
        'sore throat difficulty swallowing fever swollen lymph nodes red throat',
        'abdominal pain bloating gas constipation diarrhea cramping',
        'wheezing shortness of breath chest tightness coughing difficulty breathing',
        'muscle weakness numbness tingling fatigue balance problems vision problems',
        'anxiety restlessness racing heart sweating panic fear',
        'depression fatigue loss of interest sleep changes sadness hopelessness'
    ],
    'disease': [
        'Influenza (Flu)',
        'COVID-19',
        'Common Cold',
        'Gastroenteritis',
        'Allergic Reaction',
        'Migraine',
        'Rheumatoid Arthritis',
        'Heart Attack',
        'Diabetes',
        'Strep Throat',
        'Irritable Bowel Syndrome',
        'Asthma',
        'Multiple Sclerosis',
        'Anxiety Disorder',
        'Clinical Depression'
    ],
    'medication': [
        'Tamiflu, Acetaminophen, Rest, Fluids',
        'Consult doctor immediately, Isolation, Rest, Monitor oxygen levels',
        'Antihistamines, Decongestants, Rest, Warm fluids',
        'Oral rehydration, Anti-diarrheal medication, Clear fluids',
        'Antihistamines, Avoid allergens, Cool compress, Calamine lotion',
        'Pain relievers, Anti-nausea medication, Rest in dark room',
        'NSAIDs, DMARDs, Physical therapy, Joint protection',
        'Immediate medical attention, Aspirin, Nitroglycerin if prescribed',
        'Insulin, Oral medications, Blood sugar monitoring, Diet management',
        'Antibiotics, Pain relievers, Rest, Warm salt water gargles',
        'Antispasmodics, Fiber supplements, Stress management',
        'Inhalers, Bronchodilators, Avoid triggers, Action plan',
        'Disease-modifying therapies, Physical therapy, Rest',
        'Anti-anxiety medication, Therapy, Relaxation techniques',
        'Antidepressants, Therapy, Exercise, Sleep hygiene'
    ],
    'precautions': [
        'Rest, Stay hydrated, Avoid contact with others, Practice good hygiene',
        'Isolate, Wear mask, Monitor oxygen levels, Seek immediate care if severe',
        'Rest, Stay warm, Drink fluids, Practice good hygiene',
        'Stay hydrated, Eat bland foods, Rest, Maintain hygiene',
        'Identify triggers, Avoid allergens, Keep skin moisturized, Emergency plan',
        'Avoid triggers, Maintain sleep schedule, Stress management',
        'Regular exercise, Joint protection, Healthy diet, Regular check-ups',
        'Regular check-ups, Healthy diet, Exercise, Stress management',
        'Regular monitoring, Healthy diet, Exercise, Foot care',
        'Complete antibiotics, Rest, Avoid contact, Good hygiene',
        'Diet management, Stress reduction, Regular exercise',
        'Avoid triggers, Regular medication, Action plan, Regular check-ups',
        'Regular check-ups, Stress management, Exercise as tolerated',
        'Regular therapy, Stress management, Regular exercise, Support system',
        'Regular therapy, Exercise, Social support, Healthy lifestyle'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(disease_data)

# Initialize and train the model
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
X = vectorizer.fit_transform(df['symptoms'])
model = MultinomialNB()
model.fit(X, df['disease'])

def preprocess_symptoms(symptoms):
    # Convert to lowercase and remove extra spaces
    symptoms = symptoms.lower().strip()
    # Replace multiple spaces with single space
    symptoms = re.sub(r'\s+', ' ', symptoms)
    return symptoms

def get_symptom_matches(user_input):
    # Split input into individual symptoms
    user_symptoms = set(user_input.split())
    matches = []
    
    # Check each disease's symptoms
    for idx, row in df.iterrows():
        disease_symptoms = set(row['symptoms'].split())
        # Count matching symptoms
        matching_symptoms = user_symptoms.intersection(disease_symptoms)
        if matching_symptoms:
            # Calculate match score based on number of matching symptoms
            match_score = len(matching_symptoms) / max(len(user_symptoms), len(disease_symptoms))
            matches.append((idx, match_score))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_input = preprocess_symptoms(data.get('symptoms', ''))
        
        # Check for greetings or farewells
        if is_greeting(user_input):
            return jsonify({'response': get_greeting_response()})
        elif is_farewell(user_input):
            return jsonify({'response': get_farewell_response()})
        
        # Get symptom matches
        matches = get_symptom_matches(user_input)
        
        # If no direct matches, use the ML model
        if not matches:
            symptoms_vector = vectorizer.transform([user_input])
            probabilities = model.predict_proba(symptoms_vector)[0]
            top_indices = probabilities.argsort()[-3:][::-1]
            matches = [(idx, probabilities[idx]) for idx in top_indices if probabilities[idx] >= 0.1]
        
        if not matches:
            return jsonify({'response': "I couldn't identify any specific conditions based on your input. "
                          "Please provide more detailed symptoms or try rephrasing your query."})
        
        response = "Based on your symptoms, here are the possible conditions:\n\n"
        
        for i, (idx, confidence) in enumerate(matches[:3], 1):
            if confidence >= 0.1:  # Only show if confidence is at least 10%
                disease = df['disease'].iloc[idx]
                medications = df['medication'].iloc[idx]
                precautions = df['precautions'].iloc[idx]
                
                response += f"{i}. {disease} (Confidence: {confidence*100:.1f}%)\n"
                response += f"   Recommended medications: {medications}\n"
                response += f"   Precautions: {precautions}\n\n"
        
        if not response.strip().endswith("conditions:\n\n"):
            response += ("Please note: This is not a substitute for professional medical advice. "
                        "If symptoms are severe or persist, please consult a healthcare provider.")
            return jsonify({'response': response})
        else:
            return jsonify({'response': "I couldn't identify any specific conditions based on your input. "
                          "Please provide more detailed symptoms or try rephrasing your query."})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Greeting patterns
GREETING_PATTERNS = [
    r'hello\b', r'hi\b', r'hey\b', r'greetings\b', r'good morning\b',
    r'good afternoon\b', r'good evening\b', r'howdy\b'
]

# Farewell patterns
FAREWELL_PATTERNS = [
    r'thank\s*you\b', r'thanks\b', r'bye\b', r'goodbye\b',
    r'see\s*you\b', r'take\s*care\b'
]

def is_greeting(text):
    return any(re.search(pattern, text.lower()) for pattern in GREETING_PATTERNS)

def is_farewell(text):
    return any(re.search(pattern, text.lower()) for pattern in FAREWELL_PATTERNS)

def get_greeting_response():
    return ("Hello! I'm your medical assistant. I can help you identify possible conditions based on your symptoms. "
            "Please describe your symptoms in detail, and I'll provide you with information about potential conditions, "
            "medications, and precautions. Remember, this is not a substitute for professional medical advice.")

def get_farewell_response():
    return ("Thank you for using our medical assistant. Take care of yourself! Remember to consult with a healthcare "
            "professional for proper medical diagnosis and treatment. Wishing you good health!")

if __name__ == '__main__':
    app.run(debug=True)
