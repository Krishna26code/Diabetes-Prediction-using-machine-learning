import numpy as np
import pickle
import streamlit as st

# Load model and scaler
classifier = pickle.load(open('C:/Users/hp/Downloads/Deploying Machine Learning Model/diabetes_model.sav', 'rb'))
scaler = pickle.load(open('C:/Users/hp/Downloads/Deploying Machine Learning Model/scaler.sav', 'rb'))

# --- Page Config ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="💉", layout="centered")

# --- Background with Color + Image ---
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(rgba(0, 123, 255, 0.3), rgba(255, 255, 255, 0.3)),
            url("https://img.freepik.com/free-vector/medical-background-with-hexagons_1017-19369.jpg");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
background-color: #f2f2f2;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Prediction Function ---
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        return '🟢 The person is NOT diabetic.'
    else:
        return '🔴 The person is diabetic.'

# --- Smart Chatbot Function ---
def chatbot(user_message, context=None):
    user_message = user_message.lower().strip()
    response = ""

    # Greeting
    if any(greet in user_message for greet in ['hello', 'hi', 'hey']):
        response = "👋 Hi there! I’m your health assistant. Want to know how to check diabetes? Just enter your values above!"

    # About diabetes
    elif 'what is diabetes' in user_message or 'meaning' in user_message:
        response = "🩸 Diabetes means your body has trouble managing blood sugar levels, often due to low insulin or resistance to insulin."

    # Symptoms
    elif 'symptom' in user_message:
        response = "⚠️ Common symptoms include frequent urination, tiredness, blurry vision, thirst, and unexpected weight loss."

    # Prevention
    elif 'prevent' in user_message or 'avoid' in user_message:
        response = "🥗 You can reduce risk with exercise, balanced diet, avoiding junk food, and regular health check-ups."

    # Diet
    elif 'diet' in user_message or 'food' in user_message:
        response = "🍎 Choose high-fiber foods, green vegetables, nuts, and drink plenty of water. Avoid sugary drinks and refined carbs."

    # Exercise
    elif 'exercise' in user_message or 'workout' in user_message:
        response = "🏃‍♂️ Regular brisk walking, cycling, or yoga for 30 minutes daily helps manage blood sugar naturally."

    # Diabetes types
    elif 'type 1' in user_message or 'type 2' in user_message:
        response = "💉 Type 1 diabetes is insulin-dependent (body makes no insulin). Type 2 is lifestyle-related (body resists insulin)."

    # Thank you or bye
    elif 'thank' in user_message or 'bye' in user_message:
        response = "😊 You’re welcome! Stay healthy and positive. Type 'help' if you need guidance again."

    # Help message
    elif 'help' in user_message:
        response = "🤖 You can ask me about diabetes symptoms, prevention, diet, exercises, or simply test your data above."

    else:
        response = "🤔 I’m not sure about that, but you can ask me about symptoms, prevention, or diet tips for diabetes."

    # Add small context memory (last user message)
    if context is not None:
        context["last_message"] = user_message

    return response, context

# --- Main App ---
def main():
    st.title('💉 Diabetes Prediction Web App')
    st.markdown("<h4 style='color:#2E86C1;'>Enter your medical details below:</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure value')
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age of the Person')

    # Styled Button
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color:white;
            font-size:18px;
            border-radius:10px;
            height:3em;
            width:100%;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)

    # Prediction
    diagnosis = ''
    if st.button('🔍 Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                         Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)

    # Chatbot Section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("💬 Smart Health Chatbot Assistant")

    if 'context' not in st.session_state:
        st.session_state['context'] = {}

    user_query = st.text_input("Ask me something about diabetes, diet, or prevention:")
    if user_query:
        response, st.session_state['context'] = chatbot(user_query, st.session_state['context'])
        st.info(response)

if __name__ == '__main__':
    main()
