import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# Custom CSS styles
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        background-color: #f8f9fa;
    }
    .positive {
        color: green;
        font-weight: bold;
    }
    .negative {
        color: red;
        font-weight: bold;
    }
    .prediction-text {
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Text preprocessing (if needed)
def preprocess_text(text):
    """Cleans and preprocesses input text if necessary"""
    # This preprocessing should match what you did during training
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    return text.strip()

# Function to load models
@st.cache_resource
def load_model(model_name):
    """Loads model from a joblib file"""
    try:
        model_path = f'models/{model_name}'
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model {model_name} not found in 'models/' folder")
        return None

# Function to predict with a model
def predict_with_model(model, text):
    """Makes a prediction with the specified model"""
    # Preprocess text if necessary (depends on how you trained the models)
    processed_text = preprocess_text(text)
    
    # Make prediction (the model already includes the vectorizer)
    try:
        prediction = model.predict([processed_text])
        # Try to get probabilities if the model allows it
        try:
            prediction_proba = model.predict_proba([processed_text])
            confidence = max(prediction_proba[0])
        except:
            # If the model doesn't have predict_proba, use default confidence
            confidence = 1.0
        
        return prediction[0], confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0

# Main application interface
def main():
    st.markdown('<h1 class="main-header">Sentiment Analysis with ML</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner('Loading models...'):
        lr_model = load_model('model_LR.joblib')
        nb_model = load_model('model_NB.joblib')
        rf_model = load_model('model_RF.joblib')
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Comparison"])
    
    with tab1:
        st.header("Sentiment Prediction")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Write your review or comment here..."
        )
        
        # Model selection
        model_option = st.selectbox(
            "Select model to use:",
            ("Logistic Regression", "Naive Bayes", "Random Forest")
        )
        
        # Prediction button
        if st.button("Predict Sentiment", type="primary"):
            if text_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                # Select the appropriate model
                if model_option == "Logistic Regression" and lr_model:
                    model = lr_model
                    model_name = "Logistic Regression"
                elif model_option == "Naive Bayes" and nb_model:
                    model = nb_model
                    model_name = "Naive Bayes"
                elif model_option == "Random Forest" and rf_model:
                    model = rf_model
                    model_name = "Random Forest"
                else:
                    st.error("The selected model is not available.")
                    return
                
                # Make prediction
                with st.spinner('Analyzing sentiment...'):
                    prediction, confidence = predict_with_model(model, text_input)
                
                if prediction is not None:
                    # Show result
                    sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
                    sentiment_class = "positive" if prediction == 1 else "negative"
                    
                    st.markdown(f"""
                    <div class="model-card">
                        <h3>Result with {model_name}</h3>
                        <p class="prediction-text">Sentiment: <span class="{sentiment_class}">{sentiment}</span></p>
                        <p class="prediction-text">Confidence: <b>{confidence:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Model Comparison")
        
        # Text input for comparison
        compare_text = st.text_area(
            "Enter text to compare between models:",
            height=150,
            placeholder="Write your review or comment here...",
            key="compare_text"
        )
        
        if st.button("Compare Models", type="primary"):
            if compare_text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                results = []
                models = {
                    "Logistic Regression": lr_model,
                    "Naive Bayes": nb_model,
                    "Random Forest": rf_model
                }
                
                # Make predictions with all models
                for name, model in models.items():
                    if model is not None:
                        prediction, confidence = predict_with_model(model, compare_text)
                        if prediction is not None:
                            sentiment = "Positive" if prediction == 1 else "Negative"
                            results.append({
                                "Model": name,
                                "Sentiment": sentiment,
                                "Confidence": f"{confidence:.2%}"
                            })
                
                # Show results in a table
                if results:
                    results_df = pd.DataFrame(results)
                    st.table(results_df)
                    
                    # Also show visually
                    st.subheader("Results Visualization")
                    
                    # Create columns to display results
                    cols = st.columns(len(results))
                    
                    for idx, (col, result) in enumerate(zip(cols, results)):
                        with col:
                            sentiment_class = "positive" if result["Sentiment"] == "Positive" else "negative"
                            st.markdown(f"""
                            <div class="model-card">
                                <h4>{result['Model']}</h4>
                                <p>Sentiment: <span class="{sentiment_class}">{result['Sentiment']}</span></p>
                                <p>Confidence: <b>{result['Confidence']}</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("No models available for comparison.")

    # Additional information in the sidebar
    with st.sidebar:
        st.title("ℹ️ Information")
        st.markdown("""
        This application uses machine learning models to analyze text sentiment.
        
        ### Available models:
        - **Logistic Regression**
        - **Naive Bayes**
        - **Random Forest**
        
        ### Instructions:
        1. Write text in the appropriate area
        2. Select a model or compare all
        3. Click the button to see results
        
        """)
        
        # Show model status
        st.subheader("Model Status")
        model_status = {
            "Logistic Regression": "✅ Loaded" if lr_model else "❌ Not found",
            "Naive Bayes": "✅ Loaded" if nb_model else "❌ Not found",
            "Random Forest": "✅ Loaded" if rf_model else "❌ Not found"
        }
        
        for model, status in model_status.items():
            st.write(f"{model}: {status}")

if __name__ == "__main__":
    main()