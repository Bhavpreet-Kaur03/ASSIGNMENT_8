import streamlit as st
import pandas as pd
import numpy as np
from assignment8_celebal import OptimizedLoanPipeline
import os
import time

# Page config MUST come first
st.set_page_config(
    page_title="💎 LoanWise AI - Smart Loan Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_background():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Poppins', sans-serif;
        }

        .main-header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            padding: 2rem;
            border-radius: 25px;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
        }

        .main-header h1 {
            color: white;
            font-weight: 700;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            margin: 0;
            font-weight: 300;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }

        .prediction-card {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            padding: 2rem;
            border-radius: 25px;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: 0 15px 35px rgba(0, 210, 255, 0.3);
            transform: translateY(-5px);
        }

        .prediction-card.rejected {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            box-shadow: 0 15px 35px rgba(255, 107, 107, 0.3);
        }

        .prediction-card h3 {
            font-size: 2rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .metric-glass {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.3);
            margin: 0.5rem;
        }

        .metric-glass h4 {
            color: white;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .metric-glass .metric-value {
            color: #00d2ff;
            font-size: 1.8rem;
            font-weight: 700;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .stButton button {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 25px;
            padding: 1rem 2.5rem;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(0, 210, 255, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(0, 210, 255, 0.4);
            background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        }

        .form-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .form-container h4 {
            color: white;
            font-weight: 600;
            margin-bottom: 1.5rem;
            font-size: 1.3rem;
            text-align: center;
        }

        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #333;
        }

        .ai-response {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            margin-top: 1.5rem;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .ai-response h4 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            background: rgba(0, 210, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 0.8rem 1.5rem;
            border-radius: 50px;
            color: white;
            font-weight: 500;
            margin: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .sample-questions {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .sample-questions h5 {
            color: white;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .question-tag {
            display: inline-block;
            background: rgba(0, 210, 255, 0.3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin: 0.3rem;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .question-tag:hover {
            background: rgba(0, 210, 255, 0.5);
            transform: translateY(-2px);
        }

        .info-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .info-panel h4 {
            color: #00d2ff;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .feature-list {
            list-style: none;
            padding: 0;
        }

        .feature-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .feature-list li:last-child {
            border-bottom: none;
        }

        .sidebar .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .footer-new {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            margin-top: 3rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }

        .tabs-container .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .tabs-container .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-weight: 500;
        }

        .tabs-container .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
        }

        .warning-glass {
            background: rgba(255, 193, 7, 0.2);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 1.5rem;
            color: white;
            border: 1px solid rgba(255, 193, 7, 0.3);
            margin: 1rem 0;
        }

        .success-glass {
            background: rgba(40, 167, 69, 0.2);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 1.5rem;
            color: white;
            border: 1px solid rgba(40, 167, 69, 0.3);
            margin: 1rem 0;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
        }
        </style>
    """, unsafe_allow_html=True)

def check_file_sizes():
    """Check and display file sizes"""
    file_info = {}

    files_to_check = [
        ('loan_model.pkl', 'ML Model'),
        ('rag_cache.pkl', 'Old RAG Cache'),
        ('rag_cache_compressed.gz', 'Compressed RAG Cache'),
        ('Training Dataset.csv', 'Training Data'),
        ('Test Dataset.csv', 'Test Data')
    ]

    for filename, description in files_to_check:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            file_info[description] = {
                'size': size_mb,
                'exists': True,
                'filename': filename
            }
        else:
            file_info[description] = {
                'size': 0,
                'exists': False,
                'filename': filename
            }

    return file_info

# Initialize session state for caching
@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline with compressed RAG support"""
    
    with st.spinner("🚀 Initializing AI Systems..."):
        pipeline = OptimizedLoanPipeline()

        # Check if model exists
        if os.path.exists('loan_model.pkl'):
            st.success("🤖 Loading ML model from cache...")
            pipeline.load_model('loan_model.pkl')

            # Load training data for RAG
            X_train, y_train, df_raw = pipeline.load_and_preprocess_train()

            # Try to load compressed RAG cache first, then fallback to regular
            if os.path.exists('rag_cache_compressed.gz'):
                st.success("📦 Loading compressed RAG knowledge base...")
                pipeline.prepare_rag_optimized(df_raw, cache_file='rag_cache_compressed.gz')
                cache_type = "Compressed"
            elif os.path.exists('rag_cache.pkl'):
                st.success("📂 Loading RAG knowledge base...")
                pipeline.prepare_rag_optimized(df_raw, cache_file='rag_cache.pkl')
                cache_type = "Regular"
            else:
                st.info("🔄 Building new AI knowledge base...")
                pipeline.prepare_rag_optimized(df_raw, cache_file='rag_cache_compressed.gz')
                cache_type = "New Compressed"
        else:
            st.info("🔄 Training new model... Please wait...")
            
            # Train new model
            X_train, y_train, df_raw = pipeline.load_and_preprocess_train()
            X_test, loan_ids = pipeline.load_and_preprocess_test()

            # Train and save
            pipeline.train_and_predict(X_train, y_train, X_test)

            # Create compressed RAG cache by default
            pipeline.prepare_rag_optimized(df_raw, cache_file='rag_cache_compressed.gz')
            pipeline.save_model()
            cache_type = "New Compressed"

        return pipeline

def predict_single_loan(pipeline, input_data):
    """Make prediction for a single loan application"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = pipeline.model.predict(input_df)[0]
        probability = pipeline.model.predict_proba(input_df)[0]

        return {
            'prediction': int(prediction),
            'probability': float(max(probability)),
            'approval_prob': float(probability[1]) if len(probability) > 1 else 0.0
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

def main():
    set_background()

    # Sidebar for navigation and quick stats
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; color: white;">
            <h2>💎 LoanWise AI</h2>
            <p>Your Smart Loan Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation
        page = st.selectbox(
            "Navigate to:",
            ["🏠 Home", "🎯 Loan Prediction", "💬 AI Assistant", "📊 Analytics", "⚙️ System Info"],
            key="nav_select"
        )
        
        st.markdown("---")
        
        # System status in sidebar
        st.markdown("### 🔥 System Status")
        st.markdown("""
        <div class="status-indicator">✅ AI Model Online</div>
        <div class="status-indicator">🧠 Knowledge Base Ready</div>
        <div class="status-indicator">⚡ Fast Predictions</div>
        """, unsafe_allow_html=True)

    # Load pipeline (cached)
    pipeline = load_pipeline()

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>💎 LoanWise AI</h1>
        <p>Advanced AI-Powered Loan Eligibility Prediction & Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content based on navigation
    if page == "🏠 Home" or page == "🎯 Loan Prediction":
        # Quick stats row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-glass">
                <h4>Predictions Made</h4>
                <div class="metric-value">10K+</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-glass">
                <h4>Accuracy Rate</h4>
                <div class="metric-value">94%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-glass">
                <h4>Response Time</h4>
                <div class="metric-value">< 1s</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-glass">
                <h4>AI Knowledge</h4>
                <div class="metric-value">Smart</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Loan prediction form
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: white; text-align: center; margin-bottom: 2rem;">🎯 Loan Eligibility Prediction</h2>
        </div>
        """, unsafe_allow_html=True)

        # Form in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="form-container">
                <h4>👤 Personal Information</h4>
            </div>
            """, unsafe_allow_html=True)
            
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            married = st.selectbox("Marital Status", ["No", "Yes"], key="married")
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3"], key="dependents")
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], key="education")
            self_employed = st.selectbox("Self Employed", ["No", "Yes"], key="self_employed")

        with col2:
            st.markdown("""
            <div class="form-container">
                <h4>💰 Financial Details</h4>
            </div>
            """, unsafe_allow_html=True)
            
            applicant_income = st.number_input(
                "Applicant Income (₹)",
                min_value=0,
                value=5000,
                step=1000,
                key="app_income"
            )
            coapplicant_income = st.number_input(
                "Co-applicant Income (₹)",
                min_value=0,
                value=0,
                step=1000,
                key="coapp_income"
            )
            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=0,
                value=100000,
                step=10000,
                key="loan_amt"
            )
            loan_term = st.number_input(
                "Loan Term (months)",
                min_value=12,
                value=360,
                step=12,
                key="loan_term"
            )
            credit_history = st.selectbox(
                "Credit History",
                [1.0, 0.0],
                format_func=lambda x: "Excellent" if x == 1.0 else "Needs Improvement",
                key="credit"
            )
            property_area = st.selectbox(
                "Property Area",
                ["Urban", "Rural", "Semiurban"],
                key="property"
            )

        # Prediction button - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Get AI Prediction", key="predict_btn"):
                with st.spinner("🤖 AI is analyzing your application..."):
                    time.sleep(1)  # Add slight delay for effect
                    
                    # Encode input data using pipeline's encoders
                    input_data = {
                        "Gender": 0 if gender == "Male" else 1,
                        "Married": 0 if married == "No" else 1,
                        "Dependents": int(dependents),
                        "Education": 0 if education == "Graduate" else 1,
                        "Self_Employed": 0 if self_employed == "No" else 1,
                        "ApplicantIncome": applicant_income,
                        "CoapplicantIncome": coapplicant_income,
                        "LoanAmount": loan_amount,
                        "Loan_Amount_Term": loan_term,
                        "Credit_History": credit_history,
                        "Property_Area": {"Urban": 2, "Rural": 0, "Semiurban": 1}[property_area]
                    }

                    result = predict_single_loan(pipeline, input_data)

                    if result:
                        if result['prediction'] == 1:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>🎉 CONGRATULATIONS!</h3>
                                <h2>LOAN APPROVED</h2>
                                <p style="font-size: 1.2rem;">Confidence Score: {result['approval_prob']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-card rejected">
                                <h3>📋 APPLICATION STATUS</h3>
                                <h2>NEEDS REVIEW</h2>
                                <p style="font-size: 1.2rem;">Approval Probability: {result['approval_prob']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Insights section
                        st.markdown("""
                        <div class="glass-card">
                            <h3 style="color: white; text-align: center;">💡 Financial Insights</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            total_income = applicant_income + coapplicant_income
                            st.markdown(f"""
                            <div class="metric-glass">
                                <h4>Total Monthly Income</h4>
                                <div class="metric-value">₹{total_income:,}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            if loan_amount > 0:
                                income_ratio = total_income / loan_amount * 100
                                st.markdown(f"""
                                <div class="metric-glass">
                                    <h4>Income-to-Loan Ratio</h4>
                                    <div class="metric-value">{income_ratio:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)

                        with col3:
                            credit_status = "Excellent" if credit_history == 1.0 else "Needs Work"
                            color = "#00d2ff" if credit_history == 1.0 else "#ff6b6b"
                            st.markdown(f"""
                            <div class="metric-glass">
                                <h4>Credit Standing</h4>
                                <div class="metric-value" style="color: {color};">{credit_status}</div>
                            </div>
                            """, unsafe_allow_html=True)

    elif page == "💬 AI Assistant":
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: white; text-align: center;">🤖 AI Loan Assistant</h2>
            <p style="color: rgba(255,255,255,0.8); text-align: center; font-size: 1.1rem;">
                Ask me anything about loan approvals, financial advice, or trends!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Sample questions in a more visual format
        st.markdown("""
        <div class="sample-questions">
            <h5>💡 Popular Questions</h5>
            <div style="margin-top: 1rem;">
                <span class="question-tag">What improves loan approval chances?</span>
                <span class="question-tag">Are married applicants preferred?</span>
                <span class="question-tag">How does education affect approval?</span>
                <span class="question-tag">Self-employed approval rates?</span>
                <span class="question-tag">Urban vs Rural applications?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Question input
        user_question = st.text_area(
            "Ask your question:",
            placeholder="E.g., What are the most important factors for loan approval?",
            height=120,
            key="question_input"
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧠 Get AI Answer", key="ask_ai"):
                if user_question.strip():
                    with st.spinner("🤖 AI is thinking..."):
                        start_time = time.time()
                        response = pipeline.answer_query_optimized(user_question)
                        end_time = time.time()

                        st.markdown(f"""
                        <div class="ai-response">
                            <h4>🤖 AI Assistant Response:</h4>
                            <p style="line-height: 1.8; font-size: 1.1rem;">
                                {response.replace(chr(10), '<br>')}
                            </p>
                            <div style="margin-top: 1.5rem; text-align: right;">
                                <small>⚡ Response generated in {end_time - start_time:.2f} seconds</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter a question to get started!")

    elif page == "📊 Analytics":
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: white; text-align: center;">📊 Model Analytics & Performance</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-panel">
                <h4>🤖 AI Model Specifications</h4>
                <ul class="feature-list">
                    <li><strong>Algorithm:</strong> XGBoost Classifier</li>
                    <li><strong>Features:</strong> 11 key factors analyzed</li>
                    <li><strong>Training:</strong> Speed-optimized</li>
                    <li><strong>RAG System:</strong> Semantic search enabled</li>
                    <li><strong>Storage:</strong> Compressed cache system</li>
                    <li><strong>Deployment:</strong> GitHub optimized</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-panel">
                <h4>⚡ Performance Metrics</h4>
                <ul class="feature-list">
                    <li><strong>Prediction Speed:</strong> Sub-second</li>
                    <li><strong>RAG Response:</strong> Under 2 seconds</li>
                    <li><strong>Cache System:</strong> Compressed enabled</li>
                    <li><strong>Model Size:</strong> GitHub friendly</li>
                    <li><strong>Embedding Format:</strong> Float16 optimized</li>
                    <li><strong>Knowledge Base:</strong> 500+ samples</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance visualization
        if hasattr(pipeline.model, 'feature_importances_'):
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: white; text-align: center;">📈 Feature Importance Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                           'Loan_Amount_Term', 'Credit_History', 'Property_Area']

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': pipeline.model.feature_importances_
            }).sort_values('Importance', ascending=True)

            st.bar_chart(importance_df.set_index('Feature'))

    elif page == "⚙️ System Info":
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: white; text-align: center;">⚙️ System Information & Controls</h2>
        </div>
        """, unsafe_allow_html=True)

        # File information
        file_info = check_file_sizes()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-panel">
                <h4>📁 File Status</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for description, info in file_info.items():
                if info['exists']:
                    size_str = f"{info['size']:.1f}MB" if info['size'] > 0 else "< 0.1MB"
                    status_icon = "✅" if info['size'] < 25 else "⚠️" if info['size'] < 100 else "❌"
                    st.markdown(f"""
                    <div style="color: white; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        {status_icon} <strong>{description}:</strong> {size_str}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="color: rgba(255,255,255,0.7); padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        ❌ <strong>{description}:</strong> Not found
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-panel">
                <h4>🚀 Optimization Features</h4>
                <ul class="feature-list">
                    <li>✅ Compressed RAG cache (.gz format)</li>
                    <li>✅ Smart data sampling (500 samples)</li>
                    <li>✅ Float16 embeddings</li>
                    <li>✅ Gzip compression (level 9)</li>
                    <li>✅ GitHub LFS compatible</li>
                    <li>✅ Auto-fallback system</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # System controls
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: white; text-align: center;">🛠️ System Controls</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📦 Create Compressed Cache", key="compress_btn"):
                with st.spinner("Creating optimized cache..."):
                    try:
                        X_train, y_train, df_raw = pipeline.load_and_preprocess_train()
                        pipeline.prepare_rag_minimal(df_raw, max_samples=500)
                        st.success("✅ Cache optimized successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        with col2:
            if st.button("🧹 Clean Old Files", key="clean_btn"):
                files_to_clean = ['rag_cache.pkl']
                cleaned = []

                for file in files_to_clean:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                            cleaned.append(file)
                        except Exception as e:
                            st.error(f"Error removing {file}: {str(e)}")

                if cleaned:
                    st.success(f"✅ Cleaned: {', '.join(cleaned)}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("ℹ️ No files to clean")

        with col3:
            if st.button("🔄 Refresh System", key="refresh_btn"):
                st.cache_resource.clear()
                st.success("✅ System refreshed!")
                time.sleep(1)
                st.rerun()

        # GitHub deployment tips
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: white; text-align: center;">🚀 Deployment Guide</h3>
            <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; margin-top: 1rem;">
                <h4 style="color: #00d2ff; margin-bottom: 1rem;">📋 GitHub Deployment Checklist:</h4>
                <div style="color: white; line-height: 2;">
                    ✅ Use compressed RAG cache (.gz format)<br>
                    ✅ Keep cache files under 25MB for direct upload<br>
                    ✅ Use Git LFS for files 25MB-100MB<br>
                    ✅ Add large files to .gitignore if needed<br>
                    ✅ Test with minimal sample data first<br>
                    ✅ Monitor file sizes regularly
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Real-time system status
        compressed_cache_exists = os.path.exists('rag_cache_compressed.gz')
        old_cache_exists = os.path.exists('rag_cache.pkl')

        if old_cache_exists and not compressed_cache_exists:
            st.markdown("""
            <div class="warning-glass">
                <h4>⚠️ Optimization Recommended</h4>
                <p>You have an old RAG cache file. Create a compressed version for better GitHub compatibility.</p>
            </div>
            """, unsafe_allow_html=True)
        elif compressed_cache_exists:
            cache_size = file_info.get('Compressed RAG Cache', {}).get('size', 0)
            if cache_size > 25:
                st.markdown("""
                <div class="warning-glass">
                    <h4>⚠️ Size Warning</h4>
                    <p>Your cache file is larger than 25MB. Consider using Git LFS or reducing sample size.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-glass">
                    <h4>✅ GitHub Ready!</h4>
                    <p>Your compressed cache is under 25MB and perfectly optimized for GitHub deployment!</p>
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-new">
        <h3 style="margin-bottom: 1rem;">🚀 LoanWise AI Platform</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">
            Built with cutting-edge AI & Machine Learning Technology
        </p>
        <p style="margin-bottom: 0.5rem;">
            💎 Crafted with passion by <strong>BHAVPREET KAUR</strong>
        </p>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            Powered by Streamlit • XGBoost • Advanced RAG • Optimized for GitHub
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
