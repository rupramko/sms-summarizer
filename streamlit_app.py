"""
Streamlit Frontend for Unsupervised SMS Categorization System
Interactive web application to categorize SMS messages and display results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import joblib
from unsupervised_sms_classifier import UnsupervisedSMSClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SMS Summarizer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .category-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sms_data():
    """Load and filter SMS data from CSV file using centralized filtering"""
    try:
        # Use the centralized filtering from UnsupervisedSMSClassifier
        from unsupervised_sms_classifier import UnsupervisedSMSClassifier
        df = UnsupervisedSMSClassifier.get_filtered_data('Initial_SMS_Data.csv')
        return df
    except FileNotFoundError:
        st.error("SMS data file not found! Please ensure 'Initial_SMS_Data.csv' is in the project directory.")
        return None

@st.cache_resource
def load_model():
    """Load the trained unsupervised model"""
    model_path = 'unsupervised_sms_model.pkl'
    
    if os.path.exists(model_path):
        try:
            classifier = UnsupervisedSMSClassifier('Initial_SMS_Data.csv')
            classifier.load_model(model_path)
            return classifier
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        return None

def train_model_if_needed():
    """Train model if it doesn't exist"""
    model_path = 'unsupervised_sms_model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner("Training unsupervised model... This may take a few moments."):
            try:
                classifier = UnsupervisedSMSClassifier('Initial_SMS_Data.csv', n_clusters=3)
                classifier.train_complete_pipeline(optimize_clusters=False)
                st.success("Model trained successfully!")
                return classifier
            except Exception as e:
                st.error(f"Error training model: {e}")
                return None
    else:
        return load_model()

def display_category_overview(df, classifier):
    """Display overview of categories with counts"""
    
    st.markdown('<div class="main-header">📱 SMS Summarizer</div>', unsafe_allow_html=True)
    
    if classifier and classifier.cluster_labels:
        # Get cluster predictions for all messages
        predictions = []
        for message in df['message_text']:
            try:
                result = classifier.predict_category(message)
                predictions.append(result['predicted_category'])
            except:
                predictions.append('Unknown')
        
        df['predicted_category'] = predictions
        category_counts = df['predicted_category'].value_counts()
        
        # Display category cards
        st.markdown("### 📊 Category Distribution")
        
        cols = st.columns(3)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (category, count) in enumerate(category_counts.items()):
            with cols[i % 3]:
                percentage = (count / len(df)) * 100
                st.markdown(f"""
                <div class="category-card" style="background: linear-gradient(135deg, {colors[i]} 0%, {colors[i]}AA 100%);">
                    <h3>{category}</h3>
                    <h1>{count}</h1>
                    <p>{percentage:.1f}% of total messages</p>
                </div>
                """, unsafe_allow_html=True)
        
        return df, category_counts
    else:
        st.error("Model not available. Please train the model first.")
        return df, None

def create_visualizations(df, category_counts):
    """Create interactive visualizations"""
    
    if category_counts is None:
        return
        
    st.markdown("### 📈 Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for category distribution
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Category Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Bar chart for category counts
        fig_bar = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Messages per Category",
            color=category_counts.index,
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_bar.update_layout(showlegend=False)
        fig_bar.update_xaxes(title="Category")
        fig_bar.update_yaxes(title="Number of Messages")
        st.plotly_chart(fig_bar, width='stretch')
    
    # Message length analysis
    if 'predicted_category' in df.columns:
        st.markdown("### 📏 Message Length Analysis")
        
        df['message_length'] = df['message_text'].str.len()
        
        fig_box = px.box(
            df,
            x='predicted_category',
            y='message_length',
            title="Message Length by Category",
            color='predicted_category',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width='stretch')

def prediction_interface(classifier):
    """Interface for predicting new messages"""
    
    if classifier is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Text input for new message
    new_message = st.text_area(
        "Enter SMS message to categorize:",
        placeholder="Type your SMS message here...",
        height=100
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        predict_button = st.button("🎯 Predict Category", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    if predict_button and new_message.strip():
        # Validate message before processing
        valid_messages, invalid_messages = validate_messages([new_message.strip()])
        
        if not valid_messages:
            st.error("❌ Invalid message! Please enter a meaningful SMS message.")
            if invalid_messages:
                st.warning("**Validation failed:**")
                for msg in invalid_messages:
                    st.write(f"• {msg}")
            return
            
        try:
            # Make prediction using the validated message
            result = classifier.predict_category(valid_messages[0])
            
            # Display result
            st.markdown(f"""
            <div class="prediction-result">
                <h4>Prediction Result:</h4>
                <p><strong>Category:</strong> {result['predicted_category']}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence scores for all categories
            st.markdown("#### Confidence Scores for All Categories:")
            
            scores_df = pd.DataFrame(
                list(result['all_scores'].items()),
                columns=['Category', 'Score']
            )
            scores_df['Score'] = scores_df['Score'].round(3)
            
            fig_scores = px.bar(
                scores_df,
                x='Category',
                y='Score',
                title="Confidence Scores",
                color='Score',
                color_continuous_scale='Viridis'
            )
            fig_scores.update_layout(showlegend=False)
            st.plotly_chart(fig_scores, width='stretch')
            
            # Show processed text
            with st.expander("View Processed Text"):
                st.code(result['processed_text'])
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    elif predict_button and not new_message.strip():
        st.warning("Please enter a message to predict.")

def batch_prediction_interface(classifier):
    """Interface for batch predictions"""
    
    if classifier is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Multiple messages input
    messages_text = st.text_area(
        "Enter multiple messages (one per line):",
        placeholder="Message 1\nMessage 2\nMessage 3\n...",
        height=150
    )
    
    if st.button("🚀 Predict All Messages"):
        if messages_text.strip():
            messages = [msg.strip() for msg in messages_text.split('\n') if msg.strip()]
            
            if messages:
                # Validate all messages before processing
                valid_messages, invalid_messages = validate_messages(messages)
                
                if not valid_messages:
                    st.error("❌ No valid messages found! Please enter meaningful SMS messages.")
                    if invalid_messages:
                        st.warning("**Invalid messages detected:**")
                        for i, msg in enumerate(invalid_messages[:5], 1):  # Show first 5 invalid
                            st.write(f"{i}. {msg}")
                        if len(invalid_messages) > 5:
                            st.write(f"... and {len(invalid_messages) - 5} more")
                    return
                
                # Show validation summary if there are invalid messages
                if invalid_messages:
                    st.warning(f"⚠️ {len(invalid_messages)} invalid message(s) will be skipped. Processing {len(valid_messages)} valid message(s).")
                    with st.expander("View invalid messages"):
                        for i, msg in enumerate(invalid_messages, 1):
                            st.write(f"{i}. {msg}")
                
                results = []
                progress_bar = st.progress(0)
                
                for i, message in enumerate(valid_messages):
                    try:
                        result = classifier.predict_category(message)
                        results.append({
                            'Message': message[:50] + '...' if len(message) > 50 else message,
                            'Full Message': message,
                            'Category': result['predicted_category'],
                            'Confidence': result['confidence']
                        })
                    except Exception as e:
                        results.append({
                            'Message': message[:50] + '...' if len(message) > 50 else message,
                            'Full Message': message,
                            'Category': 'Error',
                            'Confidence': 0.0
                        })
                    
                    progress_bar.progress((i + 1) / len(valid_messages))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.markdown("#### Batch Prediction Results:")
                st.dataframe(results_df[['Message', 'Category', 'Confidence']], width='stretch')
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Messages", len(results))
                
                with col2:
                    avg_confidence = results_df['Confidence'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
                
                with col3:
                    high_conf = len(results_df[results_df['Confidence'] >= 0.8])
                    st.metric("High Confidence (≥80%)", high_conf)
                
                # Category distribution for batch
                batch_category_counts = results_df['Category'].value_counts()
                fig_batch = px.pie(
                    values=batch_category_counts.values,
                    names=batch_category_counts.index,
                    title="Batch Prediction Category Distribution"
                )
                st.plotly_chart(fig_batch, width='stretch')
                
                # Download results
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Please enter at least one message.")
        else:
            st.warning("Please enter messages to predict.")

def validate_messages(messages):
    """Validate SMS messages to ensure they contain meaningful content"""
    import re
    
    valid_messages = []
    invalid_messages = []
    
    for message in messages:
        # Clean the message
        cleaned_message = message.strip()
        
        # Skip empty messages
        if not cleaned_message:
            invalid_messages.append(message)
            continue
        
        # Validation criteria
        is_valid = True
        reasons = []
        
        # 1. Minimum length check (at least 10 characters)
        if len(cleaned_message) < 10:
            is_valid = False
            reasons.append("too short")
        
        # 2. Maximum length check (reasonable SMS length)
        if len(cleaned_message) > 1000:
            is_valid = False
            reasons.append("too long")
        
        # 3. Must contain at least some letters
        if not re.search(r'[a-zA-Z]', cleaned_message):
            is_valid = False
            reasons.append("no letters")
        
        # 4. Must contain at least one word (not just symbols/numbers)
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned_message)
        if len(words) < 2:
            is_valid = False
            reasons.append("insufficient words")
        
        # 5. Check for spam/test patterns
        spam_patterns = [
            r'^test\s*$',
            r'^hello\s*$',
            r'^hi\s*$',
            r'^\s*\d+\s*$',  # Only numbers
            r'^[^a-zA-Z]*$',  # No letters at all
            r'^(.)\1{10,}',   # Repeated characters
        ]
        
        for pattern in spam_patterns:
            if re.match(pattern, cleaned_message.lower()):
                is_valid = False
                reasons.append("test/spam pattern")
                break
        
        # 6. Check for reasonable content (basic SMS characteristics)
        # Must have some meaningful content indicators
        has_meaningful_content = any([
            # Common SMS keywords/patterns
            re.search(r'\b(offer|discount|free|deal|save|book|pay|amount|rs|rupees|upi|bank|account)\b', cleaned_message.lower()),
            # URLs or codes
            re.search(r'(http|www|\.com|\.in)', cleaned_message.lower()),
            # Numbers with context (amounts, dates, codes)
            re.search(r'\d+.*(?:rs|rupees|%|off|discount|amount|code|otp)', cleaned_message.lower()),
            # Business/service related
            re.search(r'\b(confirm|booking|order|payment|transaction|balance|credit|debit)\b', cleaned_message.lower()),
            # Time/date related
            re.search(r'\b(today|tomorrow|date|time|expires|valid|till)\b', cleaned_message.lower()),
            # General meaningful content (longer messages with proper structure)
            len(words) >= 5 and len(cleaned_message) >= 30
        ])
        
        if not has_meaningful_content:
            is_valid = False
            reasons.append("lacks meaningful SMS content")
        
        # Categorize the message
        if is_valid:
            valid_messages.append(cleaned_message)
        else:
            invalid_messages.append(f"{message} (Reason: {', '.join(reasons)})")
    
    return valid_messages, invalid_messages

def optimized_single_prediction(classifier, message):
    """Optimized single message prediction with enhanced display"""
    try:
        with st.spinner("🔮 Analyzing message..."):
            result = classifier.predict_category(message)
        
        # Enhanced result display with better layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Prediction Result")
            st.success(f"**Category:** {result['predicted_category']}")
            st.info(f"**Confidence:** {result['confidence']:.1%}")
            
            # Quick stats
            if result['confidence'] >= 0.8:
                confidence_color = "🟢 High"
            elif result['confidence'] >= 0.6:
                confidence_color = "🟡 Medium" 
            else:
                confidence_color = "🔴 Low"
            st.write(f"**Confidence Level:** {confidence_color}")
        
        with col2:
            st.markdown("#### 📊 All Category Scores")
            scores_df = pd.DataFrame(
                list(result['all_scores'].items()),
                columns=['Category', 'Score']
            )
            scores_df['Score'] = scores_df['Score'].round(3)
            scores_df = scores_df.sort_values('Score', ascending=False)
            
            # Create a more compact bar chart
            fig_scores = px.bar(
                scores_df,
                x='Score',
                y='Category',
                orientation='h',
                title="Confidence Scores",
                color='Score',
                color_continuous_scale='Viridis',
                height=300
            )
            fig_scores.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_scores, width='stretch')
        
        # Expandable details
        with st.expander("🔍 View Processed Text & Details"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original Message:**")
                st.code(message)
            with col_b:
                st.markdown("**Processed Text:**")
                st.code(result['processed_text'])
                
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

def optimized_batch_prediction(classifier, messages):
    """Optimized batch prediction with better progress tracking"""
    st.markdown("#### 📊 Batch Prediction Results")
    
    try:
        # Initialize results
        results = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        # Process messages
        for i, message in enumerate(messages):
            status_placeholder.text(f"Processing message {i+1}/{len(messages)}...")
            
            try:
                result = classifier.predict_category(message)
                results.append({
                    'S.No.': i+1,
                    'Message': message[:80] + '...' if len(message) > 80 else message,
                    'Category': result['predicted_category'],
                    'Confidence': result['confidence'],
                })
            except Exception:
                results.append({
                    'S.No.': i+1,
                    'Message': message[:80] + '...' if len(message) > 80 else message,
                    'Category': 'Error',
                    'Confidence': 0.0,
                })
            
            # Update progress
            progress_bar.progress((i + 1) / len(messages))
        
        # Clear progress indicators
        progress_bar.empty()
        status_placeholder.empty()
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Show summary stats
        st.markdown("#### 📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(results))
        
        with col2:
            valid_results = results_df[results_df['Confidence'] > 0]
            if len(valid_results) > 0:
                avg_conf = valid_results['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        with col3:
            high_conf_count = len(results_df[results_df['Confidence'] >= 0.8])
            st.metric("High Confidence", high_conf_count)
        
        with col4:
            error_count = len(results_df[results_df['Category'] == 'Error'])
            st.metric("Errors", error_count)
        
        # Display results table
        st.markdown("#### 📋 Results Table")
        
        # Format confidence for display
        display_df = results_df.copy()
        display_df['Confidence'] = display_df['Confidence'].apply(
            lambda x: f"{x:.1%}" if x > 0 else "Error"
        )
        
        # Show the table
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True
        )
        
        # Visualizations
        valid_data = results_df[results_df['Category'] != 'Error']
        if len(valid_data) > 0:
            st.markdown("#### 📊 Category Analysis")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Category distribution
                category_counts = valid_data['Category'].value_counts()
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Category Distribution"
                )
                st.plotly_chart(fig_pie, width='stretch')
            
            with col_viz2:
                # Confidence levels
                confidence_ranges = pd.cut(
                    valid_data['Confidence'], 
                    bins=[0, 0.6, 0.8, 1.0], 
                    labels=['Low (0-60%)', 'Medium (60-80%)', 'High (80-100%)']
                )
                confidence_counts = confidence_ranges.value_counts()
                
                fig_bar = px.bar(
                    x=confidence_counts.index,
                    y=confidence_counts.values,
                    title="Confidence Levels"
                )
                st.plotly_chart(fig_bar, width='stretch')
        
        # Download option
        st.markdown("#### 💾 Download Results")
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"batch_predictions_{len(messages)}_messages.csv",
            mime="text/csv",
            width='stretch'
        )
        
        st.success(f"✅ Successfully processed {len(messages)} messages!")
                
    except Exception as e:
        st.error(f"❌ Error during batch prediction: {e}")
        st.error("Please try again or check your input messages.")

def data_explorer(df, classifier=None):
    """Data exploration interface"""
    
    st.markdown("### 🔍 SMS Explorer")
    
    # Category dropdown selection
    if classifier and classifier.cluster_labels:
        # Get predictions for all messages if not already done
        if 'predicted_category' not in df.columns:
            predictions = []
            for message in df['message_text']:
                try:
                    result = classifier.predict_category(message)
                    predictions.append(result['predicted_category'])
                except:
                    predictions.append('Unknown')
            df['predicted_category'] = predictions
        
        # Category selection dropdown
        available_categories = ['All'] + list(df['predicted_category'].unique())
        selected_category = st.selectbox(
            "Select Category to Explore:",
            available_categories,
            key="data_explorer_category"
        )
        
        # Filter dataframe based on selection
        if selected_category == 'All':
            filtered_df = df
            display_title = "All"
        else:
            filtered_df = df[df['predicted_category'] == selected_category]
            display_title = selected_category
        
    else:
        filtered_df = df
        st.info("Model not loaded. Showing all data without category filtering.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Messages", len(filtered_df))
        st.metric("Average Message Length", f"{filtered_df['message_text'].str.len().mean():.0f} chars")
    
    with col2:
        st.metric("Unique Senders", filtered_df['sender_code'].nunique())
        # Show different metrics based on selection
        if classifier and classifier.cluster_labels:
            if selected_category == 'All':
                # When "All" is selected, show total number of categories
                st.metric("Total Categories", len(df['predicted_category'].unique()))
            else:
                # When specific category is selected, show the selected category
                st.metric("Selected Category", selected_category)
    
    # Show sample messages
    st.markdown("#### Sample Messages:")
    
    if len(filtered_df) > 0:
        # Maximum value is based on total message count
        total_messages = len(filtered_df)
        max_samples = total_messages
        
        sample_size = st.slider(
            f"Number of samples to show (out of {total_messages} messages):", 
            min_value=5, 
            max_value=max_samples, 
            value=min(10, max_samples)
        )
        
        sample_df = filtered_df.sample(n=min(sample_size, len(filtered_df)))
        # Add serial numbers starting from 1
        sample_df = sample_df.reset_index(drop=True)
        sample_df.index = sample_df.index + 1
        sample_df.index.name = 'S.No.'
        
        display_columns = ['message_text', 'sender_code']
        if 'predicted_category' in sample_df.columns:
            display_columns.append('predicted_category')
        st.dataframe(sample_df[display_columns], width='stretch')
    else:
        st.info("No messages found for the selected category.")
    
    # Search functionality
    st.markdown("#### Search Messages:")
    search_term = st.text_input("Search for messages containing:")
    
    if search_term:
        search_filtered_df = filtered_df[filtered_df['message_text'].str.contains(search_term, case=False, na=False)]
        st.write(f"Found {len(search_filtered_df)} messages containing '{search_term}' in {display_title}:")
        if len(search_filtered_df) > 0:
            # Add serial numbers for search results too
            search_filtered_df = search_filtered_df.reset_index(drop=True)
            search_filtered_df.index = search_filtered_df.index + 1
            search_filtered_df.index.name = 'S.No.'
            
            display_columns = ['message_text', 'sender_code']
            if 'predicted_category' in search_filtered_df.columns:
                display_columns.append('predicted_category')
            st.dataframe(search_filtered_df[display_columns], width='stretch')

def main():
    """Main Streamlit application"""
    
    # Sidebar
    st.sidebar.title("📱 SMS Summarizer")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "🔍 Predict Message", "📊 SMS Explorer"]
    )
    
    # Load data
    df = load_sms_data()
    
    if df is None:
        st.error("Cannot load SMS data. Please check if 'Initial_SMS_Data.csv' exists.")
        return
    
    # Model status
    st.sidebar.markdown("### Model Status")
    
    classifier = load_model()
    
    if classifier is None:
        st.sidebar.warning("⚠️ Model not trained")
        if st.sidebar.button("🚀 Train Model Now"):
            classifier = train_model_if_needed()
            if classifier:
                st.sidebar.success("✅ Model trained successfully!")
                st.rerun()
    else:
        st.sidebar.success("✅ Model loaded")
    
    # Train model button
    if st.sidebar.button("🔄 Retrain Model"):
        if os.path.exists('unsupervised_sms_model.pkl'):
            os.remove('unsupervised_sms_model.pkl')
        classifier = train_model_if_needed()
        if classifier:
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app uses unsupervised machine learning (K-Means clustering) "
        "to categorize SMS messages into Travel, Finance, and Recharge categories."
    )
    
    # Main content based on selected page
    if page == "🏠 Overview":
        df, category_counts = display_category_overview(df, classifier)
        if category_counts is not None:
            create_visualizations(df, category_counts)
    
    elif page == "🔍 Predict Message":
        st.markdown("### 🔍 Predict Message")
        
        # Create tabs for single and batch prediction
        tab1, tab2 = st.tabs(["• Single Message", "• • Batch Messages"])
        
        with tab1:
            st.markdown("#### Single Message Prediction")
            prediction_interface(classifier)
        
        with tab2:
            st.markdown("#### Batch Message Prediction")
            batch_prediction_interface(classifier)

    elif page == "📊 SMS Explorer":
        data_explorer(df, classifier)

if __name__ == "__main__":
    main()