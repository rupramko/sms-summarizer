"""
Unsupervised SMS Categorization using K-Means Clustering
This module creates clusters of SMS messages without using the existing category labels
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedSMSClassifier:
    def __init__(self, csv_file_path, n_clusters=3):
        """
        Initialize the unsupervised SMS classifier
        
        Args:
            csv_file_path (str): Path to the CSV file containing SMS data
            n_clusters (int): Number of clusters to create (default: 3)
        """
        self.csv_file_path = csv_file_path
        self.n_clusters = n_clusters
        self.df = None
        self.vectorizer = None
        self.kmeans_model = None
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Cluster labels mapping (will be determined after clustering)
        self.cluster_labels = {}
        
    def load_data(self):
        """Load and examine the SMS data with filtering"""
        self.df = pd.read_csv(self.csv_file_path)
        
        # Apply data filtering silently
        self.df = self.filter_data(self.df)
        
        return self.df
    
    def filter_data(self, df):
        """
        Filter SMS data based on sender code criteria
        
        Args:
            df (DataFrame): Original SMS data
            
        Returns:
            DataFrame: Filtered SMS data
        """
        def should_exclude_sender(sender_code):
            """Check if sender code should be excluded"""
            if pd.isna(sender_code):
                return True
                
            sender_str = str(sender_code).strip()
            
            # Exclude if ends with '-T' or '-G' (case insensitive)
            if sender_str.upper().endswith('-T') or sender_str.upper().endswith('-G'):
                return True
            
            # Check if it's a mobile number pattern
            # Remove common separators and check if remaining are mostly digits
            cleaned = re.sub(r'[\s\-\+\(\)]', '', sender_str)
            
            # Pattern for mobile numbers (8-15 digits, optionally starting with country code)
            mobile_patterns = [
                r'^\+?\d{8,15}$',  # 8-15 digits with optional +
                r'^[0-9\s\-\(\)\+]{8,20}$',  # Numbers with separators
            ]
            
            for pattern in mobile_patterns:
                if re.match(pattern, sender_str):
                    # Additional check: if more than 70% of characters are digits
                    digit_ratio = len(re.sub(r'[^0-9]', '', sender_str)) / len(sender_str) if len(sender_str) > 0 else 0
                    if digit_ratio > 0.7:
                        return True
            
            return False
        
        # Apply filtering silently
        df_filtered = df[~df['sender_code'].apply(should_exclude_sender)].copy()
        
        return df_filtered.reset_index(drop=True)
    
    @staticmethod
    def get_filtered_data(csv_file_path):
        """
        Static method to get filtered SMS data for UI display
        
        Args:
            csv_file_path (str): Path to the CSV file
            
        Returns:
            DataFrame: Filtered SMS data
        """
        import pandas as pd
        import re
        
        # Load raw data
        df = pd.read_csv(csv_file_path)
        
        def should_exclude_sender(sender_code):
            """Check if sender code should be excluded"""
            if pd.isna(sender_code):
                return True
                
            sender_str = str(sender_code).strip()
            
            # Exclude if ends with '-T' or '-G' (case insensitive)
            if sender_str.upper().endswith('-T') or sender_str.upper().endswith('-G'):
                return True
            
            # Check if it's a mobile number pattern
            # Remove common separators and check if remaining are mostly digits
            cleaned = re.sub(r'[\s\-\+\(\)]', '', sender_str)
            
            # Pattern for mobile numbers (8-15 digits, optionally starting with country code)
            mobile_patterns = [
                r'^\+?\d{8,15}$',  # 8-15 digits with optional +
                r'^[0-9\s\-\(\)\+]{8,20}$',  # Numbers with separators
            ]
            
            for pattern in mobile_patterns:
                if re.match(pattern, sender_str):
                    # Additional check: if more than 70% of characters are digits
                    digit_ratio = len(re.sub(r'[^0-9]', '', sender_str)) / len(sender_str) if len(sender_str) > 0 else 0
                    if digit_ratio > 0.7:
                        return True
            
            return False
        
        # Apply filtering
        df_filtered = df[~df['sender_code'].apply(should_exclude_sender)].copy()
        
        return df_filtered.reset_index(drop=True)
    
    def preprocess_text(self, text):
        """
        Preprocess SMS text for clustering
        
        Args:
            text (str): Raw SMS text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep letters and numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove numbers (amounts, dates, etc.) to focus on text patterns
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)
        
        return ' '.join(processed_tokens)
    
    def prepare_features(self):
        """Prepare TF-IDF features for clustering"""
        print("Preprocessing text data...")
        
        # Preprocess messages
        self.df['processed_text'] = self.df['message_text'].apply(self.preprocess_text)
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced for cleaner clustering
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            strip_accents='unicode',
            lowercase=True
        )
        
        # Fit and transform the text data
        tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_text'])
        
        return tfidf_matrix
    
    def find_optimal_clusters(self, tfidf_matrix, max_clusters=8):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            int: Optimal number of clusters
        """
        print("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(self.df) // 2))
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(tfidf_matrix, kmeans.labels_))
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(cluster_range, inertias, 'bo-')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(cluster_range, silhouette_scores, 'ro-')
        plt.title('Silhouette Score')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def perform_clustering(self, tfidf_matrix, n_clusters=None):
        """
        Perform K-Means clustering on the SMS data
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            n_clusters (int): Number of clusters (if None, uses self.n_clusters)
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        print(f"Performing K-Means clustering with {n_clusters} clusters...")
        
        # Perform clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.kmeans_model.fit_predict(tfidf_matrix)
        
        # Add cluster labels to dataframe
        self.df['cluster'] = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
        print(f"Silhouette score: {silhouette_avg:.3f}")
        
        return cluster_labels
    
    def analyze_clusters(self):
        """Analyze and interpret the clusters"""
        print("Analyzing clusters...")
        
        # Cluster distribution
        print(f"\nCluster distribution:")
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"Cluster {cluster}: {count} messages ({percentage:.1f}%)")
        
        # Get top terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\nTop terms for each cluster:")
        for cluster_id in range(self.n_clusters):
            cluster_center = self.kmeans_model.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            print(f"Cluster {cluster_id}: {', '.join(top_terms)}")
        
        # Analyze cluster characteristics
        self.interpret_clusters()
        
    def interpret_clusters(self):
        """Interpret clusters and assign meaningful labels"""
        print("\nInterpreting clusters...")
        
        # Analyze sample messages from each cluster
        for cluster_id in range(self.n_clusters):
            cluster_messages = self.df[self.df['cluster'] == cluster_id]['message_text'].head(5)
            print(f"\nCluster {cluster_id} sample messages:")
            for i, msg in enumerate(cluster_messages, 1):
                print(f"  {i}. {msg}")
        
        # Auto-assign labels based on keywords
        cluster_keywords = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in range(self.n_clusters):
            cluster_center = self.kmeans_model.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-20:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            cluster_keywords[cluster_id] = top_terms
        
        # Enhanced heuristic to assign labels
        for cluster_id, keywords in cluster_keywords.items():
            keywords_str = ' '.join(keywords)
            
            # Score each category based on keyword presence
            travel_score = sum(1 for word in ['flight', 'hotel', 'book', 'travel', 'trip', 'train', 'domest', 'packag', 'uber'] if word in keywords_str)
            finance_score = sum(1 for word in ['bank', 'card', 'payment', 'account', 'loan', 'credit', 'cashback', 'spend', 'transact', 'otp'] if word in keywords_str)
            recharge_score = sum(1 for word in ['recharge', 'data', 'pack', 'prepaid', 'gb', 'plan', 'call', 'unlimit', 'gbday'] if word in keywords_str)
            
            # Assign based on highest score
            scores = {'Travel': travel_score, 'Finance': finance_score, 'Recharge': recharge_score}
            best_category = max(scores, key=scores.get)
            
            # Handle ties and ensure all 3 categories are represented
            if best_category == 'Travel' and travel_score > 0:
                self.cluster_labels[cluster_id] = 'Travel'
            elif best_category == 'Finance' and finance_score > 0:
                self.cluster_labels[cluster_id] = 'Finance'
            elif best_category == 'Recharge' and recharge_score > 0:
                self.cluster_labels[cluster_id] = 'Recharge'
            else:
                # Fallback - ensure we have all 3 categories
                assigned_labels = list(self.cluster_labels.values())
                if 'Travel' not in assigned_labels:
                    self.cluster_labels[cluster_id] = 'Travel'
                elif 'Finance' not in assigned_labels:
                    self.cluster_labels[cluster_id] = 'Finance'
                else:
                    self.cluster_labels[cluster_id] = 'Recharge'
        
        # Ensure we have exactly 3 different categories
        unique_labels = set(self.cluster_labels.values())
        if len(unique_labels) < 3:
            # Force assign missing categories
            all_categories = ['Travel', 'Finance', 'Recharge']
            missing_categories = [cat for cat in all_categories if cat not in unique_labels]
            
            # Reassign some clusters to missing categories
            for i, missing_cat in enumerate(missing_categories):
                if i < len(self.cluster_labels):
                    cluster_to_reassign = list(self.cluster_labels.keys())[i]
                    self.cluster_labels[cluster_to_reassign] = missing_cat
        
        print(f"\nAssigned cluster labels:")
        for cluster_id, label in self.cluster_labels.items():
            count = len(self.df[self.df['cluster'] == cluster_id])
            print(f"Cluster {cluster_id} → {label} ({count} messages)")
    
    def visualize_clusters(self):
        """Create visualizations for the clusters"""
        print("Creating visualizations...")
        
        # Prepare data for visualization
        tfidf_matrix = self.vectorizer.transform(self.df['processed_text'])
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(tfidf_matrix.toarray())
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Cluster scatter plot
        plt.subplot(2, 3, 1)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for cluster_id in range(self.n_clusters):
            cluster_data = pca_features[self.df['cluster'] == cluster_id]
            label = self.cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c=colors[cluster_id], label=label, alpha=0.6)
        
        plt.title('SMS Clusters (PCA Visualization)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cluster distribution
        plt.subplot(2, 3, 2)
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        cluster_names = [self.cluster_labels.get(i, f'Cluster {i}') for i in cluster_counts.index]
        plt.pie(cluster_counts.values, labels=cluster_names, autopct='%1.1f%%')
        plt.title('Cluster Distribution')
        
        # Compare with original categories (if available)
        if 'category' in self.df.columns:
            plt.subplot(2, 3, 3)
            original_counts = self.df['category'].value_counts()
            plt.pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%')
            plt.title('Original Categories')
        
        # Word clouds for each cluster
        for i, cluster_id in enumerate(range(self.n_clusters)):
            plt.subplot(2, 3, 4 + i)
            cluster_text = ' '.join(self.df[self.df['cluster'] == cluster_id]['processed_text'])
            
            if cluster_text.strip():  # Check if there's text to process
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white').generate(cluster_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                label = self.cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
                plt.title(f'{label} Word Cloud')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title(f'Cluster {cluster_id} Word Cloud')
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='unsupervised_sms_model.pkl'):
        """Save the trained model and components"""
        print(f"Saving model to {model_path}...")
        
        model_components = {
            'kmeans_model': self.kmeans_model,
            'vectorizer': self.vectorizer,
            'cluster_labels': self.cluster_labels,
            'stemmer': self.stemmer,
            'stop_words': self.stop_words,
            'n_clusters': self.n_clusters
        }
        
        joblib.dump(model_components, model_path)
        print("Model saved successfully!")
    
    def load_model(self, model_path='unsupervised_sms_model.pkl'):
        """Load a saved model"""
        print(f"Loading model from {model_path}...")
        
        model_components = joblib.load(model_path)
        self.kmeans_model = model_components['kmeans_model']
        self.vectorizer = model_components['vectorizer']
        self.cluster_labels = model_components['cluster_labels']
        self.stemmer = model_components['stemmer']
        self.stop_words = model_components['stop_words']
        self.n_clusters = model_components['n_clusters']
        
        print("Model loaded successfully!")
    
    def predict_category(self, message):
        """
        Predict category for a new SMS message
        
        Args:
            message (str): SMS message text
            
        Returns:
            dict: Prediction results
        """
        if self.kmeans_model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Preprocess the message
        processed_message = self.preprocess_text(message)
        
        # Vectorize
        message_tfidf = self.vectorizer.transform([processed_message])
        
        # Predict cluster
        cluster_id = self.kmeans_model.predict(message_tfidf)[0]
        
        # Get distances to all cluster centers (for confidence)
        distances = self.kmeans_model.transform(message_tfidf)[0]
        
        # Convert distances to similarity scores (closer = higher score)
        max_distance = np.max(distances)
        similarities = 1 - (distances / max_distance) if max_distance > 0 else np.ones_like(distances)
        
        # Normalize to get probabilities
        probabilities = similarities / np.sum(similarities)
        
        # Get predicted category
        predicted_category = self.cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        
        # Create confidence scores for all categories
        confidence_scores = {}
        for i, prob in enumerate(probabilities):
            category = self.cluster_labels.get(i, f'Cluster {i}')
            confidence_scores[category] = round(prob, 3)
        
        return {
            'predicted_category': predicted_category,
            'cluster_id': int(cluster_id),
            'confidence': round(probabilities[cluster_id], 3),
            'all_scores': confidence_scores,
            'processed_text': processed_message
        }
    
    def train_complete_pipeline(self, optimize_clusters=False):
        """Complete training pipeline"""
        print("Starting unsupervised SMS categorization training...")
        
        # Load data
        self.load_data()
        
        # Prepare features
        tfidf_matrix = self.prepare_features()
        
        # Optionally find optimal number of clusters
        if optimize_clusters:
            optimal_k = self.find_optimal_clusters(tfidf_matrix)
            self.n_clusters = optimal_k
        
        # Perform clustering
        cluster_labels = self.perform_clustering(tfidf_matrix)
        
        # Analyze clusters
        self.analyze_clusters()
        
        # Create visualizations
        self.visualize_clusters()
        
        # Save model
        self.save_model()
        
        print(f"\nUnsupervised training completed!")
        print(f"Created {self.n_clusters} clusters")
        print(f"Cluster labels: {self.cluster_labels}")
        
        return self.kmeans_model

def main():
    """Main function to demonstrate the unsupervised SMS classifier"""
    # Initialize classifier
    classifier = UnsupervisedSMSClassifier('Initial_SMS_Data.csv', n_clusters=3)
    
    # Train the model
    model = classifier.train_complete_pipeline(optimize_clusters=False)
    
    # Test with some sample messages
    print("\n" + "="*50)
    print("TESTING NEW MESSAGE PREDICTIONS")
    print("="*50)
    
    test_messages = [
        "Book your flight tickets now and get 50% off",
        "Your account balance is Rs.5000. UPI payment successful",
        "Recharge your mobile for Rs.299 and get 1GB data daily",
        "Special offer on hotel bookings this weekend",
        "Credit card payment processed successfully",
        "Top-up your prepaid account with extra benefits"
    ]
    
    for message in test_messages:
        result = classifier.predict_category(message)
        print(f"\nMessage: {message}")
        print(f"Predicted Category: {result['predicted_category']}")
        print(f"Cluster ID: {result['cluster_id']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        print("All category scores:")
        for category, score in result['all_scores'].items():
            print(f"  {category}: {score:.3f}")

if __name__ == "__main__":
    main()