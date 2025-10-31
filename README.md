# SMS Summarizer - Unsupervised SMS Categorization

An intelligent SMS categorization system that automatically classifies SMS messages into meaningful categories using unsupervised machine learning techniques.

## 🚀 Features

- **Automatic SMS Categorization**: Uses K-means clustering to categorize SMS messages without predefined labels
- **Smart Data Filtering**: Automatically excludes test messages, mobile numbers, and promotional codes ending with -T/-G
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Single & Batch Prediction**: Analyze individual messages or process multiple messages at once
- **Message Validation**: Ensures only meaningful SMS content is processed
- **Data Visualization**: Interactive charts and graphs for insights
- **Export Results**: Download prediction results as CSV files

## 🏗️ Architecture

The system consists of three main categories:
- **Travel**: Hotel bookings, flight offers, travel deals
- **Recharge**: Mobile recharge plans, data packages, telecom offers  
- **Finance**: Banking, payments, cashback, credit card offers

## 📋 Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- plotly
- nltk
- numpy

## 🛠️ Installation

### Option 1: Quick Start (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/rupramko/sms-summarizer.git
cd sms-summarizer
```

2. Run the installation script:
```bash
# For Windows (PowerShell)
.\install_and_run.ps1

# For Windows (Command Prompt)
install_and_run.bat
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sms-summarizer.git
cd sms-summarizer
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run streamlit_app.py
```

## 🎯 Usage

1. **Start the Application**: Run the installation script or use `streamlit run streamlit_app.py`

2. **Access the Web Interface**: Open your browser and go to `http://localhost:8501`

3. **Navigate Through Pages**:
   - **Overview**: View data statistics and category distribution
   - **Predict Message**: Categorize new SMS messages
     - Single Message: Analyze one message at a time
     - Batch Messages: Process multiple messages together
   - **Data Explorer**: Explore the training data and search messages

4. **Message Prediction**:
   - Enter your SMS message text
   - Get category prediction with confidence score
   - View detailed analysis and processed text

## 📁 Project Structure

```
sms-summarizer/
├── streamlit_app.py              # Main web application
├── unsupervised_sms_classifier.py # ML engine and data processing
├── Initial_SMS_Data.csv          # Training data
├── requirements.txt              # Python dependencies
├── install_and_run.bat          # Windows batch installer
├── install_and_run.ps1          # PowerShell installer
├── INSTALLATION_GUIDE.md        # Detailed setup instructions
├── unsupervised_sms_model.pkl   # Trained model (auto-generated)
└── README.md                    # This file
```

## 🔧 Configuration

### Data Filtering
The system automatically filters out:
- Sender codes ending with `-T` (test messages)
- Sender codes ending with `-G` (promotional codes)
- Mobile numbers and invalid sender codes
- Messages that don't meet minimum content requirements

### Model Parameters
- **Clustering Algorithm**: K-means with 3 clusters
- **Text Processing**: TF-IDF vectorization with stemming
- **Validation**: Comprehensive message content validation

## 📊 Data Format

Your SMS data should be in CSV format with the following columns:
- `message_text`: The SMS message content
- `sender_code`: The sender identifier/code

Example:
```csv
message_text,sender_code
"Limited offer: Rs.60 off on domestic hotels. Book before 56 days end.",DM-IRCTC-P
"Recharge Rs.499 for 28 days and get 3GB/day + unlimited calls.",VM-JIOTXT-S
```

## 🚀 Deployment

### Local Deployment
Follow the installation steps above.

### Cloud Deployment
The application can be deployed on:
- **Streamlit Cloud**: Push to GitHub and deploy directly
- **Heroku**: Use the provided requirements.txt
- **AWS/GCP/Azure**: Deploy as a containerized application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Mamata Nandennavar: mamata.nandennavar@nokia.com
Swaathi Prakash: swathi.prakash@nokia.com
Swetha Kerahalli: swetha.kerahalli@nokia.com
Nupur Rupram Kohadkar: nupur.rupram_kohadkar@nokia.com

Project Link: [https://github.com/rupramko/sms-summarizer](https://github.com/rupramko/sms-summarizer)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Data visualization with [Plotly](https://plotly.com/)
- Natural language processing with [NLTK](https://www.nltk.org/)