# 🧠 Lysis AI - Advanced Text Analysis & Mathematical Solver

A powerful Flask-based web application that provides comprehensive text analysis and mathematical problem-solving capabilities. Lysis AI combines natural language processing, machine learning, and symbolic mathematics to analyze and understand text content.

![Lysis AI Interface](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.3+-red)

## ✨ Features

### 🔧 Core Features (Always Available)
- **📊 Basic Text Statistics**: Word count, character count, sentence analysis, reading time estimation
- **🧮 Mathematical Analysis**: Equation solving, expression simplification, derivatives using SymPy
- **💬 Interactive Chat**: Conversational AI assistant for guidance and help
- **📈 Auto-Detection**: Automatically determines the best analysis type for your text
- **⚡ Rate Limiting**: Built-in protection against abuse
- **💾 Data Persistence**: SQLite database for storing analysis history

### 🤖 Advanced Features (Optional ML Libraries)
- **😊 Sentiment Analysis**: Determine emotional tone (positive, negative, neutral)
- **🏷️ Keyword Extraction**: Extract important terms and phrases using TF-IDF or KeyBERT
- **📝 Text Summarization**: Generate concise summaries using BART or extractive methods
- **👤 Named Entity Recognition**: Identify people, places, organizations using spaCy
- **🔗 Text Similarity**: Compare texts using sentence transformers
- **🎯 Text Clustering**: Group similar texts using UMAP and HDBSCAN

## 🚀 Quick Start

### Option 1: Basic Installation (Core Features Only)
```bash
# Clone or download the project
cd lysis-ai

# Run the startup script
./run.sh
```

### Option 2: Full Installation (All Features)
```bash
# Run with ML libraries
./run.sh --with-ml
```

### Option 3: Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install Flask requests numpy sympy

# Optional: Install ML libraries
pip install scikit-learn sentence-transformers keybert transformers torch spacy

# Download spaCy model (if using spaCy)
python -m spacy download en_core_web_sm

# Run the application
python app.py
```

## 🌐 Usage

1. **Start the Server**: Run `./run.sh` or `python app.py`
2. **Open Browser**: Navigate to `http://localhost:5000`
3. **Analyze Text**: 
   - Select analysis type or use auto-detection
   - Paste your text in the input area
   - Click "Analyze Text"
4. **Chat**: Use the chat interface for questions and guidance

## 📋 API Endpoints

### Text Analysis
```bash
POST /api/analyze
Content-Type: application/json

{
  "text": "Your text here",
  "analysis_type": "auto|mathematical|sentiment|keywords|summary|entities|comprehensive"
}
```

### Chat Interface
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Your question here"
}
```

### System Status
```bash
GET /api/status
```

### User History
```bash
GET /api/history
```

## 🧮 Mathematical Analysis Examples

Lysis AI can solve various mathematical problems:

```
# Quadratic equations
"Solve x^2 + 5x + 6 = 0"

# Algebraic expressions
"Simplify (x + 2)(x - 3)"

# Calculus
"Find the derivative of x^3 + 2x^2 + x + 1"

# Trigonometry
"Solve sin(x) = 0.5"
```

## 📊 Text Analysis Examples

### Sentiment Analysis
```
Input: "I love this new product! It works amazingly well."
Output: Positive sentiment (85.2% confidence)
```

### Keyword Extraction
```
Input: "Machine learning and artificial intelligence are transforming technology."
Output: ["machine learning", "artificial intelligence", "technology", "transforming"]
```

### Mathematical Detection
```
Input: "Calculate the area of a circle with radius 5"
Output: Auto-detects as mathematical content
```

## 🏗️ Architecture

```
lysis-ai/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
├── run.sh               # Startup script
├── README.md            # This file
├── lysis_ai.db          # SQLite database (created automatically)
└── lysis_ai.log         # Application logs
```

## 🔧 Configuration

### Environment Variables
- `SECRET_KEY`: Flask session secret (defaults to development key)
- `DATABASE_PATH`: Path to SQLite database (defaults to `lysis_ai.db`)

### Rate Limiting
- Default: 60 requests per minute for analysis
- Default: 30 requests per minute for chat
- Configurable in `app.py`

## 📦 Dependencies

### Core Dependencies
- **Flask**: Web framework
- **NumPy**: Numerical computing
- **SymPy**: Symbolic mathematics
- **Requests**: HTTP library
- **SQLite3**: Database (built-in)

### Optional ML Dependencies
- **scikit-learn**: Machine learning algorithms
- **sentence-transformers**: Text embeddings
- **KeyBERT**: Keyword extraction
- **transformers**: Hugging Face transformers
- **spaCy**: Natural language processing
- **UMAP/HDBSCAN**: Clustering algorithms

## 🛠️ Development

### Adding New Analysis Types
1. Create analysis function in `app.py`
2. Add to `perform_analysis()` function
3. Update `get_available_features()`
4. Add frontend handling in `index.html`

### Database Schema
```sql
-- Conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    message TEXT,
    response TEXT,
    timestamp DATETIME,
    analysis_type TEXT,
    metadata TEXT
);

-- Analysis results table
CREATE TABLE analysis_results (
    id TEXT PRIMARY KEY,
    analysis_type TEXT,
    content TEXT,
    result TEXT,
    timestamp DATETIME,
    processing_time REAL,
    confidence REAL,
    metadata TEXT
);
```

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install package_name
```

**2. spaCy Model Not Found**
```bash
# Download English model
python -m spacy download en_core_web_sm
```

**3. Port Already in Use**
```bash
# Kill process using port 5000
sudo lsof -t -i tcp:5000 | xargs kill -9
```

**4. Virtual Environment Issues**
```bash
# Remove and recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🔒 Security Features

- **Rate Limiting**: Prevents API abuse
- **Input Validation**: Sanitizes user input
- **Session Management**: Secure user sessions
- **Error Handling**: Graceful error management
- **Logging**: Comprehensive activity logging

## 🎯 Performance

- **Caching**: Results cached for 1 hour
- **Async Processing**: Non-blocking analysis
- **Memory Management**: Automatic cache cleanup
- **Database Optimization**: Indexed queries

## 📈 Monitoring

Check application logs:
```bash
tail -f lysis_ai.log
```

Monitor system status:
```bash
curl http://localhost:5000/api/status
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- **SymPy**: Symbolic mathematics library
- **Hugging Face**: Transformers and models
- **spaCy**: Industrial-strength NLP
- **scikit-learn**: Machine learning library
- **Flask**: Micro web framework

---

**Made with ❤️ for text analysis and mathematical problem solving**

For questions or support, please check the chat interface within the application or review the logs for detailed error information.