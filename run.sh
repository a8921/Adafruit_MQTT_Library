#!/bin/bash

# Lysis AI Flask Application Startup Script

echo "🚀 Starting Lysis AI Text Analysis Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q Flask requests numpy sympy

# Check for optional ML libraries and install if requested
if [ "$1" = "--with-ml" ]; then
    echo "🤖 Installing optional ML libraries..."
    pip install -q scikit-learn sentence-transformers keybert transformers torch spacy
    
    # Download spaCy model
    echo "📚 Downloading spaCy language model..."
    python -m spacy download en_core_web_sm
fi

echo "✅ Dependencies installed successfully!"
echo ""
echo "🌟 Starting Lysis AI server..."
echo "📍 Access the application at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python app.py