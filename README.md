# 🔬 Universal R&D AI Assistant

An advanced AI research assistant that can answer any technical, scientific, or engineering question with intelligent, structured, multi-layered responses.

## ✨ Features

### 🎯 Universal Intent Detection
- **What is** - Definitions and explanations
- **How to make** - Manufacturing and synthesis
- **Comparison** - Analysis and benchmarking
- **Troubleshoot** - Problem solving
- **Optimize** - Performance improvement
- **Applications** - Use cases and implementations
- **Future trends** - Emerging technologies
- **Research gaps** - Open problems and challenges
- **Numerical** - Mathematical calculations and equations

### 🧮 Numerical Expression Solver
- Solve mathematical equations using SymPy
- Evaluate arithmetic expressions
- Handle mathematical functions (sin, cos, log, etc.)
- Support for variables and symbolic computation

### 🔬 Research-Grade Responses
- Comprehensive technical analysis
- Multi-domain support (Science, Medicine, Energy, Business, etc.)
- Semantic search through research papers
- Quality assessment and reliability scoring
- Implementation guidance and best practices

### 🚀 Advanced AI Integration
- Ollama + LLaMA 7B integration
- LangChain pipelines
- ChromaDB for vector storage
- PDF parsing capabilities
- Real-time conversation management

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Ollama (for LLM generation)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd universal-rd-ai-assistant
pip install -r requirements.txt
```

### 2. Install Ollama
```bash
# Download and install Ollama from https://ollama.ai
# Or use the following commands:

# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 3. Pull Required Model
```bash
ollama pull llama3.1:8b-instruct-q4_k_m
```

### 4. Optional: Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Usage

### Start the Server
```bash
python main.py
```

The server will start at `http://127.0.0.1:5000`

### API Endpoints

#### 1. Main Chat Interface
- **URL**: `http://127.0.0.1:5000/`
- **Method**: GET
- **Description**: Web interface for interacting with the AI

#### 2. Ask Question
- **URL**: `http://127.0.0.1:5000/ask`
- **Method**: POST
- **Body**: `{"question": "Your question here"}`
- **Response**: Comprehensive AI response with debug information

#### 3. System Status
- **URL**: `http://127.0.0.1:5000/api/status`
- **Method**: GET
- **Description**: Check system health and feature availability

## 📝 Usage Examples

### 1. Technical Definitions
```
Question: "What is quantum computing?"
Intent: what_is
Response: Comprehensive explanation with technical details, applications, and current state
```

### 2. Manufacturing Processes
```
Question: "How to make solar panels?"
Intent: how_to_make
Response: Step-by-step manufacturing process with materials, equipment, and quality control
```

### 3. Mathematical Calculations
```
Question: "solve x^2 + 2x - 3 = 0"
Intent: numerical
Response: Direct mathematical solution using SymPy
```

### 4. Performance Comparison
```
Question: "Compare lithium-ion vs solid-state batteries"
Intent: comparison
Response: Detailed comparison table with performance metrics, costs, and applications
```

### 5. Troubleshooting
```
Question: "My solar panel is not working, how to fix it?"
Intent: troubleshoot
Response: Systematic troubleshooting guide with diagnostic steps and solutions
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom model
export OLLAMA_MODEL="llama3.1:8b-instruct-q4_k_m"

# Optional: Set custom database paths
export PAPERS_DB_PATH="output/papers.db"
export CONVERSATIONS_DB_PATH="output/conversations.db"
```

### Model Configuration
Edit `main.py` to change:
- `MODEL_NAME`: Ollama model to use
- `OLLAMA_URL`: Ollama server URL
- `EMBEDDING_MODEL`: Sentence transformer model
- `MAX_CONTEXT_MESSAGES`: Conversation history limit

## 📊 Database Structure

### Papers Database
- Research papers with abstracts, domains, and metadata
- Vector embeddings for semantic search
- Quality scores and relevance metrics

### Conversations Database
- User conversation history
- Intent detection results
- Thinking steps and debug information

## 🧠 Technical Architecture

### Core Components
1. **UniversalIntentDetector**: Advanced intent classification
2. **NumericalExpressionSolver**: Mathematical computation engine
3. **UniversalRDPromptGenerator**: Research-grade prompt generation
4. **DomainDetector**: Multi-domain classification
5. **ResearchDataGenerator**: Technical data synthesis

### AI/ML Features
- **Semantic Search**: Sentence transformers for paper retrieval
- **Intent Detection**: Pattern-based classification with confidence scoring
- **Mathematical Solving**: SymPy integration for symbolic computation
- **Domain Classification**: Multi-domain keyword analysis

## 🔍 Debug Information

Each response includes comprehensive debug information:
```json
{
  "debug": {
    "intent_detected": "comparison",
    "domain_detected": "Energy & Environment",
    "papers_used": 3,
    "complexity_level": "expert",
    "confidence_scores": {
      "comparison": 0.8,
      "applications": 0.3
    }
  }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Check available models
   ollama list
   ```

2. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install -r requirements.txt
   
   # For GPU support (optional)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Database Issues**
   ```bash
   # Reset databases
   rm -rf output/
   python main.py  # Will recreate databases
   ```

4. **Memory Issues**
   - Reduce `MAX_CONTEXT_MESSAGES` in main.py
   - Use smaller embedding model
   - Limit concurrent requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ollama team for the excellent local LLM framework
- Hugging Face for transformer models
- SymPy team for mathematical computation
- Sentence Transformers for semantic search capabilities

---

**🔬 Universal R&D AI Assistant** - Your advanced research companion for technical, scientific, and engineering questions.