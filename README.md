# LexEval - Simplified Chat Interface

A streamlined tool for testing legal language models with Together.ai. This simplified version provides a clean two-tab interface for API configuration and model interaction.

## Features

- 🔐 **Easy API Configuration** - Simple setup with Together.ai API key
- 💬 **Chat Interface** - Direct interaction with legal language models
- 📊 **Model Information** - View model details and pricing
- ⚙️ **Advanced Settings** - Fine-tune generation parameters
- 💾 **Export Results** - Download chat history as CSV or JSON
- 🚀 **Streamlit Deployment Ready** - Optimized for Streamlit Community Cloud

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/yourusername/lexeval-simplified.git
cd lexeval-simplified
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Get your Together.ai API key from [Together.ai](https://www.together.ai)

## Usage

1. Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Start the app:
```bash
streamlit run frontend/streamlit_app.py
```

3. Open http://localhost:8501 in your browser

4. Enter your API key in the API Configuration tab

5. Switch to the Chat Interface tab to interact with models

## Project Structure

```
lexeval-simplified/
├── frontend/          # Web interface
│   └── streamlit_app.py
├── core/             # Core functionality
│   └── together_client.py
├── requirements.txt  # Python dependencies
├── README.md        # This file
└── LICENSE          # MIT License
```

## Deploying to Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "Create app" → "Yup, I have an app"
4. Fill in:
   - Repository URL
   - Branch: `main`
   - Main file path: `frontend/streamlit_app.py`
5. Optional: Set a custom subdomain
6. Click "Deploy"

### Adding API Key as Secret

1. In deployment settings, click "Advanced settings"
2. Add secrets in TOML format:
```toml
TOGETHER_API_KEY = "your-api-key-here"
```

## Available Models

The app automatically loads available models from Together.ai, including:
- Meta Llama models (3.2, 3.3, 3.1 variants)
- Mistral models
- Mixtral models
- And many more...

## Generation Parameters

- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = very random)
- **Max Tokens**: Maximum response length
- **Top P**: Nucleus sampling parameter
- **Top K**: Top-k sampling parameter
- **Repetition Penalty**: Reduces repetitive text
- **Chat Mode**: Toggle between chat and completion APIs

## Export Options

- **CSV Export**: Structured data with all parameters
- **JSON Export**: Complete chat history with metadata

## Future Extensions

This simplified version serves as a foundation for:
- Batch processing capabilities
- CSV/JSON upload for bulk evaluation
- Custom evaluation metrics
- Integration with other LLM providers
- Advanced benchmarking features

## Troubleshooting

If you encounter issues:
1. Ensure your API key is valid
2. Check your internet connection
3. Verify all dependencies are installed
4. For deployment issues, check Streamlit logs

## License

MIT License

## Author

This is a [Dazza Greenwood](https://dazzagreenwood.com) project evolved from an initial project built by [Ryan McDonough](https://www.ryanmcdonough.co.uk/)