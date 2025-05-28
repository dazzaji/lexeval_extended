# uv venv
# source .venv/bin/activate
# uv pip install -r requirements.txt
# streamlit run frontend/streamlit_app.py

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import os
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.together_client import TogetherClient

# Set page config
st.set_page_config(
    page_title="LexEval",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 1024

# Basic styling
st.markdown(
    """
    <style>
    .stApp {
        background: #f5f5f5;
    }
    .stButton>button {
        background-color: #2b6777;
        color: white;
        border-radius: 4px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2b6777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    '''
    <div style="background: #2b6777; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h1 style="color: white; margin: 0;">⚖️ LexEval</h1>
        <p style="color: #e0e7ef; margin: 0.5rem 0 0 0;">Legal model benchmarks</p>
    </div>
    ''',
    unsafe_allow_html=True
)

def load_available_models():
    """Load available models from Together.ai API."""
    if not st.session_state.client:
        return {}
    
    # Return cached models if available
    if st.session_state.available_models:
        return st.session_state.available_models
    
    # Load models and cache them
    models = st.session_state.client.get_available_models()
    st.session_state.available_models = models
    return models

def main():
    # Create tabs
    api_tab, chat_tab = st.tabs(["API Configuration", "Chat Interface"])
    
    # API Configuration Tab
    with api_tab:
        st.header("Together.ai API Configuration")
        st.markdown("""
        To use LexEval, you'll need a Together.ai API key. You can get one by:
        1. Creating an account at [Together.ai](https://api.together.xyz/)
        2. Going to your [API Keys page](https://api.together.xyz/settings/api-keys)
        3. Creating a new API key
        """)
        
        api_key = st.text_input(
            "Enter your Together.ai API Key",
            type="password",
            value=st.session_state.api_key or ""
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                st.session_state.client = TogetherClient(api_key)
                # Clear caches when API key changes
                st.session_state.available_models = {}
                st.success("API key configured successfully!")
            else:
                st.session_state.client = None
                st.warning("Please enter a valid API key to continue.")

    # Chat Interface Tab
    with chat_tab:
        if not st.session_state.client:
            st.warning("Please configure your Together.ai API key in the API Configuration tab to begin.")
        else:
            # Model Selection
            st.subheader("Model and Task Selection")
            
            # Two columns for model selection and input
            select_col1, select_col2 = st.columns(2)
            
            with select_col1:
                models = load_available_models()
                if not models:
                    st.error("No models available. Please check your configuration.")
                    return
                model_options = list(models.keys())
                model_display_names = {model_id: model_info['display_name'] for model_id, model_info in models.items()}
                model_options.sort(key=lambda x: model_display_names[x])
                selected_model = st.selectbox(
                    "Select Model",
                    options=model_options,
                    format_func=lambda x: model_display_names.get(x, x),
                    key="model_selector"
                )
            
            with select_col2:
                user_prompt = st.text_area(
                    "Input Prompt",
                    placeholder="Enter your prompt here...",
                    height=100,
                    key="user_prompt"
                )

            # Model Info and Pricing in a new row
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                if selected_model in models:
                    model_info = models[selected_model]
                    st.subheader("Model Information")
                    st.markdown(f"**Organization:** {model_info['organization']}")
                    st.markdown(f"**Context Length:** {model_info['context_length']:,} tokens")
            with info_col2:
                if selected_model in models:
                    model_info = models[selected_model]
                    pricing = model_info.get('pricing', {})
                    if pricing:
                        st.subheader("Pricing")
                        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Input:</span><span><b>${pricing.get('input', 0):.2f}/1M tokens</b></span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Output:</span><span><b>${pricing.get('output', 0):.2f}/1M tokens</b></span></div>", unsafe_allow_html=True)

            # Generation Parameters under Advanced Settings
            with st.expander("Advanced Settings"):
                st.subheader("Generation Parameters")
                col1, col2, col3 = st.columns(3)
                
                # Get context length for selected model
                context_length = 4096
                if selected_model in models:
                    context_length = models[selected_model].get('context_length', 4096)
                
                with col1:
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
                    max_tokens = st.number_input(
                        "Max Tokens",
                        1,
                        context_length,
                        value=min(1024, context_length // 3),
                        key=f"max_tokens_{selected_model}"
                    )

                with col2:
                    top_p = st.slider("Top P", 0.0, 1.0, 0.7, 0.1)
                    top_k = st.number_input("Top K", 1, 100, 50)
                
                with col3:
                    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.1)
                    use_chat = st.checkbox("Use Chat Mode", value=True)

            # Submit Button
            if st.button("Send to Model", type="primary"):
                if not user_prompt:
                    st.error("Please enter a prompt.")
                else:
                    # Generate response
                    with st.spinner("Generating response..."):
                        try:
                            start_time = time.time()
                            
                            if use_chat:
                                messages = [{"role": "user", "content": user_prompt}]
                                response = st.session_state.client.generate_chat(
                                    messages=messages,
                                    model=selected_model,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    repetition_penalty=repetition_penalty
                                )
                            else:
                                response = st.session_state.client.generate_completion(
                                    prompt=user_prompt,
                                    model=selected_model,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    repetition_penalty=repetition_penalty
                                )
                            
                            end_time = time.time()
                            
                            # Extract response text
                            if response and response.get('text'):
                                response_text = response['text']
                                latency = end_time - start_time
                                
                                # Add to chat history
                                chat_entry = {
                                    'timestamp': datetime.now().isoformat(),
                                    'model': selected_model,
                                    'prompt': user_prompt,
                                    'response': response_text,
                                    'latency': latency,
                                    'parameters': {
                                        'temperature': temperature,
                                        'max_tokens': max_tokens,
                                        'top_p': top_p,
                                        'top_k': top_k,
                                        'repetition_penalty': repetition_penalty,
                                        'use_chat': use_chat
                                    }
                                }
                                st.session_state.chat_history.append(chat_entry)
                                
                                # Force UI update
                                st.rerun()
                            else:
                                st.error("No response received from the model.")
                                
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")

            # Display Model Response
            if st.session_state.chat_history:
                st.subheader("Model Response")
                
                # Display the most recent response
                latest_entry = st.session_state.chat_history[-1]
                
                # Response info metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{latest_entry['latency']:.2f}s")
                with col2:
                    st.metric("Model", models[latest_entry['model']]['display_name'])
                with col3:
                    st.metric("Timestamp", latest_entry['timestamp'].split('T')[1].split('.')[0])
                
                # Display response table
                st.write("Response Table:")
                
                # Create DataFrame for display
                df_data = {
                    'Timestamp': [latest_entry['timestamp']],
                    'Model': [models[latest_entry['model']]['display_name']],
                    'Prompt': [latest_entry['prompt']],
                    'Response': [latest_entry['response']],
                    'Response Time (s)': [f"{latest_entry['latency']:.2f}"],
                    'Temperature': [latest_entry['parameters']['temperature']],
                    'Max Tokens': [latest_entry['parameters']['max_tokens']]
                }
                
                df = pd.DataFrame(df_data)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Display full response in expandable section
                with st.expander("View Full Response"):
                    st.markdown("**Prompt:**")
                    st.text(latest_entry['prompt'])
                    st.markdown("**Response:**")
                    st.text(latest_entry['response'])
                
                # Add export options
                col1, col2 = st.columns(2)
                with col1:
                    # Prepare data for CSV export
                    export_data = []
                    for entry in st.session_state.chat_history:
                        export_data.append({
                            'timestamp': entry['timestamp'],
                            'model': entry['model'],
                            'prompt': entry['prompt'],
                            'response': entry['response'],
                            'latency': entry['latency'],
                            **entry['parameters']
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        "Download Results (CSV)",
                        csv.encode('utf-8'),
                        "chat_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                with col2:
                    # JSON export with all chat history
                    json_data = {
                        'export_timestamp': datetime.now().isoformat(),
                        'chat_history': st.session_state.chat_history
                    }
                    
                    st.download_button(
                        "Download Results (JSON)",
                        json.dumps(json_data, indent=2).encode('utf-8'),
                        "chat_results.json",
                        "application/json",
                        key='download-json'
                    )
                
                # Clear history button
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

    # Add footer
    st.markdown(
        '''
        <div style="margin-top: 50px; padding: 1rem; text-align: center; border-top: 1px solid #e0e0e0;">
            Built by <a href="https://www.ryanmcdonough.co.uk/" target="_blank">Ryan McDonough</a>
        </div>
        ''',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()