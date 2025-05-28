import together
from typing import Dict, Optional, Union, List
import time

class TogetherClient:
    def __init__(self, api_key: str):
        """Initialize the Together.ai client with an API key."""
        self.api_key = api_key
        together.api_key = api_key

    def get_available_models(self) -> dict:
        """Get available models from Together.ai API."""
        try:
            response = together.Models.list()
            if not response:
                return self._get_default_models()
            
            models = {}
            for model in response:
                model_id = model.get('id')
                if not model_id:
                    continue
                    
                display_name = model.get('display_name')
                if isinstance(display_name, dict):
                    display_name = display_name.get('name', model_id)
                elif not display_name:
                    display_name = model_id
                    
                models[model_id] = {
                    'display_name': str(display_name),
                    'organization': str(model.get('organization', 'Unknown')),
                    'context_length': int(model.get('context_length', 0)),
                    'type': str(model.get('type', '')),
                    'pricing': {
                        'input': float(model.get('pricing', {}).get('input', 0)),
                        'output': float(model.get('pricing', {}).get('output', 0))
                    }
                }
            
            return models if models else self._get_default_models()
            
        except Exception:
            return self._get_default_models()
            
    def _get_default_models(self) -> dict:
        """Return default models when API fails."""
        return {
            "meta-llama/Llama-3.2-3B-Instruct-Turbo": {
                "display_name": "Meta Llama 3.2 3B Instruct Turbo",
                "organization": "Meta",
                "context_length": 131072,
                "type": "chat",
                "pricing": {"input": 0.06, "output": 0.06}
            },
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
                "display_name": "Meta Llama 3.3 70B Instruct Turbo",
                "organization": "Meta",
                "context_length": 131072,
                "type": "chat",
                "pricing": {"input": 0.88, "output": 0.88}
            },
            "meta-llama/Llama-3.1-8B-Instruct-Turbo": {
                "display_name": "Meta Llama 3.1 8B Instruct Turbo",
                "organization": "Meta",
                "context_length": 131072,
                "type": "chat",
                "pricing": {"input": 0.18, "output": 0.18}
            },
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                "display_name": "Mixtral 8x7B Instruct",
                "organization": "Mistral AI",
                "context_length": 32768,
                "type": "chat",
                "pricing": {"input": 0.6, "output": 0.6}
            },
            "mistralai/Mistral-7B-Instruct-v0.1": {
                "display_name": "Mistral 7B Instruct",
                "organization": "Mistral AI",
                "context_length": 8192,
                "type": "chat",
                "pricing": {"input": 0.2, "output": 0.2}
            }
        }

    def generate_completion(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[list] = None
    ) -> Dict:
        """
        Generate a completion using the Together.ai API.
        
        Args:
            prompt: The input prompt
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            stop: List of stop sequences
            
        Returns:
            Dict containing the response and metadata
        """
        start_time = time.time()
        
        try:
            response = together.Complete.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop
            )
            
            latency = time.time() - start_time
            
            if isinstance(response, dict) and 'choices' in response:
                choices = response['choices']
                if choices and isinstance(choices[0], dict) and 'text' in choices[0]:
                    return {
                        "text": choices[0]['text'],
                        "latency": latency,
                        "tokens": response.get('usage', {}).get('total_tokens', 0),
                        "error": None
                    }
            
            return {
                "text": None,
                "latency": latency,
                "tokens": 0,
                "error": "Invalid response format"
            }
            
        except Exception as e:
            return {
                "text": None,
                "latency": time.time() - start_time,
                "tokens": 0,
                "error": str(e)
            }

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[list] = None
    ) -> Dict:
        """
        Generate a chat completion using the Together.ai API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            stop: List of stop sequences
            
        Returns:
            Dict containing the response and metadata
        """
        start_time = time.time()
        
        try:
            # Use the Chat.completions API for chat models
            response = together.Chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop
            )
            
            latency = time.time() - start_time
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return {
                        "text": choice.message.content,
                        "latency": latency,
                        "tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                        "error": None
                    }
            
            # Fallback to completion API format
            return self.generate_completion(
                prompt="\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop
            )
            
        except Exception as e:
            # If chat API fails, try completion API
            try:
                return self.generate_completion(
                    prompt="\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]),
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop=stop
                )
            except Exception as e2:
                return {
                    "text": None,
                    "latency": time.time() - start_time,
                    "tokens": 0,
                    "error": f"Chat API error: {str(e)}, Completion API error: {str(e2)}"
                }