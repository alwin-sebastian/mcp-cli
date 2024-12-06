import os
import uuid
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List
import re

# Load environment variables
load_dotenv()

class LLMClient:
    def __init__(self, model_id: str = None, api_token: str = None):
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.api_token = "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"
        
        if not self.api_token:
            raise ValueError("The HUGGING_FACE_API_TOKEN environment variable is not set.")
            
        self.client = InferenceClient(token=self.api_token)

    def _format_prompt(self, messages: List[Dict], tools: List = None) -> str:
        prompt = ""
        
        # Process messages
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System Instructions:\n{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
                
        # Add available tools information if tools exist and are properly formatted
        if tools and isinstance(tools, list):
            prompt += "\nAvailable Tools:\n"
            for tool in tools:
                if isinstance(tool, dict):
                    # Safely get tool information with defaults
                    tool_name = tool.get('function', {}).get('name') or tool.get('name', 'Unnamed Tool')
                    tool_description = tool.get('description', 'No description available')
                    tool_parameters = tool.get('parameters', {})
                    
                    prompt += f"- Tool: {tool_name}\n"
                    prompt += f"  Description: {tool_description}\n"
                    if tool_parameters:
                        prompt += f"  Parameters: {json.dumps(tool_parameters)}\n"
                    prompt += "\n"
        
        # Add final prompt for assistant
        prompt += "\nAssistant: "
        return prompt

    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        try:
            # Create the formatted prompt
            prompt = self._format_prompt(messages, tools)
            
            # Get completion from model
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=512,
                return_full_text=False
            )
            
            response_text = str(response).strip()
            tool_calls = []
            
            # Check if response contains tool calls
            if "TOOL_CALL:" in response_text:
                try:
                    # Extract tool call JSON
                    tool_call_match = re.search(r'TOOL_CALL:\s*({.*})', response_text)
                    if tool_call_match:
                        tool_data = json.loads(tool_call_match.group(1))
                        tool_calls.append({
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": tool_data.get("name", ""),
                                "arguments": tool_data.get("arguments", {})
                            }
                        })
                        # Remove the TOOL_CALL from response text
                        response_text = response_text.replace(tool_call_match.group(0), "").strip()
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse tool call: {e}")
            
            return {
                "response": response_text,
                "tool_calls": tool_calls
            }
            
        except Exception as e:
            logging.error(f"HuggingFace Inference API Error: {str(e)}")
            raise ValueError(f"HuggingFace Inference API Error: {str(e)}")