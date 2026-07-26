import os
import time
import json
from dotenv import load_dotenv
from groq import Groq

# Try to load environment variables from .env
load_dotenv()

class EmailResponder:
    def __init__(self, retriever, model_name: str = 'llama-3.3-70b-versatile'):
        self.retriever = retriever
        self.model_name = model_name
        
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def _build_prompt(self, incoming_email: dict, retrieved_examples: list[dict]) -> str:
        prompt = (
            "System Instructions:\n"
            "You are a professional customer support representative. "
            "Write a response to the incoming email addressing every question or concern clearly. "
            "Provide clear next steps if applicable. "
            "Match the tone and style of the reference examples provided. "
            "Always sign off as 'Support Team'. "
            "Do NOT include subject lines, headers, or metadata. Output ONLY the response body.\n\n"
        )
        
        if retrieved_examples:
            prompt += "Reference Examples:\n\n"
            for i, example in enumerate(retrieved_examples, 1):
                inc = example.get("incoming_email", {})
                ref = example.get("reference_response", {})
                prompt += f"--- Example {i} ---\n"
                prompt += f"Incoming Email:\nSubject: {inc.get('subject', '')}\nBody: {inc.get('body', '')}\n\n"
                prompt += f"Reference Response:\n{ref.get('body', '')}\n\n"
        
        prompt += "--- Task ---\n"
        prompt += "Respond to the following incoming email:\n"
        prompt += f"Subject: {incoming_email.get('subject', '')}\n"
        prompt += f"Body: {incoming_email.get('body', '')}\n\n"
        prompt += "Response Body:\n"
        return prompt

    def generate_response(self, incoming_email: dict) -> dict:
        incoming_text = f"{incoming_email.get('subject', '')}\n{incoming_email.get('body', '')}"
        
        retrieved_examples = self.retriever.retrieve(incoming_text, top_k=3)
        prompt = self._build_prompt(incoming_email, retrieved_examples)
        
        if not self.client:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in .env or environment.")
            
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.choices[0].message.content.strip()
                
                # Constructing the expected output dictionary
                subject = incoming_email.get("subject", "")
                if not subject.lower().startswith("re:"):
                    subject = f"Re: {subject}"
                    
                return {
                    "subject": subject,
                    "body": response_text
                }
                
            except Exception as e:
                print(f"Generation error: {e}")
                if attempt == max_retries - 1:
                    return {
                        "subject": f"Re: {incoming_email.get('subject', '')}",
                        "body": f"Error generating response: {str(e)}"
                    }
                time.sleep(base_delay * (2 ** attempt))

    def generate_batch(self, emails: list[dict], max_workers: int = 1) -> list[dict]:
        results = []
        for email in emails:
            # Simple rate limiting via sleep
            time.sleep(1)
            results.append(self.generate_response(email))
        return results
