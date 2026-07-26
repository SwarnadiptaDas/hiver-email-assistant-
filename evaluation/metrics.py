import os
import json
import time
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class EmbeddingMetrics:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def semantic_similarity(self, generated: str, reference: str) -> float:
        """Compute cosine similarity between generated and reference responses."""
        if not generated or not reference:
            return 0.0
            
        embeddings = self.model.encode([generated, reference])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

class LLMMetrics:
    def __init__(self, model_name: str = 'llama-3.3-70b-versatile'):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.sleep_time = 0.5
        
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response."""
        data = None
        try:
            # Try parsing directly
            data = json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from markdown block if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            if not data:
                # Try to find something that looks like JSON structure
                json_match = re.search(r'(\{[\s\S]*\})', text)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                        
        if not data:
            print(f"Failed to parse JSON from response: {text}")
            return {"score": 0.0, "reasoning": "Failed to parse LLM evaluation JSON"}

        # Normalize score field if it's nested or missing
        if "score" not in data:
            # Check for common variants
            for k in ["overall_score", "rating"]:
                if k in data:
                    data["score"] = data[k]
                    break
            else:
                data["score"] = 0.0

        # If score is a dictionary or non-float/int, extract it
        if isinstance(data["score"], dict):
            # Check for nested values
            for k in ["score", "rating", "overall"]:
                if k in data["score"]:
                    data["score"] = data["score"][k]
                    break
            else:
                data["score"] = 0.0
                
        try:
            data["score"] = float(data["score"])
        except (ValueError, TypeError):
            data["score"] = 0.0

        return data

    def _call_llm(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call LLM with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                time.sleep(self.sleep_time)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.choices[0].message.content.strip()
                return self._parse_json_response(response_text)
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {"score": 0.0, "reasoning": f"LLM evaluation failed after {max_retries} attempts: {str(e)}"}
                time.sleep(2 ** attempt)  # Exponential backoff
        return {"score": 0.0, "reasoning": "LLM evaluation failed"}

    def intent_coverage(self, incoming_email: str, generated_response: str) -> dict:
        prompt = (
            "Given this incoming email and the generated response, identify all distinct questions/requests/concerns "
            "in the incoming email, then determine which ones are addressed in the response. "
            "Return JSON with score (0-1), intents_found (list), intents_addressed (list), reasoning (string).\n\n"
            f"Incoming Email:\n{incoming_email}\n\n"
            f"Generated Response:\n{generated_response}"
        )
        result = self._call_llm(prompt)
        if 'intents_found' not in result:
            result['intents_found'] = []
        if 'intents_addressed' not in result:
            result['intents_addressed'] = []
        return result

    def tone_score(self, incoming_email: str, generated_response: str) -> dict:
        prompt = (
            "Rate the professional tone, empathy, and appropriateness of this customer support response to the incoming email. "
            "Consider: Is it professional? Empathetic where needed? Appropriate for the context? "
            "Return JSON with score (0-1) and reasoning.\n\n"
            f"Incoming Email:\n{incoming_email}\n\n"
            f"Generated Response:\n{generated_response}"
        )
        return self._call_llm(prompt)

    def completeness_score(self, incoming_email: str, generated_response: str, reference_response: str) -> dict:
        prompt = (
            "Compare the generated response against the reference response in the context of the incoming email. "
            "Does the generated response cover all the key information and points that the reference covers? "
            "Return JSON with score (0-1), missing_points (list), and reasoning.\n\n"
            f"Incoming Email:\n{incoming_email}\n\n"
            f"Reference Response:\n{reference_response}\n\n"
            f"Generated Response:\n{generated_response}"
        )
        result = self._call_llm(prompt)
        if 'missing_points' not in result:
            result['missing_points'] = []
        return result

    def actionability_score(self, generated_response: str) -> dict:
        prompt = (
            "Does this response provide clear next steps for the customer? Are there specific actions, timelines, or follow-up instructions? "
            "Return JSON with score (0-1), actions_found (list), and reasoning.\n\n"
            f"Generated Response:\n{generated_response}"
        )
        result = self._call_llm(prompt)
        if 'actions_found' not in result:
            result['actions_found'] = []
        return result

    def fluency_score(self, generated_response: str) -> dict:
        prompt = (
            "Rate the grammatical correctness, readability, and natural flow of this email response. "
            "Return JSON with score (0-1) and reasoning.\n\n"
            f"Generated Response:\n{generated_response}"
        )
        return self._call_llm(prompt)

    def overall_judge(self, incoming_email: str, generated_response: str, reference_response: str) -> dict:
        prompt = (
            "You are evaluating a customer support email response. Given the incoming email, the generated response, and the reference response, "
            "provide a holistic quality score. Consider: Does it answer the question? Is it helpful? Professional? Would a customer be satisfied? "
            "Return JSON with score (0-1) and reasoning.\n\n"
            f"Incoming Email:\n{incoming_email}\n\n"
            f"Reference Response:\n{reference_response}\n\n"
            f"Generated Response:\n{generated_response}"
        )
        return self._call_llm(prompt)
