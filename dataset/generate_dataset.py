import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file if it exists
load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY environment variable is not set. Generation will fail if key is not found.")
    
genai.configure(api_key=api_key)

# We use the recommended gemini-2.0-flash model
model = genai.GenerativeModel('gemini-2.0-flash')

CATEGORIES = {
    "billing": 7,
    "technical_support": 8,
    "feature_request": 6,
    "account_management": 6,
    "onboarding": 6,
    "complaint": 7,
    "general_inquiry": 5,
    "follow_up": 5
}

PROMPT_TEMPLATE = """
You are an expert customer service dataset creator.
Generate a JSON array containing EXACTLY {count} realistic customer support email pairs for the category "{category}".

The JSON must exactly match this schema for the elements:
[
  {{
    "id": "A unique ID string (e.g., email_{category}_001)",
    "category": "{category}",
    "incoming_email": {{
      "from": "sender email address",
      "subject": "realistic subject line",
      "body": "The email body, 2-6 paragraphs. Include realistic details."
    }},
    "reference_response": {{
      "subject": "Re: realistic subject line",
      "body": "The response body, 2-5 paragraphs. Professional and helpful."
    }},
    "metadata": {{
      "urgency": "low|medium|high",
      "sentiment": "positive|neutral|negative",
      "requires_action": true|false
    }}
  }}
]

Make the emails realistic, vary the tone, complexity, and sender details.
Use realistic plan names, feature names, error codes, and company names.
Respond ONLY with the JSON array, without any markdown formatting like ```json.
"""

async def generate_category(category: str, count: int) -> list:
    print(f"Generating {count} emails for category: {category}...")
    prompt = PROMPT_TEMPLATE.format(category=category, count=count)
    
    try:
        response = await model.generate_content_async(prompt)
        text = response.text.strip()
        
        # Clean up markdown if the model still returns it
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
            
        if text.endswith("```"):
            text = text[:-3]
            
        text = text.strip()
        
        data = json.loads(text)
        print(f"Successfully generated {len(data)} emails for {category}.")
        return data
    except Exception as e:
        print(f"Error generating or parsing JSON for {category}: {e}")
        return []

async def generate_dataset():
    print(f"Starting generation of synthetic email dataset across {len(CATEGORIES)} categories...")
    
    tasks = []
    for category, count in CATEGORIES.items():
        tasks.append(generate_category(category, count))
    
    results = await asyncio.gather(*tasks)
    
    emails = []
    category_counts = {}
    for i, data in enumerate(results):
        category = list(CATEGORIES.keys())[i]
        category_counts[category] = len(data)
        emails.extend(data)
        
    dataset = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_count": len(emails),
            "categories": category_counts
        },
        "emails": emails
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "email_dataset.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset generation complete. Saved {len(emails)} emails to {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_dataset())
