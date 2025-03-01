import os
import openai

# Optionally, one could integrate LangChain for more complex chains or local models.
# For simplicity, we use OpenAI's API for text generation (chat completion).

# Set API key (ensure this is set as an env variable for security in real deployment)
openai.api_key = os.getenv("OPENAI_API_KEY", default="")  # Expect the key to be set in environment

def generate_response(prompt: str) -> str:
    """
    Generate a response for a given prompt using an LLM (OpenAI GPT-3.5 or similar).
    If OpenAI API is not available, returns a placeholder or uses a local model.
    """
    if openai.api_key:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = resp['choices'][0]['message']['content'].strip()
            return reply
        except Exception as e:
            # In case of API error, return an error message
            return f"(Error generating response: {e})"
    else:
        # Fallback if no API key is provided – here we could integrate a local model via LangChain or transformers
        # For hackathon simplicity, we return a default message.
        return "(No API key provided; unable to generate response.)"

# Example usage
if __name__ == "__main__":
    test_prompt = "I had a wonderful experience at the restaurant, thanks for asking!"
    print("User:", test_prompt)
    print("LLM Response:", generate_response(test_prompt))
