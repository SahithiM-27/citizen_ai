from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load small model and tokenizer (gpt2 ~500MB)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/chat", methods=["POST"])
def chat():
    # Get user input
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Tokenize and generate response
    inputs = tokenizer(user_message, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"reply": reply})

# Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
