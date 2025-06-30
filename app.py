from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# IBM Granite model on Hugging Face
MODEL_API_URL = "https://api-inference.huggingface.co/models/ibm/granite-13b-chat-v2"

# You must set your Hugging Face token as an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Call Hugging Face API
    response = requests.post(
        MODEL_API_URL,
        headers=headers,
        json={"inputs": user_input}
    )

    if response.status_code != 200:
        return jsonify({"error": "Model API failed", "details": response.text}), 500

    model_output = response.json()
    generated_text = model_output[0]["generated_text"]

    return jsonify({"reply": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

