from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_path = "ibm-granite/granite-3.3-2b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float32
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    messages = [{"role": "user", "content": user_input}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True,
        thinking=True, add_generation_prompt=True
    ).to(device)
    output = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
