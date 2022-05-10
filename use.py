from flask import Flask, request, jsonify
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_hub as hub

model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
app = Flask(__name__)

@app.route('/embed', methods=['GET'])
def embed():
	text = request.args.get('text')
	try:
		embedding = model([text]).numpy().tolist()
		return jsonify({'text': text, 'embedding': embedding, 'status': 'ok'})

	except Exception as e:
		print(e)
		return jsonify({'text': text, 'status': 'error'})

@app.route('/')
def index():
	return "Hello From MLthon"

if __name__ == '__main__':
    app.run()
