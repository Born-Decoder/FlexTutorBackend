from flask import Flask, make_response, send_file, jsonify, request
from werkzeug.utils import secure_filename
import google.generativeai as genai
import os
from transformers import SeamlessM4Tv2Model, AutoProcessor
import torch
from flask import send_file
import json


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained('D:\Manoj\cs6460\Projects\Models\Translation')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model_device = model.to(device)
genmodel = genai.GenerativeModel('gemini-pro')
aud_folder = "res/Audio"
txt_folder = "res/Text"
vid_folder = "res/Video"


# Flask Constructor
app = Flask(__name__)
gemini_cred = 'credentials.json'
resids = 'resids.json'
with open(gemini_cred) as cred_data:
    creddata = json.load(cred_data)
with open(resids) as cont_data:
    contdata = json.load(cont_data)

apikey = creddata['gemini-key']
os.environ['GOOGLE_CLOUD_API_KEY'] = apikey


genai.configure(api_key=os.environ['GOOGLE_CLOUD_API_KEY'])
# decorator to associate 
# a function with the url


@app.route("/")
def showHomePage():
	# response from the server
	return "This is home page"


@app.route("/listcontent", methods=["POST"])
def listcontent():
	_ = request.form["allcontent"]
	print(contdata)
	return jsonify(contdata)


@app.route('/getaudio/<string:filename>', methods=["GET"])
def getaudio(filename):
	try:
		filename = secure_filename(filename)
		file_path = os.path.join(f"{aud_folder}/", filename)
		print(f'got filename {file_path}')
		return send_file(file_path, as_attachment=True)
	except Exception as e:
		print(f'error on filename {filename}')
		return make_response(f"Error: {str(e)}", 500)


@app.route("/gettext", methods=["POST"])
def gettext():
	text = request.form["intext"]
	print(text)
	contname = contdata['title'][contdata['id'].index(text)]
	with open(f'{txt_folder}/{contname}.txt', 'r') as f:
		txtcont = f.read()
	return txtcont


@app.route("/tutor", methods=["POST"])
def tutor():
	text = request.form["intext"]
	id = request.form["id"]
	print(text)
	contname = contdata['title'][contdata['id'].index(id)]
	with open(f'{txt_folder}/{contname}.txt', 'r') as f:
		txtcont = f.read()
	prompt = f"""
	Content: {txtcont}

	Using the above content as reference answer the below question if the question is different from the content,
	explicitly mention that you are answering out of the content in general -
	Question: {text}
	"""
	response = genmodel.generate_content(prompt)
	res_text = response.text.replace('*', '')
	print(res_text)
	return res_text


@app.route("/translate", methods=["POST"])
def translate():
	text = request.form["intext"]
	full_translation = ''
	for sntn in text.split('\n'):
		text_inputs = processor(text = sntn, src_lang="eng", return_tensors="pt").to(device)
		output_tokens = model_device.generate(**text_inputs, tgt_lang="tam", generate_speech=False)
		translated_sntn = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
		print(translated_sntn)
		full_translation = full_translation + '\n' + translated_sntn
	return full_translation


if __name__ == "__main__":
	model = genai.GenerativeModel('gemini-pro')
	app.run(debug=True, port=5000, host='0.0.0.0')

