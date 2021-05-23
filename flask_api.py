from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'Normal Person', 1 : 'COVID Patient'}

model = load_model('CNN_COVID.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(512,512))
	i = image.img_to_array(i)
	i = i.reshape(3,512,512,1)
	p = model.predict_classes(i)
	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "THIS API HELPS TO IDENTIFY COVID PATIENTS CT SCANS"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)