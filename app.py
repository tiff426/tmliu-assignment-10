from flask import Flask, render_template, request, jsonify, send_from_directory
import os
# from neural_networks import visualize
from image_search import get_similar_images

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/image_search', methods=['POST'])
def image_search():
    query_type = request.form['query_type']
    use_pca = request.form.get('use_pca', 'false').lower() == 'true'
    k = int(request.form.get('k', 10))

    if query_type == "text":
        text_query = request.form['text_query']
        images = get_similar_images(query_type, text_query=text_query, use_pca=use_pca, k=k)
    elif query_type == "image":
        image_file = request.files['image_query']
        image_query = Image.open(image_file)
        images = get_similar_images(query_type, image_query=image_query, use_pca=use_pca, k=k)
    elif query_type == "hybrid":
        text_query = request.form['text_query']
        weight = float(request.form['weight'])
        image_file = request.files['image_query']
        image_query = Image.open(image_file)
        images = get_similar_images(query_type, text_query=text_query, image_query=image_query, weight=weight, use_pca=use_pca, k=k)
    
    return jsonify({"images": images})

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)