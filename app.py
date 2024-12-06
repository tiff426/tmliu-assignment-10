from flask import Flask, render_template, request, jsonify, send_from_directory
import os
# from neural_networks import visualize
from image_search import compute_image_embeddings, find_image, embed_image, embed_text, embed_hybrid
from PIL import Image
import pandas as pd
import torch

app = Flask(__name__)

# is there where i define df
df = df = pd.read_pickle('image_embeddings.pickle')

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
        # Get the query embedding for the text
        query_embedding = embed_text(text_query)
        images, max_similarity = find_image(df, query_embedding)  # df is your dataset with embeddings
    elif query_type == "image":
        image_file = request.files['image_query']
        image_path = image_file.filename  # You might need to save the image or use a temporary path
        # Get the query embedding for the image
        query_embedding = embed_image(image_path)
        images, max_similarity = find_image(df, query_embedding)  # df is your dataset with embeddings
    elif query_type == "hybrid":
        text_query = request.form['text_query']
        weight = float(request.form['weight'])
        image_file = request.files['image_query']
        image_path = image_file.filename  # Similarly, save or handle the image path
        # Get the query embeddings for both image and text
        image_query_embedding = embed_image(image_path)
        text_query_embedding = embed_text(text_query)
        # Combine the embeddings using the hybrid method
        query_embedding = embed_hybrid(image_path, text_query)
        images, max_similarity = find_image(df, query_embedding)  # df is your dataset with embeddings
    
    return jsonify({"images": images, "similarity": max_similarity})


# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)