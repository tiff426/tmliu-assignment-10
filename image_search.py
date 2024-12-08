# HEAVILY using the code from the notebook

# stuff to load data set and set up CLIP...?


import os
import torch
#import torchvision.transforms as transforms
from PIL import Image
#from open_clip import create_model_and_transforms, tokenizer
import open_clip
import torch.nn.functional as F
# import pandas as pd
from tqdm import tqdm
import numpy as np
import io
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# # Configuration
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "ViT-B/32"
# pretrained = "openai"
# batch_size = 128
# image_folder = "/scratch/cliao25/train2014/train2014"  # Replace with your folder path

# # Load the model and preprocess function
# model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
# model = model.to(device)
# model.eval()

# def compute_image_embeddings(image_folder, batch_size=128, output_file='image_embeddings.pickle'):
#     # Image transformations (using preprocess_val from open_clip)
#     transform = preprocess_val

#     # Collect all image paths
#     image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     print('Number of images:', len(image_paths))
#     # DataFrame to store results
#     results = []

#     # Function to load and preprocess images
#     def load_images(batch_paths):
#         images = []
#         for path in batch_paths:
#             try:
#                 image = Image.open(path).convert("RGB")
#                 images.append(transform(image))
#             except Exception as e:
#                 print(f"Error loading image {path}: {e}")
#         return torch.stack(images) if images else None

#     # Process images in batches
#     with torch.no_grad():
#         for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
#             batch_paths = image_paths[i:i + batch_size]
#             images = load_images(batch_paths)
#             if images is None:  # Skip if no valid images in this batch
#                 continue

#             images = images.to(device)
#             embeddings = model.encode_image(images)
#             embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize the embeddings

#             for path, emb in zip(batch_paths, embeddings):
#                 results.append({"file_name": os.path.basename(path), "embedding": emb.cpu().numpy()})

#     # Save results to a DataFrame
#     df = pd.DataFrame(results)
#     df.to_pickle('image_embeddings.pickle')

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def find_image(df, query_embedding):
    # keep an array of images and similarities so I can get top 5 later
    impaths = []
    similarities = []

    # Ensure query_embedding is a PyTorch tensor and has the correct dimensions
    # query_embedding = torch.tensor(query_embedding).unsqueeze(0) if query_embedding.ndim == 1 else torch.tensor(query_embedding)
    # trying this instead
    query_embedding = query_embedding.clone().detach()
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.ndim == 1 else query_embedding

    for _, row in df.iterrows():
        # Check if the embedding exists and is valid
        if not isinstance(row['embedding'], (list, torch.Tensor, np.ndarray)):
            print(f"Invalid embedding at row {_}: {row['embedding']}")
            continue
        
        # Convert the dataset embedding to a PyTorch tensor
        dataset_embedding = torch.tensor(row['embedding'], dtype=torch.float32)
        
        # Ensure correct dimensions for cosine similarity
        dataset_embedding = dataset_embedding.unsqueeze(0) if dataset_embedding.ndim == 1 else dataset_embedding

        # Compute cosine similarity
        similarity = F.cosine_similarity(query_embedding, dataset_embedding).item()

        # if similarity > max_similarity:
        #     max_similarity = similarity
        #     impath = row['file_name']  # Store the image path of the closest match
        #     impath = './coco_images_resized/' + impath
        impaths.append(row['file_name'])
        similarities.append(similarity)

    # fix image paths
    #impaths = ['../coco_images_resized/' + path for path in impaths] 
    top_5 = np.argsort(similarities)[::-1][:5]

    top_5_images = [impaths[i] for i in top_5]
    top_5_sims = [similarities[i] for i in top_5]

    return top_5_images, top_5_sims

    # print(f"Closest image path: {impath}")
    # print(f"Highest cosine similarity: {max_similarity:.4f}")

    #return impath, max_similarity

def embed_image(image_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai') # adding -quickgelu
    # This converts the image to a tensor
    #image = preprocess(Image.open("house.jpg")).unsqueeze(0)
    # this image is my own, and then we make query emebdding with this, then we see what in the dataframe best matches in the next cell
    print("IMAGE PATH!!!!", image_path)
    # image = preprocess(Image.open(image_path)).unsqueeze(0)
    if isinstance(image_path, str):
        image = preprocess(Image.open(image_path)).unsqueeze(0)
    else:
        image = Image.open(io.BytesIO(image_path.read())) 
        image = preprocess(image).unsqueeze(0)

    # from IPython.display import Image, display
    # print('This is my query image')
    # display(Image(filename="dog.jpg"))

    # This calculates the query embedding
    query_embedding = F.normalize(model.encode_image(image))
    return query_embedding

def embed_text(text_query):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai') # wait do i want image and text to u se the same model tho
    token = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = token([text_query]) # change this to be what you want...
    query_embedding = F.normalize(model.encode_text(text))
    return query_embedding

def embed_hybrid(image_path, text_query, weight):
    # oh but now i can use my other functions!!
    #image =  #preprocess(Image.open("house.jpg")).unsqueeze(0)
    image_query = embed_image(image_path)# F.normalize(model.encode_image(image))
    #text = tokenizer(["snowy"])
    text_query = embed_text(text_query)#F.normalize(model.encode_text(text))

    # we use lam + query for the weighted average
    # i find that lower lam (aka weighting text less and image more) results in higher cosine similrity
    # but then it doesnt appear as though the actual image is as good..?
    # so liek overfitting... ?
    lam  = weight # tune this

    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query)
    return query_embedding

# Already completed for you
def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

# # PCA stuff
# train_images, train_image_names = load_images(image_dir, max_images=2000, target_size=(224, 224))
# print(f"Loaded {len(train_images)} images for PCA training.")

# # Apply PCA
# k = 50 # Number of principal components (eg: 50)
# pca = PCA(k) #initialize PCA with no. of components
# #TODO  # Fit PCA on the training subset
# fit_pca = pca.fit_transform(train_images)
# print(f"Trained PCA on {len(train_images)} samples.")

# #transform 10000 photos
# transform_images, transform_image_names = load_images(image_dir, max_images=10000, target_size=(224, 224))
# print(f"Loaded {len(transform_images)} images for transformation.")
# reduced_embeddings = pca.transform(transform_images)  # Transform only the first 10,000 images
# print(f"Reduced embeddings for {len(transform_images)} images.")

def find_image_pca(df, k, image_path):
    impaths = []
    distances = []

    #this is the framework function for getting images when requested pca embedding type 

    # Directory containing images
    image_dir = "results" # Your folder path

    #train pca
    train_images, train_image_names = load_images(image_dir, max_images=2000, target_size=(224, 224))
    print(f"Loaded {len(train_images)} images for PCA training.")

    # Apply PCA
    #k = 50 # Number of principal components (eg: 50)
    pca = PCA(k) #initialize PCA with no. of components
    #TODO  # Fit PCA on the training subset
    #fit_pca = pca.fit_transform(train_images)
    pca.fit(train_images)
    print(f"Trained PCA on {len(train_images)} samples.")

    #transform 10000 photos
    transform_images, transform_image_names = load_images(image_dir, max_images=10000, target_size=(224, 224))
    print(f"Loaded {len(transform_images)} images for transformation.")
    reduced_embeddings = pca.transform(transform_images)  # Transform only the first 10,000 images
    print(f"Reduced embeddings for {len(transform_images)} images.")

    #query_embedding = reduced_embeddings[query_idx] instead of this, it's jsut our embedded image
    #images = []
    image = Image.open(io.BytesIO(image_path.read())) 
    image = image.convert('L')  # Convert to grayscale ('L' mode)
    image = image.resize((224, 224))  # Resize to target size
    img_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image = img_array.flatten()  # Flatten to 1D
    #images.append(image)

    query_embedding = pca.transform(np.array(image).reshape(1, -1))

    # top_indices, top_distances = nearest_neighbors(query_embedding, reduced_embeddings)

    # #top_5 = np.argsort(top_distances)[::-1][:5]

    # top_5_images = [transform_image_names[i] for i in top_indices]
    # top_5_sims = [top_distances[i] for i in top_indices]

    # return top_5_images, top_5_sims
    top_indices, top_distances = nearest_neighbors(query_embedding, reduced_embeddings)
    print("Top indices:", top_indices)
    print("Top distances:", top_distances)
    for i, index in enumerate(top_indices):
        impath = df['file_name'].iloc[index]
        similarity = top_distances[i].item()
        impaths.append(impath)
        distances.append(similarity)
        
    return impaths, distances


def pca_image(image_path):
    # #gets query_embeddings with image search
    # transform_query, transform_query_names = load_images(image_path, max_images=10, target_size=(224, 224))
    # print(f"Loaded {len(transform_query)} images for transformation.")
    # # reduced_embeddings = pca.transform(transform_query)  # Transform only the first 10,000 images
    # # print(f"Reduced embeddings for {len(transform_query)} images.")
    # return transform_query[0]
    impaths = []
    distances = []
    images = []
    image = Image.open(io.BytesIO(image_path.read())) 
    image = image.convert('L')  # Convert to grayscale ('L' mode)
    image = image.resize((224, 224))  # Resize to target size
    img_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image = img_array.flatten()  # Flatten to 1D
    images.append(image)
    query_embedding = pca.transform(np.array(image).reshape(1, -1))
    return query_embedding

def nearest_neighbors(query_embedding, embeddings, top_k=7):
    # query_embedding: The embedding of the query item (e.g., the query image) in the same dimensional space as the other embeddings.
    # embeddings: The dataset of embeddings that you want to search through for the nearest neighbors.
    # top_k: The number of most similar items (nearest neighbors) to return from the dataset.
    # Hint: flatten the "distances" array for convenience because its size would be (1,N)
    distances = euclidean_distances(query_embedding, embeddings).flatten() #Use euclidean distance
    nearest_indices = np.argsort(distances)[:top_k] #get the indices of ntop k results
    return nearest_indices, distances[nearest_indices]
