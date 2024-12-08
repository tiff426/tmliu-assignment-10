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

def embed_hybrid(image_path, text_query):
    # oh but now i can use my other functions!!
    #image =  #preprocess(Image.open("house.jpg")).unsqueeze(0)
    image_query = embed_image(image_path)# F.normalize(model.encode_image(image))
    #text = tokenizer(["snowy"])
    text_query = embed_text(text_query)#F.normalize(model.encode_text(text))

    # we use lam + query for the weighted average
    # i find that lower lam (aka weighting text less and image more) results in higher cosine similrity
    # but then it doesnt appear as though the actual image is as good..?
    # so liek overfitting... ?
    lam  = 0.1 # tune this

    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query)
    return query_embedding
