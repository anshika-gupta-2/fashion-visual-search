import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torchvision import models, transforms
from PIL import Image
import faiss
import numpy as np
import io
import base64
import os
import pandas as pd

class ImageVectorDB:
    """
    Manages storing and searching for images and their associated metadata.
    """
    def __init__(self):
        """
        Initializes the model, transformation pipeline, and FAISS index.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model for feature extraction
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)
        
        self.transform = weights.transforms()
        self.vector_dim = 1280
        
        # FAISS index for cosine similarity (we normalize vectors)
        self.index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity
        
        # Dictionary to map index IDs to metadata
        self.id_to_metadata = {}
        self.current_id = 0

    def get_image_embedding(self, image_pil):
        """
        Converts a PIL image into a normalized feature vector.
        """
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image_tensor).squeeze().cpu().numpy()
        
        # Normalize the vector for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def add_image(self, image_pil, metadata):
        """
        Adds an image and its metadata to the database.
        """
        embedding = self.get_image_embedding(image_pil)
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Store image as base64 for easy display in Streamlit
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        metadata['image_data'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        self.id_to_metadata[self.current_id] = metadata
        self.current_id += 1
        return self.current_id - 1

    def build_database_from_sources(self, image_folder, metadata_csv, st_ui=None):
        """
        Builds the entire database from an image folder and a metadata CSV file.
        This is a more structured way to ingest data for the project.
        """
        if not os.path.exists(metadata_csv):
            if st_ui:
                st_ui.error(f"Metadata file not found at: {metadata_csv}")
            else:
                print(f"Error: Metadata file not found at {metadata_csv}")
            return 0

        if not os.path.exists(image_folder):
            if st_ui:
                st_ui.error(f"Image folder not found at: {image_folder}")
            else:
                print(f"Error: Image folder not found at {image_folder}")
            return 0

        df = pd.read_csv(metadata_csv)
        
        # DEBUG: Show what columns are actually available
        if st_ui:
            st_ui.info(f"Available columns in CSV: {list(df.columns)}")
            st_ui.info(f"First few rows:\n{df.head()}")
        else:
            print(f"Available columns in CSV: {list(df.columns)}")
            print(f"First few rows:\n{df.head()}")
        
        # --- ROBUST COLUMN FINDER ---
        # Search for a valid image column from a list of common names.
        image_col = None
        possible_cols = [
            'image', 'filename', 'image_path', 'path', 'file_path', 'image_url',
            'Image', 'Filename', 'ImagePath', 'FilePath', 'img', 'picture',
            'photo', 'image_name', 'file', 'img_path', 'Image Path',
            'feature_image_path', 'pdp_image_paths'  # Added your specific columns
        ]
        
        # Case-insensitive search
        df_cols_lower = [col.lower() for col in df.columns]
        for col in possible_cols:
            if col.lower() in df_cols_lower:
                # Find the original column name with proper case
                image_col = df.columns[df_cols_lower.index(col.lower())]
                break
        
        if image_col:
            # Create a standardized 'image_filename' column
            # Handle both full paths and just filenames
            df['image_filename'] = df[image_col].apply(lambda x: str(x) if pd.notna(x) else '')
            if st_ui:
                st_ui.success(f"Found image column: '{image_col}'")
        else:
            # If no valid column is found, display an informative error.
            available_cols = ", ".join(df.columns.tolist())
            error_message = f"Could not find an image filename column in the CSV.\nSearched for: {', '.join(possible_cols)}\nAvailable columns: {available_cols}"
            if st_ui:
                st_ui.error(error_message)
            else:
                print(error_message)
            return 0
        # --- END ROBUST COLUMN FINDER ---

        added_count = 0
        for _, row in df.iterrows():
            # Try the full path first, then try relative to image_folder
            image_filename = row['image_filename']
            if not image_filename:  # Skip empty paths
                continue
                
            # Try multiple path combinations
            possible_paths = [
                os.path.join(image_folder, image_filename),  # Original logic
                os.path.join(os.path.dirname(image_folder), image_filename),  # One level up
                image_filename,  # Absolute path
                os.path.join(image_folder, os.path.basename(image_filename))  # Just filename
            ]
            
            image_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    # Convert row to a dictionary for metadata
                    metadata = row.to_dict()
                    self.add_image(image, metadata)
                    added_count += 1
                except Exception as e:
                    if st_ui:
                        st_ui.warning(f"Skipping {image_filename}: {e}")
                    else:
                        print(f"Error processing {image_path}: {e}")
            else:
                if st_ui:
                    st_ui.warning(f"Image not found: {image_filename}")
                else:
                    print(f"Image not found: {image_filename}")
        
        if added_count == 0 and st_ui:
            st_ui.warning("Found data files, but could not add any images. Check image filenames in the CSV match files in the image folder.")

        return added_count

    def search_similar_images(self, query_image_pil, k=5, threshold=0.5):
        """
        Searches for the most similar images to a query image using cosine similarity.
        Returns results sorted by descending similarity (1.0 = most similar).
        
        Args:
            query_image_pil: PIL Image to query with
            k: number of results to return
            threshold: minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of dicts with 'id', 'similarity', and 'metadata'
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.get_image_embedding(query_image_pil)
        
        # Search using inner product (since vectors are normalized, this is cosine similarity)
        similarities, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx >= 0 and similarity >= threshold:
                metadata = self.id_to_metadata.get(idx, {}).copy()
                results.append({
                    'id': idx,
                    'similarity': float(similarity),  # Convert numpy float to Python float
                    'metadata': metadata
                })
        
        # Sort results by descending similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results