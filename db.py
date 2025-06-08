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
import pickle

class ImageVectorDB:
    """
    Manages storing and searching for images and their associated metadata with caching support.
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

    def save(self, filepath):
        """Save the FAISS index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        # Save metadata
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump({
                'id_to_metadata': self.id_to_metadata,
                'current_id': self.current_id
            }, f)

    def load(self, filepath):
        """Load the FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            data = pickle.load(f)
            self.id_to_metadata = data['id_to_metadata']
            self.current_id = data['current_id']

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

    def get_image_embeddings_batch(self, image_pils):
        """Process multiple images simultaneously for better performance."""
        images_tensor = torch.stack([self.transform(img) for img in image_pils]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(images_tensor).squeeze().cpu().numpy()
        # Normalize all embeddings
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

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

    def build_database_from_sources(self, image_folder, metadata_csv, st_ui=None, cache_dir="cache"):
        """
        Builds or loads the database with caching support.
        Returns the number of items in the database.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = os.path.basename(image_folder.rstrip('/'))
        cache_path = os.path.join(cache_dir, cache_name)
        
        # Try to load from cache
        if os.path.exists(f"{cache_path}.index") and os.path.exists(f"{cache_path}.metadata"):
            if st_ui:
                st_ui.info("Loading pre-built database from cache...")
            self.load(cache_path)
            return len(self.id_to_metadata)
        
        # Build from scratch if no cache exists
        added_count = self._build_from_scratch(image_folder, metadata_csv, st_ui)
        if added_count > 0:
            self.save(cache_path)  # Save to cache for next time
        return added_count

    def _build_from_scratch(self, image_folder, metadata_csv, st_ui):
        """Internal method to build database from source files."""
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
        
        if st_ui:
            st_ui.info(f"Available columns in CSV: {list(df.columns)}")
            st_ui.info(f"First few rows:\n{df.head()}")
        
        # Find image column
        image_col = None
        possible_cols = [
            'image', 'filename', 'image_path', 'path', 'file_path', 'image_url',
            'Image', 'Filename', 'ImagePath', 'FilePath', 'img', 'picture',
            'photo', 'image_name', 'file', 'img_path', 'Image Path',
            'feature_image_path', 'pdp_image_paths'
        ]
        
        df_cols_lower = [col.lower() for col in df.columns]
        for col in possible_cols:
            if col.lower() in df_cols_lower:
                image_col = df.columns[df_cols_lower.index(col.lower())]
                break
        
        if image_col:
            df['image_filename'] = df[image_col].apply(lambda x: str(x) if pd.notna(x) else '')
            if st_ui:
                st_ui.success(f"Found image column: '{image_col}'")
        else:
            available_cols = ", ".join(df.columns.tolist())
            error_message = f"Could not find an image filename column in the CSV.\nSearched for: {', '.join(possible_cols)}\nAvailable columns: {available_cols}"
            if st_ui:
                st_ui.error(error_message)
            else:
                print(error_message)
            return 0

        # Batch processing
        batch_size = 32
        image_batch = []
        metadata_batch = []
        added_count = 0
        
        for _, row in df.iterrows():
            image_filename = row['image_filename']
            if not image_filename:
                continue
                
            possible_paths = [
                os.path.join(image_folder, image_filename),
                os.path.join(os.path.dirname(image_folder), image_filename),
                image_filename,
                os.path.join(image_folder, os.path.basename(image_filename))
            ]
            
            image_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_batch.append(image)
                    metadata_batch.append(row.to_dict())
                    
                    if len(image_batch) >= batch_size:
                        embeddings = self.get_image_embeddings_batch(image_batch)
                        self.index.add(embeddings.astype('float32'))
                        
                        for i, (img, meta) in enumerate(zip(image_batch, metadata_batch)):
                            buffered = io.BytesIO()
                            img.save(buffered, format="JPEG")
                            meta['image_data'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            self.id_to_metadata[self.current_id + i] = meta
                        
                        self.current_id += len(image_batch)
                        added_count += len(image_batch)
                        image_batch = []
                        metadata_batch = []
                        
                except Exception as e:
                    if st_ui:
                        st_ui.warning(f"Skipping {image_filename}: {e}")
        
        # Process remaining images in the last batch
        if image_batch:
            embeddings = self.get_image_embeddings_batch(image_batch)
            self.index.add(embeddings.astype('float32'))
            
            for i, (img, meta) in enumerate(zip(image_batch, metadata_batch)):
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                meta['image_data'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
                self.id_to_metadata[self.current_id + i] = meta
            
            added_count += len(image_batch)
            self.current_id += len(image_batch)
        
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
                    'similarity': float(similarity),
                    'metadata': metadata
                })
        
        # Sort results by descending similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results