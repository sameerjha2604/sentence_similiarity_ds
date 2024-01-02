# Imort Name Spaces"
import pandas as pd  
import numpy as np   
import tensorflow as tf      
import tensorflow_hub as hub  
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Load Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
  
  
def embed(input):
    return model(input)
def create_model():
  df = pd.read_csv('Precily_Text_Similarity.csv', encoding = 'unicode_escape')
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
  message = [df['text1'][0], df['text2'][0]]
  message_embeddings = model.encode(message)
# Initialize Sentence Transformer Model
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Load Text Data from DataFrame
  texts1 = df['text1'].tolist()
  texts2 = df['text2'].tolist()
# Generate Sentence Embeddings
  embeddings1 = model.encode(texts1)
  embeddings2 = model.encode(texts2)

# Compute Cosine Similarity Matrix
  cosine_similarities = np.diag(cosine_similarity(embeddings1, embeddings2))

# Create DataFrame with Similarity Scores

  Ans = pd.DataFrame({'Similarity_Score': cosine_similarities})

#Join DataFrames based on index
  df = df.join(Ans)
#Perform normalization 
  df['Similarity_Score'] = (df['Similarity_Score'] + 1)
  df['Similarity_Score'] = df['Similarity_Score']/df['Similarity_Score'].abs().max()
  print(df.head())
create_model()
# create a rest api
