from dotenv import load_dotenv
import openai
import numpy as np

load_dotenv()

client = openai.OpenAI()

d = 2

def get_cosine_similarity(embeddingi, embeddingj):
    embeddingi = np.array(embeddingi.data[0].embedding)
    embeddingj = np.array(embeddingj.data[0].embedding)
    cos_sim = embeddingi.dot(embeddingj) / (np.linalg.norm(embeddingi) * np.linalg.norm(embeddingj))
    return cos_sim

embedding1 = client.embeddings.create(model='text-embedding-3-small', input='- The app shall allow users to upload images.\n - The app shall provide a gallery to view the uploaded images.', dimensions=d)
embedding2 = client.embeddings.create(model='text-embedding-3-small', input='- The app shall allow users to upload images.\n- The app shall provide a way for users to view uploaded images.\n- The app shall organize uploaded images into a gallery format.', dimensions=d)

print(embedding1.data[0].embedding)
print(embedding2.data[0].embedding)

print(get_cosine_similarity(embedding1, embedding2))




