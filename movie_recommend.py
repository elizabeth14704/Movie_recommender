                        #Movie Recommender System with TF-IDF and Deep Learning

# Content-Based Movie Recommendations Using NLP & Embedding Similarity
# This project is a movie recommender web application built using Streamlit, allowing users to select a movie and receive personalized recommendations. It supports two recommendation strategies: traditional TF-IDF-based similarity and deep learning embedding similarity.



# Library Imports and Setup
import streamlit as st 
import pandas as pd 
import numpy as np 
import string 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer 
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Input,Embedding, Dot, Flatten
# All essential libraries are imported for preprocessing (pandas, numpy), feature engineering (TfidfVectorizer, LabelEncoder), similarity computation (cosine_similarity), model creation (TensorFlow), and web interface development (Streamlit).



#data = {
#    'movie_id': [1,2,3,4,5,6,7,8,9,10],
#    'movie_title': ['3 Idiots', 'Baahubali', 'Super Deluxe', 'Drishyam', #'Dangal','RRR', 'KGF', 'Kantara', 'Article 15', 'Piku'],
#    'movie_genre': ['Comedy', 'Historical', 'Thriller Drama',     'Crime Thriller', 'Sports Drama','Action Historical', 'Action Thriller', 'Mystery Drama', 'Crime Drama', 'Comedy Drama'] ,
#    'movie_language': ['Hindi', 'Telugu', 'Tamil', 'Malayalam', 'Hindi',
#        'Telugu', 'Kannada', 'Kannada', 'Hindi', 'Hindi'], 
#    'movie_poster': ['https://c8.alamy.com/comp/P4APGH/original-film-title-3-idiots-english-title-3-idiots-film-director-rajkumar-hirani-year-2009-credit-eros-international-media-album-P4APGH.jpg', 'https://c8.alamy.com/comp/HAH93T/bahubali-the-beginning-year-2015-india-director-ss-rajamouli-prabhas-HAH93T.jpg', 'https://i.pinimg.com/736x/7c/0e/df/7c0edfab528d92e21a5a9b3db15abaa9.jpg', 'https://m.media-amazon.com/images/M/MV5BM2Q2YTczM2QtNDBkNC00M2I5LTkyMzgtOTMwNzQ0N2UyYWQ0XkEyXkFqcGc@._V1_.jpg', 'https://media5.bollywoodhungama.in/wp-content/uploads/2016/03/Dangal-1.jpg', 'https://media5.bollywoodhungama.in/wp-content/uploads/2019/03/RRR-2022.jpeg', 'https://wallpapers.com/images/hd/cool-kgf-chapter-2-poster-0axobl2ual8wurln.jpg', 'https://hombalefilms.com/wp-content/uploads/2021/08/KANTARA_ENG.jpg', 'https://mir-s3-cdn-cf.behance.net/project_modules/1400/060dd880942383.5cf027b4aad73.jpg', 'https://www.filmibeat.com/wimg/mobi/2015/04/piku-wallpaper_142976700300.jpg']
#    }

#df = pd.DataFrame(data)
#df['movie_features'] = df['movie_genre']+ ' ' + df['movie_language']




# Data Loading and Preprocessing
df = pd.read_csv('C:\\Users\\basil\\Downloads\\movies (1).csv')
df = df.dropna(subset= ['Movie_Name', 'Genre']).drop_duplicates(subset= 'Movie_Name')
df['movie_features'] = df['Genre']  
# Reads the movie dataset, removes duplicates or missing values, and prepares the genre information as the core "movie feature" for similarity analysis.



# TF-IDF Vectorization & Cosine Similarity
tfid = TfidfVectorizer(stop_words= 'english')
tfid_matrix = tfid.fit_transform(df['Genre'])
cosine_sim = cosine_similarity(tfid_matrix)
# Uses the TF-IDF method to convert genres into numerical vectors, followed by computing cosine similarity between all pairs of movies.



# Deep Learning Embedding Model
movie_enconder = LabelEncoder()
df['Movie_id'] = movie_enconder.fit_transform(df['Movie_Name'])
num_movie = len(df)
Embedding_dim = 50

input_movie = Input(shape = (1,))
embedding = Embedding(num_movie, Embedding_dim, input_length = 1)(input_movie)
flatten = Flatten()(embedding)

model = Model (inputs = input_movie, outputs = flatten)
movie_embedding = model.predict(np.array(df['Movie_id']))
deep_sim = cosine_similarity(movie_embedding)
# Creates an embedding layer using Keras to generate vector representations (embeddings) for each movie based on its ID. Then computes cosine similarity between these vectors for a deep learning-based recommendation.



# Recommendation Functions
def recommend_tfid (title, top_n = 5):
    title = title.lower()
    if title not in df['Movie_Name'].str.lower().values:
        return[]
    

    idx = df[df['Movie_Name'].str.lower() == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse= True)[1:top_n+1] 
    recommend_titles = df.iloc[[i[0] for i in sim_scores]]['Movie_Name'].tolist()
    return recommend_titles  

def recommend_deep (title, top_n = 5):
    title = title.lower()
    if title not in df['Movie_Name'].str.lower().values:
        return[]
    

    idx = df[df['Movie_Name'].str.lower() == title].index[0]
    sim_scores = list(enumerate(deep_sim[idx]))
    sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse= True)[1:top_n+1] 
    recommend_titles = df.iloc[[i[0] for i in sim_scores]]['Movie_Name'].tolist()
    return recommend_titles  
# Two functions to recommend movies using:
#TF-IDF vector similarity
#Embedding-based similarity
#These return the top N most similar movies for a given input title.



# Streamlit UI Layout and Design
title_html = f"""<h1 style = 'text-align: center; color: 	#FF8C00;'>Movie Recommender</h1>"""
st.markdown(title_html, unsafe_allow_html= True)

selected_movie = st.selectbox(':orange[Choose a movie]', sorted(df['Movie_Name'].unique()))
#method = st.radio("Choose recommendation method:", ("TF-IDF", "Deep Learning"))


if st.button('recommendations'):
    
    recommend = recommend_deep(selected_movie)
    if recommend:
        st.success('Top Recommendations')
        for i,rec in enumerate(recommend,1):
            st.markdown(f'{i}.**{rec}**')
            
    else:
        st.error('No Movie recommendation found')   


page = """
<style>
[data-testid = "stAppViewContainer"]{
    background-image : url("https://img.freepik.com/free-photo/composition-cinema-elements-beige-background-with-copy-space_23-2148416816.jpg?ga=GA1.1.704494411.1747992680&semt=ais_hybrid&w=740");
    background-size : cover;
    background-position : center;
    background-repeat : no-repeat
}
[data-testid = "stHeader"]{
    background-color : rgba(0,0,0,0);
}
</style>
"""

st.markdown(page, unsafe_allow_html= True)

st.markdown("""<style> 
    div[data-baseweb = 'select']>div{
    background-color: 	#FFFFE0;
    color : black;      
    }
    </style>      
    """, unsafe_allow_html= True) 

st.markdown("""
    <style>
    div.stButton > button {
    background-color : #FFFFE0;
    color : black;        
    }
    </style>
    """, unsafe_allow_html= True)
# Builds a stylish UI with:
#Custom title and dropdown for movie selection.
#A "Recommendations" button.
#Background and widget theming via HTML/CSS injection.
#Clean output display of recommended movies.



# Conclusion
# This recommender system blends natural language processing (via TF-IDF) and deep learning (via embeddings) to generate effective, content-based movie suggestions. By leveraging the genre data and embedding similarities, it offers two complementary methods for recommendation, enhancing both interpretability and performance.

#🔸 TF-IDF is ideal for content-explainable recommendations.
#🔸 Deep embeddings allow richer, latent feature interactions, potentially finding less obvious but meaningful recommendations.
