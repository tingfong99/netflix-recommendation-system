import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('streamlit/clean_data.csv')
df.drop(columns=df.columns[0], axis=1, inplace=True)
X = np.load('streamlit/X.npy')
cos_sim_data = pd.DataFrame(cosine_similarity(X))

st.title("Netflix Recommender System")
movie = st.text_input(label="Please enter the movie or TV show that you had watched",
                      label_visibility='visible')

#movie = st.selectbox("Please enter the movie or TV show that you had watched", df.title.tolist())

if st.button('Predict !') :
    if movie in df.title.values :
        idx = np.where(df['title'] == movie)[0]
        idx = idx.tolist()
        index_recomm = cos_sim_data.loc[idx[0]].sort_values(ascending=False).index.tolist()[1:6]
        movies_recomm = df['title'].loc[index_recomm].values
        result = {'Movies':movies_recomm, 'Index':index_recomm}
        #st.write(result)
        k = 1
        for movie in movies_recomm:
            st.write('The number %i recommended movie is : %s \n' %(k,movie))
            k+=1
    else :
        st.write('Sorry,')
        st.write('the movie is not register at the database.')
        st.write('Please try another movie :slightly_frowning_face:')