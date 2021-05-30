import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

class MovieRecommendations:

    def __init__(self):
        ##Cleaning data

        #importing csv files
        credits = pd.read_csv("tmdb_5000_credits.csv")
        self.movies = pd.read_csv("tmdb_5000_movies.csv")
        credits = credits.rename(columns = {"movie_id":'id'})
        self.movies = self.movies.merge(credits, on='id')
        self.movies.drop(self.movies.columns.difference(['id', 'original_title', 'overview']), 1, inplace=True)

        #Creating TfidfVectorizer object 
        tvf = TfidfVectorizer(min_df = 3, max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range= (1,3), stop_words = 'english')
        self.movies['overview'] = self.movies['overview'].fillna('')
        tvf_matrix = tvf.fit_transform(self.movies['overview']) 
        self.sig = sigmoid_kernel(tvf_matrix, tvf_matrix)
        self.indices = pd.Series(self.movies.index, index= self.movies['original_title']).drop_duplicates()

    def give_rec(self,title, sig = None):
        #Get recommendations
        
        if(sig==None):
            sig = self.sig
        idx = self.indices[title]
        sig_scores = list(enumerate(sig[idx]))
        sig_scores = sorted(sig_scores,key = lambda x:x[1], reverse = True)
        sig_scores  = sig_scores[1:11]
        movie_indices = [i[0] for i in sig_scores]
        
        return self.movies['original_title'].iloc[movie_indices]





