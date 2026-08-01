from search_utils import load_movies, stop_words, DEFAULT_SEARCH_LIMIT
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle
import os



class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap : dict[int, dict]= {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)

    
    def get_documents(self, term: str):
        result = self.index[term]
        result.sort()
        return result
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], movie['title'] +" " +  movie['description'])
            self.docmap[movie["id"]] = movie

    def save(self):
        os.makedirs("cache", exist_ok = True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

   



def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    stemmer = PorterStemmer()
    movies = load_movies()
    results = [] 
    stop_words = stop_words_processing()

    for movie in movies:
        query_tokens = tokenize(query)
        title_tokens = tokenize(movie["title"])
        query_tokens = remove_stop_words(query_tokens, stop_words)
        title_tokens = remove_stop_words(title_tokens, stop_words)
        query_tokens = stem(query_tokens)
        movie_tokens = stem(title_tokens)

        if has_matching_tokens(query_tokens, title_tokens):
            results.append(movie)
            if len(results)>= limit:
                break

    return results

def stem(tokens: list[str]) -> list[str]:
    """ Stem the tokens using the provided stemmer """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """ Check if any token from the query matches tokens in the title """
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True

    return False

def remove_stop_words(tokens: list[str], stop_words_list: set[str]) -> list[str]:
    """ Remove stop words from the list of tokens """
    result = []
    for token in tokens:
        if token in stop_words_list:
            continue
        else:
            result.append(token)
    return result

def stop_words_processing() -> list[str]:
    """ Load stop words from the stopwords.txt file """
    stop_words_list = stop_words()
    
    for i in range(len(stop_words_list)):
        cleaned_word = clean_text(stop_words_list[i])
        stop_words_list[i] = cleaned_word
    return stop_words_list

def clean_text(query: str) -> str:
    """ Remove punctuation and convert to lowercase """
    translation_table = str.maketrans('', '', string.punctuation)
    return query.translate(translation_table).lower()

def tokenize(text: str) -> list[str]:
    """ Tokenize the input text into words """
    text = clean_text(text)
    tokens = text.split()
    valid_tokens = []

    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens

