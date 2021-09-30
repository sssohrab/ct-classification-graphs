import torch
import numpy as np
import os, json, re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix


ROOT_DIR = '/home/sohrab/storage/ctgov'


class BagOfWordsBasic:
    """
    Fits, transforms and projects (using sparse random projections) the corpus to BOW-like statistics, i.e., TF-IDF.
    """

    def __init__(self,  ngram_range=(1, 1), dim=1000, nnz_proj=1000, max_features=500000,
                 token_pattern="(?u)\\b[\\w-]+\\b", np_random_seed=123):

        self.vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english", preprocessor=self.preprocess,
                                          # tokenizer=self.tokenizer,
                                          token_pattern=token_pattern, max_features=max_features)
        self.tfidf_transformer = TfidfTransformer(smooth_idf=False, use_idf=True)
        self.dim = dim
        self.nnz_proj = nnz_proj
        self.Dim = None
        self.projector = None
        np.random.seed(np_random_seed)


    def preprocess(self, inp):
        """Assumes input is no structured, or if so, the structure is irrelevant."""
        inp = str(inp).lower().strip()
        # Remove punctuations
        # inp = re.sub('[^a-zA-Z]', ' ', inp)
        # remove tags
        inp = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", inp)
        # Remove pure digits:
        inp = re.sub("^\d+\s|\s\d+\s|\s\d+$", '', inp)

        return inp

    def docs_list_dict_concat(self, docs_list_dict):
        """This is for the first fit, where docs should be elements of a list."""
        if isinstance(docs_list_dict[0], dict):
            return [' '.join([str(v) for v in d.values()]) for d in docs_list_dict]
        elif isinstance(docs_list_dict[0], str):
            # TODO: I guess this is not needed anymore, as input is a list of dictionaries.
            return [d if d else '' for d in docs_list_dict]

    def docs_list_dict_to_list(self, docs_list_dict):
        if isinstance(docs_list_dict[0], dict):
            docs_list = [str(v) for d in docs_list_dict for v in d.values()]
            return [d if d else '' for d in docs_list]
        elif isinstance(docs_list_dict[0], str):
            return [d if d else '' for d in docs_list_dict]

    @staticmethod
    def tfidf_spmatrix_to_tensor(tfidf, num_docs):
        tfidf = np.array(tfidf.todense())
        dim = tfidf.shape[1]
        return tfidf.reshape(num_docs, -1, dim)

    def create_sparse_projection_matrix(self):
        data = np.random.randn(1, self.nnz_proj * self.dim) / np.sqrt(self.nnz_proj)
        columns = np.repeat(np.arange(self.dim), self.nnz_proj)
        rows = np.array([]).reshape(self.nnz_proj, 0)
        for i in range(self.dim):
            r = np.random.permutation(self.Dim)[0:self.nnz_proj]
            rows = np.hstack((rows, r.reshape(-1, 1)))

        projector = csr_matrix((data.reshape(-1), (rows.reshape(-1), columns.reshape(-1))), shape=(self.Dim, self.dim))
        return projector

    def fit_vectorizer(self, docs_list):
        self.vectorizer.fit(docs_list)

    def transform_docs_to_bow(self, docs_list):
        bow = self.vectorizer.transform(docs_list)
        return bow

    def fit_tfidf(self, bow):
        self.tfidf_transformer.fit(bow)

    def transform_tfidf(self, bow):
        tfidf = self.tfidf_transformer.transform(bow)
        return tfidf

    def fit(self, docs_list):
        docs_list = self.docs_list_dict_concat(docs_list)
        self.fit_vectorizer(docs_list)
        bow = self.transform_docs_to_bow(docs_list)
        self.fit_tfidf(bow)
        self.Dim = bow.shape[1]
        self.projector = self.create_sparse_projection_matrix()
        # self.projector = np.random.randn(self.Dim, self.dim) * np.sqrt(1/self.dim)

    def transform(self, docs_list):
        docs_list = self.docs_list_dict_to_list(docs_list)
        bow = self.transform_docs_to_bow(docs_list)
        tfidf = self.transform_tfidf(bow)
        return tfidf

    def transform_project(self, docs_list):
        num_docs = len(docs_list)
        tfidf = self.transform(docs_list)
        tfidf = tfidf @ self.projector
        tfidf = self.tfidf_spmatrix_to_tensor(tfidf, num_docs)

        return tfidf

    def __call__(self, docs_list):
        tfidf = self.transform_project(docs_list)
        return torch.from_numpy(tfidf).squeeze(1)


class FeaturizerChannelHuggingFace:
    """
    Embeds pieces of text using transformers based on Huggingface API's
    """
    def __init__(self, model_name, tokenizer_name, device='cpu', padding=True, truncation=True,
                 max_length=512, return_tensors='pt'):

        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.return_tensors = return_tensors

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Copy-pasted from sentence-transformers.

        Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # return sum_embeddings / sum_mask
        # Added after observing an extremely large value after division by small values of sum_mask:
        return torch.clamp(sum_embeddings / sum_mask, min=-1, max=+1)

    def __call__(self, docs):

        if isinstance(docs, dict):
            encoded_input = self.tokenizer(list(docs.values()), padding=self.padding, truncation=self.truncation,
                                           max_length=self.max_length, return_tensors=self.return_tensors)

        elif isinstance(docs, list):
            assert isinstance(docs[0], str)
            encoded_input = self.tokenizer(docs, padding=self.padding, truncation=self.truncation,
                                           max_length=self.max_length, return_tensors=self.return_tensors)

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return embeddings
