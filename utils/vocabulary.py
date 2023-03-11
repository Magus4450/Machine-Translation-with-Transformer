import string
import warnings
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize


class Vocabulary:

    def __init__(self, doc_list: List, language:str = None, min_occurence:int = 0, tokenization_method:str = 'word',  remove_punctuations:bool = True, remove_digits:bool = True, remove_stopwords: bool = False) -> None:

        """Class for generating vocabulary from a list of documents

        Args:
            doc_list (List) : List of documents to generate vocabulary from
            language (str) : Language to use for stopwords
            min_occurence (int) : Defaults to 0. Minimum occurences required to be in the vocabulary 
            tokenization_method (str) : Defaults to 'word'. Type of tokenizer to use. ['word', 'punctuation']  
            remove_punctuations (bool) : Defaults to True. Whether to remove punctuations
            remove_digits (bool) : Defaults to True. Whether to remove digits
            remove_stopwords( bool) : Defaults to False. Whether to remove stopwords
        """
        assert type(doc_list) == list, "Invalid type for `doc_list`. Type should be a list"

        assert tokenization_method in ['word', 'punctuation'], "Invalid value for `tokenization_method`. Value should be one of  ['word', 'punctuation']"

        if tokenization_method == 'punctuation' and remove_punctuations:
            warnings.warn("`tokenization_method` as 'punctuation' is redundant when `remove_punctuations` is True")

        if remove_stopwords:
            assert language in [*stopwords.fileids(), None], f"`language` should be in {stopwords.fileids()} or None"

        self.doc_list = [doc.lower() for doc in doc_list]
        self.docs_tokens = None
        self.language = language
        self.min_occurence = min_occurence
        self.tokenization_method = tokenization_method
        self.remove_punctuations = remove_punctuations
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.tokenizer = None


        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1
        self.PAD_TOKEN = 2
        self.UNK_TOKEN = 3

        self.max_length = 0

        self.special_tokens = ["<sos>", "<eos>", "<pad>", "<unk>"]
        self.t2i = {
            self.special_tokens[0]: self.SOS_TOKEN,
            self.special_tokens[1]: self.EOS_TOKEN,
            self.special_tokens[2]: self.PAD_TOKEN,
            self.special_tokens[3]: self.UNK_TOKEN
        }
        self.i2t = {}

        self.vocab = self._build_vocab(doc_list)

        self.docs_tokens_encoded = self._generate_docs_tokens_encoded()


    @staticmethod
    def _remove_punctuations(doc: str) -> str:
        """Remove punctuations from document

        Args:
            doc (str): Document

        Returns:
            str: Punctuation free document
        """
        
        doc = "".join([char for char in doc if char not in string.punctuation])

        return doc

    @staticmethod
    def _remove_digits(doc: str) -> str:
        """Remove digits from document

        Args:
            doc (str): Document

        Returns:
            str: Digits free document
        """
        doc = "".join([char for char in doc if char not in string.digits])

        return doc


    def _remove_stopwords(self, tokens: List) -> List:
        """Removes stopwords from tokens. Stopwords gathered from nltk.corpus.stopwords.words(`language`)

        Args:
            tokens (List): Tokens

        Returns:
            List: Tokens without stopwords
        """
        
        stop_words = stopwords.words(self.language)

        tokens = [token for token in tokens if token not in stop_words]

        return tokens

    def _preprocess(self, doc_list: List[List]) -> List:
        """Apply preprocessing steps for all docs. 
            1. Remove Punctuations
            2. Remove Digits
            3. Tokenization
            4. Remove Stopwords

        Args:
            doc_list (List): List of documents

        Returns:
            List[List]: List of list of tokens for all documents 
        """

        # Remove punctuations
        if self.remove_punctuations:
            doc_list = [self._remove_punctuations(elem) for elem in doc_list]

        # Remove digits
        if self.remove_digits:
            doc_list = [self._remove_digits(elem) for elem in doc_list]

        # Setup tokenizer        
        if self.tokenization_method == 'word':
            self.tokenizer = word_tokenize
        elif self.tokenization_method == 'punctuation':
            self.tokenizer = wordpunct_tokenize
        
        # Tokenize
        docs_tokens = [self.tokenizer(doc) for doc in doc_list]

        # Remove stopwords
        if self.remove_stopwords:
            docs_tokens = [self._remove_stopwords(doc_tokens) for doc_tokens in docs_tokens]

        # Lower
        for i, doc_tokens in enumerate(docs_tokens):
            new_tokens = [tok.lower() for tok in doc_tokens]
            docs_tokens[i] = new_tokens

        return docs_tokens

    def _build_vocab(self, doc_list: List[str]) -> List:
        """Generated vocabulary, token to index dict and index to token dict

        Args:
            doc_list (List[str]): List of documents

        Returns:
            List: Vocabulary
        """

        # Clean and tokenize all docs
        docs_tokens = self._preprocess(doc_list)
        self.docs_tokens = docs_tokens
        
        # To store all tokens
        all_tokens = []

        # Store all tokens in one list
        for doc_tokens in docs_tokens:

            # Change all tokens to lowercase
            lowerered_tokens = [tok.lower() for tok in doc_tokens]

            all_tokens += lowerered_tokens
        
        # Get count of each token
        token_count = Counter(all_tokens)

        # Filter tokens with less occurences
        filtered_token_count = {key: value for key, value in token_count.items() if value > self.min_occurence}

        # Unique tokens with higher occurences
        unique_tokens = list(filtered_token_count.keys())

        # Generate token to index
        t2i = {tok: (i + self.UNK_TOKEN + 1) for i, tok in enumerate(unique_tokens)}
        self.t2i.update(t2i)

        # Generate index to token
        self.i2t = {value: key for key, value in self.t2i.items()}

        # Return vocabulary
        return self.special_tokens + unique_tokens

    def _generate_docs_tokens_encoded(self) -> List[List[str]]:
        """Converts doc list to its token index

        Returns:
            List[List[int]]: Each document with its token encoded
        """
        docs_tokens_encoded = []

        for doc in self.docs_tokens:
            new_doc = [self.t2i[token] for token in doc]

            # Add <sos> and <eos> token
            new_doc.insert(0, self.SOS_TOKEN)
            new_doc.append(self.EOS_TOKEN)
            

            self.max_length = max(self.max_length, len(new_doc))
            docs_tokens_encoded.append(new_doc)




        return docs_tokens_encoded

    def convert_token_to_doc(self, doc_tokens, remove_special = False):
        doc = [self.i2t[idx] for idx in doc_tokens]
        if remove_special:
            doc = [tok for tok in doc if tok not in self.special_tokens]
        return doc
    
    def convert_doc_to_token(self, doc):
        tokens_out = []    
        for token in doc:
            token = token.lower()
            if token in self.vocab:
                tokens_out.append(self.t2i[token])
            else:
                tokens_out.append(self.UNK_TOKEN)
        
        return tokens_out

        