from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from .vocabulary import Vocabulary


class SmartDataLoader:

    def __init__(self, input_vocab:Vocabulary, output_vocab:Vocabulary, batch_size:int=1, max_input_length:int = 100, max_output_length:int = 100, input_docs_tokens_encoded:List[List[int]]=None, output_docs_tokens_encoded:List[List[int]]=None,random_seed:int = 0):
        """Create a Dataloader that outputs input and output document with sequence length being maximum length of sentence for that batch.

        Args:
            input_vocab (Vocabulary): Input Data Vocabulary
            output_vocab (Vocabulary): Output Data Vocabulary
            batch_size (int, optional): Batch Size. Defaults to 1.
            input_docs_tokens_encoded (List[List[int]], optional): List of input encoded documents to use in dataloader. Defaults to None.
            output_docs_tokens_encoded (List[List[int]], optional): List of input encoded documents to use in dataloader. Defaults to None.
            random_seed (int, optional): Random Seed for shuffling. Defaults to 0.
        """
        self.input_vocab = deepcopy(input_vocab)
        self.output_vocab = deepcopy(output_vocab)
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.input_docs_tokens_encoded = input_docs_tokens_encoded
        self.output_docs_tokens_encoded = output_docs_tokens_encoded
        if input_docs_tokens_encoded is not None and output_docs_tokens_encoded is not None:
            self.input_vocab.docs_tokens_encoded = input_docs_tokens_encoded
            self.output_vocab.docs_tokens_encoded = output_docs_tokens_encoded


        self.random_seed = random_seed
        self._shuffle_vocab_data()

        self.PAD_TOKEN = input_vocab.PAD_TOKEN

    def _shuffle_vocab_data(self):
        """Shuffles the data
        """

        # Group input and output data for same shuffling
        joined_docs_tokens_encoded = list(zip(self.input_vocab.docs_tokens_encoded, self.output_vocab.docs_tokens_encoded))

        np.random.seed(self.random_seed)

        # Shuffle 
        np.random.shuffle(joined_docs_tokens_encoded)

        # Reassign shuffled data
        self.input_vocab.docs_tokens_encoded, self.output_vocab.docs_tokens_encoded = zip(*joined_docs_tokens_encoded)


    
    def _build_tensor(self, doc_pairs: List[List[int]]) -> Tuple[Tensor, Tensor]:
        """Takes in List of input and output tokens and return their respective Tensors

        Args:
            doc_pairs (List[List[int]]): Batched pairs of input and output tokens

        Returns:
            Tuple[Tensor, Tensor]: Batched Tensors of input and output tokens
        """

    

        input_docs = []
        output_docs = []
       

        for idx, doc_pair in enumerate(doc_pairs):


            input_len = len(doc_pair[0])
            output_len = len(doc_pair[1])

            # Pad Input
            if input_len < self.max_input_length:
                pads_needed = self.max_input_length - input_len
                for _ in range(pads_needed):
                    doc_pair[0].append(self.PAD_TOKEN)
                doc_pairs[idx][0] = doc_pair[0]

            # Pad Output
            if output_len < self.max_output_length:
                pads_needed = self.max_output_length - output_len
                for _ in range(pads_needed):
                    doc_pair[1].append(self.PAD_TOKEN)
                doc_pairs[idx][1] = doc_pair[1]

            input_docs.append(doc_pairs[idx][0])
            output_docs.append(doc_pairs[idx][1])
            

        input_docs = np.transpose(np.array(input_docs))
        output_docs = np.transpose(np.array(output_docs))
        
        return (torch.tensor(input_docs), torch.tensor(output_docs))

    def __iter__(self) -> Tuple[Tensor, Tensor]:
        """Yeild batched Tensors of input and output tokens

        Yields:
            Tuple[Tensor, Tensor]:  Batched Tensors of input and output tokens
        """
        batch = 0
        collected = []
        for sent_idx in range(len(self.input_vocab.docs_tokens_encoded)):
            batch += 1
            collected.append([self.input_vocab.docs_tokens_encoded[sent_idx], self.output_vocab.docs_tokens_encoded[sent_idx]])
            
            if batch == self.batch_size:
                yield self._build_tensor(collected)
                batch = 0
                collected = []

    
    def __len__(self) -> int:
        """Returns number of iteration this dataloader runs for

        Returns:
            int: Number of iteration this dataloder runs for
        """
        return len(self.input_vocab.docs_tokens_encoded) // self.batch_size
            

    


