#testing push

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                
                print("train_words " + str(train_words))
                
                
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
                    
                print("best train ex = " + str(best_train_ex.y_tok))
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size,output_size, embedding_dropout=0.2, bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        
        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        
        #added
        self.decoder = RNNDecoder(hidden_size, output_size)
        self.output_embed = EmbeddingLayer(emb_dim,len(output_indexer), embedding_dropout)
        self.criterion =  nn.NLLLoss()

    
    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len x voc size] or a batched input/output
        [batch size x sent len x voc size]
        :param inp_lens_tensor/out_lens_tensor: either a vecor of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))

        #print("enc_output_each_word size" + str(enc_output_each_word.size()))
        #print("enc_context_mask size " + str(enc_context_mask.size()))

        #print("enc_final_states_reshaped[0] size" + str(enc_final_states_reshaped[0].size()))
        #print("enc_final_states_reshaped[1] size" + str(enc_final_states_reshaped[1].size()))
    
    
        # initalize with [1] or "start of sentence" token 
        
        
        decoder_hidden_all = enc_final_states_reshaped[1]
        
        #print(" size of x tensor " + str(x_tensor.size()))
        batch_size, input_length = x_tensor.size()
        #print("batch size is " + str(batch_size))
        bs, output_length = y_tensor.size()
        batch_loss = 0
        for batch_num in range(batch_size):
            sentence_loss = 0
            
            #hidden for one batch
            one_batch_ex = decoder_hidden_all[0,batch_num].view(1,1,-1)
            counter = 0
            
            #feed SOS token
            decoder_input = torch.tensor([[1]], device=device)
            
            for seq in range(output_length):
                #print( " bat_num " + str(batch_num) + "seq number is : " + str(seq))
                decoder_hidden = one_batch_ex
                #print("decoder hidden size is " + str(decoder_hidden))
                decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
                step_label = torch.unsqueeze(y_tensor[batch_num][seq], dim = 0)
                
                #print("step label is " + str(step_label.size()) + "step label is " + str(step_label))
                #print("decoder_output  and type is  " + str(type(decoder_output)) + str(decoder_output) )
                seq_loss =  self.criterion(decoder_output, step_label)
                #print( " seq_loss_type is " + str(type(seq_loss)))
                sentence_loss+= seq_loss
                
                # teacher force correct answer for input to next sequence
                decoder_input = y_tensor[batch_num][seq] 
                
            batch_loss += sentence_loss
                
        # why not loss.item()? will not transmit gradient info
            
        batch_average_loss = batch_loss/batch_size    
        
        return batch_average_loss
        
        # use final weights from training, apply forward for inference
        
        # This should call both embedding layer, encoder fwd, decoder fwd, prob calc w/ gold labels (attn later)
     
        #raise Exception("implement me!")

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        
        #Inference based on final weights 
        derivations = []
        
        input_max_len = np.max(np.asarray(([len(ex.x_indexed) for ex in test_data])))
        all_test_input_data = make_padded_input_tensor(test_data, self.output_indexer,input_max_len, reverse_input = False)
        
        #print( " begin loop decode")
        for ex_idx in range(len(test_data)):
            
            #forward pass using x tensor

            sent_tensor= torch.from_numpy(all_test_input_data[ex_idx]).unsqueeze(0)
            
            one_len = torch.tensor([1], dtype = torch.int64)
            one_len[0] = input_max_len
            
            #print( "sentence tensor size " + str(sent_tensor.size()))
            tensor_embed = self.input_emb.forward(sent_tensor)
            
            
            #print("size of one_len " + str(one_len.size()))
            (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(tensor_embed, one_len)
            enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
            
            
            decoder_hidden = enc_final_states_reshaped[1]
            # <SOS>
            decoder_input = torch.tensor([[1]], device=device)
            
            
            prob_sentence = torch.zeros(1)
            
            y_toks = []
            
            for i in range(len(sent_tensor)):
                decoder_output, decoder_hidden = self.decoder.forward(decoder_input,decoder_hidden)
                
                #find argmax
                
                pred_index = torch.argmax(decoder_output)
                prob = torch.max(decoder_output)
                
                print("pred_index " + str(pred_index))
                
                token = self.output_indexer.get_object(pred_index)
                
                print("token is " + str(token))
                y_toks.append(token)
                
                
                
                # is probability calculation conditional across timesteps?
                
                prob_sentence += prob
                
                #print( " one seq inference loop ")
                
            print("y_tokens " + str(y_toks))
            print("prob_sentence is " + str(prob_sentence))
                                  
                # TOMORROW: continue derivation construction, debug decoding
       
        #create list of Derivations for each sentence, iterate over all sentences 
        #raise Exception("implement me!")


    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple)
        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_size: int, hidden_size: int, bidirect: bool):
        """
        :param input_size: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.max().item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)
###################################################################################################################
# End optional classes
###################################################################################################################

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNNDecoder,self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input, hidden):
        output = self.embedding(input).view(1,1,-1)
        #print("output 1 is " + str(output.size()))
        output = F.relu(output)
        #print("output 2 is " + str(output.size()))
        #print("hidden is " + str(hidden.size()))
        
        output, hidden = self.gru(output, hidden)
        #print("output 3 is " + str(output.size()))
       
        output = self.softmax(self.out(output[0]))
        return output, hidden
   
    def initHidden(self):
        return torch.zeros(1,1, self.hidden_size, device = device)


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args, word_embeddings) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # iterate through examples, call encoder, scroll through outputs with the decoder,
    #accumulate log loss terms from the prediction at each point, take optimizer step
    # teacher force
    
    # expan Seq2SeqSeamnticParser constructor to take model weights? or to take encoder and decoder RNN's?
    
    input_dim = word_embeddings.get_embedding_length()
    print("input dim is " + str(input_dim))
    
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))
    
    # First create a model. Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters
        
    learning_rate_e = 0.001
    learning_rate_d = 0.001
    epochs = 5
    emb_dim = 5
    input_size = input_max_len
    
    hidden_size =  len(output_indexer)  # this determines decoder output dimensions 
    
    batch_size = 4
    bidirect = True
    output_size = output_max_len
    full_dict_size_input = len(input_indexer)
    full_dict_size_output = len(output_indexer)
    embedding_dropout_rate = 0.5
    
    total_loss = 0.0

       
    model = Seq2SeqSemanticParser(input_indexer, output_indexer, emb_dim,output_size, hidden_size)
    optimizer_e = optim.Adam(model.encoder.parameters(),lr = learning_rate_e)
    optimizer_d = optim.Adam(model.decoder.parameters(),lr = learning_rate_d)
    
    embedding_layer = EmbeddingLayer(input_max_len, full_dict_size_input, embedding_dropout_rate)
    
    for epoch in range(0, epochs):
        
        ex_indices = [ i for i in range(0,len(train_data))]
        random.shuffle(ex_indices)
        
        for idx in range(0,len(all_train_input_data) - batch_size, batch_size ):
            
                  
            model.encoder.zero_grad()
            model.decoder.zero_grad()
            
            #---------batch
            x_tensor = torch.zeros([batch_size,input_max_len],dtype = torch.int64)
            inp_lens_tensor = torch.zeros([batch_size],dtype = torch.int64)
            y_tensor = torch.zeros([batch_size,output_max_len],dtype = torch.int64)
            out_lens_tensor = torch.zeros([batch_size])
            
            #print("input_max_len is " + str(input_max_len))
           
            
            for i in range(0,batch_size):
               print( " idx " + str(idx))
               
               x_tensor[i] = torch.from_numpy(all_train_input_data[idx+i])
               sent_length = torch.count_nonzero(x_tensor[i]).long()
               #print (" sent_length type is " + str(type(sent_length)))
               
               
               inp_lens_tensor[i] = sent_length
               
               y_tensor[i] = torch.from_numpy(all_train_output_data[idx+i])
               target_length = torch.count_nonzero(y_tensor[i]).long()
               out_lens_tensor[i] = target_length
            
            
            #print("x_tensor is " + str(x_tensor) ) 
            
           # print (" in lens tensor type is " + str(type(inp_lens_tensor[0])))
            batch_loss = model.forward(x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor)
            
            batch_loss.backward()
            
            total_loss += batch_loss
            
            optimizer_e.step()
            optimizer_d.step()
                
    return model     



            
            #--- pass x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor 
            #into model.enc_input()  or model.forward() 
             
            
            
            
    
    #raise Exception("I, don't want no bugs (to the tune of TLC 'No Scrubs' " )
    
 
                 
       
  
                    
                
   
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    