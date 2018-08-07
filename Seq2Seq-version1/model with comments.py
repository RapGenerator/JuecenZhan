# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import LSTMCell, GRUCell


class Seq2SeqModel(object):
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
                 beam_search, beam_size, cell_type='LSTM', max_gradient_norm=5.0):
        self.learing_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.cell_type = cell_type
        self.max_gradient_norm = max_gradient_norm

        # placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')

        # embedding矩阵,encoder和decoder共用该词向量矩阵
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

        self.__graph__()
        self.saver = tf.train.Saver()

    def __graph__(self):

        # encoder
        encoder_outputs, encoder_state = self.encoder()

        # decoder
        with tf.variable_scope('decoder'): ##作用域，'/'
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tile_batch(encoder_inputs_length, multiplier=self.beam_size)

            # 定义要使用的attention机制。
            attention_mechanism = BahdanauAttention(num_units=self.rnn_size,
                                                    memory=encoder_outputs,
                                                    memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要是用的RNNCell，然后为其封装attention wrapper
            decoder_cell = self.create_rnn_cell()
            decoder_cell = AttentionWrapper(cell=decoder_cell,
                                            attention_mechanism=attention_mechanism,
                                            attention_layer_size=self.rnn_size,
                                            name='Attention_Wrapper')
            # 如果使用beam_seach则batch_size = self.batch_size * self.beam_size
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,
                                                            dtype=tf.float32).clone(cell_state=encoder_state)

            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(
                                                            mean=0.0,9
                                                            stddev=0.1))

            if self.mode == 'train':
                self.decoder_outputs = self.decoder_train(decoder_cell, decoder_initial_state, output_layer)
                
                # loss
                #This is the weighted cross-entropy loss for a sequence of logits.
                #Param:
                    #logits: [batch_size, sequence_length, num_decoder_symbols].
                    #        The logits is the prediction across all classes at each timestep.
                    #targets: [batch_size, sequence_length], representing true class at each time step
                    #weights: [batch_size, sequence_length], This is the weighting of each prediction in the sequence. 
      
                self.loss = sequence_loss(logits=self.decoder_outputs, targets=self.decoder_targets, weights=self.mask)

                # summary
                tf.summary.scalar('loss', self.loss) #Outputs a Summary protocol buffer containing a single scalar value.
                self.summary_op = tf.summary.merge_all() #Merges all summaries collected in the default graph.

                # optimizer
                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables() #train all variables that have "trainable = True"
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm) 
                #clips values of multiple tensors by the ratio of the sum of their norms.
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            elif self.mode == 'decode':
                self.decoder_predict_decode = self.decoder_decode(decoder_cell, decoder_initial_state, output_layer)

    def encoder(self):
        '''
        创建模型的encoder部分
        :return: encoder_outputs: 用于attention，batch_size*encoder_inputs_length*rnn_size //对于每个batch每个时刻都有一个hidden-size * 1的
                 encoder_state: 用于decoder的初始化状态，batch_size*rnn_size
        '''
        with tf.variable_scope('encoder'):
            encoder_cell = self.create_rnn_cell()
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, sequence_length=
                                                               self.encoder_inputs_length, dtype=tf.float32)
            return encoder_outputs, encoder_state

    def decoder_train(self, decoder_cell, decoder_initial_state, output_layer):
        '''
        创建train的decoder部分
        :param encoder_outputs: encoder的输出
        :param encoder_state: encoder的state
        :return: decoder_logits_train: decoder的predict
        '''
        ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1]) ##一次batch的Target
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<GO>']), ending], 1)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

        training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                         sequence_length=self.decoder_targets_length,
                                         time_major=False, name='training_helper')
        training_decoder = BasicDecoder(cell=decoder_cell,
                                        helper=training_helper,
                                        initial_state=decoder_initial_state,
                                        output_layer=output_layer)
        decoder_outputs, _, _ = dynamic_decode(decoder=training_decoder,
                                               impute_finished=True,
                                               maximum_iterations=self.max_target_sequence_length)
        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)  #?
        return decoder_logits_train

    def decoder_decode(self, decoder_cell, decoder_initial_state, output_layer):
        start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<GO>']
        end_token = self.word_to_idx['<EOS>']

        if self.beam_search:
            inference_decoder = BeamSearchDecoder(cell=decoder_cell,
                                                  embedding=self.embedding,
                                                  start_tokens=start_tokens,
                                                  end_token=end_token,
                                                  initial_state=decoder_initial_state,
                                                  beam_width=self.beam_size,
                                                  output_layer=output_layer)
        else:
            decoding_helper = GreedyEmbeddingHelper(embedding=self.embedding,
                                                    start_tokens=start_tokens,
                                                    end_token=end_token)
            ##Uses the argmax of the output (treated as logits) and passes the result through an embedding layer to get the next input.
            ##embedding: A callable that takes a vector tensor of ids (argmax ids), or the params argument for embedding_lookup. The returned tensor will be passed to the decoder input.
            ##start_tokens: int32 vector shaped [batch_size], the start tokens.
            ##end_token: int32 scalar, the token that marks end of decoding.
            
            inference_decoder = BasicDecoder(cell=decoder_cell,
                                             helper=decoding_helper,
                                             initial_state=decoder_initial_state,
                                             output_layer=output_layer)

        decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=50)
        ##predicted_ids: Final outputs returned by the beam search after all decoding is finished. A tensor of shape [batch_size, num_steps, beam_width] (or [num_steps, batch_size, beam_width] if output_time_major is True). Beams are ordered from best to worst.
        if self.beam_search:
            decoder_predict_decode = decoder_outputs.predicted_ids
        else:
            decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)  
        #sample_id is the argmax of the rnn output
        #predicted_ids: Final outputs returned by the beam search after all decoding is finished. A tensor of shape [batch_size, num_steps, beam_width] (or [num_steps, batch_size, beam_width] if output_time_major is True). Beams are ordered from best to worst.
        #Why do we need to add a dimension ????????

        return decoder_predict_decode

    def create_rnn_cell(self):
        '''
        创建标准的RNN Cell，相当于一个时刻的Cell
        :return: cell: 一个Deep RNN Cell
        '''
        def single_rnn_cell():
            single_cell = GRUCell(self.rnn_size) if self.cell_type == 'GRU' else LSTMCell(self.rnn_size)
            basiccell = DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            #Dropout makes each hidden unit more robust and avoid overfitting
            return basiccell
        cell = MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        #We stack up multiple RNN Cells give the number of layer
        return cell

    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,  #[batch-size, encoder-seq-length]
                     self.encoder_inputs_length: batch.encoder_inputs_length, #[batch-size]
                     self.decoder_targets: batch.decoder_targets, #[batch-size, decoder-seq-length]
                     self.decoder_targets_length: batch.decoder_targets_length, #[batch-size]
                     self.keep_prob: 0.5, #drop_out probability
                     self.batch_size: len(batch.encoder_inputs)} 
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict



