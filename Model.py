import tensorflow as tf

CODE_VOCAB_SIZE = 30000
AST_VOCAB_SIZE = 30000
SBT_VOCAB_SIZE = 30000
NL_VOCAB_SIZE = 23428
HIDDEN_SIZE = 500
NUM_LAYERS = 1
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5

class SeCNN(object):
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.Variable(initial)
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.00005)(weight))
        return weight

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def max_height_pooling(self, input):
        height = int(input.get_shape()[1])
        width = int(input.get_shape()[2])
        input = tf.expand_dims(input, -1)
        output = tf.nn.max_pool(input, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    def max_width_pooling(self, input):
        height = int(input.get_shape()[1])
        width = int(input.get_shape()[2])
        input = tf.expand_dims(input, -1)
        output = tf.nn.max_pool(input, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, height])
        return output

    def my_conv(self, input_t, stage, strides=1):
        x = self.Conv1d(input_t, self.conv_layernum, self.conv_layersize, strides=strides, padding='same')
        x = self.Relu(x)
        for i in range(stage):
            x = self.MulCnn(x)
        return x

    def MulCnn(self, input_tensor):
        x = self.Conv1d(input_tensor, self.conv_layernum, self.conv_layersize, padding='same')
        x = self.Relu(x)
        x = self.Conv1d(x, self.conv_layernum, self.conv_layersize, padding='same')
        x = tf.add_n([x, input_tensor])  # x = x + input_tensor
        x = self.Relu(x)
        return x

    def max_Attention(self, state, max_pool):
        state_height = int(state.shape[2])
        pool_height = int(max_pool.shape[1])
        attention_matrix = self.weight_variable(shape=[state_height, pool_height])
        tmp_matrix = tf.einsum("ijk,kl->ijl", state, attention_matrix)
        w_pool = tf.expand_dims(max_pool, -1)
        tmp_matrix = tf.matmul(tmp_matrix, w_pool)
        weight_vec = tf.nn.softmax(tf.reduce_max(tmp_matrix, reduction_indices=[2]))
        weight_vec = tf.expand_dims(weight_vec, -1)
        Out = tf.matmul(state, weight_vec, transpose_a=True)
        out = tf.reduce_max(Out, reduction_indices=[2])
        return out

    def __init__(self):
        self.Relu = tf.nn.relu
        self.Conv1d = tf.layers.conv1d
        self.conv_layernum = HIDDEN_SIZE
        self.conv_layersize = 4

        self.nlLeng = 30
        self.astLeng = 200
        self.codeLneg = 200
        self.sbtLneg = 300

        self.nl_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        self.nl_embedding = tf.get_variable('nl_emb', [NL_VOCAB_SIZE, HIDDEN_SIZE])
        self.code_embedding = tf.get_variable('code_emb', [CODE_VOCAB_SIZE, HIDDEN_SIZE])
        self.sbt_embedding = tf.get_variable('sbt_emb', [SBT_VOCAB_SIZE, HIDDEN_SIZE])
        self.pos_embedding = tf.get_variable('pos_emb', [500, HIDDEN_SIZE])

        self.sbt_input = tf.placeholder(tf.int32, [None, self.sbtLneg])
        self.sbt_pos = tf.placeholder(tf.int32, [None, self.sbtLneg])
        self.sbt_size = tf.placeholder(tf.int32, [None])


        self.code_input = tf.placeholder(tf.int32, [None, self.codeLneg])
        self.code_size = tf.placeholder(tf.int32, [None])

        self.nl_input = tf.placeholder(tf.int32, [None, self.nlLeng])
        self.nl_output = tf.placeholder(tf.int32, [None, self.nlLeng])
        self.nl_size = tf.placeholder(tf.int32, [None])
        self.mask_size = tf.placeholder(tf.int32, [None])

        batch_size = tf.shape(self.code_input)[0]

        sbt_emb = tf.nn.embedding_lookup(self.sbt_embedding, self.sbt_input)
        pos_emb = tf.nn.embedding_lookup(self.pos_embedding, self.sbt_pos)
        code_emb = tf.nn.embedding_lookup(self.code_embedding, self.code_input)
        nl_emb = tf.nn.embedding_lookup(self.nl_embedding, self.nl_input)

        sbt_emb = tf.nn.dropout(sbt_emb, KEEP_PROB)
        pos_emb = tf.nn.dropout(pos_emb, KEEP_PROB)
        code_emb = tf.nn.dropout(code_emb, KEEP_PROB)
        nl_emb = tf.nn.dropout(nl_emb, KEEP_PROB)

        stack_emb = tf.stack([sbt_emb, pos_emb], -2)
        temp_emb = tf.layers.conv2d(stack_emb, HIDDEN_SIZE, [1, 2])
        temp_emb = tf.reduce_max(temp_emb, reduction_indices=[-2])
        sbt_emb = tf.nn.tanh(temp_emb)

        with tf.variable_scope("sbt_conv", reuse=False):
            sbt_conv = self.my_conv(sbt_emb, 4)

        with tf.variable_scope("code_conv", reuse=False):
            code_conv = self.my_conv(code_emb, 4)


        with tf.variable_scope("h_sbt_conv", reuse=False):
            h_sbt_conv = self.my_conv(sbt_emb, 4)

        with tf.variable_scope("h_code_conv", reuse=False):
            h_code_conv = self.my_conv(code_emb, 4)

        pool1 = self.max_height_pooling(h_code_conv)
        pool2 = self.max_height_pooling(h_sbt_conv)

        c_pool = self.max_Attention(h_sbt_conv, pool1)
        h_pool = self.max_Attention(h_code_conv, pool2)

        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(c_pool, h_pool) for _ in range(NUM_LAYERS)])

        enc_outputs = tf.concat(axis=1, values=[sbt_conv, code_conv])

        enc_size = self.code_size + self.sbt_size

        with tf.variable_scope('decoder'):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=HIDDEN_SIZE, memory=enc_outputs,
                                                                    memory_sequence_length=enc_size)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=self.nl_cell,
                                                                 attention_mechanism=attention_mechanism,
                                                                 attention_layer_size=HIDDEN_SIZE,
                                                                 name='Attention_Wrapper')
            output_layers = tf.layers.Dense(NL_VOCAB_SIZE,
                                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            decoder_state = attention_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
                cell_state=tuple_state)

            # 训练
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=nl_emb, sequence_length=self.nl_size)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=training_helper,
                                                               initial_state=decoder_state,
                                                               output_layer=output_layers)
            dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, output_time_major=False,
                                                                 impute_finished=True,
                                                                 maximum_iterations=self.nlLeng)
            logits = tf.identity(dec_output.rnn_output)
            self.cost = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.nl_output,
                                                         weights=tf.sequence_mask(self.mask_size,
                                                                                  maxlen=tf.shape(self.nl_output)[1],
                                                                                  dtype=tf.float32))
            # 反向传播
            trainable_variables = tf.trainable_variables()
            # 梯度大小，优化，训练

            grads = tf.gradients(self.cost, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)

            global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(1.0,
                                                            global_step=global_step,
                                                            decay_steps=2000,
                                                            decay_rate=0.99,
                                                            staircase=True)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
            self.add_global = global_step.assign_add(1)

            # 验证或者测试
            start_tokens = tf.ones([batch_size, ], tf.int32) * 2
            end_token = 3

            decoder_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.nl_embedding,
                                                                      start_tokens=start_tokens,
                                                                      end_token=end_token)
            interence_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=decoder_helper,
                                                                initial_state=decoder_state,
                                                                output_layer=output_layers)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=interence_decoder,
                                                                     maximum_iterations=self.nlLeng)
            self.predict = decoder_output.sample_id
