import dataset
import Model
import tensorflow as tf
import os
import nltk
import random
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import tokenization
import shutil, os, zipfile
import preprocessing.preprocess as preprocess

REGULARIZER = 0.0001
BATCH_SIZE = 12

MODEL_SAVE_PATH = "./model"
MODEL_NAME = "secnn"
data_path = 'data'

def train():
    print('load data......')
    Tokens = tokenization.Tokenization()
    trainData = dataset.get_data(BATCH_SIZE, 'train', Tokens)
    validData = dataset.get_data(BATCH_SIZE, 'valid', Tokens)
    testData = dataset.get_data(BATCH_SIZE, 'test', Tokens)
    print('load finish')

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('SeCNN', reuse=None, initializer=initializer):
        model = Model.SeCNN()

    # return
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.753)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        maxBleu = 0
        nowBleu = 0
        while True:
            index = list(range(len(trainData[0])))
            random.shuffle(index)
            for j in index:
                gstep, rate, cost, _ = sess.run([model.add_global, model.learning_rate, model.cost, model.train_op],
                                                feed_dict={
                                                    model.sbt_input: trainData[0][j],
                                                    model.sbt_pos: trainData[1][j],
                                                    model.sbt_size: trainData[2][j],
                                                    model.code_input: trainData[3][j],
                                                    model.code_size: trainData[4][j],
                                                    model.nl_input: trainData[5][j],
                                                    model.nl_output: trainData[6][j],
                                                    model.nl_size: trainData[7][j],
                                                    model.mask_size: trainData[8][j],
                                                })
                if gstep % 2000 == 0:
                    nowBLEU = val(sess, model, validData, Tokens)
                    if nowBLEU > maxBleu:
                        maxBleu = nowBLEU
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=gstep)
                if gstep % 100 == 0:
                    s = 'After %d steps, cost is %.5f, nowBleu: %.5f, maxBlue: %.5f. ' % (
                        gstep, cost, nowBleu, maxBleu)
                    print(s)
                if gstep >= 300000:
                    BLEU = val(sess, model, testData, Tokens)
                    s = 'After 30000 steps, BLEU in test: %.5f ' % (BLEU)
                    print(s)
                    return


def val(sess, model, data, tokens):
    smooth = SmoothingFunction()
    NL = data[6]
    bleu = 0
    count = 0
    for i in range(len(data[0])):
        predic = sess.run(model.predict,
                          feed_dict={
                              model.sbt_input: data[0][i],
                              model.sbt_pos: data[1][i],
                              model.sbt_size: data[2][i],
                              model.code_input: data[3][i],
                              model.code_size: data[4][i]
                          })
        for j in range(len(predic)):
            hpy = []
            for k in predic[j]:
                if dic_word[k] == '<end>':
                    break
                hpy.append(tokens.nl_word(k))
            if len(hpy) > 2:
                bleu += nltk.translate.bleu([NL[i][j]], hpy, smoothing_function=smooth.method4)
                count += 1

    if count > 1:
        bleu = bleu / count
    return bleu



def main():
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)
    extracting = zipfile.ZipFile('data.zip')
    extracting.extractall()
    extracting.close()
    preprocess.start()
    train()


if __name__ == "__main__":
    main()