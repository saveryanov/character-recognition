
"""Trains and Evaluates the neuralnet network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import numpy as np

import tensorflow as tf

import input_data
import neuralnet

from PIL import Image, ImageDraw


from tensorflow.contrib.tensorboard.plugins import projector

# Initial Values
RESTORE_MODE = 0
SOFT_DETECTOR = 1
DETECTOR_OFF = 0
SEGMENTATION_OFF = 0
CIFAR_MODE = 0

NUM_CLASSES_D3 = 3
NUM_CLASSES_C12 = 12

hidden_units_D3 = [neuralnet.IMAGE_SIZE_H * neuralnet.IMAGE_SIZE_W, 150, NUM_CLASSES_D3]
hidden_units_C12 = [neuralnet.IMAGE_SIZE_H * neuralnet.IMAGE_SIZE_W, 200, NUM_CLASSES_C12]

EPOCH_LENGTH_D3 = 27000
EPOCH_LENGTH_C12 = 12600

NUM_EPOCHS_PER_DECAY_D3 = 5      # Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY_C12 = 5      # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR_D3 = 0.5  # Learning rate decay factor.
LEARNING_RATE_DECAY_FACTOR_C12 = 0.5  # Learning rate decay factor.

INITIAL_LEARNING_RATE_D3 = 0.01       # Initial learning rate.
INITIAL_LEARNING_RATE_C12 = 0.01       # Initial learning rate.

EPOCH_ITERATIONS_D3 = 2
EPOCH_ITERATIONS_C12 = 2

# Basic model parameters as external flags.
FLAGS = None



def do_eval_accuracy(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return precision




def do_eval_confmatr(sess,
            logits,
            num_classes,
            images_placeholder,
            labels_placeholder,
            data_set):
    """
    Runs one evaluation against the full epoch of data.
    """
    # And run one epoch of eval.

    softmax = tf.nn.softmax(logits)
    num_examples = data_set.num_examples
    confmatr = np.zeros([num_classes, num_classes])
    labels_count = np.zeros([num_classes])
    for step in range(num_examples):
        feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)

        log_res, label = sess.run([softmax, labels_placeholder], feed_dict=feed_dict)
        label = label[0]
        max_ic = 0
        max_val = -999999999

        for ic in range(num_classes):
          if log_res[0][ic] > max_val:
              max_val = log_res[0][ic]
              max_ic = ic
        confmatr[max_ic][label] += 1
        labels_count[label] += 1
    print("Confusion matrix: ")
    print(confmatr)
    print("Labels count: ")
    print(labels_count)
    return confmatr


def do_eval_prec_rec(
        confmatr,
        num_classes):
    """
    Runs one evaluation against the full epoch of data.
    """
    # And run one epoch of eval.
    prec_rec = np.zeros([num_classes, 2])
    for label_y in range(num_classes):
        for label_x in range(num_classes):
            prec_rec[label_y][0] += confmatr[label_x][label_y]  # precision
            prec_rec[label_y][1] += confmatr[label_y][label_x]  # recall

        prec_rec[label_y][0] = confmatr[label_y][label_y]/prec_rec[label_y][0]
        prec_rec[label_y][1] = confmatr[label_y][label_y]/prec_rec[label_y][1]


    print("Precision and recall matrix: ")
    print(prec_rec)

    return prec_rec



def do_eval_f1(
        prec_rec,
        num_classes):
    """
    Runs one evaluation against the full epoch of data.
    """
    # And run one epoch of eval.
    f1score = np.zeros([num_classes])
    for label_y in range(num_classes):
        f1score[label_y] = 2 * (prec_rec[label_y][0] * prec_rec[label_y][1])/(prec_rec[label_y][0] + prec_rec[label_y][1])

    print("F1score matrix: ")
    print(f1score)
    return f1score





def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         neuralnet.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


#google tesseract
#google scholar

def run_training(
        netname,
        hidden_units,
        data_sets,
        epoch_iterations,
        initial_learning_rate,
        num_epochs_per_decay,
        learning_rate_decay_factor):
  """Train neuralnet for a number of steps."""
  #data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
  epoch_length = data_sets.train.num_examples
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    if not CIFAR_MODE:
        logits = neuralnet.inference(images_placeholder, hidden_units)
    else:
        logits = neuralnet.CIFAR_inference(images_placeholder, hidden_units[2])

    # Add to the Graph the Ops for loss calculation.
    loss = neuralnet.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op, learning_rate = neuralnet.training(loss, epoch_length, initial_learning_rate, num_epochs_per_decay, learning_rate_decay_factor)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = neuralnet.evaluation(logits, labels_placeholder)


    # Embedding
    N = epoch_length
    EMB = np.zeros((N, neuralnet.IMAGE_PIXELS), dtype='float32')

    for i in range(N):
        for j in range(neuralnet.IMAGE_PIXELS):
            EMB[i][j] = data_sets.train.images[i][j]

    # The embedding variable, which needs to be stored
    # Note this must a Variable not a Tensor!
    embedding_var = tf.Variable(EMB, name='Embedding_%s'%(netname))

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    best_pres = 0
    last_pres = 0

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    metadata_file = open(os.path.join(FLAGS.log_dir, 'metadata_%s.tsv'%(netname)), 'w')
    metadata_file.write('Name\tClass\n')
    for i in range(N):
        metadata_file.write('%06d\t%s\n' % (i, names[data_sets.train.labels[i]]))
    metadata_file.close()

    # Comment out if you don't have metadata
    embedding.metadata_path = os.path.join(FLAGS.log_dir, 'metadata_%s.tsv'%(netname))

    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(FLAGS.log_dir, 'model%s.ckpt'%(netname)), 1)

    #run_testing(0, logits, images_placeholder, sess)
    accuracy_array = np.zeros([epoch_iterations])
    test_ind = 0
    for step in range(epoch_length * num_epochs_per_decay * epoch_iterations):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        lr_value = sess.run(learning_rate)
        #print('[%s] Step %d: loss = %.8f (%.3f sec) lr = %.6f  bp = %0.04f  lp = %0.04f' % (netname, step, loss_value, duration, lr_value, best_pres, last_pres))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % (epoch_length * num_epochs_per_decay) == 0 or (step + 1) == epoch_length * num_epochs_per_decay * epoch_iterations:
        print('[%s] CHECKPOINT Step %d: loss = %.8f (%.3f sec) lr = %.6f  bp = %0.04f  lp = %0.04f' % (netname, step, loss_value, duration, lr_value, best_pres, last_pres))
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model%s.ckpt'%(netname))
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval_accuracy(sess,
               eval_correct,
               images_placeholder,
               labels_placeholder,
               data_sets.train)

        confmatr = do_eval_confmatr(sess,
                         logits,
                         hidden_units[-1],
                         images_placeholder,
                         labels_placeholder,
                         data_sets.train)

        prec_rec = do_eval_prec_rec(
            confmatr,
            hidden_units[-1])

        f1score = do_eval_f1(
            prec_rec,
            hidden_units[-1])

        # Evaluate against the test set.
        print('Test Data Eval (accuracy):')
        accuracy_array[test_ind] = do_eval_accuracy(sess,
               eval_correct,
               images_placeholder,
               labels_placeholder,
               data_sets.test)

        print('Test Data Eval (confmatr):')
        confmatr = do_eval_confmatr(sess,
                         logits,
                         hidden_units[-1],
                         images_placeholder,
                         labels_placeholder,
                         data_sets.test)

        prec_rec = do_eval_prec_rec(
            confmatr,
            hidden_units[-1])

        f1score = do_eval_f1(
            prec_rec,
            hidden_units[-1])

        print("Test #%d ended" % (test_ind))
        test_ind += 1

  print("Accuracy array: ")
  print(accuracy_array)

  checkpoint_file = os.path.join(FLAGS.log_dir, 'model%s_last.ckpt' % (netname))
  saver.save(sess, checkpoint_file)
  return {'logits': logits, 'images_ph': images_placeholder, 'sess': sess}


def run_testing(global_step, d3_parts, c12_parts):
  """Testing func"""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    image = Image.open("img.bmp")  # Открываем изображение.
    image = image.convert('YCbCr')
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.


    # ---------------- загрузка идеальной маски символов
    image_mask = Image.open("img_mask.bmp")  # Открываем изображение.
    image_mask = image_mask.convert('YCbCr')
    pix_mask = image_mask.load()  # Выгружаем значения пикселей.

    # ---------------- загрузка идеальной маски межсимвольных интервалов
    image_wsmask = Image.open("img_wsmask.bmp")  # Открываем изображение.
    image_wsmask = image_wsmask.convert('YCbCr')
    pix_wsmask = image_wsmask.load()  # Выгружаем значения пикселей.


    # ----------------------------------------------------------------------------------
    # бинаризация исходного изображения
    print('Preprocessing image: binarisation')
    theresold_img = 0.7
    pix_bin = np.zeros((height, width))
    for i in range(width):
        for j in range(height):
            pix_bin[j][i] = pix[i, j][0]
            #if pix[i, j][0] / 255 > theresold_img:
            #    pix_bin[j][i] = 255
            #else:
            #    pix_bin[j][i] = 0

    im = Image.fromarray(pix_bin)
    im = im.convert("L")
    im.save("result%d_img_bin.bmp" % (global_step))


    # ----------------------------------------------------------------------------------
    # Медианный фильтр
    """
    quad_size_med = 1
    pix_bin_med = np.zeros((height, width))
    for j in range(quad_size_med, height - quad_size_med):
        for i in range(quad_size_med, width - quad_size_med):
            aperture = np.zeros((quad_size_med * 2 + 1) * (quad_size_med * 2 + 1))
            for aj in range(-quad_size_med, quad_size_med + 1):
                for ai in range(-quad_size_med, quad_size_med + 1):
                    aperture[(aj + quad_size_med) * (quad_size_med * 2 + 1) + (ai + quad_size_med)] = pix_bin[j + aj][i + ai]

            aperture.sort()
            pix_bin_med[j][i] = aperture[0]

    pix_bin = pix_bin_med
    im = Image.fromarray(pix_bin)
    im = im.convert("L")
    im.save("result%d_img_med.bmp" % (global_step))
    """
    # ----------------------------------------------------------------------------------
    # обработка нейросетью d3
    print('Processimg image: creating maps')

    draw = np.zeros((height, width))
    drawCHAR = np.zeros((height, width))
    drawWS = np.zeros((height, width))

    softmax = tf.nn.softmax(d3_parts['logits'])

    aperture = np.zeros((1, neuralnet.IMAGE_PIXELS))
    for i in range(width - neuralnet.IMAGE_SIZE_W + 1):
      print('Processing image: %d of %d'%(i, width - neuralnet.IMAGE_SIZE_W))
      for j in range(height - neuralnet.IMAGE_SIZE_H + 1):

        for ai in range(neuralnet.IMAGE_SIZE_W):
          for aj in range(neuralnet.IMAGE_SIZE_H):
            aperture[0][ai + aj * neuralnet.IMAGE_SIZE_H] = 255 - (pix_bin[j + aj][i + ai])

        feed_dict = {
            d3_parts['images_ph']: aperture,
        }

        #class_result = sess.run(topK, feed_dict=feed_dict)
        log_res = d3_parts['sess'].run(softmax, feed_dict=feed_dict)

        max_ic = 0
        max_val = -999999999

        for ic in range(NUM_CLASSES_D3):
            if log_res[0][ic] > max_val:
                max_val = log_res[0][ic]
                max_ic = ic

        if max_ic > 0:
          for ai in range(neuralnet.IMAGE_SIZE_W):
            for aj in range(neuralnet.IMAGE_SIZE_H):
              if SOFT_DETECTOR:
                  draw[j+aj][i+ai] += max_val
                  if max_ic == 1:
                      drawWS[j + aj][i + ai] += max_val
                  if max_ic == 2:
                      drawCHAR[j + aj][i + ai] += max_val
              else:
                  draw[j + aj][i + ai] = 255
                  if max_ic == 1:
                      drawWS[j + aj][i + ai] += 255
                  if max_ic == 2:
                      drawCHAR[j + aj][i + ai] += 255

    # ----------------------------------------------------------------------------------
    # нормализация карты вероятностей текста
    print('Normalising text detection map: charonly')
    drawCharonly = np.zeros((height, width))
    max_draw_val = 0
    for i in range(width):
      for j in range(height):
          if max_draw_val < drawCHAR[j][i]:
              max_draw_val = drawCHAR[j][i]

    for i in range(width):
      for j in range(height):
          drawCharonly[j][i] = (drawCHAR[j][i] / max_draw_val) * 255

    im = Image.fromarray(drawCharonly)
    im = im.convert("L")
    im.save("result%d_charmap_charonly.bmp" % (global_step))

    # ----------------------------------------------------------------------------------
    # нормализация карты вероятностей текста
    print('Normalising text detection map: wsonly')
    drawWsonly = np.zeros((height, width))
    max_draw_val = 0
    for i in range(width):
      for j in range(height):
          if max_draw_val < drawWS[j][i]:
              max_draw_val = drawWS[j][i]

    for i in range(width):
      for j in range(height):
          drawWsonly[j][i] = (drawWS[j][i] / max_draw_val) * 255

    im = Image.fromarray(drawWsonly)
    im = im.convert("L")
    im.save("result%d_charmap_wsonly.bmp" % (global_step))

    # ----------------------------------------------------------------------------------
    # нормализация карты вероятностей текста
    print('Normalising text detection map')
    max_draw_val = 0
    for i in range(width):
        for j in range(height):
            if max_draw_val < draw[j][i]:
                max_draw_val = draw[j][i]

    for i in range(width):
        for j in range(height):
            draw[j][i] = (draw[j][i] / max_draw_val) * 255

    im = Image.fromarray(draw)
    im = im.convert("L")
    im.save("result%d_charmap.bmp" % (global_step))

    # ----------------------------------------------------------------------------------
    #draw = drawCharonly # тут можно подменить карту областей с текстом на что-нибудь еще
    if DETECTOR_OFF:
        # ---------------- подмена маски на идеальную
        for i in range(width):
            for j in range(height):
                draw[j][i] = 255 - (pix_mask[i, j][0])

    # ----------------------------------------------------------------------------------
    # рисуем маску текста
    print('Creating text detection mask: binarisation')
    draw_b = np.zeros((height, width))
    theresold_char = 0.30
    for i in range(width):
        for j in range(height):
            if draw[j][i] / 255 > theresold_char:
                draw_b[j][i] = 255
            else:
                draw_b[j][i] = 0

    im = Image.fromarray(draw_b)
    im = im.convert("L")
    im.save("result%d_binary03.bmp" % (global_step))
    draw_original_charmask = draw_b
    # ----------------------------------------------------------------------------------
    # оквадратим области с текстом
    print('Creating text detection mask: quad morphing')
    quad_size = 1
    n_repeats = 5
    quad_step = 1
    for repeat_ind in range(n_repeats):
        print("Processing quad morphing: %d of %d"%(repeat_ind, n_repeats))
        for i in range(quad_size, width - quad_size):
            for j in range(quad_size, height - quad_size):
                is_quad = 1
                for ai in range(-quad_size, quad_size + 1, quad_step):
                    if draw_b[j + ai][i] == 0 or draw_b[j][i + ai] == 0:
                        is_quad = 0
                        break

                if is_quad:
                    for ai in range(-quad_size, quad_size + 1):
                        for aj in range(-quad_size, quad_size + 1):
                            draw_b[j + aj][i + ai] = 255

        for i in range(width - quad_size - 1, quad_size + 1, -1):
            for j in range(height - quad_size - 1, quad_size + 1, -1):
                is_quad = 1
                for ai in range(-quad_size, quad_size + 1, quad_step):
                    if draw_b[j + ai][i] == 0 or draw_b[j][i + ai] == 0:
                        is_quad = 0
                        break

                if is_quad:
                    for ai in range(-quad_size, quad_size + 1):
                        for aj in range(-quad_size, quad_size + 1):
                            draw_b[j + aj][i + ai] = 255

        for i in range(quad_size, width - quad_size):
            for j in range(height - quad_size - 1, quad_size + 1, -1):
                is_quad = 1
                for ai in range(-quad_size, quad_size + 1, quad_step):
                    if draw_b[j + ai][i] == 0 or draw_b[j][i + ai] == 0:
                        is_quad = 0
                        break

                if is_quad:
                    for ai in range(-quad_size, quad_size + 1):
                        for aj in range(-quad_size, quad_size + 1):
                            draw_b[j + aj][i + ai] = 255

        for i in range(width - quad_size - 1, quad_size + 1, -1):
            for j in range(quad_size, height - quad_size):
                is_quad = 1
                for ai in range(-quad_size, quad_size + 1, quad_step):
                    if draw_b[j + ai][i] == 0 or draw_b[j][i + ai] == 0:
                        is_quad = 0
                        break

                if is_quad:
                    for ai in range(-quad_size, quad_size + 1):
                        for aj in range(-quad_size, quad_size + 1):
                            draw_b[j + aj][i + ai] = 255

    im = Image.fromarray(draw_b)
    im = im.convert("L")
    im.save("result%d_binary03_quad.bmp" % (global_step))

    # ----------------------------------------------------------------------------------
    # оценка точности детектора
    charmap_accuracy_pix = 0
    charmap_prec = np.zeros([2])
    charmap_rec = np.zeros([2])
    charmap_f1 = np.zeros([2])

    charmap_confmatr = np.zeros([2, 2])
    for i in range(width):
        for j in range(height):
            if draw[j][i] == (255 - (pix_mask[i, j][0])):
                charmap_accuracy_pix += 1

            label_y = 0
            if 255 - (pix_mask[i, j][0]) > 0:
                label_y = 1
            label_x = 0
            if draw[j][i] > 0:
                label_x = 1

            charmap_confmatr[label_x][label_y] += 1

    for label_y in range(2):
        for label_x in range(2):
            charmap_prec[label_y] += charmap_confmatr[label_y][label_x]
            charmap_rec[label_y] += charmap_confmatr[label_x][label_y]

        charmap_prec[label_y] = charmap_confmatr[label_y][label_y] / charmap_prec[label_y]
        charmap_rec[label_y] = charmap_confmatr[label_y][label_y] / charmap_rec[label_y]
        charmap_f1[label_y] = 2 * (charmap_prec[label_y] * charmap_rec[label_y]) / (charmap_prec[label_y] + charmap_rec[label_y])

    charmap_accuracy = charmap_accuracy_pix / (width * height)  # нормируем

    print("Charmap accuracy: ")
    print(charmap_accuracy)
    print("Charmap accuracy pixels: %d of %d"%(charmap_accuracy_pix, width * height))


    print("Charmap confmatr: ")
    print(charmap_confmatr)
    print("Charmap precision")
    print(charmap_prec)
    print("Charmap recall")
    print(charmap_rec)
    print("Charmap F1 score")
    print(charmap_f1)


    # ----------------------------------------------------------------------------------
    # обрезаем WS вне областей с текстом и находим максимальное и минимальное значения
    print('Cutting whitespaces map in text detection mask')
    max_ws_val = 0
    min_ws_val = 99999999999
    for i in range(width):
        for j in range(height):
            if draw_b[j][i] == 0:
                drawWS[j][i] = 0
            else:
                if max_ws_val < drawWS[j][i]:
                    max_ws_val = drawWS[j][i]
                if min_ws_val > drawWS[j][i]:
                    min_ws_val = drawWS[j][i]

    # ----------------------------------------------------------------------------------
    # нормализация обрезанной карты WS
    print('Normalising whitespaces map')
    for i in range(width):
        for j in range(height):
            drawWS[j][i] = (drawWS[j][i] - min_ws_val) / (max_ws_val - min_ws_val) * 255

    im = Image.fromarray(drawWS)
    im = im.convert("L")
    im.save("result%d_wsmap.bmp" % (global_step))

    # ----------------------------------------------------------------------------------
    # фильтр локальных максимумов для карты ws
    print('Finding local maximums on whitespaces map')
    drawWSmax = drawWS
    quad_size_w = 3
    quad_size_h = 3
    n_repeats = 1
    quad_step = 1
    for repeat_ind in range(n_repeats):
        drawWSmax_tmp = np.zeros((height, width))
        print("Finding local maximums: repeat #%d of %d" % (repeat_ind, n_repeats))
        for i in range(quad_size_w, width - quad_size_w):
            for j in range(quad_size_h, height - quad_size_h):
                max_val = 0
                for ai in range(-quad_size_w, quad_size_w + 1, quad_step):
                    for aj in range(-quad_size_h, quad_size_h + 1, quad_step):
                        if drawWSmax[j + aj][i + ai] > max_val:
                            max_val = drawWSmax[j + aj][i + ai]

                if drawWSmax[j][i] == max_val:
                    drawWSmax_tmp[j][i] = max_val

        drawWSmax = drawWSmax_tmp
        im = Image.fromarray(drawWSmax)
        im = im.convert("L")
        im.save("result%d_ws_max_experiment%d.bmp" % (global_step, repeat_ind))


    # ----------------------------------------------------------------------------------
    # бинаризация максимума для карты ws
    print('Creating whitespaces mask: binarisation')
    drawWSmask = np.zeros((height, width))
    for i in range(width):
        for j in range(height):
            if drawWSmax[j][i] > 0:
                drawWSmask[j][i] = 255

    im = Image.fromarray(drawWSmask)
    im = im.convert("L")
    im.save("result%d_ws_mask.bmp" % (global_step))


    # ----------------------------------------------------------------------------------
    if SEGMENTATION_OFF:
        # ---------------- подмена маски на идеальную
        for i in range(width):
            for j in range(height):
                drawWSmask[j][i] = 255 - (pix_wsmask[i, j][0])

    # ----------------------------------------------------------------------------------
    # фильтрация регионов и выпрямление границ
    print('Creating whitespaces mask: filtration of regions and borders')
    skip_len = 3
    if SEGMENTATION_OFF:
        skip_len = 0

    max_r_height = 1.8
    min_r_height = 0.7

    if DETECTOR_OFF:
        max_r_height = 3
        min_r_height = 0.1

    for j in range(1, height):
        for i in range(width):
            if draw_b[j][i] > 0 and draw_b[j-1][i] == 0:
                is_border = 0
                border_skipped = 0
                region_height = 0
                # ищем границу и вычисляем высоту региона
                while (j+region_height < height) and (draw_b[j + region_height][i] > 0):
                    for bpi in range(1, skip_len + 1):
                        if i - bpi < 0:
                            break
                        if drawWSmask[j + region_height][i - bpi] > 0:
                            border_skipped = 1
                            break

                    if drawWSmask[j + region_height][i] > 0:
                        is_border = 1
                    region_height += 1


                #region_length = 0
                # ищем границу и вычисляем длинну региона
                #while (i + region_length < width) and (draw_b[j][i + region_length] > 0):
                #    region_length += 1

                # если высота региона не попадает в интервал max min то удаляем его
                if region_height > neuralnet.IMAGE_SIZE_H * max_r_height or region_height < neuralnet.IMAGE_SIZE_H * min_r_height:
                    for ih in range(region_height):
                        draw_b[j + ih][i] = 0
                        drawWSmask[j + ih][i] = 0
                    continue

                # если длинна региона меньше 17 то удаляем все границы в нем
                #if region_length <= 17:
                #    for bpj in range(region_height):
                #        for bpi in range(region_length + 1):
                #            if i + bpi < width and j + bpj < height:
                #                drawWSmask[j + bpj][i + bpi] = 0
                #            else:
                #                break
                #    continue

                # рисуем прямую границу до конца и удаляем все на протяжении skip_len
                if is_border or border_skipped:
                    for bpj in range(region_height):
                        if j + bpj >= height:
                            break
                        if border_skipped:
                            drawWSmask[j + bpj][i] = 0
                        else:
                            if is_border:
                                drawWSmask[j + bpj][i] = 255
                                for bpi in range(1, skip_len + 1):
                                    if i + bpi < width:
                                        drawWSmask[j + bpj][i + bpi] = 0
                                    else:
                                        break

    im = Image.fromarray(drawWSmask)
    im = im.convert("L")
    im.save("result%d_ws_mask_filtered.bmp" % (global_step))


    # ----------------------------------------------------------------------------------
    # Разбиение на отдельные буквы

    if tf.gfile.Exists('letters'):
        tf.gfile.DeleteRecursively('letters')
    tf.gfile.MakeDirs('letters')


    softmax_c12 = tf.nn.softmax(c12_parts['logits'])
    c12_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'D', 'P']
    words_found = 0
    symbols_found = 0
    string_result = ''
    for j in range(1, height - 1):
        print("Symbols classification: %d of %d (words: %d, symbols: %d)"%(j, height - 1, words_found, symbols_found))
        for i in range(1, width - 1):
            if draw_b[j][i] > 0 and draw_b[j-1][i] == 0 and draw_b[j][i-1] == 0:    # ищем вехнюю левую границу области

                words_found += 1 # нашли слово/предложение
                if words_found > 1:
                    string_result = string_result + ' '
                #shift_y = round(neuralnet.IMAGE_SIZE_H/2)
                region_height = 0
                # ищем границу и вычисляем высоту региона
                while (j + region_height < height) and (draw_b[j + region_height][i] > 0):
                    region_height += 1

                shift_y = round(region_height/2)
                ap_shift_y = shift_y - round(neuralnet.IMAGE_SIZE_H/2) + 1

                region_length = 0
                # ищем границу и вычисляем высоту региона
                while (i + region_length < width) and (draw_b[j][i + region_length] > 0):
                    region_length += 1

                shift_x = 1   # смещение вправо
                word_len = 0  # индекс буквы внутри области
                prev_border = 0
                next_border = 0
                while draw_b[j + shift_y][i + shift_x] > 0:         # не вышли еще из области?
                    if drawWSmask[j + shift_y][i + shift_x] > 0 or draw_b[j + shift_y][i + shift_x + 1] == 0:  # если наткнулись на следующую границу или дошли до края области
                        word_len += 1

                        next_border = shift_x

                        if region_length <= 17:
                            prev_border = 0
                            next_border = region_length

                        border_distance = next_border - prev_border # как далеко предыдущая граница
                        if border_distance < 6 and region_length > 17:
                            shift_x += 1
                            prev_border = next_border
                            continue


                        margin = round((neuralnet.IMAGE_SIZE_W - border_distance)/2) # отступы справа и слева от границ чтобы символ был посередине

                        border_top = 0
                        border_bottom = 0
                        for aj in range(region_height + 1):
                            if border_top == 0:
                                if draw_original_charmask[j + aj - 1][i + shift_x - round(border_distance/2)] < draw_original_charmask[j + aj][i + shift_x - round(border_distance/2)]:
                                    border_top = aj
                            if border_bottom == 0:
                                if draw_original_charmask[j + aj][i + shift_x] > draw_original_charmask[j + aj + 1][i + shift_x]:
                                    border_bottom = aj

                        vertical_border_distance = border_bottom - border_top
                        vertical_margin = round((neuralnet.IMAGE_SIZE_H - vertical_border_distance)/2)
                        ap_shift_y = border_top - vertical_margin + 1

                        aperture = np.zeros((neuralnet.IMAGE_SIZE_H, neuralnet.IMAGE_SIZE_W))
                        aperture2 = np.zeros((1, neuralnet.IMAGE_PIXELS))
                        for ai in range(neuralnet.IMAGE_SIZE_W):
                            if i + shift_x - border_distance - margin + ai >= 0 and i + shift_x - border_distance - margin + ai < width:
                                for aj in range(neuralnet.IMAGE_SIZE_H):
                                    if j + aj >= 0 and j + aj < height:
                                        aperture[aj][ai] = pix_bin[j + ap_shift_y + aj][i + shift_x - border_distance - margin + ai]
                                        aperture2[0][ai + aj * neuralnet.IMAGE_SIZE_H] = 255 - pix_bin[j + ap_shift_y + aj][i + shift_x - border_distance - margin + ai]
                        prev_border = next_border
                        next_border = 0

                        feed_dict = {
                            c12_parts['images_ph']: aperture2,
                        }

                        # class_result = sess.run(topK, feed_dict=feed_dict)
                        log_res = c12_parts['sess'].run(softmax_c12, feed_dict=feed_dict)

                        max_ic = 0
                        max_val = -999999999

                        for ic in range(12):
                            if log_res[0][ic] > max_val:
                                max_val = log_res[0][ic]
                                max_ic = ic
                        if max_ic == 10:
                            max_ic = 0
                        symbols_found += 1

                        string_result = string_result + c12_names[max_ic]

                        im = Image.fromarray(aperture)
                        im = im.convert("L")
                        im.save("letters/result%d_word%d_letter%d_class%d-%s.bmp" % (global_step, words_found, word_len, max_ic, c12_names[max_ic]))
                        if region_length <= 17:
                            break

                    shift_x += 1

    # ----------------------------------------------------------------------------------
    # объединяем результаты на исходном изображении
    print("Creating combined image of word regions and whitespace borders")

    drawCombined = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            if draw_b[j][i]:
                if drawWSmask[j][i]:
                    drawCombined[j, i] = [255, 0, 0]
                else:
                    pix_val = pix[i, j][0]/7 + draw_b[j][i]/3
                    drawCombined[j, i] = [pix[i, j][0]/7, pix_val, pix[i, j][0]/7]
            else:
                drawCombined[j, i] = [pix[i, j][0], pix[i, j][0], pix[i, j][0]]

    img = Image.fromarray(drawCombined, 'RGB')
    img.save('result%d_combined.bmp' % (global_step))
    img.show()

    del draw, drawCHAR, drawWSmask, drawWS, drawWSmax, drawWSmax_tmp, drawCombined, draw_b
    string_cuneiform = '82050662 86~3 "82050662 82050663 82050664 v ~. ~28,4. 82050665 82050666 82050667 82050668 8v43 ес 083 03 ,0 8 70"'
    string_finereader = '82050061 i --------------— PC-43 0,CI j 3S3.Sc| "82050662 « i 1 j pw3 i C.Cii £7 S. «1C i 82050663 PC-43 u.ui | *436:13} 82050664 7" i ?G*13 и,olj 52С.С5» I 82050665 s PC-43 0,0-8» — 2 321,12! 82050666 i PO-43 11 713.35 \ 82050667 i PO-43 0,011 «477.СС • 82050668 1 PCm13 0,011 2 31.70 j'
    string_y = '82050661 1 P043 001 35880 82050662 1 P043 001 47840 82050663 1 P043 001 49613 82050664 1 P043 001 52005 82050665 8 P043 008 232112 82050666 1 P043 001 71395 82050667 1 P043 001 47700 82050668 1 P043 001 28170'
    lev = levenshtein(string_result, string_y)
    print('LEVENSTEIN:')
    print(lev)
    lev_finereader = levenshtein(string_finereader, string_y)
    print(lev_finereader)

    lev_cf = levenshtein(string_cuneiform, string_y)
    print(lev_cf)



    print('STR res:')
    print(string_result)
    print('STR y:')
    print(string_y)

    print("Test ended")



def nn_restore(
    netname,
    step,
    hidden_units,
    epoch_length,
    initial_learning_rate,
    num_epochs_per_decay,
    learning_rate_decay_factor):

  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    if not CIFAR_MODE:
        logits = neuralnet.inference(images_placeholder, hidden_units)
    else:
        logits = neuralnet.CIFAR_inference(images_placeholder, hidden_units[2])

    # Add to the Graph the Ops for loss calculation.
    loss = neuralnet.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op, learning_rate = neuralnet.training(loss, epoch_length, initial_learning_rate, num_epochs_per_decay, learning_rate_decay_factor)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = neuralnet.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    best_pres = 0
    last_pres = 0

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    checkpoint_file = os.path.join(FLAGS.log_dir, 'model%s_last.ckpt'%(netname))
    saver.restore(sess, checkpoint_file)

    return {'logits': logits, 'images_ph': images_placeholder, 'sess': sess}


def main(_):

  string_cuneiform = '82050662 86~3 "82050662 82050663 82050664 v ~. ~28,4. 82050665 82050666 82050667 82050668 8v43 ес 083 03 ,0 8 70"'
  string_finereader = '82050061 i --------------— PC-43 0,CI j 3S3.Sc| "82050662 « i 1 j pw3 i C.Cii £7 S. «1C i 82050663 PC-43 u.ui | *436:13} 82050664 7" i ?G*13 и,olj 52С.С5» I 82050665 s PC-43 0,0-8» — 2 321,12! 82050666 i PO-43 11 713.35 \ 82050667 i PO-43 0,011 «477.СС • 82050668 1 PCm13 0,011 2 31.70 j'
  string_y = '82050661 1 P043 001 35880 82050662 1 P043 001 47840 82050663 1 P043 001 49613 82050664 1 P043 001 52005 82050665 8 P043 008 232112 82050666 1 P043 001 71395 82050667 1 P043 001 47700 82050668 1 P043 001 28170'

  lev_finereader = levenshtein(string_finereader, string_y)
  print(lev_finereader)

  lev_cf = levenshtein(string_cuneiform, string_y)
  print(lev_cf)


  if not RESTORE_MODE:
      if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
      tf.gfile.MakeDirs(FLAGS.log_dir)

      train_images_d3 = 'shuffled-images-d3.idx3-ubyte.gz'
      train_labels_d3 = 'shuffled-labels-d3.idx1-ubyte.gz'
      train_images_c12 = 'shuffled-images-c12.idx3-ubyte.gz'
      train_labels_c12 = 'shuffled-labels-c12.idx1-ubyte.gz'

      test_images_d3 = 'shuffled-images-d3-test.idx3-ubyte.gz'
      test_labels_d3 = 'shuffled-labels-d3-test.idx1-ubyte.gz'
      test_images_c12 = 'shuffled-images-c12-test.idx3-ubyte.gz'
      test_labels_c12 = 'shuffled-labels-c12-test.idx1-ubyte.gz'

      data_sets_c12 = input_data.read_data_sets(train_images_c12, train_labels_c12, test_images_c12, test_labels_c12, FLAGS.input_data_dir, FLAGS.fake_data)
      c12_parts = run_training(
            'c12',
            hidden_units_C12,
            data_sets_c12,
            EPOCH_ITERATIONS_C12,
            INITIAL_LEARNING_RATE_C12,
            NUM_EPOCHS_PER_DECAY_C12,
            LEARNING_RATE_DECAY_FACTOR_C12)
      del data_sets_c12

      data_sets_d3 = input_data.read_data_sets(train_images_d3, train_labels_d3, test_images_d3, test_labels_d3, FLAGS.input_data_dir, FLAGS.fake_data)
      d3_parts = run_training(
            'd3',
            hidden_units_D3,
            data_sets_d3,
            EPOCH_ITERATIONS_D3,
            INITIAL_LEARNING_RATE_D3,
            NUM_EPOCHS_PER_DECAY_D3,
            LEARNING_RATE_DECAY_FACTOR_D3)
      del data_sets_d3

  d3_parts = nn_restore(
      'd3',
      NUM_EPOCHS_PER_DECAY_D3 * EPOCH_LENGTH_D3 * EPOCH_ITERATIONS_D3 - 1,
      hidden_units_D3,
      EPOCH_LENGTH_D3,
      INITIAL_LEARNING_RATE_D3,
      NUM_EPOCHS_PER_DECAY_D3,
      LEARNING_RATE_DECAY_FACTOR_D3)

  c12_parts = nn_restore(
      'c12',
      NUM_EPOCHS_PER_DECAY_C12 * EPOCH_LENGTH_C12 * EPOCH_ITERATIONS_C12 - 1,
      hidden_units_C12,
      EPOCH_LENGTH_C12,
      INITIAL_LEARNING_RATE_C12,
      NUM_EPOCHS_PER_DECAY_C12,
      LEARNING_RATE_DECAY_FACTOR_C12)

  run_testing(0, d3_parts, c12_parts)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/input_data',
      #default='/tmp/tensorflow/neuralnet/input_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/projects/charclass/board74',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


  # to run tensorboard for this app try:
  # tensorboard --logdir=C:\projects\charclass\board50
  # and then go to http:\\localhost:6006
  # tsne
