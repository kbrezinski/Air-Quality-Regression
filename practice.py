
import tensorflow as tf
import xlrd
import numpy as np

DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

dataset = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))

iterator = dataset.make_one_shot_iterator()
X,Y = iterator.get_next()

#X = tf.placeholder(name='x',dtype=tf.int32)
#Y = tf.placeholder(name='y',dtype=tf.int32)

with tf.Session() as sess:
    print(sess.run([X,Y]))
    print(sess.run([X,Y]))
