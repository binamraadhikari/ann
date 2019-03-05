import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
bank_notes = pd.read_csv('data.csv')
sns.pairplot(data=bank_notes)
bank_notes_without_class = bank_notes.drop('Class', axis=1)
scaler = StandardScaler()
scaler.fit(bank_notes_without_class)
scaled_features = pd.DataFrame(data=scaler.transform(bank_notes_without_class), columns=bank_notes_without_class.columns)
# Rename 'Class' to 'Authentic'
bank_notes = bank_notes.rename(columns={'Class': 'Authentic'})
# 'Forged'
bank_notes.loc[bank_notes['Authentic'] == 0, 'Forged'] = 1
bank_notes.loc[bank_notes['Authentic'] == 1, 'Forged'] = 0

# X and y
X = scaled_features
y = bank_notes[['Authentic', 'Forged']]
# Convert X and y to Numpy arrays
X = X.as_matrix()
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

learning_rate = 0.01
training_epochs = 100
batch_size = 100

n_hidden_1 = 4 # # nodes in first hidden layer
n_hidden_2 = 4 # # nodes in second hidden layer
n_input = 4 # input shape
n_classes = 2 # total classes (authentic / forged)
n_samples = X_train.shape[0] # # samples

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#define this before calling, previously yo pred bata call garnu pachi thyo ani dherai lyang bhayo
def multilayer_perceptron(x, weights, biases):

    '''
    x: Placeholder for data input
    weights: Dictionary of weights
    biases: Dictionary of biases

    '''

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out'] + biases['out'])

    return out_layer







preds = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=preds))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
costs = []
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(n_samples/batch_size)
    for batch in range(total_batch):
        batch_x = X_train[batch*batch_size : (1+batch)*batch_size]
        batch_y = y_train[batch*batch_size : (1+batch)*batch_size]
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch

    print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))
    costs.append(avg_cost)

print("Model has completed {} epochs of training.".format(training_epochs))

correct_predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_predictions)
print("Accuracy:", accuracy.eval(feed_dict={x: X_test, y: y_test}))

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
preds_rfc = rfc.predict(X_test)
print(classification_report(y_test, preds_rfc))
# Get only the 'Forged' column values from y_test and preds_rfc
y_test_forged = [item[1] for item in y_test]
preds_rfc_forged = [item[1] for item in preds_rfc]
# Print confusion matrix
print(confusion_matrix(y_test_forged, preds_rfc_forged))
