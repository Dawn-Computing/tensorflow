import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
import os

tf.config.run_functions_eagerly(True)
# Load pre-trained BERT model and tokenizer for token classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForTokenClassification.from_pretrained('bert-base-uncased')

# Example text
text = "TensorFlow operations trace back to C++ implementations."

# Tokenize the input text
inputs = tokenizer(text, return_tensors='tf')

# Perform inference
outputs = model(inputs)

# Print the output
print(outputs)
