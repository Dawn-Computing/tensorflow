import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
import transformers
import logging
import datetime
import os

os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '10'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Load pre-trained BERT model and tokenizer for token classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForTokenClassification.from_pretrained('bert-base-uncased')
# Configure the logging
transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO)
tf.debugging.set_log_device_placement(True)

@tf.function  # Add tf.function decorator to compile this part into a TensorFlow graph
def perform_inference(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='tf')
    # Perform inference
    outputs = model(inputs)
    return outputs

# Example text
text = "TensorFlow operations trace back to C++ implementations."
outputs = perform_inference(text)
# Print the output
print(f"Final output: {outputs}\n")
