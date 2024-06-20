import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
import transformers
import logging
import os

os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '10'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TPU_NAME'] = 'fake_tpu'

# Configure the logging
transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO)
tf.debugging.set_log_device_placement(True)
tf.config.optimizer.set_jit(True)  # Enable XLA globally

class DummyTPUClusterResolver(tf.distribute.cluster_resolver.TPUClusterResolver):
    def __init__(self):
        self._tpu = os.environ['TPU_NAME']
        self._environment = None
        self._zone = None
        self._project = None

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        return "grpc://" + os.environ['TPU_NAME']

    def cluster_spec(self):
        return tf.train.ClusterSpec({})

    def get_tpu_system_metadata(self):
        class DummyTPUMetadata:
            def __init__(self):
                self.devices = [
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:0', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:1', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:2', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:3', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:4', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:5', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:6', 'device_kind': 'TPU', 'core_count': 8}),
                    type('Device', (object,), {'name': '/job:tpu_worker/replica:0/task:0/device:TPU:7', 'device_kind': 'TPU', 'core_count': 8})
                ]
                self.num_hosts = 1
                self.num_of_cores_per_host = 8
        return DummyTPUMetadata()

# Use the dummy TPU resolver
resolver = DummyTPUClusterResolver()

# Skip actual TPU system initialization for simulation
# Normally, we would call `tf.config.experimental_connect_to_cluster(resolver)`
# and `tf.tpu.experimental.initialize_tpu_system(resolver)`, but we skip these for the simulation

# Create TPU strategy
strategy = tf.distribute.TPUStrategy(resolver)

# Load pre-trained BERT model and tokenizer for token classification within the strategy scope
with strategy.scope():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForTokenClassification.from_pretrained('bert-base-uncased')

    @tf.function  # Add tf.function decorator to compile this part into a TensorFlow graph
    def perform_inference(text):
        with tf.device('/job:tpu_worker/replica:0/task:0/device:TPU:0'):
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
