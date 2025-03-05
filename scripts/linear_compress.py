import sys
import numpy as np

rng = np.random.default_rng(0)
print("Loading dataset", file=sys.stderr)
data = np.load(sys.stdin.buffer)[sys.argv[1]]
print("Sampling compressor", file=sys.stderr)
sample_size, vocab_size = data.shape
rank = int(sys.argv[2])
compressor = rng.standard_normal((vocab_size, rank))
subset_size = int(sys.argv[3])
print("Compressing data subset", file=sys.stderr)
data_train = data[:subset_size]
compressed = data_train @ compressor
print("Training decompressor", file=sys.stderr)
decompressor = np.linalg.pinv(compressed) @ data_train
print("Testing decompression", file=sys.stderr)
data_test = data[subset_size: 2 * subset_size]
compressed_test = data_test @ compressor
np.testing.assert_allclose(compressed_test @ decompressor, data_test, atol=1e-4)
print("Compressing full dataset", file=sys.stderr)
full_compressed = data @ compressor
print("Saving compressed dataset")
np.savez(sys.stdout.buffer, decompressor=decompressor, compressed=compressed)
