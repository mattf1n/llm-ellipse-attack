import sys
import numpy as np

# Loads, compresses, and saves logit with low rank
# Usage: python scripts/linear_compress.py logits <rank> <subset_size> < <path_to_logits_npz_file> > <path_to_compressed_output_npz_file>
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
compressed_train = data_train @ compressor
print("Training decompressor", file=sys.stderr)
decompressor = np.linalg.pinv(compressed_train) @ data_train
print("Testing decompression", file=sys.stderr)
data_test = data[subset_size: 2 * subset_size]
compressed_test = data_test @ compressor
np.testing.assert_allclose(compressed_test @ decompressor, data_test, atol=1e-4)
print("Compressing full dataset", file=sys.stderr)
compressed = data @ compressor
print("Saving compressed dataset")
np.savez(sys.stdout.buffer, decompressor=decompressor, compressed=compressed)
