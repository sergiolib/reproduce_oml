""" Loader file for Omniglot dataset"""

import tensorflow_datasets as tfds

omniglot_ds, info = tfds.load(name="omniglot", split="train", with_info=True)

print("Downloaded {} dataset (v:{})".format(info.name, str(info.version)))
print("Size in disk: {} MB".format(info.size_in_bytes / 1000000))

# Inspect features
# print(info.features)

# print(omniglot_ds.element_spec)
