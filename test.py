from datasets.tf_datasets import load_omniglot


dataset = "omniglot"


def main():

    train_data, test_data = load_omniglot(dataset, verbose=1)

    # More info on how to use tf.Dataset here: https://www.tensorflow.org/guide/data
    # use .take(x) if you want to iterate over specific number of samples
    for sample in train_data.take(1):
        image, label = sample["image"], sample["label"]
        print(image, label)


if __name__ == '__main__':
    main()
