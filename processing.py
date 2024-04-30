import idx2numpy

#处理二进制文件
file_train_images = 'data/train-images-idx3-ubyte'
file_train_labels = 'data/train-labels-idx1-ubyte'
file_test_images='data/t10k-images-idx3-ubyte'
file_test_labels='data/t10k-labels-idx1-ubyte'
def load_data():
    train_images=idx2numpy.convert_from_file(file_train_images)
    train_labels=idx2numpy.convert_from_file(file_train_labels)
    test_images=idx2numpy.convert_from_file(file_test_images)
    test_labels=idx2numpy.convert_from_file(file_test_labels)
    return train_images,train_labels,test_images,test_labels





