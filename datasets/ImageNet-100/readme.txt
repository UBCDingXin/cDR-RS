Download unprocessed ImageNet-100 (`imagenet.tar.gz`) from https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing

Please note that `val_image` in `imagenet.tar.gz` is not used!!!

Run `make_dataset.py` to process the ImageNet-100 dataset.

The processed ImageNet-100 h5 file:
h5 structure
-images_train: 118503x3x128
-labels_train: 118503
-labels_train_label1000: 118503; membership of training images among the original 1000 classes
-images_valid: 10000x3x128x128
-labels_valid: 10000
-labels_valid_label1000: 10000; membership of validation images among the original 1000 classes
-existing_classes: 100; membership of all images among the original 1000 classes
