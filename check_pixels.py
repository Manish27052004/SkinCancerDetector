import tensorflow as tf
import numpy as np

# Load exact same image from dataset generator
ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    labels='inferred',
    label_mode='binary',
    class_names=['benign', 'melanoma'],
    batch_size=1,
    image_size=(224, 224),
    shuffle=False
)

print("Starting to iterate over test dataset to find melanoma_10105.jpg...")
# Benign has 500 images. So index 500 (0-indexed) is the first melanoma image.
# melanoma_10105.jpg is alphabetically the first one.
for i, (images, labels) in enumerate(ds):
    if i == 500:
        img_ds = images[0]
        label_ds = labels[0]
        break

print("\n--- image_dataset_from_directory output ---")
print("Shape:", img_ds.shape)
print("Label:", label_ds.numpy())
print("Min/Max:", np.min(img_ds), np.max(img_ds))
print("Mean:", np.mean(img_ds))
print("First 5 pixels (R):", img_ds[0, :5, 0].numpy())


print("\n--- tf.io.read_file output ---")
img_path = "dataset/test/melanoma/melanoma_10105.jpg"
img_tf = tf.io.read_file(img_path)
img_tf = tf.image.decode_jpeg(img_tf, channels=3)
img_tf = tf.image.resize(img_tf, (224, 224))

print("Shape:", img_tf.shape)
print("Min/Max:", np.min(img_tf), np.max(img_tf))
print("Mean:", np.mean(img_tf))
print("First 5 pixels (R):", img_tf[:5, 0, 0].numpy())

print("\nAre they exactly equal?", np.array_equal(img_ds.numpy(), img_tf.numpy()))
