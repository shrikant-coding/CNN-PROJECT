# CIFAR-100 CNN Image Classifier

A Streamlit app that classifies uploaded images into one of the 100 CIFAR-100
classes using a trained TensorFlow/Keras convolutional neural network.

## Run the app

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The trained model and its metadata are included in `models/`, so the app can
run without training first.

## Train the model

```powershell
python train.py
```

Training downloads CIFAR-100 through Keras, saves the model under `models/`,
and writes training metrics to `training_history.csv`. The downloaded dataset
is excluded from Git.
