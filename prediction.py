import joblib

def predict(data):
    # Load the pre-trained model
    clf = joblib.load("rf_model.sav")
    
    # Make a prediction
    return clf.predict(data)
