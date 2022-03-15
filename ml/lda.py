
from ml_class import train_model, save_model

# LDA Code:
if __name__ == "__main__":

    # Import and create a ML model
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_model = LinearDiscriminantAnalysis()

    # Train the ML model
    dataset_name = "suyash10gpieday.csv"
    lda_model, lda_scaler = train_model(lda_model, dataset_name)

    # Save the ML model
    model_name = "lda_test"
    save_model(lda_model, lda_scaler, model_name)

    # Use the ML model with the Myobands
    #myo_predict(lda_model, lda_scaler)



