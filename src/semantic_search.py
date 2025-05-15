import pandas as pd
import utils

print("Initializing semantic search...")
db_data = utils.db_connection_and_data()
processed_data, titles = utils.data_preprocessing(db_data)
#utils.fine_tuning(processed_data)
model_path = utils.upzip_saved_model()
model = utils.load_saved_model(model_path)
index = utils.faiss_index(model, processed_data)
print("Model and index loaded.")

# Make a function just to return the static variables
def semantic():
    return model, index, processed_data, titles
