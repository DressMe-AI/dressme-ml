from utils import import_attributes, call_data, train_validate_model, train_final_model

if __name__ == "__main__":
    data_path = "./data/"

    # Step 1: Load data
    encoded_df = import_attributes(data_path, visualize=False)
    X, y = call_data(encoded_df, data_path, verbose=False)

    # Step 2: Train and validate
    best_epoch = train_validate_model(X, y, verbose=True)

    # Step 3: Train final model
    train_final_model(X, y, best_epoch=best_epoch)
