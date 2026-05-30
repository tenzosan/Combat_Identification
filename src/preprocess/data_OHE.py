from src.config import *


def OHE_dataset(
    train_data,
    test_data=None,
    val_data=None,
    save_path=None
):
    """
    One-Hot Encode target labels.
    """

    train_data = pd.DataFrame(train_data)

    if test_data is not None:
        test_data = pd.DataFrame(test_data)

    if val_data is not None:
        val_data = pd.DataFrame(val_data)

    y_train = train_data.iloc[:, -1].values.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)

    y_train_encoded = encoder.fit_transform(y_train)

    outputs = [y_train_encoded]

    if test_data is not None:

        y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
        y_test_encoded = encoder.transform(y_test)

        outputs.append(y_test_encoded)

    if val_data is not None:

        y_val = val_data.iloc[:, -1].values.reshape(-1, 1)
        y_val_encoded = encoder.transform(y_val)

        outputs.append(y_val_encoded)

    if save_path:
        joblib.dump(encoder, save_path)

    return tuple(outputs)