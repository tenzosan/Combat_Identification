from src.config import *


def scale_dataset(
    train_data,
    test_data=None,
    val_data=None,
    save_path=None,
    return_labels=True
):
    """
    Scale datasets using StandardScaler.
    """

    train_data = pd.DataFrame(train_data)

    if test_data is not None:
        test_data = pd.DataFrame(test_data)

    if val_data is not None:
        val_data = pd.DataFrame(val_data)

    if return_labels:

        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]

        if test_data is not None:
            X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

        if val_data is not None:
            X_val, y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]

    else:

        X_train = train_data.iloc[:, :-1]

        if test_data is not None:
            X_test = test_data.iloc[:, :-1]

        if val_data is not None:
            X_val = val_data.iloc[:, :-1]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    outputs = [X_train_scaled]

    if test_data is not None:
        X_test_scaled = scaler.transform(X_test)
        outputs.append(X_test_scaled)

    if val_data is not None:
        X_val_scaled = scaler.transform(X_val)
        outputs.append(X_val_scaled)

    if save_path:
        joblib.dump(scaler, save_path)

    if return_labels:

        outputs.append(y_train)

        if test_data is not None:
            outputs.append(y_test)

        if val_data is not None:
            outputs.append(y_val)

    return tuple(outputs)