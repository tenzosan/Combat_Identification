from src.config import *

def split_dataset(
    x,
    y,
    test_size=0.2,
    val_size=None,
    random_state=42,
    shuffle=True
):
    """
    Split dataset into train, test, and optional validation sets.
    """

    # ========================== Train-Test Split ========================== #

    x_main, x_test, y_main, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=shuffle
    )

    # ======================= Train-Validation Split ======================= #

    if val_size is not None:

        x_train, x_val, y_train, y_val = train_test_split(
            x_main,
            y_main,
            test_size=val_size,
            random_state=random_state,
            stratify=y_main,
            shuffle=shuffle
        )

    else:

        x_train, y_train = x_main, y_main

    # ========================== Combine Dataset ========================== #

    train_concat = pd.concat(
        [
            pd.DataFrame(x_train).reset_index(drop=True),
            pd.DataFrame(y_train).reset_index(drop=True)
        ],
        axis=1
    )

    test_concat = pd.concat(
        [
            pd.DataFrame(x_test).reset_index(drop=True),
            pd.DataFrame(y_test).reset_index(drop=True)
        ],
        axis=1
    )

    if val_size is not None:

        val_concat = pd.concat(
            [
                pd.DataFrame(x_val).reset_index(drop=True),
                pd.DataFrame(y_val).reset_index(drop=True)
            ],
            axis=1
        )

        return train_concat, test_concat, val_concat

    return train_concat, test_concat