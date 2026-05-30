from src.config import *


def SFE_dataset(data_slices, k, m=50):
    """
    Extract statistical features from sliced dataset.

    Parameters
    ----------
    data_slices : array-like or DataFrame
        Input sliced dataset.

    k : int
        Number of features/signals.

    m : int, default=50
        Number of timesteps per feature.

    Returns
    -------
    result : pandas.DataFrame
        Statistical feature extracted dataset with labels.
    """

    df = pd.DataFrame(data_slices)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    n = len(X)

    # Reshape into (samples, features, timesteps)
    X = X.reshape(n, k, m)

    df_features = pd.DataFrame()

    for i in range(k):

        x = X[:, i, :]

        # Statistical Features
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        median = np.median(x, axis=1)

        maximum = np.max(x, axis=1)
        minimum = np.min(x, axis=1)

        rng = maximum - minimum

        q1 = np.percentile(x, 25, axis=1)
        q3 = np.percentile(x, 75, axis=1)

        iqr = q3 - q1

        kt = np.nan_to_num(kurtosis(x, axis=1), nan=0)
        sk = np.nan_to_num(skew(x, axis=1), nan=0)

        # Feature DataFrame
        df_i = pd.DataFrame({
            f'Feature_{i+1}_Mean': mean,
            f'Feature_{i+1}_Std': std,
            f'Feature_{i+1}_Median': median,
            f'Feature_{i+1}_Max': maximum,
            f'Feature_{i+1}_Min': minimum,
            f'Feature_{i+1}_Range': rng,
            f'Feature_{i+1}_IQR': iqr,
            f'Feature_{i+1}_Q1': q1,
            f'Feature_{i+1}_Q3': q3,
            f'Feature_{i+1}_Kurtosis': kt,
            f'Feature_{i+1}_Skewness': sk,
        })

        df_features = pd.concat([df_features, df_i], axis=1)

    # Append labels
    result = pd.concat(
        [
            df_features.reset_index(drop=True),
            pd.DataFrame(Y, columns=['Label']).reset_index(drop=True)
        ],
        axis=1
    )

    return result