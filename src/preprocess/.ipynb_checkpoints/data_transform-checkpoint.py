from src.config import *


def transform_dataset(data_slices, k, m=50):
    """
    Transform sliced dataset for LSTM input.

    Parameters
    ----------
    data_slices : array-like or DataFrame
        Input sliced dataset.

    k : int
        Number of features/signals.

    m : int, default=50
        Number of timesteps.

    Returns
    -------
    samples : list
        Transformed dataset.
    """

    df = pd.DataFrame(data_slices)
    samples = []

    n = len(df.index)

    for i in range(n):

        x = df.iloc[i].tolist()
        sample = []

        for j in range(m):

            index = j

            for _ in range(k):

                sample.append(x[index])
                index += m

        sample.append(x[-1])

        samples.append(sample)

    return samples