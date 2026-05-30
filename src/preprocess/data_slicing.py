from src.config import *

def slice_dataset(data_concat, k, m, w):
    
    df = pd.DataFrame(data_concat)
    samples = []

    s = m // w
    n = len(df.index)

    for i in range(n):
        x = df.iloc[i].tolist()

        start = 0

        for _ in range(s):
            sample = []
            ind = start

            for _ in range(k):
                for q in range(ind, ind + w):

                    if q < len(x):
                        sample.append(x[q])

                ind += m

            # Append label/class
            if len(x) > 0:
                sample.append(x[-1])

            samples.append(sample)

            start += w

    return samples