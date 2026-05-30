from src.config import *
# ----------------  Check whether the trajectory is attacking or not ------------------------------- #   

# Calculate the norm of features 
def EUNorm(data,label):
    data = data.apply(pd.to_numeric, errors='coerce')
    l2_norms = []
    for i in range(data.shape[0]):
        xrow_i = data.iloc[i, :].values
        norm = np.linalg.norm(xrow_i)
        l2_norms.append(norm)
    Eu_Norm = pd.DataFrame(l2_norms)
    label = label.reset_index(drop=True)
    data_norm = pd.concat([Eu_Norm,label],axis = 1)
    return data_norm


def Cond(data, label):
    # Calculate Norm from feature vectors
    norm = EUNorm(data, label)
    data = pd.concat([data,label],axis = 1)
    data.reset_index(drop=True, inplace=True)
    # Create a boolean mask for filtering
    mask1 = (data.iloc[:, 3] == "Attacking") & (norm.iloc[:, 0] < 1.414)
    mask2 = (data.iloc[:, 3] == "Non Attacking") & (norm.iloc[:, 0] >= 1.73)
    # Use np.where to conditionally replace values
    data.iloc[:, 3] = np.where(mask1, "Non Attacking", data.iloc[:, 3])
    data.iloc[:, 3] = np.where(mask2, "Attacking", data.iloc[:, 3])
    return data