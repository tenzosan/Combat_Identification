<<<<<<< HEAD
## An Evidential Reasoning Approach for Aerial Target Classification and Intent Prediction
Timely classification and intent prediction of aerial targets is crucial for a combat aircraft to
make informed tactical decisions. The prevailing approach for aerial target classification relies
on data-driven models using time-series data. These models perform well with long-duration
data; however, this is impractical in combat situations involving rapidly evolving threats that
demand quick decisions. Minimizing false predictions is essential, as uncertainty is preferable
to incorrect assessments in high-risk environments. Here, we propose an integrated approach
to target classification and intent prediction that enables decisions from partial data in settings
where threats require rapid response. In the proposed method, predictions are generated
from short sequential sub-samples instead of the entire time series, and the results are refined
by propagating beliefs across sub-samples. Outputs from classifiers are combined through
an evidential reasoning framework to manage uncertainty. Target intent is inferred using
rule-based techniques and a distance-based combination method to fuse information over time.
Due to lack of publicly available datasets, a dataset for aerial target classification was generated
for evaluation. A case study involving eight targets is used to demonstrate the effectiveness of
the approach, whereby accuracies of 88% and 93% are achieved for target type classification
and intent prediction, respectively.

## 🚀 Installation Guide For Windows

### **1️ Install Anaconda Navigator**
Download and install Anaconda from the official website:  
🔗 [Anaconda Download](https://www.anaconda.com/download)


### **2 Create and Activate Virtual Environment**
Run the following commands in the **Anaconda Prompt** or **Terminal**:

```bash
conda create --name tf_gpu python=3.8
conda activate tf_gpu
```

### **3️ Install GPU Support (CUDA & cuDNN)**
For GPU acceleration, install **CUDA Toolkit 11.2** and **cuDNN 8.1.0**:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
### **4️ Install GPU Support (CUDA & cuDNN)**

```bash
python -m pip install "tensorflow<2.11"
```

## **5️ Verify GPU Installation Check if TensorFlow detects your GPU:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
## **6️ Install Required Packages Run the following to install additional dependencies:**
```bash
pip install notebook joblib scikit-learn pandas matplotlib seaborn
pip install -U aeon[all_extras]
```


## Combat Geometry
<p align="center">
  <img src="./images/combatGeometry.png" width="45%" style="margin-right: 10px;" />
  <img src="./images/angleOffCondn.png" width="45%" />
</p>

<p align="center">
  <img src="./images/lambdaCondn-1.png" width="45%" style="margin-right: 10px;" />
  <img src="./images/lambdaCondn-2.png" width="45%" />
</p>

## Demo Cruise Missile Vs Own-shp
<table>
  <tr>
    <td rowspan="2" width="55%"><img src="./images/Cruise_Missile_traj.gif" width="100%" /></td>
    <td width="45%"><img src="./images/Cruise_Missile_Type.gif" width="100%" /></td>
  </tr>
  <tr>
    <td><img src="./images/Cruise_Missile_Intent.gif" width="100%" /></td>
  </tr>
</table>

=======
# Combat-identification-CID-
>>>>>>> bdbe6fced514556107b54b307a6cd690f4f46b8c
