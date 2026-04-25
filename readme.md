# download the data
HEROHE (~1TB)
https://ecdp2020.grand-challenge.org/
There is a google drive folder containing the WSIs and labels

# install the base conda env, then add mobilemedsam and other dependencies

```bash
conda env create -f env.yml
conda activate ml_med_imaging 
pip install torch torchvision timm pandas openpyxl scikit-learn pillow tqdm openslide-python
```

# Install openslide for ubuntu
sudo apt-get install libopenslide0 openslide-tools

# run with 

```python
conda activate ml_med_imaging
python train_encoders.py
```
