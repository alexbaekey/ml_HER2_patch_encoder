# download the data
HEROHE (~1TB)
https://ecdp2020.grand-challenge.org/
There is a google drive folder containing the WSIs and labels

# install the base conda env along with other dependencies:
```bash
conda env create -f env.yml
conda activate ml_med_imaging 
pip install torch torchvision timm pandas openpyxl scikit-learn pillow tqdm openslide-python
```

# install openslide for ubuntu:
```bash
sudo apt-get install libopenslide0 openslide-tools
```

# download the weights for the CTransPath encoder
download link can be found at: https://github.com/Xiyue-Wang/TransPath
```
mkdir -p checkpoints
mv /path/to/ctranspath.pth checkpoints/ctranspath.pth
```

# run with:

```python
conda activate ml_med_imaging
python train_encoders.py
```

# plot all figures and evaluation results with:
```python
python figs.py
python plot_results.py
python plot_results_and_report_metrics.py
```
