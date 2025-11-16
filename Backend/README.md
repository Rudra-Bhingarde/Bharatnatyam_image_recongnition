conda create -p venv python=3.10
conda activate venv/
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
pip uninstall numpy
pip install numpy==1.26.4
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"