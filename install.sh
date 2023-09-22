conda create -n mutex python=3.8
conda activate mutex
pip install -r requirements.txt
pip install -e LIBERO/.
pip install -e .
