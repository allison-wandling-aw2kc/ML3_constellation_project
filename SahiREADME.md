# SAHI Environment Setup on Rivanna

## 1. Rivanna Configuration

Launch a GPU-enabled interactive session through UVA Open OnDemand using the following settings:

- Number of hours: 2
- Number of cores: 4
- Memory: 32 GB
- Allocation: shakeri_ds6050
- Work directory: SCRATCH
- GPU type: Default
- Number of GPUs: 1

Verify GPU availability:

```bash
nvidia-smi
```

## 2. Create the Conda Environment

```bash
module load miniforge/26.9.4
conda create -n sahi python=3.11 -y
conda activate sahi
```

Verify the environment:

```bash
python --version
which python
conda info --envs
```

## 3. Install PyTorch with CUDA Support

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4. Install SAHI and Ultralytics

```bash
python -m pip install "sahi[ultralytics]"
```

Verify installation:

```bash
python -c "import sahi; import ultralytics; print('Installation successful')"
```

## 5. Clone the SAHI Repository

```bash
cd ~
mkdir -p sahi_project
cd sahi_project
git clone https://github.com/obss/sahi.git
cd sahi
```

## 6. Run Initial SAHI Test

Copy the `test_sahi.py` script from our group repository into the cloned SAHI directory and run:

```bash
python test_sahi.py
```

## 7. Initial Test Results

The environment was successfully tested using the following configuration:

- Python: 3.11.18
- PyTorch: 2.14.1+cu126
- GPU: NVIDIA RTX A6000
- Model: YOLOv8n
- Slice dimensions: 512 × 512
- Slice overlap: 20%
- Number of slices processed: 6
- Number of objects detected: 22

The initial test confirms that SAHI sliced inference works with GPU acceleration on Rivanna. Further testing will use the Constellation Urban Intersection dataset and the group's fine-tuned YOLOv8x model.

## 8. Reactivate Environment in Future Sessions

```bash
module load miniforge/26.9.4
conda activate sahi
cd ~/sahi_project/sahi
```

There is no need to reinstall the packages when starting a new session.
