# **DeepETpicker**

A deep learning based open-source software with a friendly user interface to pick 3D particles rapidly and accurately from cryo-electron tomograms. With the advantages of weak labels, lightweight architecture and GPU-accelerated pooling operations, the cost of annotations and the time of computational inference are significantly reduced while the accuracy is greatly improved by applying a Gaussian-type mask and using a customized architecture design.

[Guole Liu, Tongxin Niu, Mengxuan Qiu, Yun Zhu, Fei Sun, and Ge Yang, “DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning,” Nature Communications, vol. 15, no. 1, pp. 2090, 2024.](https://www.nature.com/articles/s41467-024-46041-0)


**Note**: DeepETPicker is a Pytorch implementation. 

## **About this fork**

I reorganised how this software reads and writes data, because the original layout was hard to follow and the interface sometimes failed. Everything hung off a single `base_path`, predictions ended up inside TensorBoard's version folders, and it was never clear from the guide which file fed which step. The workflow now follows RELION: every step writes into its own numbered job directory, a later step reads an earlier one by path, and nothing is overwritten when you re-run. The steps are plain command-line programs (`deepetpicker prepare | train | predict | export | compare`) so that [RELION-US](https://github.com/schiracha/RELION-US), a web front end for RELION, can run them as jobs next to RELION's own, locally or on SLURM.

The old PyQt interface (`main.py`) and the `bin/` scripts are still here; apart from the bug fixes listed in the plan (for example `bin/train_bash.py` used to ignore `--train_set_ids` / `--val_set_ids`), I did not change what they do. I plan to remove the interface once everything it did, including the things it did silently, is covered by the new workflow; the list of those is in [docs/PLAN_relion_io_webgui.md](docs/PLAN_relion_io_webgui.md). The original installation and GUI tutorial below still apply to it.

Where this fork differs from the original [cbmi-group/DeepETPicker](https://github.com/cbmi-group/DeepETPicker): the `deepetpicker` command line and package are new, and `train.py` / `test.py` gained an output-directory mode (checkpoints and predictions go into the job directory; a failed training run now raises instead of being swallowed). RELION-US needs this fork, not the original. I also corrected one reporting bug in the validation step: `val_f1` and `val_iou` were swapped (the IoU was logged as F1 and the reverse), so logs from earlier runs have those two labels interchanged.

## **Quick start: job directories**

Install as described under [Setup](#setup) (conda, Python 3.8, PyTorch 1.7.1, then `pip install -r requirement.txt`), and then, from a checkout of this repository:

```bash
pip install -e .          # provides the `deepetpicker` command; keep the checkout in place
```

`prepare` needs only that. `train` and `predict` need the PyTorch environment.

A worked example: tomograms recorded at 2.1 Å/px and binned 3 times (6.3 Å/px), a 150 Å particle, and `.coords` files with the known particles of two tomograms (`TS_001`, `TS_002`) next to a third that has none (`TS_003`). Run from a project folder, and give each command its own job directory:

```bash
# 1. standardize the tomograms and paint weak labels from the known particles.
#    A 150 A particle is 24 voxels at 6.3 A/px; the diameters are made odd (25, label 13).
deepetpicker prepare --output_dir DeepETPickerPrepare/job001 \
    --tomo_dir raw_data --coords_dir raw_data \
    --pixel_size 6.3 --particle_diameter_A 150

# 2. train; TS_002 is held out for validation, the other annotated tomogram trains
deepetpicker train --output_dir DeepETPickerTrain/job002 \
    --prepare_job DeepETPickerPrepare/job001 --val_tomo TS_002 --gpu 0

# 3. pick particles in all three tomograms; the picks were made on a 6.3 A/px grid and a RELION
#    tomogram at 10.5 A/px needs them multiplied by 6.3 / 10.5 = 0.6 in particles.star
deepetpicker predict --output_dir DeepETPickerPredict/job003 \
    --train_job DeepETPickerTrain/job002 --tomo_dir raw_data --relion_scale 0.6

# 4. other formats, if you need them
deepetpicker export --coords_dir DeepETPickerPredict/job003/coords --fmt box --output_dir picks_box
```

Each job leaves everything it made in its own directory:

```
DeepETPickerPrepare/job001/   tomograms_std/  coords/  labels/  occupancy/  dataset.star  job_params.json
DeepETPickerTrain/job002/     checkpoints/ (best.ckpt, last.ckpt)  metrics.csv  loss_full.png  logs/  job_params.json
DeepETPickerPredict/job003/   coords/ (<tomogram>.coords)  particles.star  coords_with_area/  job_params.json
```

`job_params.json` records every parameter and which earlier job each one used, so a result can always be traced back to its model and settings. [docs/job_directories.md](docs/job_directories.md) has the full reference: every flag, every file, the formats, how pixel size and binning translate into voxel diameters, and what replaced which old file.

Picks are whole voxels unless you pass `--refine_centroids`, an experimental option that moves each pick to the centre of mass of the prediction around it. I have not validated it against ground truth.

## **Using it from RELION-US**

RELION-US lists these three steps under **DeepETPicker (Beta)**, builds the same command lines from a form, and chains them by job directory. See its README for installation.

## **Original documentation (PyQt interface)**

The rest of this file is the original documentation, written for the PyQt interface and the `bin/` scripts.

## **Setup**

### **Prerequisites**

- Linux (Ubuntu 18.04.5 LTS； CentOS)
- NVIDIA GPU

### **Installation**

#### **Option 1: Using conda**

The following instructions assume that `pip` and `anaconda` or `miniconda` are available. In case you have a old deepetpicker environment installed, first remove the old one with:

```bash
conda env remove --name deepetpicker
```

The first step is to crate a new conda virtual environment:

```bash
conda create -n deepetpicker -c conda-forge python=3.8.3 -y 
```

Activate the environment:

```bash
conda activate deepetpicker
```

To download the codes, please do:
```
git clone https://github.com/cbmi-group/DeepETPicker
cd DeepETPicker
```

Next, install a custom pytorch and relative packages needed by DeepETPicker:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y

pip install -r requirement.txt
```

- **if using GUI**
	
	To use GUI packages with Linux, you will need to install the following extended dependencies for Qt. 
	1. For `CentOS`, to install packages, please do:
		```bash
		sudo yum install -y mesa-libGL libXext libSM libXrender fontconfig xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil libxkbcommon-x11
		```

	2. For `Ubuntu`, to install packages, please do:
		```bash
		sudo apt-get install -y libgl1-mesa-glx libglib2.0-dev libsm6 libxrender1 libfontconfig1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-dev libdbus-1-3
		```


	To run the DeepETpicker, please do:

	```bash
	conda activate deepetpicker
	python PATH_TO_DEEPETPICKER/main.py
	```

	Note: `PATH_TO_DEEPETPICKER` is the corresponding directory where the code located.
- **Non GUI version**

	In addition to the `GUI version` of DeepETPicker, we also provide a `non-GUI version` of DeepETPicker for people who understand python and deep-learning. It consists of four processes, including `preprocessing`, `train config generation`, `training` and `testing`. A sample tutorial can be found in `.bin/bash_command.md`.

#### **Option 2：Using docker**

The following steps are required in order to run DeepETPicker:
1. Install [Docker](https://www.docker.com/)

	Note: docker engine version shuold be >= 19.03. The size of Docker mirror of Deepetpicker is 7.21 GB, please ensure that there is enough memory space.

2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

3. Download Docker image of DeepETPicker.

	```bash
	docker pull docker.io/lgl603/deepetpicker:latest
	```

4. Run the Docker image of DeepETPicker.

	```bash
	docker run --gpus=all -itd \
			--restart=always \
			--shm-size=100G \
			-e DISPLAY=unix$DISPLAY \
			--name deepetpicker \
			-p 50022:22 \
			--mount type=bind,source='/host_path/to/data',target='/container_path/to/data' \
			lgl603/deepetpicker:latest
	```

- The option `--shm-size` is used to set the required size of shared momory of the docker containers.
- The option `--mount` is used to mount a file or directory on the host machine into the Docker container, where `source='/host_path/to/data'` denotes denotes the data directory really existed in host machine. `target='/container_path/to/data'` is the data directory where the directory `'/host_path/to/data'` is mounted in the container. 

	**Note: `'/host_path/to/data'` should be writable by running bash command `chmod -R 777 '/host_path/to/data'`. `'/host_path/to/data'` should be replaced by the data directory real existed in host machine. For convenience, `'/container_path/to/data'` can set the same as `'/host_path/to/data'`**
	<img width="960" alt="image" src="https://user-images.githubusercontent.com/16335327/198990001-b04fbd1e-c284-482a-81f9-c266fc957a42.png">


1. The DeepETPicker can be used directly in this machine, and it also can be used by a machine in the same LAN.
   - Directly open DeepETPicker in this machine: 
   ```bash
   ssh -X test@'ip_address' DeepETPicker
   # where the 'ip_address' of DeepETPicker container can be obtained as follows:
   docker inspect --format='{{.NetworkSettings.IPAddress}}' deepetpicker
   ```
   <img width="702" alt="image" src="https://user-images.githubusercontent.com/16335327/198967756-3a409b6f-bc19-42cd-83ec-d4cb67776b58.png">

   
   - Connect to this server remotely and open DeepETPicker software via a client machine:
   ```bash
   ssh -X -p 50022 test@ip DeepETPicker
   ```
	Here `ip` is the IP address of the server machine，password is `password`.

`Installation time`: the size of Docker mirror of Deepetpicker is 7.21 GB, and the installation time depends on your network speed. When the network speed is fast enough, it can be configured within a few minutes.

## **Particle picking tutorial**

Detailed tutorials for two sample datasets of [SHREC2021](https://github.com/cbmi-group/DeepETPicker/blob/main/tutorials/A_tutorial_of_particlePicking_on_SHREC2021_dataset.md) and [EMPIAR-10045](https://github.com/cbmi-group/DeepETPicker/blob/main/tutorials/A_tutorial_of_particlePicking_on_EMPIAR10045_dataset.md) are provided. Main steps of DeepETPicker includeds preprocessing, traning of DeepETPicker, inference of DeepETPicker, and particle visualization.

### **Preprocessing**
- Data preparation
  
	Before launching the graphical user interface, we recommend creating a single folder to save inputs and outputs of DeepETpicker. Inside this base folder you should make a subfolder to store raw data. This raw_data folder should contain: 
  - tomograms(with extension .mrc or .rec)
  - coordinates file with the same name as tomograms except for extension. (with extension *.csv, *.coords or *.txt. Generally, *.coords is recoommand.).  

	<br>

  Here, we provides two sample datasets of EMPIAR-10045 and SHREC_2021 for particle picking to enable you to learn the processing flow of DeepETPicker better and faster. The sample dataset can be download in one of two ways:
  - Baidu Netdisk Link: [https://pan.baidu.com/s/1aijM4IgGSRMwBvBk5XbBmw](https://pan.baidu.com/s/1aijM4IgGSRMwBvBk5XbBmw ); verification code: cbmi
  - Zeenodo Link: [https://zenodo.org/records/12512436](https://zenodo.org/records/12512436)

- Data structure
  
  The data should be organized as follows:
	```
	├── /base/path
	│   ├── raw_data
	│   │   ├── tomo1.coords
	│   │   └── tomo1.mrc
	│   │   └── tomo2.coords
	│   │   └── tomo2.mrc
	│   │   └── tomo3.mrc
	│   │   └── tomo4.mrc
	│   │   └── ...
	```

	For above data, `tomo1.mrc` and `tomo2.mrc` can be used as train/val dataset, since they all have coordinate files (matual annotation). If a tomogram has no matual annotation (such as `tomo3.mrc`), it only can be used as test dataset.
	
	<br>


- Input & Output

	<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/utils/images/Preprocessing.gif" width="80%"> 
	</div>

	Launch the graphical user interface of DeepETPicker. On the `Preprocessing` page, please set some key parameters as follows:
	- `input`
		- Dataset name: e.g. SHREC_2021_preprocess
		- Base path: path to base folder
		- Coords path:  path to raw_data folder 
		- Coords format: .csv, .coords or .txt
		- Tomogram path: path to raw_data folder
		- Tomogram format: .mrc or .rec
		- Number of classes:  multiple classes of macromolecules also can be localized separately
	- `Output`
		- Label diameter(in voxels): the diameter of generated weak label, which is usually smaller than the average diameter of the particles. Empirically, you can set it as large as possible but should be smaller than the real diameter.
		- Ocp diameter(in voxels):  the real diameter of the particles. Empirically, in order to obtain good selection results, we recommend that the particle size is adjusted to the range of 20~30 by binning operation. For particles of multi-classes, their diameters should be separated with a comma.
		- Configs: if you click 'Save configs', it would be the path to the file which contains all the parameters filled in this page  
	

### **Training of DeepETPicker**
	
<br>

	Note: Before `Training of DeepETPicker`, please do `Preprocessing` first to ensure that the basic parameters required for training are provided.

<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/utils/images/Training.gif" width="80%"> 
</div>

In practice, default parameters can give you a good enough result.

*Training parameter description:*

- Dataset name: e.g. SHREC_2021_train
- Dataset list: get the list of train/val tomograms. The first column denotes particle number, the second column denotes tomogram name, the third column denotes tomogram ids. If you have n tomograms, the ids will be {0, 1, 2, ..., n-1}.
- Train dataset ids: tomograms used for training. You can click `Dataset list` to obain the dataset ids firstly. One or multiple tomograms can be used as training tomograms. But make sure that the `traning dataset ids` are selected from {0, 1, 2, ..., n-1}, where n is the total number of tomograms obtained from `Dataset list`. Here, we provides two ways to set dataset ids:
  - 0, 2, ...: different tomogram ids are separated with a comma.
  - 0-m: where the ids of {0, 1, 2, ..., m-1} will be selected. Note: this way only can be used for tomograms with continuous ids.
- Val dataset ids: tomograms used for validation. You can click `Dataset list` to obain the dataset ids firstly. Note: only one tomogram can be selected as val dataset.
- Number of classes:  particle classes you want to pick
- Batch size: a number of samples processed before the model is updated. It is determined by your GPU memory, reducing this parameter might be helpful if you encounter out of memory error.
- Patch size: the sizes of subtomogram. It needs to be a multiple of 8. It is recommended that this value is not less than 64, and the default value is 72.
- Padding size: a hyperparameter of overlap-tile strategy. Usually, it can be from 6 to 12, and the default value is 12.
- Learning rate:  the step size at each iteration while moving toward a minimum of a loss function.
- Max epoch: total training epochs. The default value 60 is usually sufficient.
- GPU id: the GPUs used for training, e.g. 0,1,2,3 denotes using GPUs of 0-4. You can run the following command to get the information of available GPUs: nvidia-smi.
- Save Configs: save the configs listed above. The saved configurations contains all the parameters filled in this page, which can be directly loaded via *`Load configs`* next time instead of filling them again.

### **Inference of DeepETPicker**
	
<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/utils/images/Inference.gif" width="80%"> 
</div>
	
In practice, default parameters can give you a good enough result.

*Inference parameter description:*

- Train Configs: path to the configuration file which has been saved in the `Training` step
- Networks weights: path to the model which has be generated in the `Training` step
- Patch size & Pad_size: tomogram is scanned with a specific stride S and a patch size of N in this stage, where S = N - 2*Pad_size.
- GPU id: the GPUs used for inference, e.g. 0,1,2,3 denotes using GPUs of 0-4. You can run the following command to get the information of available GPUs: nvidia-smi.

*Coord format conversion*

The predicted coordinates with extension `*.coords` has four columns: `class_id, x, y, z`. To facilitate users to perform the subsequent subtomogram averaging, format conversion of coordinate file is provided. 

- Coords path: the path of coordinates data predicted by well-trained DeepETPicker.
- Output format: three formats can be converted, including `*.box` for EMAN2, `*.star` for RELION, `*.coords` for RELION.

	
Note: After converting the coordinates to `*.coords`, one can get the coordinates with relion4 format using the script in `https://github.com/cbmi-group/DeepETPicker/blob/main/utils/coords_to_relion4.py` directly.


### **Particle visualization and mantual picking**

<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/utils/images/Visualization.gif" width="90%"> 
</div>

- *Showing Tomogram*

	You can click the `Load tomogram` button on this page to load the tomogram.

- *Showing Labels*
	
	After loading the coordinates file by clicking `Load labels`, you can click `Show result` to visualize the label. The label's diameter and width also can be tuned on the GUI.

- *Parameter Adjustment*

	In order to increase the visualization of particles, Gaussian filtering and histogram equalization are provided:
	- Filter: when choosing Gaussian, a Gaussian filter can be applied to the displayed tomogram, kernel_size and sigma(standard deviation) can be tuned to adjust the visual effects
	- Contrast: when choosing hist-equ, histogram equalization can be performed 
	

- *Mantual picking*
	
	After loading the tomogram and pressing ‘enable’, you can pick particles manually by double-click the left mouse button on the slices. If you want to delete an error labeling, just right-click the mouse.	You can specify a different category id per class. Always remember to save the resuls when you finish.

- *Position Slider*
	
	You can scan through the volume in x, y and z directions by changing their values. For z-axis scanning, shortcut keys of Up/Down arrow can be used.

# Troubleshooting

If you encounter any problems during installation or use of DeepETPicker, please contact us by email [guole.liu@ia.ac.cn](guole.liu@ia.ac.cn). We will help you as soon as possible.

# Citation

If you use this code for your research, please cite our paper [DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning](https://www.nature.com/articles/s41467-024-46041-0).

```
@article{DeepETPicker,
  title={DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning},
  author={Guole Liu, Tongxin Niu, Mengxuan Qiu, Yun Zhu, Fei Sun, and Ge Yang},
  journal={Nature Communications},
  year={2024}
}
```
