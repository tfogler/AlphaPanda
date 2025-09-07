Alphapanda cheat sheet

https://github.com/tfogler/AlphaPanda

### Cloning the Repo and Creating Env

`git clone https://github.com/tfogler/AlphaPanda.git  # forked alphapanda repo`

### Create the env

`conda create -f AlphaPanda_env.yml`

### Training
CUDA support not yet enabled - 7/31

CUDA support enabled - 8/13
**Example**
`python train_AlphaPanda.py --device cuda --num_workers 0 --logdir ./logs ./configs/YueTrain/codesign_single_tfogler.yml`


- Download the sabdab dataset from https://huggingface.co/datasets/YueHuLab/AlphaPanda_training_dataset/tree/main
- Prepare the config to point to data and set num iterations
- Run `python train_AlphaPanda.py --device cpu --num_worker 0 --logdir ./logs /path/to/config.yml`
- Resume training from checkpoint `python train_AlphaPanda.py ... /path/to/config.yml --resume /path/to/checkpoint/checkpoints/####.pt`

**Example**
`python train_AlphaPanda.py --device cpu --num_workers 0 --logdir ./logs ./configs/YueTrain/codesign_single_tfogler.yml &`

### Design
- Run `python design_AlphaPanda.py /path/to/structure.pdb --config /path/to/config.yml --device cpu --no_renumber --heavy H.ID --light L.ID`

**Example**
`python design_AlphaPanda.py /data/foglerm/SAbDab_PDB_Structures/all_structures/chothia/9l1s.pdb --config ./configs/YueTest/codesign_single_tfoglerTest200.yml --device cpu --no_renumber --heavy C --light B &`

python design_AlphaPanda.py /home/data/all_structures/chothia/7wd7.pdb --config ./configs/YueTest/codesign_single_tfoglerTest800K.yml --device cuda --no_renumber --heavy C --light B

*Note: if loading weights not trained on-device, goto AlphaPanda/AlphaPanda/tools/runner/design_for_pdb.py:162 and edit in `weights_only=False` in kwargs


### Cleaning WSL disc space

If disc space becomes limited you can free up unused space by compressing the virtual hard disc

1. Locate path to your virtual hard disc in powershell
`(Get-ChildItem -Path HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | Where-Object { $_.GetValue("DistributionName") -eq '<distribution-name>' }).GetValue("BasePath") + "\ext4.vhdx"`

https://learn.microsoft.com/en-us/windows/wsl/disk-space#how-to-locate-the-vhdx-file-and-disk-path-for-your-linux-distribution

2. Shutdown WSL in powershell:
`wsl --shutdown`

3. optionally, backup your virtual disc onto an external SSD

4. Compress the WSL vhd with diskpart
```
	diskpart
	# open window Diskpart
	select vdisk file="C:\WSL-Distros\…\ext4.vhdx"
	attach vdisk readonly
	compact vdisk
	detach vdisk
	exit
```

https://superuser.com/questions/1606213/how-do-i-get-back-unused-disk-space-from-ubuntu-on-wsl2
https://github.com/microsoft/WSL/issues/4699#issuecomment-627133168


### Training

Training random states used
2023
899288514
8092025
314159
23452134

Weights correction
rot: 0.9
prop: 1.9
seq: 0.2

#### Training/Design Results

2025-09-07 8e5 rounds in on cuda: loss 3.3520 | loss(rot) 0.6428 | loss(pos) 2.5325 | loss(seq) 0.1767

designing 9l1s pertuzumab and 7wd7 H_CDR3 have wildly inconsistent positions, many residues are 100s or even 1000 A away from the protein location, result is disordered
9l1s not a part of the training set, tried 7wd7 as representative of training set to check for overfitting, observed similar pattern

previously cpu-only design produced a H_CDR3 structure more adherent to the reference protein with fewer rounds of training, still need to hypothesize on why this happened