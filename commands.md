## Installation

pull docker image
```
docker pull nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
```

Start container
```
docker run --gpus all --pid=host --shm-size=1g -it -v <host>:/root/ <image> bash
```

Install packages
```
apt-get update
apt-get install sudo wget curl zip git ffmpeg libsm6 libxext6 -y
```

Install AWS CLI
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```
cd ~
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Install python 3.8
```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.9.0-0-Linux-x86_64.sh
bash Miniconda3-py38_23.9.0-0-Linux-x86_64.sh
rm -r -f Miniconda3-py38_23.9.0-0-Linux-x86_64.sh
```

Install and activate virtual env
```
conda create -n textbpn++ python=3.8
conda activate textbpn++
```

Install PyTorch 1.10.1
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Generate SSH Key
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
```
ssh-keygen -t ed25519 -C "yewen.zhou@scryanalytics.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
$ cat ~/.ssh/id_ed25519.pub
# Then select and copy the contents of the id_ed25519.pub file
# displayed in the terminal to your clipboard
```

Clone git repo
```
cd ~
git clone git@github.com:scry-yewenzhou/TextBPN-Plus-Plus.git
cd TextBPN-Plus-Plus
git fetch origin dev
git switch dev
```


## Evaluation

### Evaluate TotalText

Download dataset
```
cd ~/TextBPN-Plus-Plus/dataset/total_text
bash download.sh
```

Download model
https://drive.google.com/file/d/11AtAA429JCha8AZLrp3xVURYZOcCC2s1/view

```
cd ~/TextBPN-Plus-Plus/scripts-eval/
conda activate textbpn++
```

Comment/Uncomment lines depending on which model to evaluate
Add --viz to the end if need plot

```
bash Eval_Totaltext.sh
```

### Evaluate Detection 157

```
cd ~/TextBPN-Plus-Plus/scripts-eval/
conda activate textbpn++
```

Comment/Uncomment lines depending on which model to evaluate
Add --viz to the end if need plot

```
bash Eval_Detection157.sh
```

### Evaluate CTW1500

```
cd ~/TextBPN-Plus-Plus/scripts-eval/
conda activate textbpn++
```

Comment/Uncomment lines depending on which model to evaluate
Add --viz to the end if need plot

```
bash Eval_CTW1500.sh
```

### Evaluate TD500
```
cd ~/TextBPN-Plus-Plus/scripts-eval/
conda activate textbpn++
```

Comment/Uncomment lines depending on which model to evaluate
Add --viz to the end if need plot

```
bash Eval_TD500.sh
```

## Training

### Train TD500
```
cd TextBPN-Plus-Plus/scripts-train/
bash train_TD500_res50_1s.sh
```
