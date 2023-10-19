## Installation

pull docker image
```
docker pull nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
```

Start container
```
docker run --gpus all --pid=host --shm-size=1g -it -v <host>:/root/ <image> bash
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
