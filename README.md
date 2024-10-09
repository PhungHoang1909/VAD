# Video Anomaly Detection Model - ASTNetPlus

ASTNetPlus is an enhanced version of the ASTNet model, developed for video anomaly detection. This version integrates **ConvLSTM** for temporal modeling and **SE Block** for attention mechanisms, improving the performance on challenging datasets like Ped2 and Avenue.

## Datasets
We use the following datasets for training and evaluation:

- **Ped2 Dataset**: [Download Ped2](https://www.svcl.ucsd.edu/projects/anomaly/dataset.htm#ped2)
- **Avenue Dataset**: [Download Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)

## Pretrained Model
The backbone of our model uses **WiderResNet38**, pretrained on ImageNet. You can download the pretrained weights from the following link:

- **WiderResNet38 Pretrained**: [Download WiderResNet38](https://github.com/hszhao/semseg)

## Training the Model
To train the model, use the following command:

```bash
python train.py --cfg "path_to_yaml_file.yaml"
```
## Evaluate the Model
To evaluate the model, use the following command:
```bash
python test.py \
    --cfg "path_to_yaml_file.yaml" \
    --model-file "path_to_trained_model.pth"
```
Ensure that the correct YAML configuration file and model checkpoint file are provided.

# Download Trained Models
You can download the trained models for Ped2 and Avenue from the following links:
** Ped2 Trained Model ** on 5 epoches
** Avenue Trained Model ** on 2 epoches


