# Video Anomaly Detection Model - ASTNetPlus

**ASTNetPlus** is an enhanced version of the ASTNet model, developed for video anomaly detection. This version integrates **ConvLSTM** for temporal modeling and **SE Block** for attention mechanisms, improving performance on challenging datasets like **Ped2** and **Avenue**.

---

## 📂 Datasets
The following datasets were used for training and evaluation:

- **UCSD Ped2 Dataset**: [Download UCSD Ped2](https://www.svcl.ucsd.edu/projects/anomaly/dataset.htm#ped2)
- **CUHK Avenue Dataset**: [Download CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)

---

## 🔗 Pretrained Model
The backbone of our model uses **WiderResNet38**, pretrained on ImageNet. You can download the pretrained weights from the Internet.

---

## 🚀 Training the Model
To train the model, use the following command:

```bash
python train.py --cfg "path_to_yaml_file.yaml"
```
Make sure to update your .yaml configuration file with the correct dataset paths and hyperparameters.

## 🧪 Evaluating the Model
To evaluate the model, use the following command:
```bash
python test.py \
    --cfg "path_to_yaml_file.yaml" \
    --model-file "path_to_trained_model.pth"
```
Ensure that the correct YAML configuration file and trained model checkpoint are provided.

## 📥 Download Trained Models
You can download the trained models for **Ped2** and **Avenue** from the following links:

- **Ped2 Trained Model (5 epochs)**: [Download Ped2 Model](https://drive.google.com/drive/folders/1W204w9blNcusqB8ZnDd5tseXtObihtUo?usp=sharing)
- **Avenue Trained Model (2 epochs)**: [Download Avenue Model](https://drive.google.com/drive/folders/1W204w9blNcusqB8ZnDd5tseXtObihtUo?usp=sharing)

