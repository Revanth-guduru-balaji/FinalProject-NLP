# Text Summarization Using facebook/bart-base model
## Step1 - Clone the repository in your command prompt.
```bash
git clone  
```
## Step2 - Install all the required packages.
Run the below command for installing required packages.
```bash
pip install -r requirements.txt
```
## Step3 - Before training make sure cuda is available.Use the command below to check if it prints True or False.
```bash
python -c "import torch;torch.cuda.is_available();"
```
## Step4 - Train the pre-trained model.
Run the below command to train by default values.
```bash
python pre-trained-model.py --output_dir pre-trained-model-output
```
To pass custom values below is an example of passing custom values to train the model.
```bash
python pre-trained-model.py \
    --output_dir output_dir \
    --batch_size 64 \
    --num_warmup_steps 5000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --eval_every_steps 5000
```
## Step5 -Using Streamlit we can now interact  with our trained model.
Run the below command.
```bash
streamlit run app.py
```

