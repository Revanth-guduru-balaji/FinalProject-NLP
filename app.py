import streamlit as st
from PIL import Image
import transformers
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig
)

device = ""
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
config = BartConfig.from_json_file("./config.json")
model = BartForConditionalGeneration.from_pretrained("./pytorch_model.bin", config=config)

model.eval()




image = Image.open('summy.png')

st.image(image, use_column_width=True)

st.write("""
# Text Summarization
Using bart model!a
***
""")



st.header('Enter Input Sequence')
sequence_input = st.text_area("Sequence input", "Enter text here...", height=250)
if st.button("Summarize"):
    input_object = tokenizer(sequence_input,text_pair=None,padding=True,return_tensors="pt")

    output_ids = model.generate(input_ids=input_object["input_ids"], attention_mask=input_object["attention_mask"])
    sequence_output = tokenizer.batch_decode(output_ids)
    sequence_output=sequence_output[0].replace('<s>','')
    sequence_output=sequence_output.replace('</s>','')
    st.header('Model Output Sequence')
    st.text_area("Sequence Output", sequence_output, height=250)