import streamlit as st
from PIL import Image
from transformers import PreTrainedTokenizerFast
from transformer_mt.modeling_transformer import TransfomerEncoderDecoderModel



source_tokenizer = PreTrainedTokenizerFast.from_pretrained("../output_dir/source_tokenizer")
target_tokenizer = PreTrainedTokenizerFast.from_pretrained("../output_dir/target_tokenizer")
model = TransfomerEncoderDecoderModel.from_pretrained("../output_dir")
model.eval()


image = Image.open('summy.png')

st.image(image, use_column_width=True)

st.write("""
# Text Summarization
Using bart model!
***
""")

st.header('Enter Input Sequence')
sequence_input = st.text_area("Sequence input", "Enter text here...", height=250)

input_ids = source_tokenizer.encode(sequence_input, return_tensors="pt")
output_ids = model.generate(
    input_ids,
    max_length=500,
    bos_token_id=target_tokenizer.bos_token_id,
    eos_token_id=target_tokenizer.eos_token_id,
    pad_token_id=target_tokenizer.pad_token_id,
)

sequence_output = target_tokenizer.decode(output_ids[0])
st.header('Generated Output Sequence')
st.text_area("Sequence Output", sequence_output, height=250)