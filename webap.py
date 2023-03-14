import streamlit as st
import torch
import torchvision
# Set the page layout to wide
st.set_page_config(layout="wide")


# Declare a class to store the chat history
class TextHistory:
    def __init__(self):
        self.text = [">"]*10

    def write(self, text):
        if type(text) is list:
            self.text += text
        else:
            self.text.append(text)

    def __str__(self):
        return '\n'.join(self.text)
    
    def get_text(self):
        return '\n'.join(self.text[-10:])

@st.cache_resource()
def make_text_history():
    texthistory = TextHistory()
    return texthistory

def submit():
    #st.session_state.cached_input = '> ' + st.session_state.user_input
    st.session_state.texthistory.write('> ' + st.session_state.user_input)
    st.session_state.user_input = ""


##########################################
##########################################

# Declare the columns
left_column, center_column, right_column = st.columns([2, 3, 2])

# Declare chat history string
if 'cached_input' not in st.session_state:
    st.session_state.texthistory = make_text_history()



with left_column:
    st.header("Console")
    st.text("Output")
    chat = st.empty()
    
    # Input
    #texthistory.write(st.session_state.cached_input)
    
    st.text_input("Input", key="user_input", on_change=submit)
    chat.text(st.session_state.texthistory.get_text())
    


with center_column:
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
        bytes_data = img_file_buffer.getvalue()
        torch_img = torchvision.io.decode_image(
            torch.frombuffer(bytes_data, dtype=torch.uint8)
        )

        # Check the type of torch_img:
        # Should output: <class 'torch.Tensor'>
        st.write(type(torch_img))

        # Check the shape of torch_img:
        # Should output shape: torch.Size([channels, height, width])
        st.write(torch_img.shape)

with right_column:
    st.button('Inference')

