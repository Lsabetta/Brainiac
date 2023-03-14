import streamlit as st
import torch
import torchvision
from Brainiac import Brainiac
from opt import OPT


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

@st.cache_resource()
def make_brainiac():
    brainiac = Brainiac(OPT.MODEL, OPT.DISTANCE_TYPE)
    return brainiac

def submit():
    #st.session_state.cached_input = '> ' + st.session_state.user_input
    st.session_state.texthistory.write('> ' + st.session_state.user_input)
    st.session_state.user_input = ""

def on_inference_click():
    if st.session_state.torch_img is not None:
        print('sono arrivato fino a qua')
        st.session_state.texthistory.write('$ clickato')
        st.session_state.brainiac.forward_example(st.session_state.torch_img)


##########################################
##########################################

# Declare session state variables
if 'texthistory' not in st.session_state:
    st.session_state.texthistory = make_text_history()

if 'brainiac' not in st.session_state:
    st.session_state.brainiac = make_brainiac()

if 'torch_img' not in st.session_state:
    st.session_state.torch_img = None

##########################################
##########################################

# Declare columns
left_column, center_column, right_column = st.columns([2, 3, 2])
with left_column:
    st.header("Console")
    st.text("Output")
    chat = st.empty()
    
    # Input
    #texthistory.write(st.session_state.cached_input)
    
    st.text_input("Input", key="user_input", on_change=submit)
    chat.text(st.session_state.texthistory.get_text())
    


with center_column:
    st.header("What I see")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
        bytes_data = img_file_buffer.getvalue()
        torch_img = torchvision.io.decode_image(
            torch.frombuffer(bytes_data, dtype=torch.uint8)
        ).unsqueeze(0).to(OPT.DEVICE)
        
        st.session_state.torch_img = torch_img

        # Check the type of torch_img:
        # Should output: <class 'torch.Tensor'>
        st.write(type(torch_img))

        # Check the shape of torch_img:
        # Should output shape: torch.Size([channels, height, width])
        st.write(torch_img.shape)


with right_column:
    
    st.button('Predict', on_click=on_inference_click)


#st.container