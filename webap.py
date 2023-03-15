import streamlit as st
import torch
import torchvision
from Brainiac import Brainiac
from opt import OPT
from utils import check_known
from webfunc import *

# Set the page layout to wide
st.set_page_config(layout="wide")


# Declare a class to store the chat history


##########################################
##########################################

# Declare session state variables
if 'texthistory' not in st.session_state:
    st.session_state.texthistory = make_text_history()

if 'brainiac' not in st.session_state:
    st.session_state.brainiac = make_brainiac()
    st.session_state.prediction = None
    st.session_state.distances = None
    st.session_state.right_answer = None
    st.session_state.known = None
    st.session_state.new_class = True


if 'torch_img' not in st.session_state:
    st.session_state.torch_img = None

if 'disable_new_infer' not in st.session_state:
    st.session_state.disable_new_infer = False
    st.session_state.waiting_yn = False


##########################################
##########################################
print(st.session_state.brainiac.index_2_label)

# Declare columns
left_column, center_column, right_column = st.columns([2, 3, 2])

with left_column:
    st.header("Console")
    st.text("Output")
    chat = st.empty()
    if not st.session_state.waiting_yn:
        st.text_input("Input", key="user_input", on_change=submit)
    else:
        yes_column, no_column = st.columns([1,1])
        with yes_column:
            st.button("Yes", on_click=yes_func)
        with no_column:
            st.button("No", on_click=no_func)

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
    
    st.button('Predict', on_click=on_inference_click, disabled = st.session_state.disable_new_infer)


#st.container