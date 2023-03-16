import streamlit as st
import torch
import torchvision
from opt import OPT
from webfunc import *
import torchvision.transforms as T

#alias for session_state

ss = st.session_state

# Set the page layout to wide
st.set_page_config(layout="wide")


##########################################
##########################################

# Declare session state variables
if 'texthistory' not in ss:
    ss.texthistory = make_text_history()

if 'brainiac' not in ss:
    ss.brainiac = make_brainiac()
    ss.prediction = None
    ss.distances = None
    ss.known = None


if 'image' not in ss:
    ss.image = None
    ss.image_buffer = []

if 'disable_new_infer' not in ss:
    ss.disable_new_infer = False
    ss.waiting_yn = False


##########################################
##########################################
print(ss.brainiac.index_2_label)

# Declare columns

left_column, right_column = st.columns([2, 3])
set_page_container_style()
with st.sidebar:
    transform = T.ToPILImage()
    for i, img in enumerate(ss.image_buffer):
        #st.markdown("""---""")
        l, c, r = st.columns([20, 2, 1])
        with c:
            st.button("x", key=f"x_{i}")
        img = transform(img)
        st.image(img, caption=ss.brainiac.index_2_label[i])

with left_column:
    st.header("Console")
    st.subheader("Output")
    chat = st.empty()
    if not ss.waiting_yn:
        st.subheader("Input")
        st.text_input("Input", label_visibility = "collapsed", key="user_input", on_change=submit)
    else:
        yes_column, no_column = st.columns([1,1])
        with yes_column:
            st.button("Yes", on_click=yes_func)
        with no_column:
            st.button("No", on_click=no_func)

    chat.text(ss.texthistory.get_text())
    


with right_column:
    print(ss.prediction)
    if ss.prediction != None:
        cls_predicted = ss.brainiac.index_2_label[ss.prediction]
    else:
        cls_predicted = ""
    
    st.header(f"What I see: {cls_predicted}")

    img_file_buffer = st.camera_input("Take a picture", label_visibility = "collapsed", on_change=clear_pred)

    if img_file_buffer is not None:
        # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
        bytes_data = img_file_buffer.getvalue()
        image = torchvision.io.decode_image(
            torch.frombuffer(bytes_data, dtype=torch.uint8)
        ).unsqueeze(0).to(OPT.DEVICE)
        
        ss.image = image
    
    st.button('Predict', on_click=on_inference_click, disabled = ss.disable_new_infer, use_container_width = True)
    
    # rc_l, rc_c, rc_r = st.columns([1,1,1])
    # with rc_c: 
    #     st.header(ss.brainiac.index_2_label[ss.prediction])