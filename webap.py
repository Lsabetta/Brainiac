import streamlit as st
import torch
import torchvision
from opt import OPT
from webfunc import *
import torchvision.transforms as T
import torch.nn.functional as F
from main import main as brainiac_main
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
    if OPT.WEBAPP_PRETRAIN:
        brainiac_main()
    
    

if 'image' not in ss:
    ss.image = None
    ss.image_buffer = []
    ss.count_click_photo = 0


if 'disable_new_infer' not in ss:
    ss.disable_new_infer = False
    ss.waiting_yn = False


##########################################
##########################################
print(ss.brainiac.index_2_label)

# Declare columns

left_column, right_column = st.columns([2, 3])


with st.sidebar:
    transform = T.ToPILImage()
    for i, label in enumerate(ss.brainiac.cls_image_examples):
        #st.markdown("""---""")
        l, c, r = st.columns([20, 2, 1])
        with c:
            st.button("x", key=f"x_{i}")
        img = ss.brainiac.cls_image_examples[label]
        
        if i<10 and OPT.WEBAPP_PRETRAIN:
            img = Unnorm(img.detach())
        img = transform(img)
    
        st.image(img, caption=label)

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
    with st.container():
        st.write("Acquired frames")
        st.image([transform(img) for img in ss.image_buffer], width=155)
    


with right_column:
    if ss.brainiac.prediction != None and ss.brainiac.known:
        cls_predicted = ss.brainiac.index_2_label[ss.brainiac.prediction]
    elif ss.brainiac.prediction != None and not ss.brainiac.known:
        cls_predicted = "New Class"
    else:
        cls_predicted = ""

    st.markdown(f'## What I see: <span style="color:#94403A;"> {cls_predicted} </span> ', unsafe_allow_html=True)
    img_file_buffer = st.camera_input("Take a picture", label_visibility = "collapsed", on_change=clear_pred)

    if img_file_buffer is not None:
        # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
        bytes_data = img_file_buffer.getvalue()
        image = torchvision.io.decode_image(
            torch.frombuffer(bytes_data, dtype=torch.uint8)
        ).to(OPT.DEVICE)
        ss.image = torchvision.transforms.Resize((400, 700))(image)
    
    st.button('Predict', on_click=on_inference_click, disabled = ss.disable_new_infer, use_container_width = True)
    
set_page_container_style()


