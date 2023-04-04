import streamlit as st
import torch
from Brainiac import Brainiac
from opt import OPT
from utils import check_known
import av

#alias for session_state
ss = st.session_state

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
    with torch.no_grad():
        brainiac = Brainiac(OPT.MODEL, OPT.DISTANCE_TYPE)
        return brainiac

def submit():
    ss.texthistory.write('> ' + ss.user_input)
    
    if ss.disable_new_infer:
        dic = ss.brainiac.index_2_label
        cls_index = [k for k in dic if dic[k] == ss.user_input]
        if len(cls_index) == 0: #the class is new since it is not present in the dic
            ss.disable_new_infer = False
            ss.brainiac.store_new_class(ss.user_input, ss.image)
            ss.texthistory.write(f'$ class {ss.user_input} stored')
            ss.image_buffer = []

        else: #class is known, centroids to be updated
            ss.disable_new_infer = False
            ss.brainiac.update_class(ss.user_input)
            ss.texthistory.write(f'$ class {ss.user_input} updated')
            ss.image_buffer = []
            
                

    ss.user_input = ""

def on_inference_click():
    ss.disable_new_infer = True

    if ss.image is not None:
        first_iteration = len(ss.brainiac.index_2_label)==0
        if not first_iteration:
            if len(ss.image_buffer) == 1:
                stack = ss.image_buffer[0].unsqueeze(0)
            else:
                stack = torch.stack(ss.image_buffer)

            ss.brainiac.prediction, ss.brainiac.distances = ss.brainiac.forward_example(stack, first_iteration)
            
            ss.brainiac.known = check_known(ss.brainiac.prediction, ss.brainiac.distances, OPT.THRESHOLD)
            label_pred = ss.brainiac.index_2_label[ss.brainiac.prediction]
            if ss.brainiac.known:
                ss.texthistory.write(f'$ I believe it is a {label_pred}. Am I right?')
                ss.waiting_yn = True
            else:
                ss.texthistory.write(f"$ The closest thing is {label_pred}")
                ss.texthistory.write(f"$ But I think this is a new class, right?")
                ss.waiting_yn = True

        else: 
            #on_first_iteration(first_iteration)
            if len(ss.image_buffer) == 1:
                stack = ss.image_buffer[0].unsqueeze(0)
            else:
                stack = torch.stack(ss.image_buffer)

            ss.brainiac.forward_example(stack, first_iteration)
            ss.texthistory.write("$ First image ever seen.")
            ss.texthistory.write("$ Would you tell me what this is?")


def yes_func():
    ss.waiting_yn = False
    
    if ss.brainiac.known:
        #update when brainiac prediction when answer is right
        label_pred = ss.brainiac.index_2_label[ss.brainiac.prediction]
        ss.brainiac.update_class(label_pred)
        ss.texthistory.write(f"$ class {label_pred} updated")
        ss.disable_new_infer = False
        ss.image_buffer = []

    else:
        #behaviour when brainiac was right in telling the class is new
        ss.texthistory.write("$ Would you tell me what this is?")
        #ss.new_class = True

        

def no_func():

    if ss.brainiac.known:
        ss.waiting_yn = False
        ss.texthistory.write("$ Dang. Would you tell me what this is then?")

    else:
        label_pred = ss.brainiac.index_2_label[ss.brainiac.prediction]
        ss.texthistory.write(f"$ Dang. Was it a {label_pred}?")
        ss.brainiac.known = True

    
def del_class(label):
    #temp_label_2_index = dict(ss.brainiac.label_2_index)
    temp_index_2_label = {}
    idx = ss.brainiac.label_2_index[label]

    del ss.brainiac.cls_image_examples[label]
    del ss.brainiac.centroids[label]
    del ss.brainiac.all_embeddings[label]
    del ss.brainiac.sigmas[label]
    
    del ss.brainiac.label_2_index[label]
    del ss.brainiac.index_2_label[idx]
    
    if idx <= ss.n_class_core50:
        ss.n_class_core50 -= 1

    for key in ss.brainiac.label_2_index:
        if ss.brainiac.label_2_index[key]>idx:
            ss.brainiac.label_2_index[key] -= 1 
        else:
            pass
    for i in ss.brainiac.index_2_label:
        if i > idx:
            temp_index_2_label[i-1] = ss.brainiac.index_2_label[i]
        else:
            temp_index_2_label[i] = ss.brainiac.index_2_label[i]
    ss.brainiac.index_2_label = temp_index_2_label




def clear_pred():
    ss.brainiac.prediction = None
    ss.count_click_photo += 1
    if (ss.image is not None) and (ss.count_click_photo%2 == 0):
        ss.image_buffer.append(ss.image)
        print([el.shape for el in ss.image_buffer])

def Unnorm(img):
    m = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(-1, 1, 1)
    s = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(-1, 1, 1)
    return img.mul_(s).add_(m)

def set_page_container_style():
 
    st.markdown(
        f'''
        <style>
        /* sidebar */
        .css-1544g2n {{
            padding: 2.5rem 0rem 0rem;
            
            }}

        /* markdownline */
        .css-1kbbaad {{
            padding: 0rem 0rem 0rem;
            }}

        /* central-widget */
        .css-z5fcl4 {{
            padding: 2rem 1rem 1rem 1rem;
            }}
        /*x button width*/
        .css-8koux3 {{
    
            font-weight: 200;
            padding: 0.25rem 0.25rem;
            
            width: 1rem;
            height: 1rem;
             }}
        /*.css-13n2bn5{{
            height: 400px;
        }}
        video{{
            height: 350px;
        }}*/

        /*pages div*/
        .css-1mbwpjo{{
            padding-top: 3rem;
            
             }}

        </style>
        ''',
        unsafe_allow_html=True,
    )
