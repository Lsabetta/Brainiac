import streamlit as st
import torch
from Brainiac import Brainiac
from opt import OPT
from utils import check_known
ss = st.session_state
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
            ss.brainiac.store_new_class(ss.user_input)
            ss.texthistory.write(f'$ class {ss.user_input} stored')

        else: #class is known, centroids to be updated
            ss.disable_new_infer = False
            ss.brainiac.update_class(ss.user_input)
            ss.texthistory.write(f'$ class {ss.user_input} updated')
            
                

    ss.user_input = ""

def on_inference_click():
    ss.disable_new_infer = True

    if ss.torch_img is not None:
        first_iteration = len(ss.brainiac.index_2_label)==0
        if not first_iteration:
            ss.prediction, ss.distances = ss.brainiac.forward_example(ss.torch_img, first_iteration)
            ss.known = check_known(ss.prediction, ss.distances, OPT.THRESHOLD)
            label_pred = ss.brainiac.index_2_label[ss.prediction]
            if ss.known:
                ss.texthistory.write(f"$ I believe it is a {label_pred}. Am I right?")
                print(ss.prediction)
                ss.waiting_yn = True
            else:
                ss.texthistory.write(f"$ The closest thing is {label_pred}")
                ss.texthistory.write(f"$ But I think this is a new class, right?")
                ss.waiting_yn = True

        else: 
            #on_first_iteration(first_iteration)
            ss.brainiac.forward_example(ss.torch_img, first_iteration)
            ss.texthistory.write("$ First image ever seen.")
            ss.texthistory.write("$ Would you tell me what this is?")


def on_first_iteration(first_iteration):
    ss.brainiac.forward_example(ss.torch_img, first_iteration)
    ss.texthistory.write("$ First image ever seen.")
    ss.texthistory.write("$ Would you tell me what this is?")

def yes_func():
    ss.waiting_yn = False
    
    if ss.known:
        #update when brainiac prediction when answer is right
        label_pred = ss.brainiac.index_2_label[ss.prediction]
        ss.brainiac.update_class(label_pred)
        ss.texthistory.write(f"$ class {label_pred} updated")
        ss.disable_new_infer = False
    else:
        #behaviour when brainiac was right in telling the class is new
        ss.texthistory.write("$ Would you tell me what this is?")
        #ss.new_class = True

        

def no_func():

    if ss.known:
        ss.waiting_yn = False
        ss.texthistory.write("$ Dang. Would you tell me what this is then?")

    else:
        label_pred = ss.brainiac.index_2_label[ss.prediction]
        ss.texthistory.write(f"$ Dang. Was it a {label_pred}?")
        ss.known = True
        
