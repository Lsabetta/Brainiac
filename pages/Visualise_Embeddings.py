import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import umap 
ss = st.session_state
if len(ss.brainiac.all_embeddings) != 0:
    X = ss.brainiac.all_embeddings
    labels = X.keys()
    print("labels", labels)
    labels_names = [[l]*len(X[l]) for l in labels]
    print("lables names", labels_names)

    for l in labels:
        for x in X[l]:
            print(x.shape)

    X = np.array([x.cpu().numpy() for label in labels for x in X[label]])

    print("cazzone", X.shape)
    X_embedded = umap.UMAP(n_components=3).fit_transform(X)

    print("x embedded", X_embedded.shape, type(X_embedded))
    df = pd.DataFrame(X_embedded, columns = ["0", "1", "2"])
    df["label"] = [i  for j in labels_names for i in j]
    fig = px.scatter_3d(df, x="0", y="1", z="2", color = "label")

    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")
else:
    st.header("0 image acquired")