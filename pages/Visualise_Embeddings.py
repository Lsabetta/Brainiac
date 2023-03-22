import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import umap 
ss = st.session_state
@st.cache_data
def make_UMAP_embedding(X, n_neighbours):
    return umap.UMAP(n_components=3, n_neighbors=n_neighbours, metric="euclidean").fit_transform(X)
@st.cache_data
def make_TSNE_embedding(X, perplexity):
    return TSNE(n_components=3, perplexity = perplexity, n_iter=1000).fit_transform(X)
if len(ss) != 0:
    X = ss.brainiac.all_embeddings
    labels = X.keys()
    
    labels_names = [[l]*len(X[l]) for l in labels]

    X = np.array([x.cpu().numpy() for label in labels for x in X[label]])

    print("embeddings shape: ", X.shape)
    n_neighbours = 5+ int(45*(X.shape[0]/7500))
    print(n_neighbours)

    print(f"{labels=}")
    r, l = st.columns([1,1])
    with r:
        st.header("Umap visualization")
        n_neighbours = st.slider("n neighbours", value = 5)
        X_embedded_UMAP = make_UMAP_embedding(X, n_neighbours)
        df_UMAP = pd.DataFrame(X_embedded_UMAP, columns = ["0", "1", "2"])
        df_UMAP["label"] = [i  for j in labels_names for i in j]
        fig_UMAP = px.scatter_3d(df_UMAP, x="0", y="1", z="2", color = "label")

        st.plotly_chart(fig_UMAP, use_container_width=True, sharing="streamlit", theme="streamlit")

    with l:
        st.header("TSNE visualization")
        perplexity = st.slider("Perplexity", value = 5)
        X_embedded_TSNE = make_TSNE_embedding(X, perplexity)
        df_TSNE = pd.DataFrame(X_embedded_TSNE, columns = ["0", "1", "2"])
        df_TSNE["label"] = [i  for j in labels_names for i in j]
        fig_TSNE = px.scatter_3d(df_TSNE, x="0", y="1", z="2", color = "label")

        st.plotly_chart(fig_TSNE, use_container_width=True, sharing="streamlit", theme="streamlit")
else:
    st.header("0 image acquired")