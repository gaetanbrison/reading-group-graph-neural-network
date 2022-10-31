## NLP App Theatre Reviews

### Import packages

import os
from pathlib import Path
import base64
import time
import math
import json
import ast

import pandas as pd
import numpy as np



import matplotlib.pyplot as plt
import matplotlib

from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb



import streamlit as st

import torch

import py3Dmol
from rdkit import Chem

import matplotlib



st.set_page_config(
    page_title="Hyperparameter Tuning", layout="wide", page_icon="images/flask.png"
)


### methods used within streamlit - to organize later on 


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

    images = Image.open('./images/hi-paris.png')
    st.image(images, width=200)

    st.markdown("# Reading Group:  Graph Neural Network 🔍 🖥")
    st.subheader(
        """
        This is a place where you can get familiar with Graph Classifications   🧪
        """
    )

    #####
    # Sidebar
    st.sidebar.header("Hyperparameter Tuning")
    st.sidebar.markdown("---")

    conv = st.sidebar.selectbox('Convolution Type', ['GCNConv', 'GATConv'])
    st.sidebar.markdown("---")

    lr = st.sidebar.selectbox('Learning Rate', ['0.01', '0.001'])
    st.sidebar.markdown("---")

    hc = st.sidebar.selectbox('Hidden Channels', ['32', '64'])
    st.sidebar.markdown("---")

    epochs = st.sidebar.selectbox('Epochs', ['50', '100'])
    st.sidebar.markdown("---")

    batches = st.sidebar.selectbox('Batches', ['100', '200'])
    st.sidebar.markdown("---")

    n_layers = st.sidebar.selectbox('Layers', ['3'])
    st.sidebar.markdown("---")




    #####
    # Content
    st.header("00 - Use case")
    st.write("* Source: The dataset comes from MoleculeNet (Tox-21) with node and edge enrichment introduced by the Open Graph Benchmark.")
    st.write("* Description: The dataset used contains 7 831 molecules. Each molecule is converted into a graph by representing atoms by nodes and replacing the bonds by edges. These nodes and edges are further enriched with various features to avoid losing valuable information such as the name of the atom or the type of bond. In total, input node features are 9-dimensional and edge features 3-dimensional.")
    st.write("* Task: Predict whether a molecule is toxic or not.")


    st.header("01 - Dataset")


    st.markdown("#### Examples of molecule in the dataset:")
    # Gather some statistics about the first graph.
    col1, col2 = st.columns(2)

    with col2:
        images = Image.open('images/mol.png')
        st.image(images, width=300)

    with col1:
        st.write(" ")
        st.write("Number of nodes:",17)  
        st.write("Number of edges:",36)
        st.write("Average node degree:",2.12)
        st.write("Has isolatednodes:",False)
        st.write("Has self-loops:",False)
        st.write ("Is undirected:",True)



    st.header("02 - Model Performance")

    all_files = os.listdir("./datasets")    
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    #print(csv_files)


    



    filter_conv = [i for i in csv_files if conv in i]
    filter_learning_rates = [i for i in filter_conv if lr in i]
    filter_hidden_channels = [i for i in filter_learning_rates if hc in i]
    filter_epochs = [i for i in filter_hidden_channels if epochs in i]
    filter_batches = [i for i in filter_epochs if batches in i]
    filter_n_layers = [i for i in filter_batches if n_layers in i]
    final_dataset = filter_n_layers[0]
    print(final_dataset)

    print(os.getcwd())
    test = pd.read_csv("datasets/"+final_dataset)
    test = test.loc[:, ~test.columns.str.contains('^Unnamed')]
    means = pd.DataFrame(test.mean()).T.rename(columns={c:c+'_mean' for c in test.columns})
    min = pd.DataFrame(test[["loss"]].min()).T.rename(columns={c:c+'_min' for c in test.columns})
    max = pd.DataFrame(test[["train_score","test_score"]].max()).T.rename(columns={c:c+'_max' for c in test.columns})
    results_df = pd.concat([means,min, max], axis=1).reset_index(drop=True)    


    col1, col2 = st.columns(2)
    with col1:
        st.write("You have selected the following model configuration:")
        st.write(" ")
        st.write(" ")
        st.write("* Convolutions Type:",conv)
        st.write("* Learning Rate:",float(lr))
        st.write("* Hidden Channels:",int(hc))
        st.write("* Epochs:",int(epochs))
        st.write("* Batches:",int(batches))
        st.write("* N° of Layers:",int(n_layers))


    with col2:
        st.write("You have achieved following performance:")
        st.dataframe(results_df)


        st.line_chart(test[["loss"]])


    st.header("03 - About the model ")



    snippet = f"""

class GNN_3l(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, conv, conv_params=):
        super(GNN_3l, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = conv(
            input_size, hidden_channels, **conv_params)
        self.conv2 = conv(
            hidden_channels, hidden_channels, **conv_params)
        self.conv3 = conv(
            hidden_channels, hidden_channels, **conv_params)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch,  edge_col=None):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_col)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_col)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        batch = torch.zeros(data.x.shape[0],dtype=int) if batch is None else batch
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
    
        return x

    
    """
    code_header_placeholder = st.empty()
    snippet_placeholder = st.empty()
    code_header_placeholder.subheader(f"**Code for the GNN **")
    snippet_placeholder.code(snippet)
    
 



    st.header("04 - Hyperparameters Tuning Code ")


    st.header("05 -  References")



    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")
    st.markdown("     ")



    selected_indices = []
    master_review = "DEFAULT REVIEW - This is the season in which theatres revisit their histories. In the crumbling glory of Wilton’s Music Hall, east London, Fiona Shaw is reprising her wild version of The Waste Land, talking about death in the City, with the aid of Music Hall voices. Hackney Empire has burst into its traditional life with rousing panto. Meanwhile, the Orange Tree is producing The Lady or the Tiger, which had its premiere at the theatre in 1975 and was revived there in 1989. Now it’s back again; I wish it wasn’t. Based on a whimsical 1882 story by Frank Stockton, the show has words by Michael Richmond and Jeremy Paul and music by Nola York, who once sang with the Chantelles and was the first woman to write a complete score for a West End musical. It has a few good mots, a dash of sauce, but hardly any point It features one despotic ruler who follows his subjects’ every wiggle “from sperm to worm”, one reluctantly virgin daughter (“Think of your position”; “I am, I wish it was horizontal”), one drippy suitor and one multipurpose character who flips from role to role by changing his hat. Riona O’Connor has a suitably 70s Lulu-like shout of a voice but does too much gurgling to be really convincing as a grown-up: she sings better than she swings. As the naughty king - ooh what a scamp that tyrant is - Howard Samuels dispenses oeillades, pecks on the cheeks and pats on the knees to the ladies in the front row. Sam Walters’s production is almost eerily pleasant. It’s like a panto that doesn’t yell but quietly chortles."


   # def file_select(folder='./datasets'):
   #     filelist = os.listdir(folder)
   #     selectedfile = st.sidebar.selectbox('', filelist)
   #     return os.path.join(folder, selectedfile)







    index_review = 0

    st.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/gaetanbrison/reading-group-graph-neural-network) <small> graph classification 0.0.1 | November 2022</small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )






if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### ** 👩🏼‍💻👨🏼‍💻 Speakers : **")
st.image(['images/1.png','images/2.png'], width=150)

st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison/reading-group-graph-neural-network'}) 🚀 ")



def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made for ",
        link("https://engineeringteam.hi-paris.fr/", "Hi! PARIS Reading Groups"),
        " 👩🏼‍💻 👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()



