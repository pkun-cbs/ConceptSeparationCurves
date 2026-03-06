# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 09:03:00 2026

@author: PKUN
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from io import StringIO
# library for reporting progress on loading large files
from stqdm import stqdm # https://github.com/Wirg/stqdm


from ConceptSepCurves import random_article_insertion, cosine_sym, density_plot_data, surface_normalize, density_overlap, compute_density_function


@st.cache_resource
def get_model(model_name:str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device="cpu")
    return model


st.set_page_config(
    page_title="Concept Separation Curves",
    page_icon="🔀",
    )
st.write("# Concept Separation Curves")

st.markdown("""
            This is a demo for our research on Concept Separation Curves.

            Sentence embedding techniques aim to encode key concepts of a sentence’s meaning in a vector space.
            However, the majority of evaluation approaches for sentence embedding quality rely on the use of additional classifiers or downstream tasks.
            These additional components make it unclear whether good results stem from the embedding itself or from the classifier's behaviour. 
            In this paper, we propose a novel method for evaluating the effectiveness of sentence embedding methods in capturing sentence-level concepts. 
            Our approach is classifier-independent, allowing for an objective assessment of the model's performance. 
            The approach adopted in this study involves the systematic introduction of syntactic noise and semantic negations into sentences, with the subsequent quantification of their relative effects on the resulting embeddings. 
            The visualisation of these effects is facilitated by Concept Separation Curves, which show the model's capacity to differentiate between conceptual and surface-level variations. 
            By leveraging data from multiple domains, employing both Dutch and English languages, and examining sentence lengths, this study offers a compelling demonstration that Concept Separation Curves provide an interpretable, reproducible, and cross-model approach for evaluating the conceptual stability of sentence embeddings.
            
            In this demo you can use any huggingface model.
            The ones we use in the paper are the following:
            - sentence-transformers/LaBSE
            - sentence-transformers/all-mpnet-base-v2
            - sentence-transformers/all-distilroberta-v1

            
            """)
st.warning("Changing the model will take time. The larger a model the more time needed.")
llm_model = get_model(st.text_input("Input a huggingface sentence transformer", value='sentence-transformers/LaBSE'))

st.markdown(
    """
    ## Algorithm settings
    The section below shows the settings for the algorithm.
    First you can input an example sentence on which the example Fuzzing and Negation are performed.
    The example sentence is only used to show the Fuzzing and Negation.

""")

example_text = st.text_input("Input an example", "Type something here...")

random_options = st.number_input("How many sentences should be generated?", min_value=1, max_value=100, value=3)

positives, negatives = st.columns(2)

positive_df = None
negative_df = None

embedded_text = llm_model.encode(example_text)
compare = lambda s: cosine_sym(llm_model.encode(s), embedded_text)

with positives:
    alteration_1 = st.text_input("Fuzzing terms", "the | a")
    pos_terms = tuple(map(str.strip, alteration_1.split("|")))
    
    positive_gen = lambda sentence: random_article_insertion(sentence.split(), 
                                                             pos_terms,
                                                             variants=random_options)
    alt_texts = pd.Series(list(positive_gen(example_text)),name='text')
    
    positive_df = pd.DataFrame(dict(text=alt_texts, cosine=alt_texts.apply(compare)))
    st.dataframe(positive_df, hide_index=True)

with negatives:
    alteration_2 = st.text_input("Negation terms", "not")
    neg_terms = tuple(map(str.strip, alteration_2.split("|")))
    negative_gen = lambda sentence: random_article_insertion(sentence.split(), 
                                                             neg_terms,
                                                             variants=random_options)
    alt_texts = pd.Series(list(negative_gen(example_text)),name='text')
    negative_df = pd.DataFrame(dict(text=alt_texts, cosine=alt_texts.apply(compare))) 
    st.dataframe(negative_df, hide_index=True) 
    
st.markdown(
    """
    # Complete sets & Curves
    
    The above example shows the code on small scale and gives an idea of the method.
    Now lets use the method on larger data.
    For this implementation we allow data to be dragged and dropped.
    Although limited in size by the hosting server, the computation can still take a significant amount of time.
    
    This process will take more time to compute (up to several hours for huge sets), as such it requires a button press to ensure it is only executed when called upon.
    The testing parameters given in the above functions are automatically used on the given set.
    
    The data input is expected to be code in a plaintext file.
    Each row is 1 sentence, without headers present.
    """
    )


uploaded_file = st.file_uploader(
    "Upload data", accept_multiple_files=False, type="txt"
)
if uploaded_file is None:
    st.text("Awaiting input...")
    st.stop()


st.markdown("""
            ## Resulting curves
            The curves are intended to show the response of an LLM on changing the input.
            This automated input alteration is not guaranteed to result in a valid text.
            However, the goal is to visualise whether the impact of one alteration exceeds the other.
            If the curves overlap, it is an indicator that, to the model, there is no difference between the alteration.
            
            For the original research, we used terms for which a complete overlap should not be the case.
            As terms like "the | a" are rarely as impactfull as "not | no" on the meaning of a sentence.
            However, one could also switch to inserting nouns and thus altering the sentence further.
            
            In the method we use a kernel width for the gaussian curve.
            This is set to 0.2 as this gave the best optical fit.
            If the margins are closer (like on larger sentences), and you whish to only view subsections in the final curve plot, a smaller value might be in order.
            """)

kernel_width = st.number_input("Please select a kernel width (default=0.2):", value=0.2)
resolution = st.number_input("Number of steps in the density function", min_value=10, value=1000)


value_range = st.select_slider("Range of values considered", 
                                options=[np.round(v,2) 
                                            for v in np.arange(-1,1.01,step=0.01)], 
                                value=(-1.0,1.0))

compute = st.button("Start computation", help="The computation could take a significant amount of time, as such it is only executed on request.")
if compute:
    positive_sim = []
    negative_sim = []
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    texts = stringio.read().split('\n')
    for line in map(str.strip, stqdm(texts, desc="Computing function per line of text:", total=len(texts))):
        embedded_text = llm_model.encode(line)
        compare = lambda s: cosine_sym(llm_model.encode(s), embedded_text)
        positive_sim.extend(compare(gen) for gen in positive_gen(line))
        negative_sim.extend(compare(gen) for gen in negative_gen(line))
                                
    positive_sim = pd.Series(positive_sim, name="Alteration_1")
    negative_sim = pd.Series(negative_sim, name="Alteration_2")

    st.success(f"Computed cosine for {len(positive_sim)} positives and {len(negative_sim)} negatives.")

    # first create the 'raw' density per dataset
    fuzzed_density = compute_density_function(positive_sim, kernel_width=kernel_width) 
    negative_density = compute_density_function(negative_sim, kernel_width=kernel_width)
            
    # now normalise to have the sum of the densities equal 1
    fuzzed_density_normalized = lambda values: surface_normalize(fuzzed_density(values))
    negative_density_normalized = lambda values: surface_normalize(negative_density(values))
            
    # we are interested in both the actual overlap and the plot.
    overlap =np.round(density_overlap(fuzzed_density_normalized, 
                                    negative_density_normalized,
                                    resolution=resolution), 5)
    st.markdown("For the overlap scores as presented in the paper, the full range of [-1, 1] is required. If this range is adjusted, the overlap wil be different.")
    st.success(f"Overall overlap in given window: {overlap}")

    df = density_plot_data(
        densities={
                'Alteration 1': fuzzed_density_normalized, 
                'Alteration 2': negative_density_normalized}, 
        start_range=min(value_range),
        end_range=max(value_range),resolution=resolution)


    st.pyplot( df.plot.line().figure)
            
    st.markdown("Just to re-iterate,"
                f" the Fuzzing are texts with ''{alteration_1}''."
                f" Negation texts have ''{alteration_2}'' inserted.")
