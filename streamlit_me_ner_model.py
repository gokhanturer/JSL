
import streamlit as st
import pandas as pd
import base64
import os

import sparknlp
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession

from sparknlp.annotator import *
from sparknlp.base import *

from sparknlp_display import NerVisualizer
from sparknlp.base import LightPipeline

spark = sparknlp.start(gpu = True) 


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.sidebar.image('https://nlp.johnsnowlabs.com/assets/images/logo.png', use_column_width=True)
st.sidebar.header('Choose the pretrained model')
select_model = st.sidebar.selectbox("select model name",["ner_model_glove_100d"])
if select_model == "ner_model_glove_100d":
    st.write("you choosed : ner_model_glove_100d ")

st.title("Spark NLP NER Model Playground")

#data
text1 = """Barack Hussein Obama is the 44th President of the United States of America. He is currently serving his second term as President. Barack Obama is the first African-American to be elected President of the United States. Obama is a member of the Democratic Party. He was first elected in 2008 after a close race with Hilary Clinton. He was re-elected for a second time in 2012. Obama previously served as U.S Senator from Illinois."""
text2 = """Spain has a rich tradition in art and has played a major role in the history of western painting. Spanish Golden Age was a period from the early 16th century to the late 17th century which saw the nation flourishing in arts and literature; and, among other things, becoming an important contributor to the European art world. The best known artists of this era are El Greco, Diego Velazquez, Zurbaran andMurillo. Francisco Goya , the most important Spanish artist of late 18th and early 19th centuries, is often considered as both the last of the Old Masters and the first of the moderns. Spanish artists have had an enormous impact on modern art with some of the most influential artists of the modern era including Joan Miro, Pablo Picasso and Salvador Dali all coming from Spain """
text3 = """Groups of painters also moved to the city from abroad and formed their schools and galleries in the city, increasing its artistic diversity. For instance, the Académie Vassilieff, which was a special school founded for Russian art students, was specially popular with Russian female painters in the early 20th century, along with the Académie de La Palette, specializing in Cubism .The Académie Julian was established in 1868. Painters such as Vincent van Gogh , Paul Cézanne , Henri Rousseau , Pablo Picasso , Henri Matisse , Jean Metzinger , Albert Gleizes , Robert Delaunay , Fernand Léger , Henri Le Fauconnier , Juan Gris , María Blanchard , Amedeo Modigliani , Amédée Ozenfant , André Dunoyer de Segonzac ,Roger de La Fresnaye and many others became associated with Paris. Following the Armory Show of 1913, New York City increasingly competed with Paris as a hub for artists, and its museums acquired some of the world's most valuable paintings."""
text4 = """Yesterday, Stephen returned from a trip to Washington, D.C., the capital of the United States. His visit took place during the week prior to the Fourth of July. Logically, there were many activities and celebrations in town in preparation for Independence Day. During his stay in the city, Stephen visited a lot of important historical sites and monuments, and he left with a deeper understanding of the political history of the United States.Stephen spent a lot of time outdoors exploring the important monuments surrounding Capitol Hill. Of course, he saw the White House from its outside gate at 1600 Pennsylvania Avenue. Stephen also visited the Washington Monument, the Jefferson Memorial, and the Lincoln Memorial. These statues and pavilions are dedicated to former U.S. presidents. They commemorate the contributions that these leaders made throughout American history. Washington, D.C. also has several war memorials dedicated to fallen soldiers during the major wars of the 20th century."""
text5 = """Adolf Hitler was born on 20 April 1889. He was born in a town in Austria-Hungary. His birthplace is modern day Austria. His father Alois Hitler and mother Klara Pölzl moved to Germany when Adolf Hitler was 3 years old. Since childhood he was a German Nationalist. He sang the German national anthem with his friends. After the death of his father he lived a bohemian lifestyle. He worked as a painter in 1905. Vienna’s Academy of Fine Arts rejected Adolf Hitler"""

sample_text = st.selectbox("",[text1, text2, text3,text4,text5])

@st.cache(hash_funcs={"_thread.RLock": lambda _: None},allow_output_mutation=True, suppress_st_warning=True)
def model_pipeline():
    documentAssembler = DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

    sentenceDetector = SentenceDetector()\
          .setInputCols(['document'])\
          .setOutputCol('sentence')

    tokenizer = Tokenizer()\
          .setInputCols(['sentence'])\
          .setOutputCol('token')

    gloveEmbeddings = WordEmbeddingsModel.pretrained()\
          .setInputCols(["document", "token"])\
          .setOutputCol("embeddings")

    nerModel = NerDLModel.load("/content/drive/MyDrive/SparkNLPTask/Ner_glove_100d_e8_b16_lr0.02")\
          .setInputCols(["sentence", "token", "embeddings"])\
          .setOutputCol("ner")

    nerConverter = NerConverter()\
          .setInputCols(["document", "token", "ner"])\
          .setOutputCol("ner_chunk")
 
    pipeline_dict = {
          "documentAssembler":documentAssembler,
          "sentenceDetector":sentenceDetector,
          "tokenizer":tokenizer,
          "gloveEmbeddings":gloveEmbeddings,
          "nerModel":nerModel,
          "nerConverter":nerConverter
    }
    return pipeline_dict

model_dict = model_pipeline()

def load_pipeline():
    nlp_pipeline = Pipeline(stages=[
                   model_dict["documentAssembler"],
                   model_dict["sentenceDetector"],
                   model_dict["tokenizer"],
                   model_dict["gloveEmbeddings"],
                   model_dict["nerModel"],
                   model_dict["nerConverter"]
                   ])

    empty_data = spark.createDataFrame([['']]).toDF("text")

    model = nlp_pipeline.fit(empty_data)

    return model


ner_model = load_pipeline()

def viz (annotated_text, chunk_col):
  raw_html = NerVisualizer().display(annotated_text, chunk_col, return_html=True)
  sti = raw_html.find('<style>')
  ste = raw_html.find('</style>')+8
  st.markdown(raw_html[sti:ste], unsafe_allow_html=True)
  st.write(HTML_WRAPPER.format(raw_html[ste:]), unsafe_allow_html=True)


def get_entities (ner_pipeline, text):
    
    light_model = LightPipeline(ner_pipeline)

    full_annotated_text = light_model.fullAnnotate(text)[0]

    st.write('')
    st.subheader('Entities')

    chunks=[]
    entities=[]
    
    for n in full_annotated_text["ner_chunk"]:

        chunks.append(n.result)
        entities.append(n.metadata['entity'])

    df = pd.DataFrame({"chunks":chunks, "entities":entities})

    viz (full_annotated_text, "ner_chunk")
    
    st.subheader("Dataframe")
    st.write('')

    st.table(df)
    
    return df


entities_df  = get_entities (ner_model, sample_text)


def show_html(annotated_text):

    st.header("Named Entities ({})".format(ner_model))
    st.sidebar.header("Named Entities")

    #st.write(annotated_text['ner'])
    label_set = list(set([i.split('-')[1] for i in annotated_text['ner'] if i!='O']))

    labels = st.sidebar.multiselect(
            "Entity labels", options=label_set, default=list(label_set)
        )
        
    html = get_onto_NER_html (annotated_text, labels) 
        # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    st.write('')
    st.write('')