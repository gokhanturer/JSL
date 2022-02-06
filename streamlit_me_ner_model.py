
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
select_model = st.sidebar.selectbox("",["ner_model_glove_100d"])

st.title("Spark NLP NER Model Playground")

#data
text1 = """On february 6th television viewers from 157 countries will tune in to the final of the Africa Cup of Nations, a biennial tournament featuring some of the world’s best footballers. In the first semi-final, Senegal beat Burkina Faso 3-1 in an entertaining match, marked by quick counter-attacks, occasionally sloppy passing and some excellent refereeing. The second semi-final was a much more tense affair with no goals in regular or extra time. Egypt eventually overcame Cameroon, the hosts, in a penalty shoot-out."""
text2 = """The year is 2035. You are walking along the Bund, Shanghai’s storied waterfront, with two of your old classmates, pointing out how much has changed since the last time you were here 20 years ago. You almost joke about how the only thing that never changes is Xi Jinping being in power, but you think better of it. Someone might be listening. A message flashes in your glasses, and you say hurried goodbyes. The landscape of the Bund dissolves into the fantasy realm of a multiplayer game, where three friends in magical armour are waiting, swords at the ready."""
text3 = """For many Haitians it felt wearily familiar. On January 24th a large earthquake hit the south-west part of the country, the second in the area in less than six months. The victims would of course need help, and the dysfunctional government of the western hemisphere’s poorest country was unlikely to provide much. But the prospect of yet more foreign aid workers descending on the place once dubbed the “Republic of ngos” did not inspire much enthusiasm either. They are “like vultures”, complains Monique Clesca, a journalist and activist: they live off disasters, but do little to improve things. It is a common view."""
text4 = """The acronym stuck for a decade, no matter how bitterly the countries it lumped together moaned about it. Being branded one of the pigs—short for Portugal, Italy, Greece and Spain—as the euro teetered was to be the perennial butt of bond-market bullying, Eurocrat nagging and German tabloid contempt. But look today and the bloc’s Mediterranean fringe is doing rather well. Those once stuck in the muck in the aftermath of the global financial crisis are now flying high. Southern Europeans are running their countries with the competence and reformist zeal all too often lacking in their northern neighbours. It may be a flash in the pan. But if it endures, it will come to change the nature of the eu."""
text5 = """With fanfare and fireworks, the Beijing 2022 Winter Olympic Games will begin on February 4th. But many officials will be missing from the opening ceremony. Countries including Britain, America and Australia are instigating a diplomatic boycott of the Games. They are wasting an opportunity to engage with their Chinese counterparts on the issues that matter to them."""

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