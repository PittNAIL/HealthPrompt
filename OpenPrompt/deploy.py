# !pip install -q openprompt==0.1.1 \
# 'torch>=1.9.0' \
# 'transformers>=4.10.0' \
# sentencepiece==0.1.96 \
# 'scikit-learn>=0.24.2' \
# 'tqdm>=4.62.2' \
# tensorboardX \
# nltk \
# yacs \
# dill \
# datasets \
# rouge==1.0.0 \
# scipy==1.4.1 \
# fugashi \
# ipadic \
# unidic-lite \
# streamlit


import requests
import streamlit as st

import openprompt.plms as plms
from openprompt.plms.mlm import MLMTokenizerWrapper
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from openprompt.data_utils import InputExample
from transformers import AutoTokenizer, AutoModel
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import PtuningTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch


def prompt_classification(text):
    classes = [ 
        "lung",
        "brain",
        "virus"
    ]


    dataset = [ 
          InputExample(
    #         guid = 0,
            text_a =text
#               "Asthma affects lungs  and can be hard to diagnose. The signs of asthma can seem like the signs of COPD, pneumonia, bronchitis, pulmonary embolism, anxiety, and heart disease.", #lung
        ),
    #     InputExample(
    #         guid = 1,
    #         text_a = "COVID-19 is caused by a coronavirus called SARS-CoV-2", #virus
    #     ),
    #     InputExample(
    #         guid = 2,
    #         text_a = "When your brain is damaged, it can affect many different things, including your memory, your sensation, and even your personality. Brain disorders include any conditions or disabilities that affect your brain.", #brain
    #     ),
    #     InputExample(
    #         guid = 3,
    #         text_a = "Symptoms may appear 2-14 days after exposure to the virus", #virus
    #     ),
#             InputExample(
#     #         guid = 4,
#             text_a = """Neurodegenerative diseases cause your brain and nerves to deteriorate over time. They can change your personality and cause confusion. They can also destroy your brain’s tissue and nerves.

#     Some brain diseases, such as Alzheimer’s disease, may develop as you age. """, #brain
#         ),
    ]


#     plm, tokenizer, model_config, WrapperClass = load_plm_fn()


    # template_text = '{"placeholder":"text_a"}: This effects {"mask"}'
    template_text= 'A {"mask"} disorder :  {"placeholder": "text_a"}'

    promptTemplate = ManualTemplate(
        text = template_text,
        tokenizer = tokenizer,
    )


    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "lung": ["chest"],
            "brain": ["head"],
            "virus": ["virus"],
        },
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer
    )


    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer, 
        template = promptTemplate, 
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256, decoder_max_length=3, 
        batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head"
    )

    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
    #         print(logits)
            preds = torch.argmax(logits, dim = -1)
            print(classes[preds])
            
    return classes[preds]


if __name__ == "__main__":
    
    st.title("HealthPrompt: Classifying clinical texts")
    st.write("Upload Clinical text, Classify it..")

    @st.cache(allow_output_mutation=True)
    def load_plm_fn():
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-uncased")
        return plm, tokenizer, model_config, WrapperClass
    
    with st.spinner("Loading PLM into memory..."):
        plm, tokenizer, model_config, WrapperClass = load_plm_fn()
        
    text = st.text_input('Enter the clinical text here: ')
        
    if text:
        st.write("Response: ")
        with st.spinner("Searching for classes.."):
            res=prompt_classification(text)
            st.write("Class : {} - disease".format(res))
        st.write("")
            
    


            
