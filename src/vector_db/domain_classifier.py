from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


load_dotenv()


prompt = ChatPromptTemplate.from_template("""
You are an expert researcher.
Given this research paper abstract, classify the paper into exactly one of:
- NLP (Natural Language Processing)
- CV (Computer Vision)
- ML (Machine Learning other than CV/NLP)
- DL (Deep Learning generic)
- MM (Multimodal (CV and NLP) ) 

Respond with only the category name.
                                          
Abstract:
{abstract}
""")


def init_model(model="llama-3.3-70b-versatile"):

    llm=ChatGroq(model=model,temperature=0.0)

    return llm

def classify_paper(abstract):

    as_list=abstract.split(" ")
    num_of_words=len(as_list)

    if num_of_words>150:
        abstract=" ".join(as_list[:150])


    
    llm=init_model()

    prompt_value=prompt.invoke({"abstract":abstract})

    response=llm.invoke(prompt_value)
    print(response.content.strip())



    return response.content.strip()









    





