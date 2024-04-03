import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

#adding audio

import speech_recognition as sr
st.title(" Large Language model Q&A Board 🌳")
st.write("In Ed-tech industry")
st.write(" You can give voice command input 🎙️ ️️✔️")
st.write(" You can add knowledge ✔️ ")
st.write(" You can retrieve past interactions ✔️ ")

btn = st.button("Create Knowledgebase")
def save_to_file(text, filename):
    with open(filename, 'a') as file:
        file.write(text+"\n")

def main():
    text_input = st.text_area("Enter your knowledge here:")

    # Button to save text to file
    if st.button("Save knowledge"):
        if text_input:
            save_to_file(text_input, "createKnowledge.txt")
            st.success("Knowledge created successfully!")
        else:
            st.warning("Please enter some text before creating knowledge.")

if __name__ == "__main__":
    main()


question = st.text_input("Question: ")


#add voice
with st.sidebar:
    st.header("Voice Command Input")
    record = st.checkbox("Record Voice Command")

if record:
    # Record audio
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now... 🎙️ ")
        audio = r.listen(source)

    # Convert audio to text
    try:
        voice_command = r.recognize_google(audio)
        st.write("Voice Command:", voice_command)
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        # Process the voice command
    if voice_command:
        # we may need to preprocess or clean the voice command before using it in your Q&A system
        # Example: Removing unnecessary words, correct mistakes, etc.

        chain = get_qa_chain()
        response = chain(voice_command)

        st.header("Answer")
        st.write(response["result"])


#--
# if btn:
#     create_vector_db()
if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])


chain = get_qa_chain()
response = chain(question)
context=response["result"]
# Initializing LLM here

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.0,model_name='models/text-bison-001')
memory = ConversationBufferMemory(llm=llm,ai_prefix="LLM",human_prefix="You")

memory.save_context({"input":"why row and value option is not showing for the visual in PowerBI , any setting need to be change, please let me know?"},{"output":"You have selected Table Visual instead of Matrix. That is why you are seeing a different interface."})

template = """Given the context and a question, generate an answer based on the context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without 
    making much changes.
    If the answer is not found in the response, kindly state "I don't know ,kindly add this query in the knowledgebase..
    "Don't try to make up an answer".
    If any task is given to remember than store it in the history or database for future conversations.retrieve the task from {history} when it is been asked.Like Remember my name is vrindavan.

conversation history:
{history}
You: {input} 
\n LLM:"""

#session state variable
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['You']},{'output':context})
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = LLMChain(llm=llm, memory=memory,verbose=True,prompt=PROMPT)
#conversation.predict(input=question)
if question:
    response=conversation(question)
    message={'You':question,'LLM':context }
    st.session_state.chat_history.append(message)
    #st.write(response)
    st.header("Past Interactions")
    with st.expander(label='Your last queries',expanded=False):
        st.write(st.session_state.chat_history)

# #unanswered_questions list here
if 'not_ans' not in st.session_state:
    st.session_state.not_ans=[]

else:
    for message in st.session_state.not_ans:
        memory.save_context({'input':message['Question']},{'output':context})

if question:
    response=conversation(question)
    message={'Question':question }

if 'response' not in context:
    st.session_state.not_ans.append(message)

    # st.error("I don't know, kindly add this query to the knowledgebase.")
st.header("Unanswered Questions")
with st.expander(label='You can answer,if you can.. ', expanded=False):
    st.write(st.session_state.not_ans)

with st.sidebar:
    with st.expander(label='You can answer,if you can.. ', expanded=False):

        st.header("Unanswered Questions")
        st.write(st.session_state.not_ans)