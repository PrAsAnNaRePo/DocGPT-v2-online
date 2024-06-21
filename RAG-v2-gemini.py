import os
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PyPDF2 import PdfReader
import re
import json
from dotenv import load_dotenv

load_dotenv()

system_prompt = """You are a document assistant. You can read and understand the document uploaded by user to answer the question asked by user.
Read and understand the content/context fully inorder to make the correct answer. Don't make any answer of your own. Suppose if you don't have a content or the content is empty, then convey to user that you can't answer that question because its hard to find the detail in the document nicely.
Make your **response more {response_format}**"""

ADDN_QUERY_PROMPT = """You are follow-up query generated. The user will share a query that they are searching in a document (like pdf, docs etc.). Your job is to generate 2 or 3 more queries with same context. Something like follow-up queries (includes what could user will ask after this query relatively) and related queries (related to the query in different tone).
Use following JSON schema for response:
{
    "follow_up_query": [
        "follow_up_query_1",
        "follow_up_query_2",
    ],
    "related_queries": [
        "related_query_1",
        "related_query_2",
    ]
}

NOTE:
- If your are not sure about the context of the user query like you don't have any idea about the query, then just use user query as follow-up and related queries, DON'T use any random stuff before you knowing about the context fully.
- Make sure your response is parsable using json.loads in python."""

class RAG_v2_gemini:
    def __init__(self, system_prompt: str, chunk_size: int, temperature: int, verbose) -> None:
        self.system_prompt = system_prompt
        self.chunk_size = chunk_size
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.core_gemini = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
            system_instruction=self.system_prompt,
        )
        self.core_chat = self.core_gemini.start_chat()
        self.utils_gemini = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        self.vector_store = None
        self.verbose = verbose

    def load_vectorestore(self, context):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
        text = text_splitter.split_text(context)
        self.vector_store = FAISS.from_texts(text, self.embeddings)

    def get_answer(self, question):

        start_time = time.time()
        logs = {
            "user_prompt": question
        }

        # 1. Follow-up & related query generation.
        
        res = self.utils_gemini.generate_content(f"{ADDN_QUERY_PROMPT}\n\nHere is the query: {question}").candidates[0].content.parts[0].text
        pattern = re.compile(r'```json\s*(\{.*\})\s*```', re.DOTALL)
        match = pattern.search(res)
        res = match.group(1)
        additional_queries = json.loads(res)
        follow_ups = additional_queries['follow_up_query']
        related_queries = additional_queries['related_queries']
        query_list = follow_ups + related_queries
        query_list.append(question)
        logs['query_list'] = query_list
        
        # 2. Convert Ques queries to key words.

        key_words = []
        for query in query_list:
            res = self.utils_gemini.generate_content(f"""Given a question by user, you have to convert it into nice one simple form or like keyword. Just convert the question into simpler form that looks easier. Don't include any other text, just the query as response.\nFor example:\nWhat all are the related works they considered? -> Related works.\n\nNow here is the query: {query}""")
            key_words.append(
                res.candidates[0].content.parts[0].text
            )
        logs['keywords'] = key_words
        
        # 3. Remove duplicates

        ref_key_words = []
        res = self.utils_gemini.generate_content(f"""The following LIST of key-words may contains some duplicates in terms of spelling, but all the key-words are all in same category or in the same field. If is contains any duplicates you have to remove and return the correct list. If the list is already doesn't have any duplicates, then just return the list. Make sure all the key-words are comma seperated in your response.\nHere is the list: {key_words}""")
        refined_key_words = res.candidates[0].content.parts[0].text
        for i in refined_key_words.split(','):
            ref_key_words.append(i.strip())
        print(ref_key_words)
        st.write(ref_key_words)
        logs['refined_keywords'] = ref_key_words

        # 4. get relevent contents

        relevent_contents = []
        for ref_key in ref_key_words:
            related_docs = str(self.vector_store.similarity_search(ref_key))
            relevent_contents.append(related_docs)
        logs['relevent_contents'] = relevent_contents

        # 5. Remove the unwanted contents
        
        ref_relevent_content = []
        for i in range(len(relevent_contents)):
            res = self.utils_gemini.generate_content(f"CONTENT:\n{relevent_contents[i]}\n\nUnderstand the above content fully. QUESTION: {question}\n\nKEY_WORD: {ref_key_words[i]}.\nCheck if we can able to answer the question by reading the content or Does the content have some context to answer the question.\nAlso check the KEY_WORD is present in the content. Response with only yes if all conditions satisfied or just no.")
            is_relevent = res.candidates[0].content.parts[0].text.strip()
            if is_relevent == 'yes':
                ref_relevent_content.append(relevent_contents[i])
            else:
                print(f"Rejected content {i}")

        logs['refined_relevent_content'] = ref_relevent_content
        print(f"Considering only {len(ref_relevent_content)} contents.")
        res = self.core_chat.send_message(f"Here is the content: {str(ref_relevent_content)}\n\nQuestion: {question}")
        # res = self.core_gemini.generate_content(f"Here is the content: {str(ref_relevent_content)}\n\nQuestion: {question}")
        print(res.candidates[0].content.parts[0].text)
        response = res.candidates[0].content.parts[0].text.strip()
        end_time = time.time()
        time_took = end_time - start_time
        print(">>>>>> ", end_time - start_time)
        logs['final_answer'] = response
        logs['time_took'] = end_time - start_time
        st.info(f"Time took: {time_took:2f}s")
        return response

def main():
    st.title("Rag-V2")
    
    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "history" not in st.session_state:
        st.session_state.history = []
    
    for history in st.session_state.history:
        with st.chat_message(history["role"]):
            st.markdown(history["text"])
    
    with st.sidebar:
        st.title("Set the model Conf.")
        temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.75, step=0.05, help="Helps to generate content creative every time when the temperature is high.")
        chunk_size = st.slider(label="Chunk size", min_value=300, max_value=1000, value=700, step=50, help="Set the chars size in a chunk that'll be extracted from the uploaded docs and provide to LLM to generate answer. Larger the size more the detail but slower the generation speed.")
        verbose = st.checkbox("verbose", help="If verbose enabled, the logging will be displayed in the UI.")
        response_format = st.selectbox("Response format", options=["consice", "detailed"])
        
        _set = st.button("Set")
        
        if _set:
            st.session_state.agent = RAG_v2_gemini(
                system_prompt=system_prompt.format(response_format=response_format),
                chunk_size=chunk_size,
                temperature=temperature,
                verbose=verbose
            )
            st.success("Rag ready!!")
        
        st.divider()

        files = st.file_uploader("Upload your files", accept_multiple_files=True, type=["pdf", "txt"])
        process = st.button("Process")
        if process and files:
            if st.session_state.agent is not None:
                with st.spinner('loading your file. This may take a while...'):
                    total_content = ''
                    num_pages = 0
                    for file in files:
                        file_type = file.type
                        if file_type == "application/pdf":
                            pdf_reader = PdfReader(file)
                            content = ''
                            for page in pdf_reader.pages:
                                num_pages += 1
                                content += page.extract_text()

                        if file_type == "text/plain":
                            content = file.read()
                            content = content.decode("utf-8")

                        total_content += content

                    st.session_state.agent.load_vectorestore(total_content)
                st.success("Documents loaded.")
            else:
                st.info("Set the model Conf. first...")

    if prompt := st.chat_input("Enter your message..."):
        if st.session_state.agent is not None:
            st.session_state.history.append({"role": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response = st.session_state.agent.get_answer(prompt)
                message_placeholder.markdown(response)
            st.session_state.history.append({"role": "assistant", "text": response})
        else:
            st.info("Set the model Conf. first.")

if __name__ == '__main__':
    main()