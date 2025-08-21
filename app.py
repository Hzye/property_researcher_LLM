import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

st.title("Australian Property Researcher")

# # process file
# uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# init chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

    
# init objects
if "client" not in st.session_state:
    st.session_state["client"] = QdrantClient(path="dense")
if "qdrant" not in st.session_state:
    st.session_state["qdrant"] = QdrantVectorStore(
        client=st.session_state["client"],
        collection_name="dense_texts",
        embedding=OllamaEmbeddings(model="bge-m3:latest"),
        retrieval_mode=RetrievalMode.DENSE
    )
if "llm" not in st.session_state:
    st.session_state["llm"] = OllamaLLM(model="qwen2.5:7b-instruct")

# show history every refresh
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# react to user input
if question := st.chat_input("Ask something about the Australian property market"):
    # display user message in container
    with st.chat_message("user"):
        st.markdown(question)
    # add user message to history
    st.session_state["messages"].append({"role": "user", "content": question})

    ## get llm response
    # get more relevant docs
    retriever = st.session_state["qdrant"].as_retriever(search_type="mmr", search_kwargs={"k": 5})
    relevant = retriever.invoke(question)

    # combine content and source
    contents = [x.page_content for x in relevant]
    sources = [x.metadata["source"] for x in relevant]

    context = "\n\n---\n\n".join([c+" ##from video## "+s for c, s in zip(contents, sources)])

    # build prompt
    template = """
        #Background#
        You are an efficient AI assistant for researching the Australian Property Market.
        You are responsible for anwering questions regarding the Australian Property Market.
        The following context is relevant to the question: {context}
        #Objective#
        Using only the relevant context, answer the following question succinctly and accurately: {question}
        #Output#
        Only answer in English.
    """

    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = st.session_state["llm"].invoke(prompt)

    # display
    with st.chat_message("assistant"):
        st.markdown(response)
    # add to history
    st.session_state["messages"].append({"role": "assistant", "content": response})