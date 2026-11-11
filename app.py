import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

search = DuckDuckGoSearchResults(name="search")

tools = [wiki, arxiv, search]
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter the GROQ API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if user_prompt:= st.chat_input(placeholder="Ask your query.."):
    st.session_state.messages.append({"role":"user", 'content':user_prompt})
    st.chat_message("user").write(user_prompt)

    if not api_key:
        st.warning("Please enter the Api Key in the sidebar to proceed further..!")

    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with search capabilities."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=api_key)

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            history_for_agent = []
            if msg["role"] == "user":
                history_for_agent.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_for_agent.append(AIMessage(content=msg["content"]))
            
            response = agent_executor.invoke({"input": user_prompt, "chat_history": history_for_agent}, 
                                             config = {"callbacks" : [st_cb]})
            st.session_state.messages.append({"role":"user", "content":response["output"]})
            st.write(response["output"])