import streamlit as st
from project.crews.assistant_crew.assistant_crew import AssistantCrew
import sys
import os
import json
import re

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'assistant_crew' not in st.session_state:
        st.session_state.assistant_crew = AssistantCrew()

def extract_raw_response(response):
    try:
        
        if hasattr(response, 'raw'):
            return response.raw
            
        
        if isinstance(response, str):
            
            match = re.search(r'"raw":"(.*?)(?:","|\})', response)
            if match:
                return match.group(1).strip()
            
            
            response = re.sub(r'"pydantic":.*?(?=,"|}})', '', response)
            response = re.sub(r'"json_dict":.*?(?=,"|}})', '', response)
            response = re.sub(r'"tasks_output":.*?(?=,"|}})', '', response)
            response = re.sub(r'"token_usage":.*?(?=}|$)', '', response)
            
            
            match = re.search(r'"raw":"(.*?)"', response)
            if match:
                return match.group(1).strip()
            
            
            try:
                data = json.loads(response)
                if isinstance(data, dict):
                    return data.get('raw', response)
            except:
                pass
                
        return str(response)
    except Exception:
        return str(response)

def main():
    st.set_page_config(
        page_title="AI Personal Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    
    with st.sidebar:
        st.title("Options")
        if st.button("🗑️ Clear Chat History", type="primary"):
            st.session_state.messages = []
            st.rerun()
    
    initialize_session_state()
    
    st.title("🤖 AI Personal Assistant")
    st.markdown("""
    Welcome! I'm your AI assistant. I can help you with various tasks and remember our conversations.
    """)
    
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    
    if prompt := st.chat_input("You:"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.assistant_crew.crew().kickoff(
                        inputs={
                            "question": prompt
                        }
                    )
                    
                    clean_response = extract_raw_response(response)
                    if isinstance(clean_response, str):
                        clean_response = clean_response.replace('\\n', '\n').replace('\\"', '"')
                    st.write(clean_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

