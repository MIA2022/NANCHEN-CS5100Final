import streamlit as st
from utils import generate_summary

# Initialize session state variables
if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = ""

st.title("Dialogue Text Summarization")

st.write("---") 

height = 200

# Text area with session state
input_text = st.text_area("Dialogue", height=height, key="input_text")

# Submit button logic
if st.button("Submit"):
    if st.session_state.input_text.strip() == "":
        st.error("Please enter a dialogue!")
    else:
        st.write("---")
        st.write("## Summary")
        st_container = st.empty()
        st_info_container = st.empty()
        # Generate summary and store it in session state
        st.session_state.generated_summary = generate_summary(
            " ".join(st.session_state.input_text.split()), 
            st_container, 
            st_info_container
        )

# Display the generated summary
if st.session_state.generated_summary:
    st.write(st.session_state.generated_summary)

# Clear button logic
def clear_all():
    st.session_state.clicked = True
    st.session_state.input_text = ""  # Clear input text
    st.session_state.generated_summary = ""  # Clear summary

st.button("Clear", on_click=clear_all)

# Logic for clearing display
if st.session_state.clicked:
    st.session_state.clicked = False
    st.experimental_rerun()
