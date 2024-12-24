import streamlit as st
import re
import cohere
from PyPDF2 import PdfReader

# Initialize Cohere client
API_KEY = "Qh7wu89RKFszz5PiDKpmJTLziuHeR49YM2L8nv2q"
co = cohere.Client(API_KEY)

# Streamlit UI setup
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)
st.title("📄 PDF Chatbot")
st.subheader("Summarize, Highlight Keywords, and Ask Questions")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to generate summary using Cohere LLM
def generate_summary_with_keywords(text):
    if len(text) < 250:
        return "Text is too short to summarize. Please upload a larger PDF."

    response = co.summarize(
        text=text,
        length="medium",
        format="bullets",
        model="summarize-xlarge"
    )
    summary = response.summary
    # Highlight keywords (words with frequency > 3)
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = {word for word in set(words) if words.count(word) > 3}
    for keyword in keywords:
        summary = re.sub(
            rf"\b{keyword}\b", f"**{keyword.capitalize()}**", summary, flags=re.IGNORECASE
        )
    return summary


# Function to answer questions using Cohere LLM
def answer_question_with_cohere(question, context):
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        max_tokens=100,
        temperature=0.5
    )
    return response.generations[0].text.strip()


# File uploader
uploaded_file = st.file_uploader("📂 Upload your PDF file", type="pdf")

if uploaded_file:
    st.write("📄 Processing PDF...")
    text = extract_text_from_pdf(uploaded_file)

    # Check if text is long enough
    if len(text) < 250:
        st.warning("The extracted text is too short for summarization. Please upload a larger PDF.")
    else:
        # Display Summary with highlighted keywords
        st.subheader("📑 Summary with Highlighted Keywords")
        summary = generate_summary_with_keywords(text)
        st.markdown(summary, unsafe_allow_html=True)

        # Question-Answering
        st.subheader("🤔 Ask a Question")
        question = st.text_input("🔍 Type your question here")

        if question:
            st.write("💡 Generating answer...")
            answer = answer_question_with_cohere(question, text)
            st.write("### Answer:")
            st.write(answer)

else:
    st.info("🙇‍ Please upload a PDF file to start.")

# Footer for App
st.markdown("---")
st.markdown("Built with ❤️ by **Janani** ")
