import streamlit as st
from PIL import Image
from config import model
import base64
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# LangChain Prompt for Document Classification
classification_prompt = ChatPromptTemplate.from_template("""
Classify the provided text into exactly one of these categories:
- Legal Document
- Real Estate Document
- Invoice or Receipt
- Product Image
- Unknown: If the provided text is none of the above categories

Extracted Text:
{text}

Category:
""")

# Streamlit App UI
st.title("📄 MultiDoc AI: Image Recognition & Classification")
st.write("Upload an image to start processing with Gemini.")

uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Process and Classify Image with Gemini"):
        with st.spinner("Gemini is analyzing the image..."):
            # Convert image to base64
            uploaded_image.seek(0)
            image_bytes = uploaded_image.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = uploaded_image.type
            data_url = f"data:{mime_type};base64,{image_base64}"

            # Gemini Vision Message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Provide whatever you can read and extract from the image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )

            # Gemini Vision Response
            vision_response = model.invoke([message])
            extracted_text = vision_response.content

            # Display extracted text
            st.subheader("📝 Extracted Text:")
            st.write(extracted_text)

            # Classification Step
            classification_chain = classification_prompt | model
            classification_response = classification_chain.invoke({"text": extracted_text})
            doc_category = classification_response.content.strip()

            # Display classification
            st.subheader("✅ Document Classification:")
            st.success(doc_category)
