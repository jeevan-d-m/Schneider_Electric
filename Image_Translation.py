import time
import google.generativeai as genai
import PIL.Image as pil
import streamlit as st
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Image Translation",
    layout="wide"
)


def greet_user():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good Morning!"
    elif 12 <= current_hour < 16:
        greeting = "Good Afternoon!"
    elif 16 <= current_hour < 21:
        greeting = "Good Evening!"
    else:
        greeting = "Good Night!"
    return greeting


st.header("" + greet_user(), anchor=False, divider="violet")
with st.sidebar:
    st.image("./Schneider-Electric-logo-.png", use_container_width=True)
    st.write("")
    st.write("")
    st.markdown(":blue[About:]")
    st.write("This app allows users to upload images of **<span style='color:green;'>Schneider Electric</span>** "
             "devices for quick clarification on product-related issues. Please note that results are automated and "
             "should be verified with official support for accuracy", unsafe_allow_html=True)
    st.write("")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])


def main():
    message_container = st.container(height=500, border=False)
    if uploaded_file is not None:
        message_container.image(pil.open(uploaded_file), use_container_width=False)
    if prompt := st.chat_input("Enter a prompt ", key="input-1"):
        message_container.subheader(":orange[User Prompt]")
        message_container.write(prompt)
        message_container.write("")
        start_time = time.time()
        try:
            if uploaded_file is not None:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                image = pil.open(uploaded_file)

                with st.spinner("Generating response.."):
                    response = model.generate_content([prompt, image])

            else:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")

                with st.spinner("Generating response.."):
                    response = model.generate_content(prompt)

            st.sidebar.write("")
            message_container.subheader(":blue[Response]")

            # Check if the response contains a valid text part
            if hasattr(response, 'text'):
                message_container.write(response.text)
            else:
                message_container.write("No valid response text. Please check the prompt or try again.")

            st.sidebar.markdown(":green[Response Time: ]" + " {:.2f}".format(time.time() - start_time) + "s")

        except ValueError as e:
            message_container.write(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
