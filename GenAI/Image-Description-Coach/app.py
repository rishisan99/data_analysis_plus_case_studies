import streamlit as st
import openai
import os
import base64

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Gen AI Product",
    layout="centered"
)

# --- 2. SIDEBAR (API KEY) ---
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()
    

# --- 3. Helper Functions ---
  
def generate_image(prompt, api_key):
    """
    Sends a text prompt to DALL-E 2 (Cheaper) and returns the image URL.
    """

    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )

        return base64.b64decode(response.data[0].b64_json)

    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None
    
def transcribe_audio(audio_file_path, api_key):
    """
    Sends an audio file to Whisper and returns the text transcript.
    """

    client = openai.OpenAI(api_key=api_key)

    try:
        # Open the audio file in binary read mode
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return transcript.text

    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None
    
def get_feedback(image_context, user_description, api_key):
    """
    Asks GPT-3.5 Turbo to act as a coach and evaluate the description.
    """

    client = openai.OpenAI(api_key=api_key)

    system_prompt = """
    You are an English language coach.
    1. Compare the user's description to the actual image topic.
    2. Give a score out of 10 for Accuracy and Vocabulary.
    3. Provide 2-3 specific tips to improve.
    """

    user_prompt = f"""
    Image Topic: {image_context}
    User's Description: "{user_description}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error getting feedback: {e}")
        return None

# --- 4. SESSION STATE ---
if 'image_url' not in st.session_state:
    st.session_state.image_url = None

if 'image_prompt' not in st.session_state:
    st.session_state.image_prompt = None


# --- 5. MAIN APP UI ---

st.title("Gen AI Product")
st.markdown("This app generates a image and then you can do a comprehension practice using its capabilities")

# Step 1: Generate Image
st.header("1. Generate")
topics = [
    "A futuristic city",
    "A cozy coffee shop",
    "An astronaut on Mars",
    "A medieval castle",
    "A surreal dreamscape"
]

selected_topic = st.selectbox("Choose a theme:", topics)

if st.button("Generate Random Image"):
    with st.spinner("Drawing your image (GPT-IMAGE-2)..."):
        st.session_state.image_prompt = selected_topic
        st.session_state.image_url = generate_image(selected_topic, api_key)
        
# Show image and recording interface if image exists
if st.session_state.image_url:
    st.image(
        st.session_state.image_url,
        caption=st.session_state.image_prompt
    )

    # Step 2: Record
    st.header("2. Describe & Feedback")
    audio_value = st.audio_input("Record your voice describing the image above")

    if audio_value:
        st.audio(audio_value)

        # Save temp file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.read())

        if st.button("Analyze My Description"):
            with st.spinner("Analyzing..."):
                # Transcribe
                transcription = transcribe_audio("temp_audio.wav", api_key)

                if transcription:
                    st.success("Transcription Complete!")
                    st.write(f"**You said:** {transcription}")

                    # Feedback
                    feedback = get_feedback(
                        st.session_state.image_prompt,
                        transcription,
                        api_key
                    )
                    st.info(feedback)

                    # Cleanup
                    if os.path.exists("temp_audio.wav"):
                        os.remove("temp_audio.wav")