from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import logging
import boto3
import base64
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up voice recognition and TTS engine
try:
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
except Exception as e:
    logger.error(f"Error setting up TTS engine: {e}")

def speak(audio):
    """Convert text to speech"""
    try:
        engine.say(audio)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Error in TTS speak function: {e}")

def recognize_voice():
    """Recognize voice input using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        logger.info("Recognizing voice...")
        query = recognizer.recognize_google(audio, language='en-in')
        logger.info(f"User said: {query}")
        return query
    except Exception as e:
        logger.error(f"Voice recognition failed: {e}")
        speak("Sorry, I didn't catch that. Please try again.")
        return None

# Replace with your own Generative AI API key
try:
    genai.configure(api_key="HERE")
except Exception as e:
    logger.error(f"Error configuring Generative AI API: {e}")

# Set up AWS Rekognition client
try:
    rekognition_client = boto3.client(
        'rekognition', 
        aws_access_key_id='HERE',
        aws_secret_access_key='HERE',
        region_name='us-east-1'
    )
except Exception as e:
    logger.error(f"Error setting up AWS Rekognition client: {e}")

# Configure the generative model
try:
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    convo = model.start_chat(history=[
        {"role": "user", "parts": ["Hi"]},
        {"role": "model", "parts": ["Hello there! How can I assist you today?"]}
    ])
except Exception as e:
    logger.error(f"Error configuring generative model: {e}")

@app.route("/")
def index():
    return render_template('a.html')

@app.route("/voice_prompt", methods=["POST"])
def voice_prompt():
    try:
        logger.info("Handling voice input...")
        user_text = recognize_voice()
        if not user_text:
            return jsonify({"error": "Could not understand voice input."}), 400

        logger.info(f"User said: {user_text}")
        convo.send_message(user_text)
        response = convo.last.text

        logger.info(f"AI response: {response}")
        speak(response)

        return jsonify({'response': response}), 200

    except Exception as e:
        logger.error(f"Error in voice_prompt: {e}")
        return jsonify({'error': "Failed to process voice prompt."}), 500

@app.route("/send_message", methods=["POST"])
def send_message():
    try:
        data = request.get_json()
        user_text = data.get('message', '')
        image_data = data.get('image', '')

        logger.info(f"Received message: {user_text}")

        # Process image if provided
        if image_data:
            image_bytes = base64.b64decode(image_data)

            # Detect labels in the image
            image_response = rekognition_client.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=10
            )
            image_description = ', '.join([label['Name'] for label in image_response['Labels']])
            logger.info(f"Image description: {image_description}")

            # Recognize celebrities in the image
            celebrity_response = rekognition_client.recognize_celebrities(
                Image={'Bytes': image_bytes}
            )
            celebrities = celebrity_response['CelebrityFaces']
            celebrity_names = ', '.join([celebrity['Name'] for celebrity in celebrities])

            if celebrity_names:
                image_description += f" Celebrities recognized: {celebrity_names}."

        else:
            image_description = ""

        convo.send_message(user_text + " " + image_description)
        response = convo.last.text

        logger.info(f"Generated response: {response}")
        speak(response)

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        return jsonify({'response': "Sorry, I couldn't process your request."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
