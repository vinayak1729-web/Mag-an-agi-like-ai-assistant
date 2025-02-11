
from flask import Flask, render_template, request, jsonify, redirect, url_for, session,Response
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import json
import cv2
from datetime import datetime
from features.trained_chikitsa import chatbot_response
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret_key'

from fer import FER
# # Initialize emotion detection model
emotion_detector = FER(mtcnn=True)

genai.configure(api_key=os.getenv("API_KEY"))
defaultprompt ="""you have to act as a sexologist , gynologist , neuroloigist also health guide  
        You are a professional, highly skilled mental doctor, and health guide.
        You act as a best friend to those who talk to you , but you have to talk based on their mental health , by seeing his age intrests qualities , if you dont know ask him indirectly by asking his/her studing or any work doing currently. 
        you can use the knowlege of geeta to make the user's mind more powerfull but you dont have to give reference of krishna or arjuna etc if the user is more towards god ( hindu gods ) then u can else wont
        You can assess if someone is under mental stress by judging their communication. 
        Your goal is to make them feel happy by cracking jokes and suggesting activities that bring joy. 
        If anyone asks anything outside your field, guide them or use your knowledge to help. 
        You can speak in Hindi, Marathi, Hinglish, or any language the user is comfortable with.
        your name is chikitsa , means Cognitive Health Intelligence Knowledge with Keen Interactive Treatment Support from AI."""
prompt = "this is my assesment of close ended questions and open ended questions , so you have to talk to me accordingly "
# Create the model
generation_config = {
  "temperature": 2,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  system_instruction=defaultprompt+ prompt+ " ")


chat_session = model.start_chat()

# Chat function to handle user input
def gemini_chat(user_input, history_file="dataset/intents.json"):
    try:
        # Load intents from JSON or initialize
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                intents_data = json.load(f)
        else:
            intents_data = {"intents": []}

        # Send user input to the model
        response = chat_session.send_message(user_input)

        # Create a new intent object for the conversation
        new_intent = {
            "patterns": [user_input],
            "responses": [response.text.strip()],
        }

        # Append the new intent to the intents list
        intents_data['intents'].insert(1, new_intent)

        # Save the updated intents JSON file
        with open(history_file, 'w') as f:
            json.dump(intents_data, f, indent=4)

        return response.text

    except Exception as e:
       
        response = chatbot_response(user_input)
        # Optionally log the error to a file
        with open('error.log', 'a') as log_file:
            log_file.write(f"{str(e)}\n")
        return response

# Initialize global variables
attention_status = "Not Paying Attention"
dominant_emotion = "neutral"  # Default value

def is_paying_attention(emotions_dict, threshold=0.5):
    """ Checks if the user is paying attention based on emotion scores. """
    dominant_emotion = max(emotions_dict, key=emotions_dict.get)
    emotion_score = emotions_dict[dominant_emotion]
    return emotion_score > threshold, dominant_emotion

def detect_emotion_and_attention(frame):
    """ Detects emotion and attention from the frame. """
    global attention_status, dominant_emotion

    results = emotion_detector.detect_emotions(frame)

    for result in results:
        bounding_box = result["box"]
        emotions_dict = result["emotions"]

        # Update attention status and dominant emotion
        paying_attention, dominant_emotion = is_paying_attention(emotions_dict)
        attention_status = "Paying Attention" if paying_attention else "Not Paying Attention"

        # Draw bounding box and display emotion info
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        emotion_text = ", ".join([f"{emotion}: {prob:.2f}" for emotion, prob in emotions_dict.items()])
        cv2.putText(frame, f"{dominant_emotion} ({attention_status})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

# Video feed generator
def generate_frames():
    """ Captures frames from the webcam and detects emotion and attention. """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_emotion_and_attention(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask route for the chatbot interaction
@app.route('/talk_to_me', methods=['GET', 'POST'])
def talk_to_me():
    """ Handles the user's input and sends it to the chatbot along with emotion and attention. """
    global attention_status, dominant_emotion

    if request.method == 'GET':
        return render_template('talk_to_me.html')

    elif request.method == 'POST':
        user_input = request.form.get('user_input', '')

        # Create the prompt with the attention status and emotion
        prompt = f"The user is in a {dominant_emotion} mood and is {'paying attention' if attention_status == 'Paying Attention' else 'not paying attention'}."

        # Call the gemini_chat function with the user input and the generated prompt
        bot_response = gemini_chat(user_input + " " + prompt)

        return jsonify({'response': bot_response})

    else:
        return "Unsupported request method", 405

def log_conversation(user_input, bot_response, history_file="dataset/intents.json"):
    # Load or initialize intents data
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            intents_data = json.load(f)
    else:
        intents_data = {"intents": []}

    # Create a new intent object
    new_intent = {
        "patterns": [user_input],
        "responses": [bot_response],
    }

    # Append the new intent
    intents_data['intents'].append(new_intent)

    # Save the updated intents JSON file
    with open(history_file, 'w') as f:
        json.dump(intents_data, f, indent=4)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
    