from datetime import timedelta
from flask import Flask, request, session, render_template
from flask_session import Session
from twilio.twiml.messaging_response import MessagingResponse

from interact import new_chat, generate_reply

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)
Session(app)

@app.route('/')
def ui():
    session.clear() # Clear session with each refresh
    return render_template("ui.html")

@app.route('/chat', methods=['POST'])
def chat():
    if 'past' not in session:
        _, session['past'] = new_chat()
    output, session['past'] = generate_reply(str(request.data, 'utf-8'), session['past'])
    return output

@app.route('/sms', methods=['POST'])
def sms():
    text = request.values.get('Body', '').strip()
    if not text:
        return '', 204

    if text == '/reset':    # Magic code to restart the chat
        session.clear()
        return '', 204

    if 'past' not in session:
        _, session['past'] = new_chat()
    output, session['past'] = generate_reply(text, session['past'])

    resp = MessagingResponse()
    resp.message(output)
    return str(resp)
