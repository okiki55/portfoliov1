"""
Chat/Conversational AI Project
==============================
This is an example chatbot project. Replace with your actual implementation.

Your predict function receives:
    data = {
        'message': 'User message here',
        'type': 'chat'
    }

You should return:
    {
        'response': 'AI response text'
    }
"""

import random

# Simple response patterns (replace with your actual LLM)
RESPONSES = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What's on your mind?",
        "Hey! I'm here to assist you.",
    ],
    "farewell": [
        "Goodbye! Have a great day!",
        "See you later! Take care!",
        "Bye! Feel free to come back anytime.",
    ],
    "thanks": [
        "You're welcome! Is there anything else I can help with?",
        "Happy to help! Let me know if you need anything else.",
        "Anytime! What else would you like to know?",
    ],
    "fun_fact": [
        "Did you know? Honey never spoils. Archaeologists have found 3000-year-old honey in Egyptian tombs that was still edible!",
        "Here's a fun fact: Octopuses have three hearts and blue blood!",
        "Fun fact: A group of flamingos is called a 'flamboyance'!",
    ],
    "default": [
        "That's interesting! Tell me more about what you're thinking.",
        "I understand. How can I help you with that?",
        "Thanks for sharing. What else would you like to discuss?",
    ]
}

GREETING_KEYWORDS = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
FAREWELL_KEYWORDS = ['bye', 'goodbye', 'see you', 'later', 'quit', 'exit']
THANKS_KEYWORDS = ['thank', 'thanks', 'appreciate', 'grateful']
FUN_FACT_KEYWORDS = ['fun fact', 'tell me something', 'interesting', 'did you know']


def predict(data):
    """
    Chat prediction function.
    
    Args:
        data: Dictionary with 'message' key containing user's message
        
    Returns:
        Dictionary with 'response' key containing AI response
    """
    message = data.get('message', '').lower()
    
    if not message:
        return {'error': 'No message provided'}
    
    # Simple intent detection (replace with actual NLU/LLM)
    if any(keyword in message for keyword in GREETING_KEYWORDS):
        response = random.choice(RESPONSES['greeting'])
    elif any(keyword in message for keyword in FAREWELL_KEYWORDS):
        response = random.choice(RESPONSES['farewell'])
    elif any(keyword in message for keyword in THANKS_KEYWORDS):
        response = random.choice(RESPONSES['thanks'])
    elif any(keyword in message for keyword in FUN_FACT_KEYWORDS):
        response = random.choice(RESPONSES['fun_fact'])
    else:
        response = random.choice(RESPONSES['default'])
    
    return {
        'response': response
    }


# ============================================================
# TO IMPLEMENT WITH AN ACTUAL LLM:
# ============================================================
#
# Option 1: OpenAI
# ----------------
# import openai
# 
# def predict(data):
#     message = data.get('message', '')
#     
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": message}
#         ]
#     )
#     
#     return {'response': response.choices[0].message.content}
#
# Option 2: Hugging Face Transformers
# -----------------------------------
# from transformers import pipeline
# 
# generator = pipeline('text-generation', model='gpt2')
# 
# def predict(data):
#     message = data.get('message', '')
#     result = generator(message, max_length=100)
#     return {'response': result[0]['generated_text']}
#
# Option 3: LangChain with Memory
# -------------------------------
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# 
# llm = ChatOpenAI()
# memory = ConversationBufferMemory()
# conversation = ConversationChain(llm=llm, memory=memory)
# 
# def predict(data):
#     message = data.get('message', '')
#     response = conversation.predict(input=message)
#     return {'response': response}
