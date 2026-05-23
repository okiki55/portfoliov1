from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from anthropic import Anthropic
from dotenv import load_dotenv
from cachetools import TTLCache
import os

load_dotenv()

cache = TTLCache(maxsize=100, ttl=300)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ LOAD existing DB only
vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

CHAT_HISTORY = []
MAX_HISTORY = 6

def predict(data):
    message = data.get("message", "").strip()

    if not message:
        return {"error": "No message provided"}

    # 🔥 CACHE CHECK (NEW)
    if message in cache:
        return {"response": cache[message], "cached": True}

    # 1. Retrieve relevant docs
    docs = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. History
    history_text = ""
    for turn in CHAT_HISTORY:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    # 3. PROMPT (UNCHANGED — as requested)
    prompt = f"""
    You are a helpful assistant that answers questions using the user's documents.

    You do NOT talk about yourself.
    You do NOT introduce yourself.
    You do NOT use phrases like "based on the provided context".

    Just answer the question naturally and clearly using the information below.

    If the answer is not in the information, say:
    "I don't have that information in the document."

    Information:
    {context}

    Conversation history:
    {history_text}

    Question:
    {message}

    Answer:
    """

    # 4. Claude call
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text

    # 5. memory
    CHAT_HISTORY.append({
        "user": message,
        "assistant": answer
    })

    if len(CHAT_HISTORY) > MAX_HISTORY:
        CHAT_HISTORY.pop(0)

    # 🔥 STORE CACHE
    cache[message] = answer

    return {"response": answer}
# data= {"message":"who is okiki"}
# print(predict(data))