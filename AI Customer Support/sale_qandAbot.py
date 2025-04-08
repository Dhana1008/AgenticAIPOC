import streamlit as st
import pandas as pd
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import re
from typing import TypedDict, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("groq_key")
os.environ["GROQ_API_KEY"] = api_key

df = pd.read_csv(r"C:\Users\Suraj\OneDrive\Documents\Dhanashree work codes\AIWORk\AI Customer Support\datasales_orders.csv")
df["Order ID"] = df["Order ID"].astype(str)  # Replace with your actual file or DataFrame
print(df.head())

# Initialize LLM
llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7)
# Define State Structure
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# Query Categorization
def categorize(state: State) -> State:
    """Classify query into: Order Status, Order Details, Billing, General"""
    prompt = ChatPromptTemplate.from_template(
    "Categorize the following customer query into exactly one of these five categories: "
    "'Order Status', 'Order Details', 'Billing', 'General', or 'Random'. "
    "Respond ONLY with the category name, nothing else. Query: {query}"
)
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content.strip()
    print(f"[DEBUG] Categorized as: '{category}'")  # Debugging log
    return {"category": category}

# Sentiment Analysis
def analyze_sentiment(state: State) -> State:
    """Determine sentiment: Positive, Neutral, Negative"""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following query as 'Positive', 'Neutral', or 'Negative'. "
        "Respond ONLY with the sentiment label. Query: {query}"
    )
    sentiment = (prompt | llm).invoke({"query": state["query"]}).content.strip()
    return {"sentiment": sentiment}

# Extract Order ID
def extract_order_id(query: str) -> str:
    """Extracts a numeric order ID using regex"""
    match = re.search(r"\b\d{6}\b", query)  # Look for exactly 6-digit numbers
    if match:
        order_id = match.group()
        print(f"[DEBUG] Extracted Order ID: {order_id}")  # Debugging log
        return order_id
    print("[DEBUG] No valid numeric order ID found")
    return None  # Return None if no valid order ID is found
  
#print(extract_order_id("My order number is 554369"))

# Fetch Order Status
def fetch_order_status(state: State) -> State:
    """Retrieve order status from DataFrame using extracted Order ID"""
    print(f"[DEBUG] State: {state}")  # Debugging log
    order_id = extract_order_id(state["query"])
    print(f"[DEBUG] Extracted Order ID: {order_id}")  # Debugging log

    if not order_id:
        return {"response": "I couldn't find an order ID in your query. Please provide a valid order number."}
    
    print(f"[DEBUG] Extracted Order ID: {order_id}")  # Debugging log

    if not order_id:
        return {"response": "I couldn't find an order ID in your query. Please provide a valid order number."}

    print(f"[DEBUG] Checking Order ID: {order_id} in DataFrame")  # Debugging log
    df["Order ID"] = df["Order ID"].astype(str)  # Ensure column is string

    if order_id in df["Order ID"].values:
        status = df.loc[df["Order ID"] == order_id, "Order Status"].values[0]
        print(f"[DEBUG] Order found! Status: {status}")
        return {"response": f"The status of order {order_id} is: {status}"}
    
    print("[DEBUG] Order ID not found in DataFrame")
    return {"response": f"Order {order_id} not found. Please check the order number."}


# Fetch Order Details
def fetch_order_details(state: State) -> State:
    """Retrieve order details from DataFrame using extracted Order ID"""
    order_id = extract_order_id(state["query"])
    
    if not order_id:
        return {"response": "I couldn't find an order ID in your query. Please provide a valid order number."}
    
    print(f"[DEBUG] Fetching details for Order ID: {order_id}")  # Debugging log

    if order_id in df["Order ID"].astype(str).values:  
        order_details = df[df["Order ID"].astype(str) == order_id].iloc[0].to_dict()
        print(order_details)
        response = (
            f"Here are the details for order {order_id}:\n"
            f"- Status: {order_details.get('Order Status', 'N/A')}\n"
            f"- Date: {order_details.get('Order Date', 'N/A')}\n"
            f"- Customer: {order_details.get('Customer Name', 'N/A')}\n"
            f"- Total Amount: {order_details.get('Order Amount', 'N/A')}\n"
            f"- Items: {order_details.get('Item Details', 'N/A')}"
        )
        print("[DEBUG] Order details fetched successfully")  # Debugging log
        return {"response": response}
    else:
        print("[DEBUG] Order ID not found in DataFrame")  # Debugging log
        return {"response": f"Order {order_id} not found. Please check the order number."}


# Handle Billing Queries
def handle_billing(state: State) -> State:
    """Provide billing-related information"""
    return {"response": "For billing inquiries, please visit our billing portal or contact support."}

def handle_general(state: State) -> State:
    """Handle general sales-related queries like delivery hours, shipping days, and customer support."""
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that only answers questions related to sales, shipping, and orders. "
        "Provide concise, accurate responses for general sales-related inquiries such as delivery hours, shipping days, or customer support. "
        "If the query is unrelated to sales, shipping, or orders, respond with: 'I'm sorry, but I can only assist with sales-related queries.'\n\n"
        "For reference, assume the following standard information unless specified otherwise:\n"
        "- Delivery hours: 9 AM to 5 PM, Monday to Friday\n"
        "- Shipping days: Orders ship within 2 business days\n"
        "- Customer support: Available 24/7 via email (support@example.com) and 9 AM to 6 PM via phone (1-800-555-1234)\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    
    return {"response": response}

def handle_random(state: State) -> State:
    """Handle queries that don't fit into predefined categories"""
    prompt = ChatPromptTemplate.from_template(
        "You are a customer support assistant focused on sales and orders. "
        "The following query could not be categorized into Order Status, Order Details, Billing, or General. "
        "Provide a polite, helpful response indicating that the query is unclear or off-topic, and suggest how the user can get assistance. "
        "Keep the response concise and professional.\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    
    return {"response": response}


# Escalate Negative Sentiments
def escalate(state: State) -> State:
    return {"response": "Your query has been escalated to a human agent."}

# Routing Logic
def route_query(state: State) -> str:
    category = state["category"].strip().lower()
    sentiment = state["sentiment"].strip().lower()

    print(f"[DEBUG] Routing decision: Sentiment = {sentiment}, Category = {category}")

    if sentiment == "negative":
        return "escalate"
    elif category == "order status":
        return "fetch_order_status"
    elif category == "order details":
        return "fetch_order_details"
    elif category == "billing":
        return "handle_billing"
    elif category == "random":
        return "handle_random"
    else:
        return "handle_general"

# Build LangGraph Workflow
workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("fetch_order_status", fetch_order_status)
workflow.add_node("fetch_order_details", fetch_order_details)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("handle_random", handle_random)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "fetch_order_status": "fetch_order_status",
        "fetch_order_details": "fetch_order_details",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "handle_random": "handle_random",
        "escalate": "escalate",
    },
)

workflow.add_edge("fetch_order_status", END)
workflow.add_edge("fetch_order_details", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

workflow.set_entry_point("categorize")
app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Sales Order Chatbot", layout="wide")
st.title("📦 Sales Order Assistant")

st.markdown("Ask about your **order status, details, billing, or general inquiries.**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
query = st.chat_input("Type your question here...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Process user query
    result = app.invoke({"query": query})

    response = result.get("response", "Sorry, I couldn't process your request.")
    
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save conversation history
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["messages"].append({"role": "assistant", "content": response})
