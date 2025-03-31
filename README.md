# Sales Order Assistant Chatbot

This project is a **Sales Order Assistant Chatbot** built using Python, Streamlit, and LangGraph. The chatbot helps users inquire about their sales orders, including order status, order details, billing, and general inquiries. It uses a pre-trained language model (LLM) to process queries and provide relevant responses.

## Features

- **Order Status**: Retrieve the status of an order using the order ID.
- **Order Details**: Fetch detailed information about an order, including customer name, order date, total amount, and items.
- **Billing Queries**: Provide billing-related information.
- **General Queries**: Answer general sales-related questions.
- **Sentiment Analysis**: Analyze the sentiment of user queries to escalate negative sentiments to a human agent.
- **Order ID Extraction**: Automatically extract order IDs from user queries using regex.
- **Streamlit UI**: Interactive chat interface for users to interact with the chatbot.

## Prerequisites

- Python 3.10 or higher
- Required Python packages (install using `pip`):
  - `streamlit`
  - `pandas`
  - `langgraph`
  - `langchain-core`
  - `langchain-groq`
  - `python-dotenv`

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dhana1008/AgenticAIPOC.git
   cd AgenticAIPOC/AI Customer Support

## Running the Application

To run the chatbot using Streamlit, execute the following command in the terminal:

```bash
streamlit run sale_qandAbot.py