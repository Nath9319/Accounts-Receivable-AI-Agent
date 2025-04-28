import os
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.types import interrupt

# Define the state structure
class OrderProcessState(TypedDict):
    messages: List[Dict[str, Any]]
    customer_data: Optional[Dict[str, Any]]
    order_data: Optional[Dict[str, Any]]
    credit_assessment: Optional[Dict[str, Any]]
    approval_decision: Optional[str]
    current_step: str

# Initialize Gemini 2.0 Flash model
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Using Gemini 2.0 Flash
        temperature=0,
        google_api_key=os.environ.get("AIzaSyDZqkig8Kgyrm4db1hw2nK3NJTg35RFgq0")
    )

# Define data loading functions
def load_customer_data(customer_id):
    """Load customer data from CSV file."""
    try:
        df = pd.read_csv("customer_master.csv")
        customer = df[df["Customer_ID"] == customer_id]
        if customer.empty:
            return None
        return customer.iloc[0].to_dict()
    except Exception as e:
        print(f"Error loading customer data: {e}")
        return None

def load_order_data(order_id=None):
    """Load order data from CSV file."""
    try:
        df = pd.read_csv("sales_order.csv")
        if order_id:
            order = df[df["Order ID"] == order_id]
            if order.empty:
                return None
            return order.iloc[0].to_dict()
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Error loading order data: {e}")
        return None

# Define the nodes for our workflow
def receive_sales_order(state: OrderProcessState) -> OrderProcessState:
    """Step 1: Receive the sales order from the sales team or customer."""
    order_data = load_order_data()
    
    if not order_data:
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="Error: Failed to load sales order data."))
        return {
            **state,
            "messages": messages,
            "current_step": "error"
        }
    
    messages = state.get("messages", [])
    messages.append(SystemMessage(content="Sales Order Processing"))
    messages.append(AIMessage(content=f"Processing sales order: {order_data.get('Order ID', 'Unknown')}"))
    
    return {
        **state,
        "messages": messages,
        "order_data": order_data,
        "current_step": "sales_order_received"
    }

def check_customer_master_data(state: OrderProcessState) -> OrderProcessState:
    """Step 2: Check Customer Master Data to verify customer exists and credit status."""
    order_data = state.get("order_data", {})
    customer_id = order_data.get("Customer_ID")
    
    if not customer_id:
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="Error: Customer ID not found in order data."))
        return {
            **state,
            "messages": messages,
            "current_step": "error"
        }
    
    customer_data = load_customer_data(customer_id)
    
    if not customer_data:
        messages = state.get("messages", [])
        messages.append(SystemMessage(content=f"Error: Customer ID {customer_id} not found in master data."))
        return {
            **state,
            "messages": messages,
            "current_step": "error"
        }
    
    messages = state.get("messages", [])
    messages.append(SystemMessage(content="Customer Data Verification"))
    messages.append(AIMessage(content=f"Customer ID {customer_id} verified. Credit Status: {customer_data.get('Credit_Status', 'Unknown')}"))
    
    return {
        **state,
        "messages": messages,
        "customer_data": customer_data,
        "current_step": "customer_data_checked"
    }

def review_credit_policy(state: OrderProcessState) -> OrderProcessState:
    """Step 3: Review the predefined credit policy."""
    gemini = get_llm()
    customer_data = state.get("customer_data", {})
    
    prompt = f"""
    As a Credit Policy Analyst, review the following customer credit information:
    
    Credit Status: {customer_data.get('Credit_Status', 'Unknown')}
    Credit Score: {customer_data.get('Credit_Score', 'Unknown')}
    Default Payment Terms: {customer_data.get('Default_Payment_Terms', 'Unknown')}
    
    Based on this information, summarize the applicable credit policy guidelines.
    """
    
    message = HumanMessage(content=prompt)
    response = gemini([message])
    
    messages = state.get("messages", [])
    messages.append(SystemMessage(content="Credit Policy Review"))
    messages.append(message)
    messages.append(response)
    
    return {
        **state,
        "messages": messages,
        "current_step": "credit_policy_reviewed"
    }

def assess_customer_creditworthiness(state: OrderProcessState) -> OrderProcessState:
    """Step 4: Assess Customer Creditworthiness."""
    gemini = get_llm()
    customer_data = state.get("customer_data", {})
    order_data = state.get("order_data", {})
    
    # Extract relevant information for credit assessment
    credit_limit = float(customer_data.get("Credit_Limit", 0))
    outstanding_balance = float(customer_data.get("Outstanding_Balance", 0))
    available_credit = credit_limit - outstanding_balance
    order_amount = float(order_data.get("turnover", 0))
    
    # Determine credit status for this order
    if order_amount > available_credit:
        credit_status = "Exceeds Available Credit"
        needs_escalation = True
    else:
        credit_status = "Within Credit Limit"
        needs_escalation = False
    
    # Create a credit assessment record
    credit_assessment = {
        "credit_limit": credit_limit,
        "outstanding_balance": outstanding_balance,
        "available_credit": available_credit,
        "order_amount": order_amount,
        "credit_status": credit_status,
        "needs_escalation": needs_escalation
    }
    
    # Ask Gemini to provide a detailed credit analysis
    prompt = f"""
    As a Credit Analyst, assess the creditworthiness for this order:
    
    Customer Information:
    - Credit Limit: ${credit_limit}
    - Outstanding Balance: ${outstanding_balance}
    - Available Credit: ${available_credit}
    - Credit Score: {customer_data.get('Credit_Score', 'Unknown')}
    - Payment History: {customer_data.get('Days_Outstanding', 'Unknown')} days outstanding
    
    Order Information:
    - Order Amount: ${order_amount}
    - Order Status: {credit_status}
    
    Provide a detailed credit assessment and recommendation (approve, deny, or escalate).
    """
    
    message = HumanMessage(content=prompt)
    response = gemini([message])
    
    messages = state.get("messages", [])
    messages.append(SystemMessage(content="Credit Assessment"))
    messages.append(message)
    messages.append(response)
    
    return {
        **state,
        "messages": messages,
        "credit_assessment": credit_assessment,
        "current_step": "creditworthiness_assessed"
    }

def approve_or_escalate(state: OrderProcessState) -> OrderProcessState:
    """Step 5: Approve or Escalate the order based on credit assessment."""
    credit_assessment = state.get("credit_assessment", {})
    needs_escalation = credit_assessment.get("needs_escalation", True)
    
    messages = state.get("messages", [])
    messages.append(SystemMessage(content="Order Approval Decision"))
    
    if needs_escalation:
        # Order needs human approval/escalation
        messages.append(AIMessage(content="Order requires escalation to management for approval."))
        # Use interrupt to pause the workflow for human input
        return interrupt(
            {
                **state,
                "messages": messages,
                "current_step": "awaiting_escalation_approval"
            },
            "human_approval_needed"
        )
    else:
        # Order is automatically approved
        messages.append(AIMessage(content="Order automatically approved - within credit limits and policy."))
        
        return {
            **state,
            "messages": messages,
            "approval_decision": "approved",
            "current_step": "order_approved"
        }

