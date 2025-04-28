# %%
import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from langgraph.types import interrupt, Command
import random
from datetime import datetime

# %%
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# %%
from langchain.callbacks.base import BaseCallbackHandler

class GeminiLogger(BaseCallbackHandler):
    """Custom callback handler to log Gemini model interactions"""
    
    def __init__(self, verbose=True):
        """Initialize the logger"""
        self.verbose = verbose
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts processing"""
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"SENDING PROMPT TO GEMINI:")
            print(f"{'='*50}")
            for i, prompt in enumerate(prompts):
                print(f"{prompt}\n")
    
    def on_llm_end(self, response, **kwargs):
        """Log when LLM completes processing"""
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"RECEIVED RESPONSE FROM GEMINI:")
            print(f"{'='*50}")
            print(f"{response.generations[0][0].text}")
            
    def on_llm_error(self, error, **kwargs):
        """Log any errors during LLM processing"""
        print(f"\n{'='*50}")
        print(f"ERROR DURING LLM CALL:")
        print(f"{'='*50}")
        print(f"Error: {error}")


# %%
# Define state structure
class ARWorkflowState(TypedDict):
    customer_data: dict
    order_data: dict
    policy_content: str
    credit_assessment: dict
    approval_status: Annotated[str, "approved|rejected|escalated"]
    decision_reason: str
    requires_human: bool

# %%
# Initialize the logger
gemini_logger = GeminiLogger()

# Initialize the model with the logger
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", 
    temperature=0,
    callbacks=[gemini_logger]  # Add logger to callbacks
)


# %%
def load_policy() -> str:
    """Load and preprocess credit policy document"""
    try:
        with open("CreditPolicy.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "DEFAULT_CREDIT_POLICY: Max credit utilization 80%, Min credit score 650"

# %%
def data_loader(state: ARWorkflowState) -> ARWorkflowState:
    """Step 1: Load customer and order data"""
    try:
        # Load CSV files
        customer_df = pd.read_csv("customer_master.csv")
        order_df = pd.read_csv("sales_order.csv")
        
        # For testing purposes, use a consistent sample rather than random
        customer_ids = list(set(order_df.Customer_ID.to_list()) & set(customer_df.Customer_ID.to_list()))
        
        if not customer_ids:
            raise ValueError("No matching customers found between datasets")
            
        customer_id = random.choice(customer_ids)  # Choose first one for consistency
        
        # Get single row and convert to dictionary (properly)
        order_row = order_df[order_df.Customer_ID == customer_id].iloc[0]
        customer_row = customer_df[customer_df.Customer_ID == customer_id].iloc[0]
        
        # Convert to proper dictionaries
        current_order = {col: order_row[col] for col in order_df.columns}
        customer_data = {col: customer_row[col] for col in customer_df.columns}
        
        print(f"Loaded data for customer: {customer_id}")
        
        return {
            **state,
            "customer_data": customer_data,
            "order_data": current_order,
            "policy_content": load_policy()
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return default state with error information
        return {
            **state,
            "customer_data": {"error": str(e)},
            "order_data": {"error": str(e)},
            "policy_content": load_policy(),
            "decision_reason": f"Error loading data: {e}"
        }


# %%
def credit_assessment(state: ARWorkflowState) -> ARWorkflowState:
    """Step 3: Automated credit check"""
    try:
        print("\nSTARTING CREDIT ASSESSMENT")
        
        # Extract key data with logging
        customer = state.get("customer_data", {})
        order = state.get("order_data", {})
        print(f"Customer data keys: {list(customer.keys())}")
        print(f"Order data keys: {list(order.keys())}")
        
        # Get numeric values with safe defaults and logging
        try:
            credit_limit = float(customer.get("Credit_Limit", 0))
            outstanding_balance = float(customer.get("Outstanding_Balance", 0))
            order_amount = float(order.get("turnover", 0))
            policy_content = state.get("policy_content", "")
            print(f"Parsed values: Credit Limit=${credit_limit}, Outstanding=${outstanding_balance}, Order=${order_amount}")
        except ValueError as e:
            print(f"ERROR converting values to float: {e}")
            raise
        
        # Prepare and send prompt to LLM
        print("\nPreparing LLM prompt...")
        prompt = f"""
        Your a CA in accounts receivable team and you are working for a credit risk management team,.
        You are responsible for assessing the creditworthiness of customers based on company policy and customers credit limit, outstanding balance, and order amount.
        You have access to a credit policy document that outlines the criteria for credit approval: 
        Policy: {policy_content}.

        Analyze this customer's creditworthiness based on the following:
        
        Customer Information:
        - Credit Limit: ${credit_limit}
        - Outstanding Balance: ${outstanding_balance}
        
        Order Information:
        - Order Amount: ${order_amount}
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        ANALYSIS: [detailed assessment]
        WITHIN_LIMITS: [Yes/No]
        NEEDS_ESCALATION: [Yes/No]
        """
        
        print("Sending request to Gemini...\n")
        response = model.invoke(prompt)
        print("Received response from Gemini\n")
        
        # Parse response
        content = response.content
        print(f"PARSING RESPONSE:\n{content}")
        
        # Create assessment with parsed results
        assessment = {
            "analysis": content,
            "available_credit": credit_limit - outstanding_balance,
            "order_amount": order_amount,
            "within_limits": (credit_limit - outstanding_balance) >= order_amount,
            "needs_escalation": (credit_limit - outstanding_balance) < order_amount
        }
        
        print(f"Final assessment: {assessment}")
        
        return {
            **state,
            "credit_assessment": assessment
        }
    except Exception as e:
        print(f"ERROR in credit assessment: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            **state,
            "credit_assessment": {
                "analysis": f"Error in credit assessment: {e}",
                "within_limits": False,
                "needs_escalation": True
            }
        }


# %%
def policy_check(state: ARWorkflowState) -> ARWorkflowState:
    """Step 4: Policy compliance verification"""
    try:
        assessment = state.get("credit_assessment", {})
        
        print(f"Policy check assessment data: {assessment}")
        
        # Default reason if assessment is empty
        if not assessment:
            reason = "Order rejected due to missing credit assessment data"
            status = "rejected"
        elif not assessment.get("within_limits", False):
            reason = f"Order amount (${assessment.get('order_amount', 0)}) exceeds available credit (${assessment.get('available_credit', 0)})"
            status = "rejected"
        else:
            reason = "Order is within credit limits and complies with policy"
            status = "approved"
        
        print(f"Policy decision: {status} - {reason}")
        
        return {
            **state,
            "approval_status": status,
            "decision_reason": reason,
            "requires_human": assessment.get("needs_escalation", False)
        }
    except Exception as e:
        print(f"Error in policy check: {e}")
        return {
            **state,
            "approval_status": "rejected",
            "decision_reason": f"Error in policy check: {e}",
            "requires_human": True
        }


# %%
def human_review(state: ARWorkflowState) -> Command[Literal["approve", "reject"]]:
    """Step 4: Human escalation point"""
    review_data = {
        "customer": state["customer_data"]["Customer_ID"],
        "order_value": state["order_data"]["turnover"],
        "available_credit": state["customer_data"]["Credit_Limit"] - state["customer_data"]["Outstanding_Balance"],
        "reason": state["decision_reason"]
    }
    
    decision = interrupt(review_data)
    return Command(goto="approve" if decision else "reject")

# %%
# Define all node functions before creating the graph
def approve_order(state: ARWorkflowState) -> ARWorkflowState:
    """Step 5a: Process order approval"""
    return {
        **state,
        "approval_status": "approved",
        "decision_reason": f"Order approved within credit policy guidelines.",
    }

# %%
def reject_order(state: ARWorkflowState) -> ARWorkflowState:
    """Step 5b: Process order rejection"""
    return {
        **state,
        "approval_status": "rejected",
        "decision_reason": state.get("decision_reason", "Failed credit policy check")
    }

# %%
def check_customer_data(state: ARWorkflowState) -> ARWorkflowState:
    """Step 2: Verify customer exists in the system"""
    try:
        # Load customer master data directly from CSV
        customer_df = pd.read_csv('customer_master.csv')
        
        # Extract customer ID from order data
        customer_id = state["order_data"].get("Customer_ID")
        
        # Check if customer ID exists in customer master
        customer_exists = customer_id in customer_df['Customer_ID'].values
        
        print(f"Checking if customer {customer_id} exists: {customer_exists}")
        
        return {
            **state,
            "customer_exists": customer_exists
        }
    except Exception as e:
        print(f"Error in check_customer_data: {e}")
        return {
            **state,
            "customer_exists": False,
            "decision_reason": f"Error checking customer: {e}"
        }

# %%
def document_approval(state: ARWorkflowState) -> ARWorkflowState:
    """Step 6: Document the approval decision"""
    # In production: Update records in accounting system
    return {
        **state,
        "documentation_complete": True,
        "documentation_timestamp": datetime.now().isoformat()
    }

# %%
def communicate_decision(state: ARWorkflowState) -> ARWorkflowState:
    """Step 7: Inform relevant teams about the decision"""
    # In production: Send notifications to sales/logistics
    return {
        **state,
        "communication_complete": True,
        "notification_sent": True
    }


# %%
def human_escalation(state: ARWorkflowState) -> ARWorkflowState:
    """Step 5c: Escalate order for management review"""
    # Prepare escalation details for management review
    escalation_details = {
        "customer_id": state["customer_data"].get("Customer_ID"),
        "order_value": state["order_data"].get("turnover"),
        "credit_limit": state["customer_data"].get("Credit_Limit"),
        "outstanding_balance": state["customer_data"].get("Outstanding_Balance"),
        "escalation_reason": state["decision_reason"]
    }
    
    # In production, this would use interrupt() to pause for human input
    # For example: approval_decision = interrupt(escalation_details)
    
    # For a complete solution, using interrupt():
    approval_decision = interrupt(
        state,
        "management_approval_needed",
        escalation_details
    )
    
    # Process the management decision
    return {
        **state,
        "approval_status": approval_decision,  # "approved" or "rejected" from management
        "decision_reason": f"Decision made by management after review"
    }

# %%
# Build workflow
workflow = StateGraph(ARWorkflowState)

# Add nodes for each step in the accounts receivable process
workflow.add_node("load_data", data_loader)  # Step 1: Load customer and order data
workflow.add_node("check_customer", check_customer_data)  # Step 2: Verify customer exists
workflow.add_node("credit_check", credit_assessment)  # Step 3: Assess creditworthiness
workflow.add_node("policy_verification", policy_check)  # Step 4: Verify against policy
workflow.add_node("approve_order", approve_order)  # Step 5a: Approve within limits
workflow.add_node("reject_order", reject_order)  # Step 5b: Reject outside limits
workflow.add_node("escalate_to_management", human_escalation)  # Step 5c: Escalate for review
workflow.add_node("document_approval", document_approval)  # Step 6: Document the decision
workflow.add_node("communicate_decision", communicate_decision)  # Step 7: Inform teams

# Set the entry point
workflow.set_entry_point("load_data")

# Define the process flow with edges
workflow.add_edge("load_data", "check_customer")

# Customer verification branching
workflow.add_conditional_edges(
    "check_customer",
    lambda s: "existing_customer" if s.get("customer_exists", False) else "new_customer",
    {
        "existing_customer": "credit_check",
        "new_customer": "reject_order"  # New customers rejected per policy
    }
)

# Credit assessment flow
workflow.add_edge("credit_check", "policy_verification")

# Policy verification branching
# Policy verification branching
workflow.add_conditional_edges(
    "policy_verification",
    lambda s: "requires_escalation" if s.get("requires_human", False) else 
              ("approve_order" if s.get("approval_status") == "approved" else "reject_order"),
    {
        "requires_escalation": "escalate_to_management",
        "approve_order": "approve_order",
        "reject_order": "reject_order"
    }
)



# Management escalation paths
workflow.add_conditional_edges(
    "escalate_to_management",
    lambda s: s.get("approval_status", "rejected"), 
    {
        "approved": "approve_order",
        "rejected": "reject_order"
    }
)


# Final processing steps
workflow.add_edge("approve_order", "document_approval")
workflow.add_edge("document_approval", "communicate_decision")
workflow.add_edge("reject_order", "communicate_decision")
workflow.add_edge("communicate_decision", END)

# Compile the workflow into an executable agent
ar_workflow = workflow.compile()


# %%
from IPython.display import Image

Image(ar_workflow.get_graph().draw_mermaid_png())

# %%
print("Starting workflow...")
result = ar_workflow.invoke({
    "customer_data": {},
    "order_data": {},
    "policy_content": load_policy(),
    "credit_assessment": {},
    "approval_status": "",
    "decision_reason": "",
    "requires_human": False,
    "customer_exists": True  # Set to True to test the main path
})
print("Workflow complete.")

# %%
for i, v in result.items():
    if i not in ["customer_data", "order_data", "policy_content"]:
        print(f"{i}: {v}")

# %%
print(result["credit_assessment"]["analysis"])

# %%



