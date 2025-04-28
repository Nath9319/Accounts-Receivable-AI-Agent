## AI Agent for Accounts Receivable Order Approval

This project implements an automated Accounts Receivable (AR) workflow using LangGraph and Google's Gemini 2.0 Flash model. The agent processes incoming sales orders, assesses customer creditworthiness against a defined company policy, and determines whether to approve, reject, or escalate the order for human review.

**Objectives:**

*   Automate the process of checking sales orders against customer credit limits and company policy.
*   Leverage the Gemini 2.0 Flash LLM for nuanced credit assessment based on provided data and policy rules.
*   Build a robust, stateful, and traceable workflow using LangGraph.
*   Provide a mechanism for human intervention (escalation) for orders requiring manual review.
*   Log workflow steps and LLM interactions locally for evaluation and debugging.

**Inputs:**

*   `customer_master.csv`: File containing customer master data, including credit limits, outstanding balances, etc.
*   `sales_order.csv`: File containing details of incoming sales orders.
*   `CreditPolicy.txt`: Text file defining the company's credit approval rules.
*   Google API Key (loaded via `.env`) for accessing the Gemini model.

**Outputs:**

*   A final decision state for each order, indicating:
    *   `approval_status`: 'approved', 'rejected', or requires human input via `interrupt`.
    *   `decision_reason`: Explanation for the approval or rejection.
    *   `credit_assessment`: Details from the LLM's credit analysis.
*   Locally stored log files (e.g., `ar_workflow_[timestamp].log`, `llm_interactions.log`) detailing the workflow execution and LLM calls.
