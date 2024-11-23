#!/usr/bin/env python
from datetime import datetime
from flask import Flask, jsonify, request
import sys
from textwrap import dedent
from mainframe_userstory_demo.MainframeUserStoryDemoCrew import MainframeUserStoryDemoCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding necessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

app = Flask(__name__)

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        print("Reached upto here")
        # Get the topic from the request JSON
        request_data = request.get_json()
        topic = request_data.get('topic', '')
        print(topic)
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400

        inputs = {
            'topic': dedent(topic)
        }
        
        # Call the existing run logic
        result = MainframeUserStoryDemoCrew().crew().kickoff(inputs=inputs)

        # Debug information about the result type
        print(f"Type of result: {type(result)}")
        print(f"Result value: {result}")

        # Custom response formatting
        response_data = {
            'status': 'success',
            #'timestamp': datetime.datetime.now().isoformat(),
            'data': str(result)
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        error_response = {
            'status': 'error',
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(e),
            'type': type(e).__name__
        }
        return jsonify(error_response), 500


def run():
    """
    Welcome to User Story Generator Crew.
    """
    inputs = {
        'topic': dedent(f"""
                        'Credit Card Bill Payment from Savings/Current Account with Conditional Balance Check

                        Pre-Conditions
                        User has a Savings Account and a Current Account.
                        User has a Credit Card Bill with a Due Amount and a Minimum Bill Pay Amount.
                        System has permissions to access account balances and perform transactions.
                        Steps
                        Check Account Balances

                        Retrieve the balance of the Savings Account (savings_balance).
                        Retrieve the balance of the Current Account (current_balance).
                        Determine Payment Account

                        Compare savings_balance and current_balance.
                        Select the account with the higher balance as the primary_account for the transaction.
                        Check Due Amount and Deduct Funds

                        If the primary_account balance is greater than or equal to the Due Amount:
                        Deduct the Due Amount from primary_account.
                        Record a transaction for the Credit Card Bill Payment for the Due Amount.
                        Handle Insufficient Balance in Both Accounts

                        If neither the Savings nor Current Account has a sufficient balance to cover the Due Amount:
                        Check if the primary_account balance is greater than or equal to the Minimum Bill Pay Amount.
                        If yes:
                        Deduct the Minimum Bill Pay Amount from primary_account.
                        Record a transaction for the Credit Card Bill Payment for the Minimum Bill Pay Amount.
                        If no:
                        Record an Insufficient Funds status and notify the user.
                        Post-Transaction Update

                        Update the balances of Savings and Current Account in the system.
                        Notify the user of the transaction status:
                        Success: Indicate amount paid and account used.
                        Insufficient Funds: Notify of failure to meet even the minimum payment.'
                        """)
    }
    result = MainframeUserStoryDemoCrew().crew().kickoff(inputs=inputs)
    print("###########################################################")
    print(result)
    print("###########################################################")

    # Dump the result to a JSON object




def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        #MainframeUserstoryDemoCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
        MainframeUserStoryDemoCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MainframeUserStoryDemoCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        MainframeUserStoryDemoCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


# Add Flask app runner
if __name__ == '__main__':
    # Check if we want to run as API or command line
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        # Run as Flask app
        app.run(host='0.0.0.0', port=5002, debug=False)
    else:
        # Run as command line
        run()