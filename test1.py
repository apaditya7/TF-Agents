from courtroom_debate import CourtRoomDebate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv() 

llm = ChatGroq(model="groq/llama-3.1-8b-instant")

# SSL and logging configurations remain the same
import urllib3
import logging
import ssl
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

# Silence specific loggers completely
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
def verify_core_flow():
    """Verify the multi-round debate flow sequence."""
    
    # Set up environment variable to disable telemetry
    os.environ["CREWAI_TELEMETRY"] = "false"
    
    # Create a test debate topic
    test_topic = "Artificial Intelligence will have a net positive impact on society"
    
    # Initialize the flow
    print("Initializing CourtRoomDebate flow...")
    debate_flow = CourtRoomDebate(debate_topic=test_topic, llm=llm)
    
    # Start a simple interactive test
    print("\n=== INTERACTIVE CORE FLOW TEST ===")
    print("This will run through multiple rounds of the debate cycle.")
    print("You'll be prompted to provide feedback after each round.")
    print("Type 'exit' to end the debate early.\n")
    
    # Run multiple rounds
    rounds_to_run = 3  # You can adjust this number
    
    for round in range(1, rounds_to_run + 1):
        print(f"\n===== ROUND {round} =====")
        
        # Run the flow
        print("\nStarting the debate flow...")
        result = debate_flow.kickoff()
        
        # Inspect the result object
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        
        # Check if the result has a 'should_continue' key
        if isinstance(result, dict) and 'should_continue' in result:
            if not result['should_continue']:
                break
        else:
            # Fallback: Assume the debate should continue
            print("Warning: 'should_continue' key not found. Continuing debate.")
        
        # Prompt for feedback
        if round < rounds_to_run:
            proceed = input("\nPress Enter to continue to next round or type 'exit' to end: ")
            if proceed.lower().strip() == 'exit':
                break
    
    # Final results
    print("\n=== FINAL TEST RESULTS ===")
    
    # Check if the result has the required keys
    if isinstance(result, dict) and 'pro_argument' in result and 'con_argument' in result:
        print(f"Pro Argument: {result['pro_argument'].raw}")
        print(f"Con Argument: {result['con_argument'].raw}")
        
        if 'should_continue' in result and not result['should_continue']:
            print("\n✅ Debate concluded successfully!")
            return True
        else:
            print("\n❌ Debate did not conclude as expected. Check the implementation.")
            return False
    else:
        print("\n❌ Result object does not have required keys. Check the implementation.")
        return False
if __name__ == "__main__":
    verify_core_flow()