from courtroom_debate import CourtRoomDebate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv() 

llm = ChatGroq(model="groq/llama-3.1-8b-instant")
# Add this at the very beginning of your script
import urllib3
import logging
import ssl

# Disable all SSL verification
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

# Silence specific loggers completely
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

# Disable telemetry completely
os.environ["CREWAI_TELEMETRY"] = "false"
def verify_core_flow():
    """Verify the core debate flow sequence."""
    
    # Set up environment variable to disable telemetry
    os.environ["CREWAI_TELEMETRY"] = "false"
    
    # Create a test debate topic
    test_topic = "Artificial Intelligence will have a net positive impact on society"
    
    # Initialize the flow
    print("Initializing CourtRoomDebate flow...")
    debate_flow = CourtRoomDebate(debate_topic=test_topic,llm = llm)
    
    # Start a simple interactive test
    print("\n=== INTERACTIVE CORE FLOW TEST ===")
    print("This will run through one full round of the debate cycle.")
    print("You'll be prompted to provide feedback at the end of the round.")
    print("Type 'exit' to end the debate, or provide feedback to continue.\n")
    
    # Instructions for test mode
    print("NOTE: Since this is just a verification test, we won't")
    print("actually continue to a second round even if you provide feedback.\n")
    
    proceed = input("Press Enter to begin the test or type 'quit' to exit: ")
    if proceed.lower().strip() == 'quit':
        return False
    
    # Run the flow
    print("\nStarting the debate flow...")
    result = debate_flow.kickoff()
    
    # Check the result
    print("\n=== TEST RESULTS ===")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Rounds completed: {result.get('rounds_completed', 0)}")
    print(f"Topic: {result.get('topic', 'unknown')}")
    
    if result.get('status') == "completed" and result.get('rounds_completed') > 0:
        print("\n✅ Core flow sequence verified successfully!")
        return True
    else:
        print("\n❌ Core flow verification failed. Check the implementation.")
        return False

if __name__ == "__main__":
    verify_core_flow()