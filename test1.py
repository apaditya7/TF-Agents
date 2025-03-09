# main.py

import os
import sys
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from courtroom_debate import CourtRoomDebate

# Configure logging and SSL
import urllib3
import logging
import ssl

# Disable SSL warnings and configure logging
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv()

# Disable telemetry
os.environ["CREWAI_TELEMETRY"] = "false"

def print_header():
    """Print the application header."""
    print("\n" + "=" * 80)
    print(" " * 25 + "AI COURTROOM DEBATE SYSTEM")
    print("=" * 80)
    print("\nThis system facilitates structured courtroom-style debates on any topic,")
    print("with AI agents taking Pro and Con positions backed by research.")
    print("\nThe debate consists of multiple rounds, with your feedback guiding the")
    print("evolution of arguments. You act as the judge providing guidance.")
    print("\nAt the end, a comprehensive summary of the debate will be provided.")
    print("=" * 80 + "\n")

def get_topic():
    """Get the debate topic from the user."""
    print("What topic would you like to debate?\n")
    print("Examples:")
    print("  - Artificial Intelligence will have a net positive impact on society")
    print("  - Social media has improved human communication")
    print("  - Cryptocurrency is the future of finance")
    print("  - Remote work is more productive than office work")
    
    while True:
        topic = input("\nEnter your debate topic: ").strip()
        if topic:
            confirm = input(f"\nDebate topic: \"{topic}\"\nIs this correct? (y/n): ").lower()
            if confirm == 'y' or confirm == 'yes':
                return topic
        print("Please enter a valid debate topic.")

def get_rounds():
    """Get the number of rounds for the debate."""
    while True:
        try:
            rounds = input("\nHow many rounds would you like the debate to run? (1-5, default: 3): ").strip()
            if not rounds:
                return 3
            rounds = int(rounds)
            if 1 <= rounds <= 5:
                return rounds
            print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")

def format_argument(title, argument):
    """Format an argument for better readability."""
    formatted = f"\n{'-' * 40}\n{title}\n{'-' * 40}\n"
    formatted += argument
    formatted += f"\n{'-' * 40}\n"
    return formatted

def run_debate(topic, max_rounds, llm):
    """Run the debate with the specified topic and number of rounds."""
    print(f"\nInitializing debate on: \"{topic}\"")
    print("This may take a moment as the AI agents prepare...\n")
    
    debate_flow = CourtRoomDebate(debate_topic=topic, llm=llm)
    
    rounds_completed = 0
    
    result = debate_flow.kickoff()
    rounds_completed = 1
    
    if isinstance(result, dict) and 'pro_argument' in result and 'con_argument' in result:
        if hasattr(result['pro_argument'], 'raw'):
            pro_arg = result['pro_argument'].raw
        else:
            pro_arg = str(result['pro_argument'])
        
        if hasattr(result['con_argument'], 'raw'):
            con_arg = result['con_argument'].raw
        else:
            con_arg = str(result['con_argument'])
        
        print(format_argument("PRO ARGUMENT", pro_arg))
        print(format_argument("CON ARGUMENT", con_arg))
        
        # Check if debate should continue
        if isinstance(result, dict) and 'should_continue' in result:
            if not result['should_continue']:
                # Display debate summary if available
                if 'debate_summary' in result:
                    print("\n" + "=" * 80)
                    print(result['debate_summary'])
                    print("\n" + "=" * 80 + "\n")
                print(f"\nDebate completed with {rounds_completed} rounds.")
                print("Thank you for participating in the AI Courtroom Debate!\n")
                return
        
        for current_round in range(2, max_rounds + 1):
            print("\n" + "=" * 60)
            print("YOUR JUDGMENT IS NEEDED")
            print("=" * 60)
            print("\nAs the judge, please provide your feedback on both arguments.")
            print("Your insights will guide the next round of the debate.")
            print("Type 'exit' to end the debate early.\n")
            
            feedback = input("Your feedback: ").strip()
            if feedback.lower() == 'exit':
                print("\nEnding debate early at your request...\n")
                break
            
            result = debate_flow.kickoff()
            rounds_completed += 1
            
            if isinstance(result, dict) and 'pro_argument' in result and 'con_argument' in result:
                if hasattr(result['pro_argument'], 'raw'):
                    pro_arg = result['pro_argument'].raw
                else:
                    pro_arg = str(result['pro_argument'])
                
                if hasattr(result['con_argument'], 'raw'):
                    con_arg = result['con_argument'].raw
                else:
                    con_arg = str(result['con_argument'])
                
                # Print formatted arguments
                print(format_argument("PRO ARGUMENT", pro_arg))
                print(format_argument("CON ARGUMENT", con_arg))
            
            # Check if debate should continue
            if isinstance(result, dict) and 'should_continue' in result:
                if not result['should_continue']:
                    break
    
    # Display final debate summary
    if isinstance(result, dict) and 'debate_summary' in result:
        print("\n" + "=" * 80)
        print(result['debate_summary'])
        print("\n" + "=" * 80 + "\n")
    
    print(f"\nDebate completed with {rounds_completed} rounds.")
    print("Thank you for participating in the AI Courtroom Debate!\n")

def main():
    """Main application entry point."""
    try:
        print_header()
        
        # Initialize LLM
        print("Initializing language model...")
        llm = ChatGroq(model="groq/llama-3.3-70b-versatile")
        
        # Get debate parameters
        topic = get_topic()
        rounds = get_rounds()
        
        # Run the debate
        run_debate(topic, rounds, llm)
        
    except KeyboardInterrupt:
        print("\n\nDebate terminated by user. Exiting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()