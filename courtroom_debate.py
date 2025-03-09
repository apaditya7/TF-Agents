from crewai.flow.flow import Flow, listen, start
from crewai import Agent, Task, Crew
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

from crewai_script import google_fact_check, serper_search, combined_research
from langchain_groq import ChatGroq
import re

load_dotenv()

class CourtRoomDebate(Flow):
    """
    A structured courtroom-style debate flow using CrewAI Flow.
    This class manages the entire debate process, from research to argument
    presentation, user feedback, and subsequent iterations.
    """
    
    max_rounds = 5
    
    def __init__(self, 
             debate_topic: str,
             serper_api_key: Optional[str] = None,
             llm = None,
             factcheck_api_key: Optional[str] = None):
        super().__init__()
        
        # Store API keys
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.factcheck_api_key = factcheck_api_key or os.getenv("GOOGLE_FACTCHECK_API_KEY")
        
        # Set the LLM BEFORE initializing tools and agents
        self.llm = llm
        
        # Initialize state
        self.state["topic"] = debate_topic
        self.state["current_round"] = 0
        self.state["debate_history"] = []
        self.state["should_continue"] = True
        
        # Initialize tools and agents
        self._initialize_tools()
        self._initialize_agents()
        
        print(f"Initialized debate on topic: {debate_topic}")
        print(f"Maximum rounds: {self.max_rounds}")
    
    def _initialize_tools(self):
        """Initialize research tools using decorated tool functions."""

        self.search_tool = serper_search
        self.factcheck_tool = google_fact_check
        self.combined_tool = combined_research
    
    def _initialize_agents(self):
        """Initialize all debate participants."""
        if self.llm is None:
            self.llm = ChatGroq(model="groq/llama-3.3-70b-versatile")
            
        # Pass the LLM to each agent explicitly
        self.pro_researcher = Agent(
            name="Pro Research Specialist",
            role="Evidence Finder",
            goal="Find compelling, factual evidence supporting the proposition",
            backstory="""You are an expert researcher with skills in finding credible sources 
                        and evidence to support arguments. You specialize in thorough investigation
                        and presenting only verified information.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool, self.factcheck_tool],
            llm=self.llm 
        )
        
        self.pro_debater = Agent(
            name="Pro Debate Advocate",
            role="Proposition Defender",
            goal="Construct compelling, logical arguments supporting the proposition",
            backstory="""You are a skilled debater who excels at crafting persuasive 
                      arguments based on evidence. You have expertise in rhetoric, 
                      logical reasoning, and persuasive communication.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm 

        )
        
        # Con Side
        self.con_researcher = Agent(
            name="Con Research Specialist",
            role="Counter-Evidence Finder",
            goal="Find compelling, factual evidence opposing the proposition",
            backstory="""You are an expert researcher with skills in finding credible sources 
                      and evidence to counter arguments. You specialize in thorough investigation
                      and presenting only verified information.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool, self.factcheck_tool],
            llm=self.llm 
        )
        
        self.con_debater = Agent(
            name="Con Debate Advocate",
            role="Proposition Challenger",
            goal="Construct compelling, logical arguments opposing the proposition",
            backstory="""You are a skilled debater who excels at crafting persuasive 
                      counter-arguments based on evidence. You have expertise in rhetoric, 
                      logical reasoning, and persuasive communication.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm 
        )
        
        # Judge Agent
        self.judge_agent = Agent(
            name="Debate Judge",
            role="Feedback Interpreter",
            goal="Translate user feedback into structured, actionable guidance for debaters",
            backstory="""You are an impartial judge with expertise in debate evaluation.
                      Your role is to interpret user feedback and translate it into clear,
                      specific guidance that helps debaters improve their arguments.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm 
        )
    def generate_self_argument(self, topic, position="supporting"):
        """Generate a self-generated argument using the LLM."""
        from langchain_core.messages import HumanMessage
        
        prompt_text = f"""
        You are an expert researcher. Since no external sources were found, generate a logical argument {position} the proposition: "{topic}"
        
        Your response should include:
        1. A clear claim {position} the proposition.
        2. Logical reasoning or evidence to support the claim.
        3. A brief conclusion.
        
        Format your response as:
        
        ## SOURCE
        Self-generated argument
        
        ## KEY POINT
        [Your logical argument, 2-3 sentences]
        
        ## RECOMMENDATION
        [One sentence on how to use this argument effectively]
        """
        
        # Create a proper message object instead of passing a string
        message = HumanMessage(content=prompt_text)
        
        # Call the LLM with a proper message object
        response = self.llm.invoke([message])
        
        # Extract the content from the response
        return response.content
    def create_pro_research_task(self, round_num=1, feedback=None, con_argument=None):
        """Create a task for the Pro researcher to find supporting evidence."""
        task_description = f"""
        Find ONE piece of evidence supporting: "{self.state['topic']}"
        
        Your task:
        1. First, try to find a single credible source using the Web Search tool.
        2. If the Web Search tool returns no results, generate your own logical argument.
        3. Format your response as shown below.
        
        ## SOURCE
        [Website name] - [URL] (or "Self-generated argument" if no source is found)
        
        ## KEY POINT
        [Brief summary of the supporting evidence or argument, 2-3 sentences maximum]
        
        ## RECOMMENDATION
        [One sentence on how to use this evidence effectively]
        """
        
        return Task(
            description=task_description,
            agent=self.pro_researcher,
            expected_output="Brief research supporting the proposition"
        )

    def create_pro_debate_task(self, research, round_num=1, feedback=None, con_argument=None):
        """Create a task for the Pro debater to construct arguments."""
        task_description = f"""
        Create a brief argument supporting: "{self.state["topic"]}"
        
        RESEARCH: {research}
        
        Your task:
        1. Write a short, compelling argument using the provided research
        2. Keep your response under 200 words total
        3. Use this simple format:
        
        ## CLAIM
        [One clear statement supporting the proposition]
        
        ## EVIDENCE
        [The evidence from the research that supports your claim]
        
        ## CONCLUSION
        [A brief, powerful closing sentence]
        """
        
        return Task(
            description=task_description,
            agent=self.pro_debater,
            expected_output="Brief argument supporting the proposition"
        )

    def create_con_research_task(self, pro_argument, round_num=1, feedback=None):
        """Create a task for the Con researcher to find opposing evidence."""
        task_description = f"""
        Find ONE piece of evidence opposing: "{self.state['topic']}"
        
        PRO ARGUMENT: {pro_argument}
        
        Your task:
        1. First, try to find a single credible source using the Web Search tool.
        2. If the Web Search tool returns no results, generate your own logical counter-argument.
        3. Format your response as shown below.
        
        ## SOURCE
        [Website name] - [URL] (or "Self-generated argument" if no source is found)
        
        ## KEY COUNTER-POINT
        [Brief summary of the opposing evidence or argument, 2-3 sentences maximum]
        
        ## RECOMMENDATION
        [One sentence on how to use this evidence effectively]
        """
        
        return Task(
            description=task_description,
            agent=self.con_researcher,
            expected_output="Brief research opposing the proposition"
        )

    def create_con_debate_task(self, research, pro_argument, round_num=1, feedback=None):
        """Create a task for the Con debater to construct counter-arguments."""
        task_description = f"""
        Create a brief counter-argument opposing: "{self.state["topic"]}"
        
        RESEARCH: {research}
        PRO ARGUMENT: {pro_argument}
        
        Your task:
        1. Write a short, compelling counter-argument
        2. Keep your response under 200 words total
        3. Use this simple format:
        
        ## COUNTER-CLAIM
        [One clear statement opposing the proposition]
        
        ## EVIDENCE
        [The evidence from the research that supports your counter-claim]
        
        ## CONCLUSION
        [A brief, powerful closing sentence]
        """
        
        return Task(
            description=task_description,
            agent=self.con_debater,
            expected_output="Brief counter-argument opposing the proposition"
        )

    def create_judge_task(self, pro_argument, con_argument, user_feedback):
        """Create a task for the Judge agent to process user feedback."""
        task_description = f"""
        Summarize the user's feedback on the debate about: "{self.state["topic"]}"
        
        USER FEEDBACK: {user_feedback}
        
        Your task:
        1. Extract the main points from the user's feedback
        2. Create a brief, balanced summary for both sides
        3. Use this simple format:
        
        ## FOR PRO SIDE
        - Strength: [One key strength noted by the user]
        - Improvement: [One area to improve]
        
        ## FOR CON SIDE
        - Strength: [One key strength noted by the user]
        - Improvement: [One area to improve]
        """
        
        return Task(
            description=task_description,
            agent=self.judge_agent,
            expected_output="Brief feedback summary"
        )
    
    @start()
    def initialize_debate(self):
        """Start the debate process with the opening round."""
        print(f"\n===== DEBATE TOPIC: {self.state['topic']} =====\n")
        print("Beginning debate process. Round 1 starting...\n")
        
        # Store initial round in state
        self.state["current_round"] = 1
        
        # Return the topic to begin the flow
        return self.state["topic"]
    

    
    @listen(initialize_debate)
    def run_pro_side_round1(self, topic):
        """Execute the Pro side research and argument for round 1."""
        print("🔍 PRO SIDE: Researching and formulating opening argument...\n")
        
        # Create tasks for the Pro side
        research_task = self.create_pro_research_task(round_num=1)
        
        # Run the Pro researcher task
        pro_research_crew = Crew(
            agents=[self.pro_researcher],
            tasks=[research_task],
            verbose=True
        )
        
        try:
            research_results = pro_research_crew.kickoff()
        except Exception as e:
            # Fallback to self-generated argument if the task fails
            print(f"⚠️ Web Search failed. Falling back to self-generated argument. Error: {str(e)}")
            research_results = self.generate_self_argument(topic, position="supporting")
        
        # Store research in state
        self.state["pro_research_r1"] = research_results
        
        # Create debate task with research results
        debate_task = self.create_pro_debate_task(research=research_results, round_num=1)
        
        # Run the Pro debater task
        pro_debate_crew = Crew(
            agents=[self.pro_debater],
            tasks=[debate_task],
            verbose=True
        )
        pro_argument = pro_debate_crew.kickoff()
        
        # Store argument in state
        self.state["pro_argument_r1"] = pro_argument
        
        return pro_argument
    
    @listen(run_pro_side_round1)
    def run_con_side_round1(self, pro_argument):
        """Execute the Con side research and argument for round 1."""
        print("🔍 CON SIDE: Researching and formulating counter-argument...\n")
        
        # Create tasks for the Con side
        research_task = self.create_con_research_task(pro_argument=pro_argument, round_num=1)
        
        # Run the Con researcher task
        con_research_crew = Crew(
            agents=[self.con_researcher],
            tasks=[research_task],
            verbose=True
        )
        
        try:
            research_results = con_research_crew.kickoff()
        except Exception as e:
            # Fallback to self-generated argument if the task fails
            print(f"⚠️ Web Search failed. Falling back to self-generated argument. Error: {str(e)}")
            research_results = self.generate_self_argument(self.state["topic"], position="opposing")
        
        # Store research in state
        self.state["con_research_r1"] = research_results
        
        # Create debate task with research results
        debate_task = self.create_con_debate_task(
            research=research_results,
            pro_argument=pro_argument,
            round_num=1
        )
        
        # Run the Con debater task
        con_debate_crew = Crew(
            agents=[self.con_debater],
            tasks=[debate_task],
            verbose=True
        )
        con_argument = con_debate_crew.kickoff()
        
        # Store argument in state
        self.state["con_argument_r1"] = con_argument
        
        # Return both arguments for user judgment
        return {
            "pro_argument": pro_argument,
            "con_argument": con_argument
        }
    
    @listen(run_con_side_round1)
    def collect_user_judgment(self, arguments):
        """Present arguments to user and collect feedback."""
        pro_argument = arguments["pro_argument"]
        con_argument = arguments["con_argument"]
        
        print("\n===== ARGUMENTS ROUND 1 =====\n")
        print("===== PRO ARGUMENT =====")
        print(pro_argument)
        print("\n===== CON ARGUMENT =====")
        print(con_argument)
        print("\n===== YOUR JUDGMENT =====")
        print("Please provide your feedback on both arguments.")
        print("Type 'exit' to end the debate, or provide feedback to continue.")
        
        # Get user input
        user_feedback = input("\nYour feedback: ")
        
        # Check for exit command
        if user_feedback.lower().strip() == "exit":
            self.state["should_continue"] = False
            return {
                "user_feedback": "The debate has been ended by the user.",
                "pro_argument": pro_argument,
                "con_argument": con_argument,
                "continue": False
            }
        
        # Store user feedback
        self.state["user_feedback_r1"] = user_feedback
        
        return {
            "user_feedback": user_feedback,
            "pro_argument": pro_argument,
            "con_argument": con_argument,
            "continue": True
        }
    
    @listen(collect_user_judgment)
    def process_judgment(self, feedback_data):
        """Process user feedback into structured guidance for debaters."""
        if not feedback_data["continue"]:
            return self.conclude_debate(feedback_data)
        
        print("\n⚖️ JUDGE: Processing feedback...\n")
        
        # Create the judge task
        task = self.create_judge_task(
            pro_argument=feedback_data["pro_argument"],
            con_argument=feedback_data["con_argument"],
            user_feedback=feedback_data["user_feedback"]
        )
        
        # Run the judge task
        judge_crew = Crew(
            agents=[self.judge_agent],
            tasks=[task],
            verbose=True
        )
        processed_feedback = judge_crew.kickoff()
        
        # Store processed feedback
        self.state["processed_feedback_r1"] = processed_feedback
        
        return processed_feedback
    
    @listen(process_judgment)
    def determine_next_round(self, processed_feedback):
        """Determine whether to continue to the next round."""
        current_round = self.state["current_round"]
        next_round = current_round + 1
        self.state["current_round"] = next_round

        # Check if we should continue
        if next_round > self.max_rounds:
            self.state["should_continue"] = False
            return self.conclude_debate({
                "processed_feedback": processed_feedback,
                "round": current_round
            })
        
        print(f"\n===== BEGINNING ROUND {next_round} =====\n")
        self.state["should_continue"] = True
        return self.start_next_round(processed_feedback)
    
    def start_next_round(self, processed_feedback):
        """Start the next round of debate with the processed feedback."""
        current_round = self.state["current_round"]
        
        # Create tasks that explicitly reference previous round's arguments and feedback
        pro_research_task = self.create_pro_research_task(
            round_num=current_round,
            feedback=processed_feedback,
            con_argument=self.state.get(f"con_argument_r{current_round-1}")
        )
        
        # Run Pro Research Crew
        pro_research_crew = Crew(
            agents=[self.pro_researcher],
            tasks=[pro_research_task],
            verbose=True
        )
        
        try:
            research_results = pro_research_crew.kickoff()
        except Exception as e:
            # Fallback to self-generated argument if the task fails
            print(f"⚠️ Pro Researcher: Web Search failed. Falling back to self-generated argument. Error: {str(e)}")
            research_results = self.generate_self_argument(self.state["topic"], position="supporting")
        
        # Store research in state
        self.state[f"pro_research_r{current_round}"] = research_results

        pro_debater_task = self.create_pro_debate_task(
            research=research_results,
            round_num=current_round,
            feedback=processed_feedback,
            con_argument=self.state.get(f"con_argument_r{current_round-1}")
        )

        pro_debater_crew = Crew(
            agents=[self.pro_debater],
            tasks=[pro_debater_task],
            verbose=True
        )
        pro_argument = pro_debater_crew.kickoff()
        
        self.state[f"pro_debate_r{current_round}"] = pro_argument

        con_research_task = self.create_con_research_task(
            pro_argument=pro_argument,
            round_num=current_round,
            feedback=processed_feedback
        )

        con_research_crew = Crew(
            agents=[self.con_researcher],
            tasks=[con_research_task],
            verbose=True
        )
        
        try:
            research_results = con_research_crew.kickoff()
        except Exception as e:
            print(f"⚠️ Con Researcher: Web Search failed. Falling back to self-generated argument. Error: {str(e)}")
            research_results = self.generate_self_argument(self.state["topic"], position="opposing")
        
        self.state[f"con_research_r{current_round}"] = research_results

        con_debater_task = self.create_con_debate_task(
            research=research_results,
            pro_argument=pro_argument,
            round_num=current_round,
            feedback=processed_feedback
        )

        con_debater_crew = Crew(
            agents=[self.con_debater],
            tasks=[con_debater_task],
            verbose=True
        )
        con_argument = con_debater_crew.kickoff()
        
        self.state[f"con_debate_r{current_round}"] = con_argument
        
        judge_task = self.create_judge_task(
            pro_argument=pro_argument,
            con_argument=con_argument,
            user_feedback=processed_feedback
        )

        judge_crew = Crew(
            agents=[self.judge_agent],
            tasks=[judge_task],
            verbose=True
        )
        processed_feedback = judge_crew.kickoff()

        self.state[f"processed_feedback_r{current_round}"] = processed_feedback

        return {
            "pro_argument": pro_argument,
            "con_argument": con_argument,
            "should_continue": True 
        }
    
    def conclude_debate(self, final_data):
        """Conclude the debate and provide summary."""
        print("\n===== DEBATE CONCLUDED =====\n")
        print(f"Completed {self.state['current_round']-1} rounds of debate.")
        
        # Build comprehensive debate summary
        summary = f"""
    ===== FINAL DEBATE SUMMARY =====

    TOPIC: "{self.state['topic']}"
    ROUNDS COMPLETED: {self.state['current_round']-1}

    KEY PRO ARGUMENTS:
    """
        
        # Add pro arguments from each round
        for round_num in range(1, self.state["current_round"]):
            pro_arg = self.state.get(f"pro_debate_r{round_num}", "No argument provided")
            if pro_arg and isinstance(pro_arg, str):
                # Extract just the CLAIM and EVIDENCE parts to keep it concise
                claim_match = re.search(r"##\s*CLAIM\s*(.*?)(?=##|$)", pro_arg, re.DOTALL)
                if claim_match:
                    claim = claim_match.group(1).strip()
                    summary += f"- Round {round_num}: {claim[:150]}...\n"
        
        summary += "\nKEY CON ARGUMENTS:\n"
        
        # Add con arguments from each round
        for round_num in range(1, self.state["current_round"]):
            con_arg = self.state.get(f"con_debate_r{round_num}", "No argument provided") 
            if con_arg and isinstance(con_arg, str):
                # Extract just the COUNTER-CLAIM parts to keep it concise
                claim_match = re.search(r"##\s*COUNTER-CLAIM\s*(.*?)(?=##|$)", con_arg, re.DOTALL)
                if claim_match:
                    claim = claim_match.group(1).strip()
                    summary += f"- Round {round_num}: {claim[:150]}...\n"
        
        summary += "\nDEBATE OUTCOME:\n"
        summary += "This debate explored multiple perspectives on whether AI will have a net positive impact on society."
        summary += " The Pro side emphasized economic benefits and quality of life improvements, while the Con side"
        summary += " highlighted potential risks including job displacement, bias, and social inequality."
        
        summary += "\n\nFURTHER EXPLORATION:\n"
        summary += "Future debates might explore specific AI regulations, ethical frameworks, or focus on particular"
        summary += " sectors where AI's impact might be most profound such as healthcare, transportation, or education."
        
        # Print the summary
        print(summary)
        
        # Return final arguments and status
        return {
            "pro_argument": self.state.get(f"pro_debate_r{self.state['current_round']-1}", "No final argument"),
            "con_argument": self.state.get(f"con_debate_r{self.state['current_round']-1}", "No final argument"),
            "should_continue": False,
            "status": "completed",
            "rounds_completed": self.state["current_round"]-1,
            "topic": self.state["topic"],
            "final_feedback": final_data.get("processed_feedback", "No final feedback available"),
            "debate_summary": summary
        }

    def create_summary_task(self):
        """Create a task for generating a comprehensive debate summary."""
        
        # Collect all arguments and feedback from the debate
        debate_history = self._collect_debate_history()
        
        task_description = f"""
        Create a comprehensive summary of the debate on: "{self.state["topic"]}"
        
        DEBATE HISTORY:
        {debate_history}
        
        Your task:
        1. Analyze the entire debate across all rounds
        2. Identify the strongest arguments from both sides
        3. Highlight key areas of agreement and disagreement
        4. Assess the quality of evidence presented
        5. Provide suggestions for further exploration
        
        Format your summary as follows:
        
        ## DEBATE OVERVIEW
        [Brief overview of the debate topic and progression across rounds]
        
        ## KEY PRO ARGUMENTS
        - [First key pro argument with evidence]
        - [Second key pro argument with evidence]
        
        ## KEY CON ARGUMENTS
        - [First key con argument with evidence]
        - [Second key con argument with evidence]
        
        ## POINTS OF AGREEMENT
        [Areas where both sides converged or acknowledged similar points]
        
        ## POINTS OF CONTENTION
        [Core disagreements that remained unresolved]
        
        ## QUALITY ASSESSMENT
        [Evaluation of evidence quality and reasoning from both sides]
        
        ## FURTHER EXPLORATION
        [Suggestions for how this debate could be expanded or refined]
        """
        
        return Task(
            description=task_description,
            agent=self.judge_agent,
            expected_output="Comprehensive debate summary"
        )

    def _collect_debate_history(self):
        """Collect all arguments and feedback from all debate rounds."""
        history = ""
        
        for round_num in range(1, self.state["current_round"]):
            history += f"\n===== ROUND {round_num} =====\n"
            
            # Add Pro arguments
            pro_arg = self.state.get(f"pro_debate_r{round_num}", "No argument provided")
            history += f"\nPRO ARGUMENT (Round {round_num}):\n{pro_arg}\n"
            
            # Add Con arguments
            con_arg = self.state.get(f"con_debate_r{round_num}", "No argument provided")
            history += f"\nCON ARGUMENT (Round {round_num}):\n{con_arg}\n"
            
            # Add feedback
            feedback = self.state.get(f"processed_feedback_r{round_num}", "No feedback provided")
            history += f"\nFEEDBACK (Round {round_num}):\n{feedback}\n"
        
        return history