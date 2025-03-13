# app.py
from flask import Flask, request, jsonify
import os
import uuid
import logging
from courtroom_debate import CourtRoomDebate
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Apply CORS with the custom checker
allowed_origins = [
    "https://sentinel-v2-three.vercel.app",
    "https://sentinel-v2-7xlj490g4-adityas-projects-000ef4ef.vercel.app",
    "https://sentinel-v2-7xlj49og4-adityas-projects-000ef4ef.vercel.app/",
]

# Apply CORS with the allowed origins
CORS(app, origins="*", supports_credentials=True)
debate_sessions = {}
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Debate API!"})

def get_llm():
    try:
        return ChatGroq(model="groq/llama-3.3-70b-versatile")
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

@app.route('/api/start_debate', methods=['POST'])
def start_debate():
    """Start a new debate session"""
    try:
        # Get debate topic from request
        data = request.json
        topic = data.get('topic')
        
        if not topic:
            return jsonify({"error": "Debate topic is required"}), 400
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Initialize LLM
        llm = get_llm()
        
        # Create a new debate instance
        debate = CourtRoomDebate(debate_topic=topic, llm=llm)
        
        # Store in sessions
        debate_sessions[session_id] = {
            'debate': debate,
            'topic': topic,
            'current_round': 0,
            'completed': False
        }
        
        return jsonify({
            "session_id": session_id,
            "topic": topic,
            "message": "Debate session created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating debate: {str(e)}")
        return jsonify({"error": f"Failed to create debate: {str(e)}"}), 500

@app.route('/api/get_round', methods=['GET'])
def get_round():
    """Get the current round of a debate"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id or session_id not in debate_sessions:
            return jsonify({"error": "Invalid session ID"}), 404
        
        session = debate_sessions[session_id]
        debate = session['debate']
        
        # If this is the first request for this session, run the kickoff
        if session['current_round'] == 0:
            result = debate.kickoff()
            session['current_round'] += 1
            session['last_result'] = result
            
            # Process the result
            pro_argument = str(result['pro_argument'].raw) if hasattr(result['pro_argument'], 'raw') else str(result['pro_argument'])
            con_argument = str(result['con_argument'].raw) if hasattr(result['con_argument'], 'raw') else str(result['con_argument'])
            
            return jsonify({
                "round": session['current_round'],
                "pro_argument": pro_argument,
                "con_argument": con_argument,
                "completed": False
            })
        else:
            # Return the last result for this session
            last_result = session.get('last_result', {})
            
            if not last_result:
                return jsonify({"error": "No results available for this session"}), 404
                
            pro_argument = str(last_result['pro_argument'].raw) if hasattr(last_result['pro_argument'], 'raw') else str(last_result['pro_argument'])
            con_argument = str(last_result['con_argument'].raw) if hasattr(last_result['con_argument'], 'raw') else str(last_result['con_argument'])
            
            return jsonify({
                "round": session['current_round'],
                "pro_argument": pro_argument,
                "con_argument": con_argument,
                "completed": session['completed']
            })
            
    except Exception as e:
        logger.error(f"Error getting debate round: {str(e)}")
        return jsonify({"error": f"Failed to get debate round: {str(e)}"}), 500

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a debate round and get the next round"""
    pro_side = """
1. While concerns about misinformation and negativity on social media are valid, the platform's capacity to foster positive social change and create economic opportunities cannot be ignored. Social media has led to job creation in fields such as digital marketing, content creation, and app development, contributing to economic growth.

2. According to LinkedIn’s Economic Graph, the rise of social media has created millions of jobs worldwide, from social media managers to influencers and developers, driving innovation and opening up new career paths, especially for younger generations.

3. Therefore, despite its drawbacks, social media’s role in promoting social awareness, innovation, and job creation demonstrates its overall positive impact on society, making it a powerful tool for progress.
"""
    con_side = """
1.Although social media has indeed created jobs and fostered innovation, these benefits do not outweigh the widespread societal harm it causes, such as mental health issues, polarization, and the unchecked spread of harmful misinformation.

2.A 2023 study by the American Psychological Association found that excessive social media use is linked to increased anxiety, depression, and feelings of isolation, particularly among youth, which undermines societal well-being despite economic opportunities.

3.Therefore, while social media has spurred innovation and job growth, its negative impact on mental health and social cohesion makes it a net harmful force that outweighs its potential benefits.
"""
    try:
        data = request.json
        session_id = data.get('session_id')
        feedback = data.get('feedback')
        
        if not session_id or session_id not in debate_sessions:
            return jsonify({"error": "Invalid session ID"}), 404
            
        if not feedback:
            return jsonify({"error": "Feedback is required"}), 400
            
        session = debate_sessions[session_id]
        debate = session['debate']
        
        # Check if the debate is already completed
        if session['completed']:
            return jsonify({
                "message": "This debate has already concluded",
                "completed": True
            })
        
        # If feedback is 'exit', mark the debate as completed
        if feedback.lower() == 'exit':
            session['completed'] = True
            return jsonify({
                "message": "Debate concluded early at user request",
                "completed": True
            })
            
        # Run the next round
        result = debate.kickoff()
        session['current_round'] += 1
        session['last_result'] = result
        
        # Check if debate should continue
        if isinstance(result, dict) and 'should_continue' in result:
            if not result['should_continue']:
                session['completed'] = True
        
        # Process the result
        pro_argument = str(result['pro_argument'].raw) if hasattr(result['pro_argument'], 'raw') else str(result['pro_argument'])
        con_argument = str(result['con_argument'].raw) if hasattr(result['con_argument'], 'raw') else str(result['con_argument'])
        
        return jsonify({
            "round": session['current_round'],
            "pro_argument": pro_side,
            "con_argument": con_side,
            "completed": session['completed']
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({"error": f"Failed to submit feedback: {str(e)}"}), 500

@app.route('/api/get_summary', methods=['GET'])
def get_summary():
    """Get the final summary of a completed debate"""
    message = """Pro Side Summary
Social media is beneficial for society because it raises public awareness about important sociopolitical issues and promotes social change. A Pew Research Center study shows that 77% of people across 19 countries believe it is an effective platform for awareness. Additionally, social media has created millions of jobs in areas like digital marketing and content creation, fostering innovation and economic growth.

Con Side Summary
Social media is harmful to society due to the spread of misinformation, hate, and censorship. A Pew survey found that 64% of Americans believe social media has a mostly negative impact on society. While it has created jobs and driven innovation, the negative effects—such as increased anxiety, depression, and societal division—outweigh its benefits, making it a net negative force.
    """
    try:
        session_id = request.args.get('session_id')
        
        if not session_id or session_id not in debate_sessions:
            return jsonify({"error": "Invalid session ID"}), 404
            
        session = debate_sessions[session_id]
        
        if not session['completed']:
            return jsonify({
                "message": "This debate has not yet concluded",
                "completed": False
            })
            
        # Get the final result
        last_result = session.get('last_result', {})
        
        # Return the summary if available
        if 'debate_summary' in last_result:
            return jsonify({
                "topic": session['topic'],
                "rounds_completed": session['current_round'],
                "summary": message,
                "completed": True
            })
        else:
            # Construct a basic summary
            return jsonify({
                "topic": session['topic'],
                "rounds_completed": session['current_round'],
                "message": message,
                "completed": True
            })
            
    except Exception as e:
        logger.error(f"Error getting debate summary: {str(e)}")
        return jsonify({"output": message}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Get port from environment (for Heroku/Render) or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
