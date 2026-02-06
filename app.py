import streamlit as st
st.set_page_config(
    page_title="Satyam | ML & GenAI Portfolio",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>
.about-card {
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 28px;
    background: rgba(255,255,255,0.02);
}

.about-name {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.about-sub {
    font-size: 1.1rem;
    opacity: 0.85;
    margin-bottom: 1rem;
}

.about-text {
    font-size: 1rem;
    line-height: 1.6;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Apply font globally */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Page titles */
h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Section subtitles */
h2, h3 {
    font-weight: 600;
    letter-spacing: -0.3px;
}

/* Expander header text */
div[data-testid="stExpander"] > div > div > div > p {
    font-size: 1.05rem;
    font-weight: 500;
}

/* Expander content */
div[data-testid="stExpander"] div[role="region"] {
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Buttons */
button {
    font-weight: 500 !important;
}

/* Reduce visual noise */
hr {
    opacity: 0.3;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "About Me",
        "Featured Projects",
        "Other Works",
        "Skills & Journey",
        "What I Care About",
        "Contact"
    ]
)
st.markdown("""
<style>
.about-card {
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 28px;
    background: rgba(255,255,255,0.02);
}

.about-name {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.about-sub {
    font-size: 1.1rem;
    opacity: 0.85;
    margin-bottom: 1rem;
}

.about-text {
    font-size: 1rem;
    line-height: 1.6;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

from PIL import Image

if page == "About Me":
    profile_img = Image.open("profile.jpeg")  

    st.markdown('<div class="about-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(profile_img, width=160)

    with col2:
        st.markdown('<div class="about-name">👋 Hi, I’m Satyam</div>', unsafe_allow_html=True)
        st.markdown('<div class="about-sub">ML & Generative AI Enthusiast</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="about-text">
            I am a Chemical Engineering undergraduate with a strong interest in
            <b>Machine Learning, Deep Learning, and Generative AI</b>.<br><br>

            I enjoy building <b>end-to-end AI systems</b> — from classical ML models
            to modern <b>RAG and agentic AI architectures</b> — with a strong focus on
            correctness, grounding, and real-world usability.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x")
    with col2:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/satyam-singh-61152a334/")
    with col3:
        st.link_button("📄 Resume", "https://link-to-your-resume.pdf")

    st.markdown('</div>', unsafe_allow_html=True)



elif page == "Featured Projects":
    st.title("⭐ Featured Projects")
    st.markdown(
        "<p style='font-size:1rem; opacity:0.85;'>Click on a project to explore its objective, system design, technical stack, and implementation details.</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 1️⃣ Agentic Chatbot
    with st.expander("End-to-End Agentic Chatbot (LangGraph + Ollama)", expanded=False):
        st.markdown("""
        **Objective**  
        Design and implement a production-style agentic chatbot that goes beyond
        simple prompt–response interactions by incorporating structured workflows,
        persistence, and tool orchestration.

        **Problem it solves**  
        Most chatbots lack memory, structure, and reliability. This system demonstrates
        how to build a **stateful, extensible, and production-oriented AI agent**.

        **How it is built**
        - LangGraph for defining agent states, transitions, and execution flow
        - LangChain for LLM abstraction and tool integration
        - Local Ollama LLM to avoid external API dependencies
        - SQLite database for persistent conversation storage
        - Modular tool calling (weather, news, search, utilities)

        **Unique & trending aspects**
        - Agentic AI architecture with explicit state management  
        - Fully local inference (cost- and privacy-aware design)  
        - Persistent memory across sessions  

        **Tech Stack**
        - Python, LangGraph, LangChain, Ollama, SQLite, Streamlit
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/LangGraph-Chatbot")
        with col2:
            st.link_button("🎥 Demo Video", "https://youtu.be/e3pjGZ_ms3Q")

    # 2️⃣ Chemical Engineering RAG
    with st.expander("Chemical Engineering RAG System", expanded=False):
        st.markdown("""
        **Objective**  
        Build a domain-specific AI assistant for Chemical Engineering that provides
        accurate, grounded answers strictly from academic reference material.

        **Problem it solves**  
        General-purpose LLMs often hallucinate technical details. This system enforces
        **source-grounded responses**, which is critical for engineering reliability.

        **How it is built**
        - Retrieval-Augmented Generation (RAG) pipeline using LangChain
        - Vector database created from Chemical Engineering textbooks and PDFs
        - Strict system prompting to prevent hallucination
        - SQLite-based persistence for conversation continuity
        - Local Ollama LLM to eliminate API limits and cost issues

        **Unique & trending aspects**
        - Domain-specific RAG architecture  
        - Explicit hallucination control  
        - Safety-aware technical explanations  

        **Tech Stack**
        - Python, LangChain, Ollama, Vector Database, SQLite, Streamlit
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/Chemical-AI-Expert")
        with col2:
            st.link_button("🎥 Demo Video", "https://www.youtube.com/watch?v=ZIqISfhQ6tc")

    # 3️⃣ AI Story Generator
    with st.expander("AI Story Generator (Multimodal GenAI)", expanded=False):
        st.markdown("""
        **Objective**  
        Develop a multimodal GenAI application that transforms images into
        coherent, long-form stories with creative control.

        **Problem it solves**  
        Most generative systems are text-only. This project explores **image-based
        reasoning and narrative generation** in a user-friendly application.

        **How it is built**
        - Gemini multimodal model for image understanding
        - Prompt engineering for narrative coherence and style control
        - Support for multiple moods and genres
        - Story narration and PDF export
        - Fully deployed using Streamlit

        **Unique & trending aspects**
        - Multimodal GenAI pipeline  
        - Creative AI with structured output  
        - End-to-end deployed application  

        **Tech Stack**
        - Python, Gemini, Streamlit, PDF generation tools
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/AI-STORY-GENERATOR")
        with col2:
            st.link_button("🌐 Live App", "https://ai-story-generator-satyam.streamlit.app/")

    # 4️⃣ Book Recommender
    with st.expander("Book Recommender System", expanded=False):
        st.markdown("""
        **Objective**  
        Build a personalized book recommendation engine using classical
        machine learning techniques.

        **Problem it solves**  
        Helps users discover relevant books based on preference similarity
        rather than generic popularity rankings. (Users with same preference gets similar recommendations.)

        **How it is built**
        - Collaborative filtering approach
        - Cosine similarity on user–item interaction matrix
        - Data preprocessing and similarity computation
        - End-to-end deployment using Streamlit

        **Unique & trending aspects**
        - Strong ML fundamentals  
        - Personalization-driven recommendations  
        - Fully deployed ML system  

        **Tech Stack**
        - Python, Pandas, NumPy, Scikit-learn, Streamlit
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/Book-recommender-system")
        with col2:
            st.link_button("🌐 Live App", "https://book-recommender-system-bysatyam.streamlit.app/")

elif page == "Other Works":
    st.title("📦 Other Projects & Explorations")
    st.markdown(
        "<p style='font-size:1rem; opacity:0.85;'>Additional projects showcasing experimentation, applied learning, and breadth across ML, CV, and GenAI.</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 1️⃣ Automated Blog Writing Agent
    with st.expander("Automated Blog Writing Agent (LangGraph)", expanded=False):
        st.markdown("""
        **Objective**  
        Build an autonomous agent capable of planning, researching, and writing
        long-form technical blogs with minimal human input.

        **Problem it solves**  
        Writing structured, high-quality technical blogs requires planning and research.
        This agent automates the entire workflow.

        **How it is built**
        - LangGraph for orchestration and step-wise execution
        - Planner agent to design blog sections
        - Research agents for section-wise content gathering
        - Local Ollama LLM (no external API calls)

        **Unique features**
        - Orchestrator-driven workflow
        - Autonomous section planning
        - Fully local, infra-aware GenAI system

        **Tech stack**
        - Python, LangGraph, LangChain, Ollama
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/GraphScribe-Technical-Blog-Writer")
        with col2:
            st.link_button("🎥 Demo Video", "YOUR_DEMO_LINK")

    # 2️⃣ Udaya AI – Morning Assistant
    with st.expander("Udaya AI – Smart Morning Assistant", expanded=False):
        st.markdown("""
        **Objective**  
        Create a personalized AI assistant to help users start their day productively.

        **Problem it solves**  
        Users often rely on multiple apps for weather, news, and planning.
        This app consolidates everything into a single intelligent interface.

        **How it is built**
        - Gemini LLM for natural language reasoning
        - External APIs for weather and news
        - Streamlit-based UI for interaction
        - Rule-based logic for daily planning

        **Unique features**
        - Smart daily planner based on user’s city
        - Live weather and news integration
        - Fully deployed end-to-end application

        **Tech stack**
        - Python, Streamlit, Gemini API
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/Morning-Buddy---Smart-Day-Planner")
        with col2:
            st.link_button("🌐 Live App", "https://morning-buddy---smart-day-planner-satyam.streamlit.app/")

    # 3️⃣ Hand-Gesture Controlled Snake Game
    with st.expander("Hand-Gesture Controlled Snake Game (OpenCV)", expanded=False):
        st.markdown("""
        **Objective**  
        Explore real-time hand tracking and gesture-based interaction using computer vision.

        **Problem it solves**  
        Demonstrates intuitive human–computer interaction without physical controllers.

        **How it is built**
        - OpenCV for real-time video processing
        - Hand landmark detection for finger tracking
        - Gesture-to-action mapping for game control

        **Unique features**
        - Real-time CV pipeline
        - Gesture-based gameplay
        - Visual feedback loop

        **Tech stack**
        - Python, OpenCV
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "YOUR_GITHUB_LINK")
        with col2:
            st.link_button("🎥 Demo Video", "YOUR_DEMO_LINK")

    # 4️⃣ Virtual Painter
    with st.expander("Virtual Painter (OpenCV)", expanded=False):
        st.markdown("""
        **Objective**  
        Build an interactive drawing application controlled entirely by hand gestures.

        **Problem it solves**  
        Enables touch-free drawing and demonstrates practical gesture recognition.

        **How it is built**
        - Hand landmark tracking using OpenCV
        - Gesture detection to switch colors and draw
        - Real-time canvas overlay on video feed

        **Unique features**
        - Touchless drawing
        - Multi-color support via gestures
        - Real-time responsiveness

        **Tech stack**
        - Python, OpenCV
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/Virtual-painter")
        with col2:
            st.link_button("🎥 Demo Video", "https://www.youtube.com/watch?v=iE6sB0k3Wu4")

    # 5️⃣ Laptop Price Predictor
    with st.expander("Laptop Price Predictor (Machine Learning)", expanded=False):
        st.markdown("""
        **Objective**  
        Predict laptop prices based on hardware specifications using machine learning.

        **Problem it solves**  
        Helps users estimate fair pricing based on configuration rather than brand bias.

        **How it is built**
        - Data preprocessing and feature engineering
        - Regression-based ML model
        - End-to-end pipeline deployed with Streamlit

        **Unique features**
        - Clean ML workflow
        - Real-world tabular dataset
        - Deployed prediction system

        **Tech stack**
        - Python, Scikit-learn, Pandas, Streamlit
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub", "https://github.com/Satyam-Singh-x/Laptop-price-predictor")
        with col2:
            st.link_button("🌐 Live App", "https://laptop-price-predictor-bysatyam.streamlit.app/")

elif page == "Skills & Journey":
    st.title("🧠 Skills & Journey")
    st.markdown(
        "<p style='font-size:1rem; opacity:0.85;'>A practical learning journey focused on deep understanding, system design, and real-world AI applications.</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================
    # TECHNICAL SKILLS
    # =========================
    st.subheader("🔧 Technical Skills & Core Concepts")

    st.markdown("""
        **Machine Learning**
        - Supervised & unsupervised learning (regression, classification, clustering)
        - Feature engineering, preprocessing, and pipeline design
        - Model evaluation, bias–variance tradeoff, and error analysis
        - Similarity-based systems (cosine similarity, collaborative filtering)

        **Deep Learning**
        - Neural network fundamentals and backpropagation
        - ANN, CNN, RNN, LSTM architectures and use cases
        - Sequence modeling and representation learning
        - Transformer architecture and attention mechanisms

        **Generative AI**
        - Large Language Models (LLMs) and prompt design
        - Retrieval-Augmented Generation (RAG) pipelines
        - Vector databases, embeddings, and semantic search
        - Multimodal systems (image-to-text reasoning)
        - Working with Hugging Face ecosystem, Gemini, and Ollama

        **Agentic AI**
        - Agent orchestration and workflow design using LangGraph
        - Tool calling, state management, and persistence
        - Multi-agent systems using **Agno** and **CrewAI**
        - Designing collaborative agents (e.g., Data Science Team agents)
        - Planning, execution, and delegation patterns in agents

        **MCP (Model Context Protocol)**
        - Understanding MCP architecture and purpose
        - Building **local MCP servers** (e.g., finance tracking MCP server)
        - Implementing MCP clients using LangChain and LangGraph
        - Context sharing and tool exposure via MCP
        - Designing extensible, model-agnostic agent interfaces

        **Computer Vision**
        - Real-time image and video processing with OpenCV
        - Hand tracking, pose detection, and gesture-based interaction
        - Object detection pipelines using YOLO
        - Applying CV concepts to interactive applications

        **Tools & Deployment**
        - Python, SQL, SQLite
        - Streamlit for rapid ML/AI application deployment
        - FastAPI for model-serving APIs
        - Docker for containerization
        - Git & GitHub for version control and collaboration
        """)

    st.markdown("---")

    # =========================
    # LEARNING JOURNEY
    # =========================
    st.subheader("🚀 My Learning and Application Journey")

    st.markdown("""
        I started my journey by building a **strong foundation in Machine Learning**, 
        focusing not just on algorithms but on the **mathematics, intuition, and evaluation**
        behind each model.

        As my understanding grew, I moved into **Deep Learning**, studying neural network
        architectures in depth and implementing them from first principles to understand
        how and why they work.

        To apply theory in practice, I explored **Computer Vision**, building real-time
        systems using OpenCV for hand tracking, pose detection, object detection, and
        gesture-based interaction.

        My curiosity then led me into **Generative AI**, where I focused on:
        - Building grounded RAG systems
        - Preventing hallucinations in technical domains
        - Designing multimodal applications
        - Understanding embeddings, retrieval, and context injection

        Moving beyond single-agent systems, I specialized in **Agentic AI**, learning how
        to design structured, stateful, and persistent agents using LangGraph. I built
        multi-agent systems using Agno and CrewAI, including collaborative agents that
        simulate a **Data Science Team** with distinct roles and responsibilities.

        To deepen my system-level understanding, I worked extensively with **MCP (Model
        Context Protocol)** — building local MCP servers and MCP clients to enable clean,
        extensible communication between models, tools, and agents.

        Throughout this journey, my focus has remained on:
        - Deep conceptual understanding
        - Clean architecture and system design
        - Applying theory to **real-world, end-to-end AI applications**
        - Building reliable, deployable, and maintainable systems
        """)


elif page == "What I Care About":
    st.title("🎯 How I Build AI Systems")

    st.markdown("""
    • I prefer **structured orchestration** over simple sequential pipelines, using graph-based workflows instead of linear chains  
    • I choose **transfer learning and pretrained models** over training neural networks from scratch, focusing on efficiency and real-world constraints  
    • I design **multi-agent systems** where responsibilities are distributed, rather than forcing a single agent to handle everything  
    • I use **MCP servers and MCP clients** instead of ad-hoc tool calling, keeping model context, tools, and integrations cleanly separated  
    • I prioritize **local LLMs** over API-based models when cost, privacy, reliability, or control matters  
    • I focus on **end-to-end production systems** — from data and models to persistence, APIs, and deployment  
    • I prefer serving ML models through **FastAPI** so they can integrate cleanly with web and frontend teams  
    • I value simplicity, maintainability, and clear system boundaries over unnecessary complexity  
    """)

elif page == "Contact":
    st.title("📬 Contact")

    st.markdown(
        "<p style='font-size:1rem; opacity:0.85;'>Feel free to reach out for opportunities, collaborations, or discussions around ML, GenAI, and AI systems.</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📧 Email**  
        singhsatyam.0912@gmail.com 

        **📞 Phone**  
        +91-8100463060  
        """)

    with col2:
        st.markdown("""
        **💼 LinkedIn**  
        https://www.linkedin.com/in/satyam-singh-61152a334/  

        **🐙 GitHub**  
        https://github.com/Satyam-Singh-x  
        """)

    st.markdown("---")

    st.markdown(
        "<p style='font-size:1rem;'>Thank you for taking the time to explore my work. I appreciate your interest and look forward to connecting.</p>",
        unsafe_allow_html=True
    )


