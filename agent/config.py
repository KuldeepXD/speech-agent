"""
Configuration for the Speech/Feeding Medical AI Agent.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Model Configuration ---
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- Agent Metadata ---
AGENT_NAME = "speech_feeding_agent"
AGENT_DESCRIPTION = (
    "A medical AI agent specialized in Speech-Language Pathology that classifies "
    "user queries into Speech or Feeding categories, identifies medical ailments, "
    "and generates treatment-oriented assessment questions."
)

# --- Category Definitions ---
CATEGORY_SPEECH = "Speech"
CATEGORY_FEEDING = "Feeding"
VALID_CATEGORIES = [CATEGORY_SPEECH, CATEGORY_FEEDING]

# --- Speech Conditions ---
SPEECH_CONDITIONS = [
    "Aphasia",
    "Apraxia of Speech",
    "Dysarthria",
    "Stuttering / Fluency Disorders",
    "Voice Disorders (Dysphonia)",
    "Articulation Disorders",
    "Phonological Disorders",
    "Language Delay",
    "Specific Language Impairment (SLI)",
    "Childhood Apraxia of Speech (CAS)",
    "Resonance Disorders",
    "Cognitive-Communication Disorders",
    "Social Communication Disorder",
    "Selective Mutism",
    "Cleft Palate Speech",
    "Laryngeal Disorders",
    "Spasmodic Dysphonia",
    "Vocal Fold Paralysis",
]

# --- Feeding Conditions ---
FEEDING_CONDITIONS = [
    "Dysphagia (Oral Phase)",
    "Dysphagia (Pharyngeal Phase)",
    "Dysphagia (Esophageal Phase)",
    "Pediatric Feeding Disorder (PFD)",
    "Aspiration",
    "Texture Aversion",
    "Food Refusal",
    "Oral Motor Dysfunction",
    "Failure to Thrive (FTT)",
    "Gastroesophageal Reflux Disease (GERD)",
    "Oropharyngeal Dysphagia",
    "Neurogenic Dysphagia",
    "Post-Stroke Dysphagia",
    "Sensory-Based Feeding Difficulties",
]

# --- Session State Keys ---
STATE_CONVERSATION_HISTORY = "user:conversation_history"
STATE_PREVIOUS_AILMENTS = "user:previous_ailments"
STATE_LAST_CATEGORY = "temp:last_category"
STATE_LAST_AILMENT = "temp:last_ailment"
STATE_INTERACTION_COUNT = "user:interaction_count"
