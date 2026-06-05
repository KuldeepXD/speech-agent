"""
Curated Test Queries for Evaluation Benchmarking.

Each entry contains a query, ground-truth category, expected ailment,
and a gold-standard reference answer for RAGAS evaluation.

Start with 5 representative queries (3 Speech, 2 Feeding).
Scale to 100 once the pipeline is validated.
"""

TEST_QUERIES = [
    {
        "id": "S001",
        "query": "My 4-year-old child has trouble pronouncing the R and S sounds and often substitutes them with W and TH. It has been going on since they started talking.",
        "category": "Speech",
        "expected_ailment": "Articulation Disorders",
        "reference_answer": (
            "Articulation disorders involve difficulty producing specific speech sounds correctly. "
            "For a child substituting R→W and S→TH, a speech-language pathologist should conduct "
            "a comprehensive articulation assessment (e.g., Goldman-Fristoe Test of Articulation). "
            "Treatment typically involves traditional articulation therapy with auditory discrimination "
            "training, phonetic placement techniques, and structured practice moving from isolation "
            "to conversational speech. Given the child's age of 4, R sound errors may still be "
            "developmentally appropriate, but S distortions warrant early intervention."
        ),
    },
    {
        "id": "S002",
        "query": "My husband had a stroke 3 months ago and now he can understand what we say but struggles to form words and sentences. He gets very frustrated.",
        "category": "Speech",
        "expected_ailment": "Aphasia",
        "reference_answer": (
            "This presentation is consistent with Broca's aphasia (non-fluent/expressive aphasia), "
            "where comprehension is relatively preserved but speech production is impaired. "
            "Assessment should include standardized aphasia batteries such as the Western Aphasia "
            "Battery (WAB) or Boston Diagnostic Aphasia Examination. Treatment approaches include "
            "Constraint-Induced Language Therapy (CILT), melodic intonation therapy, and script "
            "training. Augmentative and alternative communication (AAC) strategies should be "
            "introduced to reduce frustration. Early intensive therapy within the first 6 months "
            "post-stroke is associated with better outcomes."
        ),
    },
    {
        "id": "S003",
        "query": "I am a 35-year-old teacher and my voice becomes hoarse and fatigued by the end of each school day. I also feel a scratchy sensation in my throat.",
        "category": "Speech",
        "expected_ailment": "Voice Disorders (Dysphonia)",
        "reference_answer": (
            "Chronic vocal hoarseness and fatigue in a professional voice user suggest a voice "
            "disorder, potentially vocal fold nodules or muscle tension dysphonia. An ENT referral "
            "for laryngoscopy/stroboscopy is essential to visualize the vocal folds. Voice therapy "
            "should focus on vocal hygiene education, resonant voice therapy (Lessac-Madsen), and "
            "semi-occluded vocal tract exercises (straw phonation). Reducing vocal load through "
            "amplification devices in the classroom is recommended. Hydration and avoidance of "
            "phonotraumatic behaviors (throat clearing, yelling) are key lifestyle modifications."
        ),
    },
    {
        "id": "F001",
        "query": "My elderly mother who had a stroke is coughing and choking every time she tries to drink water or thin liquids. She seems fine with thicker foods.",
        "category": "Feeding",
        "expected_ailment": "Post-Stroke Dysphagia",
        "reference_answer": (
            "Post-stroke dysphagia affecting thin liquids with preserved solid tolerance suggests "
            "pharyngeal phase dysphagia with impaired airway protection. A Modified Barium Swallow "
            "Study (MBSS) or Fiberoptic Endoscopic Evaluation of Swallowing (FEES) should be "
            "conducted to assess aspiration risk. Immediate management includes thickening liquids "
            "to nectar or honey consistency per IDDSI guidelines. Therapeutic interventions include "
            "Mendelsohn maneuver, effortful swallow, and chin-tuck posture. Neuromuscular electrical "
            "stimulation (NMES) may supplement traditional therapy. Regular reassessment is needed "
            "as swallowing function can improve during stroke recovery."
        ),
    },
    {
        "id": "F002",
        "query": "My 2-year-old refuses to eat any solid foods and gags whenever we try to introduce anything beyond purees. He has been on pureed food since 6 months.",
        "category": "Feeding",
        "expected_ailment": "Pediatric Feeding Disorder (PFD)",
        "reference_answer": (
            "Persistent food refusal with gagging on textures beyond purees at age 2 suggests "
            "a pediatric feeding disorder with possible sensory-based feeding difficulties. "
            "A multidisciplinary evaluation involving SLP, OT, pediatric gastroenterologist, and "
            "dietitian is recommended. Assessment should evaluate oral motor skills, sensory "
            "processing, and nutritional status. Treatment approaches include Sequential Oral "
            "Sensory (SOS) approach for systematic desensitization to textures, food chaining "
            "from accepted to novel foods, and positive mealtime environment strategies. "
            "Ruling out underlying medical causes (GERD, eosinophilic esophagitis) is essential."
        ),
    },
]
