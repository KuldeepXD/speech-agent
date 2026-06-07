"""
Curated Test Queries for Evaluation Benchmarking.

Each entry contains a query, ground-truth category, expected difficulty level,
and a gold-standard reference answer for RAGAS evaluation.

Queries sourced from Documents/sample.csv — 25 Speech-Language Pathology questions
across Easy, Medium, and Hard difficulty levels.
"""

TEST_QUERIES = [
    # ── Easy Level ─────────────────────────────────────────────────────
    {
        "id": "E001",
        "query": "How would you start intervention for a child with speech delay?",
        "category": "Speech",
        "expected_ailment": "Speech Delay",
        "reference_answer": (
            "The initial approach to speech delay should include ruling out hearing loss, "
            "assessing receptive and expressive language abilities, and obtaining a detailed "
            "developmental history. Intervention should focus on increasing communication "
            "opportunities through play, modeling simple language, and training parents to use "
            "language stimulation techniques during everyday activities. Make the environmental "
            "learning age appropriate."
        ),
    },
    {
        "id": "E002",
        "query": "What would you do for oro-motor weakness?",
        "category": "Speech",
        "expected_ailment": "Oro-Motor Weakness",
        "reference_answer": (
            "Management of oro-motor weakness should primarily focus on improving postural "
            "stability and promoting functional oral movements during speech and feeding "
            "activities. Therapy should aim to enhance lip closure, tongue mobility, jaw "
            "stability, and chewing skills through age-appropriate and meaningful tasks rather "
            "than isolated exercises. Isometric and isotonic exercises should be incorporated. "
            "Feeding activities and sensory input can also be incorporated to facilitate better "
            "oral control."
        ),
    },
    {
        "id": "E003",
        "query": "What would you do to reduce drooling?",
        "category": "Speech",
        "expected_ailment": "Drooling",
        "reference_answer": (
            "Reducing drooling involves addressing the underlying factors contributing to poor "
            "saliva management. Improving posture, breathing and head control, increasing oral "
            "sensory awareness, promoting lip closure, and encouraging frequent swallowing are "
            "important components of intervention."
        ),
    },
    {
        "id": "E004",
        "query": "How would you start intervention for a non-verbal child with autism?",
        "category": "Speech",
        "expected_ailment": "Non-Verbal Autism",
        "reference_answer": (
            "The initial focus should be on developing communication intent and establishing "
            "joint attention rather than immediately targeting speech production. Check for all "
            "the primitive reflex integration. Functional communication can be encouraged through "
            "play-based interactions, gestures, sign language, picture exchange systems, or "
            "augmentative and alternative communication (AAC) methods."
        ),
    },
    {
        "id": "E005",
        "query": "What would you do for a child who stores food in the cheeks?",
        "category": "Feeding",
        "expected_ailment": "Oral Pocketing",
        "reference_answer": (
            "Intervention should aim to improve oral awareness and chewing efficiency. Small "
            "bite sizes, alternating solids with liquids, encouraging tongue lateralization, and "
            "checking for residual food after meals are helpful strategies. Therapy should focus "
            "on developing effective chewing patterns and safe oral clearance during feeding."
        ),
    },
    {
        "id": "E006",
        "query": "How would you start therapy for a late talker?",
        "category": "Speech",
        "expected_ailment": "Late Talker",
        "reference_answer": (
            "Therapy for a late talker should be child-centered and based on the child's "
            "interests. The clinician should model simple words and short phrases, expand the "
            "child's attempts to communicate, and create opportunities that encourage requesting "
            "and interaction. Parents should be actively involved and taught language facilitation "
            "strategies to use during everyday routines."
        ),
    },
    {
        "id": "E007",
        "query": "How would you improve lip closure?",
        "category": "Speech",
        "expected_ailment": "Poor Lip Closure",
        "reference_answer": (
            "Improving lip closure requires addressing posture and oral control while "
            "incorporating functional activities. Speech sounds requiring bilabial closure, "
            "drinking through straws, and feeding tasks that encourage lip activity can be used "
            "to facilitate better lip strength and coordination. Therapy should emphasize "
            "functional use rather than isolated strengthening exercises."
        ),
    },
    {
        "id": "E008",
        "query": "What would you do for poor chewing skills?",
        "category": "Feeding",
        "expected_ailment": "Poor Chewing Skills",
        "reference_answer": (
            "Management should begin with textures that the child can safely handle and gradually "
            "progress to more challenging consistencies. Tool based therapy can be used. Therapy "
            "should encourage bilateral chewing, jaw stability, and tongue lateralization while "
            "ensuring proper positioning during meals. Caregiver education is important to promote "
            "consistent practice during daily feeding activities."
        ),
    },
    # ── Medium Level ───────────────────────────────────────────────────
    {
        "id": "M001",
        "query": "What would you do for a child who refuses textured foods?",
        "category": "Feeding",
        "expected_ailment": "Textured Food Refusal",
        "reference_answer": (
            "Treatment should involve gradual exposure to different textures while maintaining a "
            "positive feeding environment. Starting with preferred foods and slowly progressing to "
            "more complex textures can help reduce anxiety and sensory defensiveness. Food chaining "
            "can be used. Desensitization techniques and food exploration activities may also "
            "facilitate acceptance of new foods."
        ),
    },
    {
        "id": "M002",
        "query": "What would you do for tongue thrust during swallowing?",
        "category": "Feeding",
        "expected_ailment": "Tongue Thrust",
        "reference_answer": (
            "Management should focus on establishing an appropriate resting tongue posture and "
            "promoting mature swallowing patterns. Therapy may include activities to improve lip "
            "closure and oral awareness, along with exercises aimed at coordinating tongue movements "
            "during swallowing. Regular sucking through the straw will reduce tongue thrust."
        ),
    },
    {
        "id": "M003",
        "query": "How would you manage childhood apraxia of speech?",
        "category": "Speech",
        "expected_ailment": "Childhood Apraxia of Speech",
        "reference_answer": (
            "Intervention for childhood apraxia of speech should emphasize motor-based approaches "
            "with frequent and repetitive practice. Non speech movements can be practiced. Therapy "
            "should provide multisensory cues and focus on improving movement sequences for speech "
            "production rather than isolated sounds. Depending on the severity, augmentative and "
            "alternative communication may be introduced to support functional communication."
        ),
    },
    {
        "id": "M004",
        "query": "What would you do if a child has poor attention during therapy sessions?",
        "category": "Speech",
        "expected_ailment": "Poor Attention in Therapy",
        "reference_answer": (
            "The clinician should modify the therapy environment to minimize distractions and use "
            "motivating activities that match the child's interests. Sessions should include short "
            "tasks with frequent breaks and opportunities for movement. Reinforcement and a "
            "predictable routine can help maintain attention and participation."
        ),
    },
    {
        "id": "M005",
        "query": "How would you improve joint attention in children with autism?",
        "category": "Speech",
        "expected_ailment": "Joint Attention Deficit (Autism)",
        "reference_answer": (
            "Joint attention can be facilitated through face-to-face interactions, turn-taking "
            "games, imitation activities, and shared book reading. The therapist should follow the "
            "child's interests and create opportunities for shared experiences. Parent involvement "
            "is essential to generalize these skills across different environments."
        ),
    },
    {
        "id": "M006",
        "query": "How would you facilitate first words in a child with delayed speech?",
        "category": "Speech",
        "expected_ailment": "Delayed Speech",
        "reference_answer": (
            "Intervention should focus on increasing communication opportunities and modeling "
            "meaningful words during daily routines. Repetitive play activities, songs, and "
            "routines can be used to encourage vocalizations and word approximations. Positive "
            "reinforcement should be provided for all attempts to communicate."
        ),
    },
    {
        "id": "M007",
        "query": "What would you do for excessive gagging while introducing solids?",
        "category": "Feeding",
        "expected_ailment": "Excessive Gagging",
        "reference_answer": (
            "Management should include gradual progression of textures, oral sensory stimulation, "
            "and opportunities for self-feeding. Caregivers should avoid force feeding and provide "
            "positive experiences with food. The child's sensory and motor abilities should be "
            "considered while introducing new textures at an appropriate pace."
        ),
    },
    {
        "id": "M008",
        "query": "What would you do if a patient coughs while drinking thin liquids?",
        "category": "Feeding",
        "expected_ailment": "Dysphagia (Thin Liquid Aspiration)",
        "reference_answer": (
            "The first priority is to ensure swallowing safety. Further assessment should be "
            "carried out to determine the cause of the difficulty. Appropriate texture "
            "modifications, safe swallowing strategies, and postural adjustments may be "
            "recommended. Instrumental assessments such as VFSS or FEES may be required to guide "
            "treatment planning."
        ),
    },
    {
        "id": "M009",
        "query": "How would you improve respiratory support for speech?",
        "category": "Speech",
        "expected_ailment": "Poor Respiratory Support for Speech",
        "reference_answer": (
            "Intervention should focus on improving posture, respiratory control, and coordination "
            "between breathing and phonation. Activities should gradually progress from sustained "
            "phonation to phrase and sentence production while emphasizing efficient breath support "
            "during speech."
        ),
    },
    # ── Hard Level ─────────────────────────────────────────────────────
    {
        "id": "H001",
        "query": "What are the priorities in dysphagia management?",
        "category": "Feeding",
        "expected_ailment": "Dysphagia",
        "reference_answer": (
            "The primary goals of dysphagia management are to maintain airway safety, ensure "
            "adequate nutrition and hydration, and improve swallowing efficiency. Intervention may "
            "involve texture modifications, compensatory strategies, therapeutic exercises, "
            "caregiver education, and instrumental assessments when necessary. A multidisciplinary "
            "approach is often essential."
        ),
    },
    {
        "id": "H002",
        "query": "How would you improve tongue lateralization?",
        "category": "Feeding",
        "expected_ailment": "Poor Tongue Lateralization",
        "reference_answer": (
            "Tongue lateralization can be facilitated through functional feeding activities that "
            "encourage movement of food from one side of the mouth to the other. Appropriate food "
            "placement, bilateral chewing opportunities, and sensory stimulation may help improve "
            "tongue mobility and oral awareness. Therapy should emphasize natural movements during "
            "feeding rather than isolated non-speech exercises."
        ),
    },
    {
        "id": "H003",
        "query": "What are the management priorities for global aphasia?",
        "category": "Speech",
        "expected_ailment": "Global Aphasia",
        "reference_answer": (
            "Management should focus on maximizing functional communication and participation in "
            "daily activities. Alternative methods such as gestures, pictures, communication "
            "boards, and augmentative communication systems may be introduced. Family members "
            "should be educated on effective communication strategies to support the individual "
            "across different settings."
        ),
    },
    {
        "id": "H004",
        "query": "What are the goals of feeding therapy in children with cerebral palsy?",
        "category": "Feeding",
        "expected_ailment": "Feeding Difficulties in Cerebral Palsy",
        "reference_answer": (
            "Feeding therapy should aim to promote safe swallowing, improve oral motor control, "
            "enhance nutritional intake, and support participation during mealtimes. Proper "
            "positioning, texture modifications, caregiver training, and collaboration with other "
            "healthcare professionals are essential components of management."
        ),
    },
    {
        "id": "H005",
        "query": "How would you improve language comprehension in children with receptive language disorder?",
        "category": "Speech",
        "expected_ailment": "Receptive Language Disorder",
        "reference_answer": (
            "Intervention should focus on simplifying language, using visual supports, and "
            "providing multimodal input to facilitate understanding. Repetition, gestures, "
            "pictures, and structured activities can help improve comprehension skills. Therapy "
            "should gradually increase the complexity of language as the child's abilities develop."
        ),
    },
    {
        "id": "H006",
        "query": "What would you do if a child bites the spoon and cannot remove food from it?",
        "category": "Feeding",
        "expected_ailment": "Poor Jaw Grading / Spoon Feeding Difficulty",
        "reference_answer": (
            "Management should focus on improving jaw grading and oral coordination during "
            "feeding. The clinician should use shallow spoon placement and provide controlled "
            "amounts of food to encourage appropriate lip activity and tongue movements. Functional "
            "feeding experiences should be used to promote efficient bolus removal."
        ),
    },
    {
        "id": "H007",
        "query": "How would you manage sensory food aversion?",
        "category": "Feeding",
        "expected_ailment": "Sensory Food Aversion",
        "reference_answer": (
            "Treatment should involve gradual desensitization and systematic exposure to different "
            "foods. Food chaining techniques and sensory-based approaches can be used to expand "
            "the child's food repertoire while maintaining a positive and pressure-free feeding "
            "environment. Active caregiver participation is essential for successful outcomes."
        ),
    },
    {
        "id": "H008",
        "query": "What would you do when speech progress is very limited?",
        "category": "Speech",
        "expected_ailment": "Limited Speech Progress",
        "reference_answer": (
            "When speech progress is slow, the primary focus should be on ensuring effective "
            "communication rather than waiting for speech to emerge. Augmentative and alternative "
            "communication systems may be introduced to support functional communication, while "
            "speech and language therapy continues to target verbal skills. Family members should "
            "be trained to facilitate communication in daily activities."
        ),
    },
]
