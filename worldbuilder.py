"""
The worldbuilder generates a story for the agents to act out in the prisoners dilemma.
The prisoners can share various relationships, commit on not commit various crimes, or have evidence gathered against them.

Backstories can be randomly generated or manually configured for testing with the generate_backstory function.
There is also a payoff matrix builder function. 
It is derived from the basic prisoners dilemma but scales the time served based on the severity of the crime.
It is currently not utilzed in the main code, but is available for future use.
"""
import pprint
import random

NAMES_P1 = ["Fred", "John", "Henry", "George", "Bobby",
            "Bonny", "Martha", "Megan", "Julia", "Liz"]

NAMES_P2 = ["Clide", "Samson", "Rob Banks", "Mat", "Randy",
            "Becky", "Joann", "Mary", "Katheryn", "Monica"]

RELATIONSHIPS = [
    "You despise the other prisoner to the point that you wish for his death.",
    "You hate the other prisoner. He has been your rival for years.",
    "You strongly dislike the other prisoner.",
    "You dislike the other prisoner.",
    "You have no relationship to the other prisoner. Until today the two of you have never met.",
    "You feel neutral toward the other prisoner. You've met briefly, but nothing stood out.",
    "You get along decently with the other prisoner. You've shared a few positive interactions.",
    "You get along great with the other prisoner.",
    "You are deeply in love with the other prisoner.",
    "You recognize that the other prisoner is part of your family."
]

STATUS_LIST = [
    "You are completely innocent. You have no idea what is going on.",
    "You were an accessory and assisted the criminal but were not at the scene when it was committed.",
    "You are just as guilty of this crime as the other prisoner.",
    "You are the main culprit and the other prisoner is an accessory to your crime.",
    "You are guilty of this crime. But the other prisoner is innocent."
]

CRIMES = [
    "shoplifting from a big box store.",
    "minor vandalism and graffiti.",
    "small scale pickpocketing.",
    "motor vehicle theft.",
    "breaking into a home and committing burglary.",
    "fraud and embezzlement.",
    "illegal drug trade.",
    "assault as a member of a violent gang.",
    "robbing a gas station at gunpoint.",
    "premeditated murder."
]

# The interrogator may have some evidence that he can try to use to pressure prisoners.
EVIDENCE = [
   # Weak Evidence
   "The police have found little to no evidence connecting the prisoners to the crime",
   "Rumors suggest the prisoners might have been near the scene.",
   "Blury CCTV camera footage shows a vague figure similar to the prisoners in the area.",
   "Some witnesses remember seeing the prisoners in the area but cannot identify them clearly.",
   # Moderate Evidence
   "Partial fingerprints were found near the scene but are inconclusive.",
   "Some items belonging to the prisoners were found in the vicinity.",
   "Partial DNA samples were found at the scene but could belong to many different individuals.",

   # Strong Evidence
   "Clear CCTV footage shows one prisoner near the scene around the time of the crime.",
   "A witness confidently recalls both the prisoners at the scene of the crime.",
   "Forensic evidence conclusively links both of the prisoners to the crime scene.",
]


# By keeping the initialization out here when we import the code into a file it won't rerun this every time.
# i.e. Random variables won't change between files.
def generate_backstory(severity=None, relationship_score=None, status=None):
    print("Initializing Backstories...")
    severity = severity or random.randint(1, len(CRIMES))
    relationship_score = relationship_score or random.randint(1, len(RELATIONSHIPS))
    status = status or random.randint(0, 4)
    p1_index = random.randint(0, 9)
    p2_index = random.randint(0, 9)

    p1_name = NAMES_P1[p1_index]
    p2_name = NAMES_P2[p2_index]

    p1_gender = 0 if p1_index > 5 else 1
    p2_gender = 0 if p2_index > 5 else 1

    relationship = RELATIONSHIPS[relationship_score - 1]

    # p1 and p2 status's are inversed. This ensures that at least one of them is always guilty.
    p1_status = STATUS_LIST[status]
    p2_status = STATUS_LIST[4 - status]

    crime = CRIMES[severity - 1]

    evidence_quality = random.randint(0,8)
    print(evidence_quality)
    evidence = EVIDENCE[evidence_quality]

    return {
        "p1_name": p1_name,
        "p2_name": p2_name,
        "p1_gender": p1_gender,
        "p2_gender": p2_gender,
        "relationship": relationship,
        "p1_status": p1_status,
        "p2_status": p2_status,
        "crime": crime,
        "severity": severity,
        "evidence": evidence,
        "evidence_quality": evidence_quality
    }


####################################
#           Initialization         #
####################################

# Generate a single shared backstory
# You can pass in arguments here for
# severity, relationship_score, status in order to manually configure for testing.
STORY = generate_backstory() # TODO: MANUALY CONFIGURE INPUTS FOR BACKSTORY HERE

print("Generating Story")
pprint.pprint(STORY)



####################################
#           Initialization         #
####################################


def build_payoff(plea_deal = 0.2, scaler = 1, reducer = 0.8):
  max_time = -1 * STORY["severity"] * scaler
  plea_time = -1 * STORY["severity"] * plea_deal
  reduced_time = -1 * STORY["severity"] * reducer

  payoff_matrix = {
    "stay_silent": {
        "stay_silent": {"time": (reduced_time, reduced_time), "desc": "Both stay silent."},
        "confess":     {"time": (max_time, 0),  "desc": "You stay silent; they confess."}
    },
    "confess": {
        "stay_silent": {"time": (0, max_time),  "desc": "You confess; they stay silent."},
        "confess":     {"time": (plea_time, plea_time), "desc": "Both confess."}
    }
  }
  return payoff_matrix