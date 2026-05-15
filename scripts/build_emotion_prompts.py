"""Build emotion_prompts.parquet: 50 prompts × 4 emotions + 50 neutrals.

Schema
------
prompt          str   — the text snippet
emotion_label   str   — joy | fear | anger | sadness | neutral
split           str   — train | test (70/30 per emotion)
id              str   — unique identifier (e.g., joy_001)
category        str   — domain tag (work, relationships, health, news, daily_life, creative, social, existential)
length_words    int   — word count (for balance checks)
source          str   — hand_authored

Design choices
--------------
- Hand-authored for quality control and conceptual fit.
- Diverse domains to test generalization.
- Balanced lengths (5–30 words).
- No explicit emotion words in the prompt itself where possible,
  to avoid trivial lexical confounds.  (Not achievable for all
  items, but aimed for.)
- 35 train / 15 test per emotion (70/30).

Usage
-----
    uv run python scripts/build_emotion_prompts.py

Output
------
    data/public/emotion_prompts.parquet
"""

from __future__ import annotations

import random

import pandas as pd

# Reproducible shuffle
random.seed(42)

# ---------------------------------------------------------------------------
#  JOY  (50 items)
# ---------------------------------------------------------------------------
JOY = [
    # work / achievement
    ("Just landed the fellowship I spent two years preparing for. The letter arrived at dawn.", "work"),
    ("The manuscript was accepted without revisions. The editor called it 'a genuine contribution.'", "work"),
    ("My grant proposal scored in the top 2 percent nationally. Funding starts in January.", "work"),
    ("The code finally passed all integration tests after three weeks of debugging.", "work"),
    ("My mentee defended her dissertation today and the committee gave unanimous praise.", "work"),
    ("The startup I advised just closed its Series A at double the projected valuation.", "work"),
    ("My paper was cited by the researcher whose work originally inspired me to enter the field.", "work"),
    ("The promotion came through with the exact title and scope I had proposed.", "work"),
    ("Our team shipped the release on time for the first time in eighteen months.", "work"),
    ("The keynote audience gave a standing ovation. I could see my advisor smiling in the third row.", "work"),
    # relationships / social
    ("My sister called to say the adoption was finalized. They bring him home next week.", "relationships"),
    ("After years of distance, my childhood friend and I talked until three in the morning.", "relationships"),
    ("The dinner party ended with everyone still at the table, laughing, refusing to leave.", "social"),
    ("My parents surprised me by flying in for my birthday. They had told me they could not afford it.", "relationships"),
    ("The text from her simply said, 'I am proud of you,' and I had to sit down.", "relationships"),
    ("My dog, missing for eleven days, was found by a neighbor three blocks away.", "relationships"),
    ("The old photograph arrived in the mail from a cousin I had never met.", "relationships"),
    ("Our anniversary dinner was at the same restaurant where we had our first date.", "relationships"),
    ("The community garden produced its first tomatoes, and the neighbors started a potluck tradition.", "social"),
    ("My grandmother's recipe worked. The kitchen smelled exactly like her house.", "relationships"),
    # health / body
    ("The biopsy came back benign. The surgeon said I could go home today.", "health"),
    ("After six months of physical therapy, I ran a full mile without stopping.", "health"),
    ("The migraine that had lasted four days lifted suddenly while I was walking in the park.", "health"),
    ("My bloodwork for the first time in years showed all markers in the normal range.", "health"),
    ("I slept through the night without medication for the first time since the accident.", "health"),
    # creative / aesthetic / sensory
    ("The sunrise over the fjord turned the water gold for exactly three minutes.", "creative"),
    ("I heard a song I had forgotten existed, and it transported me to a summer I thought I had lost.", "creative"),
    ("The painting I had doubted resolved itself in the final hour of the residency.", "creative"),
    ("A stranger on the train asked if I was the poet who wrote the piece in last month's review.", "creative"),
    ("The northern lights appeared on the one clear night of the entire trip.", "creative"),
    ("I found the exact edition of the book I had been hunting for since college.", "creative"),
    ("The bread rose perfectly, the crust crackled, and the crumb was open and elastic.", "creative"),
    ("The symphony reached the coda and the conductor held the silence for ten full seconds.", "creative"),
    # daily life / small victories
    ("The refund I had given up on appeared in my account this morning.", "daily_life"),
    ("The package that was supposed to arrive next week came two days early.", "daily_life"),
    ("I found a twenty in the pocket of a coat I had not worn since last winter.", "daily_life"),
    ("The cafe had my favorite pastry, which they only make on Tuesdays, and it was Tuesday.", "daily_life"),
    ("The commute took half the usual time. I walked the last mile in daylight.", "daily_life"),
    # existential / meaning
    ("For the first time in years, I felt like the work I do actually mattered to someone.", "existential"),
    ("I realized today that I am no longer afraid of the thing that defined my twenties.", "existential"),
    ("The volunteer shift ended with a refugee family telling me I had changed their son's life.", "existential"),
    ("I looked at my bookshelf and recognized that I had become the person I wanted to read.", "existential"),
    ("The letter I wrote but never sent finally found its way to the right person, years later.", "existential"),
    # news / public
    ("The prisoner whose case I had been following was exonerated this morning after eighteen years.", "news"),
    ("The coral reef survey showed the first signs of recovery since the marine reserve was established.", "news"),
    ("The town voted unanimously to fund the public library expansion.", "news"),
    ("The teacher strike ended with a contract that included the raises they had asked for.", "news"),
    ("The endangered species count recorded ten new calves this season, the highest in two decades.", "news"),
    ("The child I tutor finally understood the concept he had been struggling with for months, and his face lit up.", "existential"),
    ("The rain stopped just as the outdoor concert began, and the crowd cheered louder than the opening band.", "creative"),
]

# ---------------------------------------------------------------------------
#  FEAR  (50 items)
# ---------------------------------------------------------------------------
FEAR = [
    # health / body
    ("The doctor paused too long before answering my question about the mass on the scan.", "health"),
    ("The chest pain returned at 3 AM, sharper this time, radiating down my left arm.", "health"),
    ("My mother's voice on the phone had a tremor I had never heard before.", "health"),
    ("The test results were posted online at midnight but the portal kept timing out.", "health"),
    ("I woke up unable to move my right hand, and for thirty seconds I thought it was permanent.", "health"),
    ("The surgeon said the procedure carries a risk of partial paralysis I had not been told about.", "health"),
    ("The pharmacist looked at the prescription, then at me, then back at the screen.", "health"),
    ("The genetic screening came back with a variant of uncertain significance and no literature.", "health"),
    # existential / safety
    ("The footsteps on the stairs matched no one who should have been in the house.", "daily_life"),
    ("The door I was certain I had locked stood open when I returned.", "daily_life"),
    ("The email said my account had been accessed from a country I have never visited.", "daily_life"),
    ("I smelled gas in the basement and could not find where it was coming from.", "daily_life"),
    ("The security camera showed motion at 2:47 AM but the recording glitched at 2:48.", "daily_life"),
    ("My phone displayed a notification that my location was being shared with an unknown device.", "daily_life"),
    # work / professional
    ("The committee meeting started without me, and my name was on the agenda under 'personnel action.'", "work"),
    ("My supervisor asked me to save all my files to a shared drive by end of day.", "work"),
    ("The funding agency announced a review of all grants in my cohort, effective immediately.", "work"),
    ("I received a calendar invite titled 'Performance Discussion' with no description and no sender.", "work"),
    ("The server logs showed unauthorized access to the data I am responsible for protecting.", "work"),
    ("The journal sent a retraction notice for the paper that underpins my entire research program.", "work"),
    ("My tenure case was returned without comment from the dean's office two weeks before the vote.", "work"),
    ("The whistleblower hotline email had my project number in the subject line.", "work"),
    # social / relationships
    ("She said she needed to talk, then went silent for three hours.", "relationships"),
    ("The group chat stopped the moment I sent a message. No one replied for two days.", "social"),
    ("My child's teacher called and asked me to come in immediately, but would not say why.", "relationships"),
    ("The letter was addressed by hand, with no return address, and it was thick.", "relationships"),
    ("My partner's location showed them at an address I did not recognize at midnight.", "relationships"),
    ("The family gathering ended with my father asking me to stay behind while everyone else left.", "relationships"),
    # existential / abstract
    ("The presentation of climate data showed the curve I had been dreading crossing the threshold this decade.", "existential"),
    ("I realized I could not remember the last conversation I had that was not about work.", "existential"),
    ("The philosopher in the interview said free will was an illusion, and for a moment I believed him.", "existential"),
    ("I woke from a dream in which I had died and the feeling of non-existence stayed with me for hours.", "existential"),
    ("The contract required me to waive rights I did not fully understand but could not afford to challenge.", "existential"),
    ("The algorithm flagged my behavior as anomalous and I could not find out what behavior it meant.", "existential"),
    # news / public
    ("The emergency broadcast system activated on my phone but the message was in a language I do not read.", "news"),
    ("The election results showed a margin narrower than the number of disputed ballots.", "news"),
    ("The evacuation order included my neighborhood but the traffic was already at a standstill.", "news"),
    ("The missile alert turned out to be a drill, but no one told us that for thirty-eight minutes.", "news"),
    ("The news report said the fault line had moved silently for years and was now critically stressed.", "news"),
    ("The biopsy results were supposed to be back Friday. It is now Tuesday and the lab has not returned the call.", "health"),
    ("I heard footsteps in the attic at 2 AM. No one else was supposed to be home.", "daily_life"),
    ("The email from my bank said unusual activity was detected, but the customer service line was disconnected.", "daily_life"),
    ("The professor said my thesis contained 'significant methodological concerns' but would not specify which ones.", "work"),
    ("My visa extension was denied with thirty days' notice and no explanation of the grounds.", "existential"),
    ("The building's fire alarm went off at midnight, and the emergency exit was chained shut.", "daily_life"),
    ("She stopped responding to my messages three days ago, but her social media shows her active hourly.", "relationships"),
    ("The contract I signed included a non-compete clause I did not notice until after I resigned.", "work"),
    ("The pediatrician said the rash was probably nothing, but her tone suggested she was not certain.", "health"),
    ("The protest turned violent three blocks from my apartment, and the police blocked all evacuation routes.", "news"),
    ("I opened the door to find a process server holding papers I did not know existed.", "daily_life"),
]

# ---------------------------------------------------------------------------
#  ANGER  (50 items)
# ---------------------------------------------------------------------------
ANGER = [
    # work / institutional
    ("My idea was presented in the meeting as original by a colleague who had dismissed it in private.", "work"),
    ("The deadline was moved up by two weeks without consultation, and my manager called it a 'stretch goal.'", "work"),
    ("I discovered my co-author had submitted our paper solo to a higher-tier journal without telling me.", "work"),
    ("The promotion went to the person who had done none of the implementation work I had documented.", "work"),
    ("My grant was rejected because the reviewer confused my project with a completely different lab.", "work"),
    ("The university announced a pay freeze the same day the board approved a new luxury headquarters.", "work"),
    ("I was asked to train my replacement before being informed that my contract would not be renewed.", "work"),
    ("The data I spent six months cleaning was deleted by an automated retention policy with no warning.", "work"),
    ("My performance review praised my 'team spirit' while denying me the raise I had earned.", "work"),
    ("The conference rejected my talk but invited me to pay full registration to attend as an observer.", "work"),
    # relationships / social
    ("She borrowed the book I had inscribed from my late mentor and returned it with coffee stains.", "relationships"),
    ("He interrupted me for the fourth time in a ten-minute conversation and then asked why I seemed quiet.", "social"),
    ("My roommate used my savings to cover their rent without asking, then accused me of being selfish.", "relationships"),
    ("The family group chat forwarded a conspiracy theory about my field and tagged me for comment.", "social"),
    ("My ex posted photographs from a trip we had planned together, using my itinerary, with someone new.", "relationships"),
    ("The neighbor's construction started at 6 AM on a Saturday after promising it would not begin before 9.", "relationships"),
    ("My father repeated the same dismissive comment about my career at Thanksgiving in front of everyone.", "relationships"),
    # daily life / systemic
    ("The airline lost my luggage on a trip for a funeral and offered me a fifty-dollar voucher.", "daily_life"),
    ("The customer service representative put me on hold for forty minutes, then hung up.", "daily_life"),
    ("My landlord raised the rent by thirty percent with thirty days' notice and no explanation.", "daily_life"),
    ("The insurance company denied the claim citing a clause they had added after I signed the policy.", "daily_life"),
    ("The delivery driver left my fragile package in the rain without knocking.", "daily_life"),
    ("The subscription renewed at triple the price and the cancellation button led to a chatbot loop.", "daily_life"),
    # public / news
    ("The report documented systematic abuse in the facility, and the director received a bonus.", "news"),
    ("The politician who voted to cut school funding enrolled his own children in private academies.", "news"),
    ("The oil company knew about the leak for three months before notifying the community.", "news"),
    ("The police report contradicted three witness statements and was accepted without question.", "news"),
    ("The CEO apologized for layoffs while his compensation increased by forty million dollars.", "news"),
    ("The city council approved the demolition of the public housing without a relocation plan.", "news"),
    ("The hospital charged a patient twelve thousand dollars for a procedure that took eleven minutes.", "news"),
    # existential / abstract
    ("I realized the institution I had devoted fifteen years to had never intended to keep its promises.", "existential"),
    ("The essay I wrote criticizing the system was published behind a paywall I could not afford.", "existential"),
    ("My therapist suggested my anger was a 'cognitive distortion,' and I almost walked out.", "existential"),
    ("The algorithm recommended I watch a documentary about my own hometown's water crisis as entertainment.", "existential"),
    ("I discovered the charity I had donated to spent eighty percent of its budget on executive salaries.", "existential"),
    ("The archive I had trusted for years was quietly selling access to my research data to advertisers.", "existential"),
    ("The intellectual I admired gave a lecture arguing that people like me were the real problem.", "existential"),
    ("The restaurant added a mandatory twenty percent service charge but the service had been openly rude.", "daily_life"),
    ("My request for disability accommodation was denied because the form was filed one day late.", "work"),
    ("The landlord entered my apartment without notice while I was at work and left the door unlocked.", "daily_life"),
    ("The university promoted the person who had been formally investigated for harassment last year.", "work"),
    ("The warranty claim was denied because the product failed three days after the expiration date.", "daily_life"),
    ("My therapist terminated our sessions by email, effective immediately, with no referral.", "existential"),
    ("The documentary about workers' rights was sponsored by the corporation it was supposed to be investigating.", "news"),
    ("My colleague took credit for the presentation I had prepared while he was on vacation.", "work"),
    ("The judge dismissed the case because the statute of limitations had expired by two weeks.", "news"),
    ("The recycling program was discontinued because it was 'too expensive' on the same day the CEO got a bonus.", "news"),
    ("My gym charged me for six months after I canceled, and the refund process requires a notarized letter.", "daily_life"),
    ("The textbook I wrote was pirated by a predatory publisher who now charges students three hundred dollars for it.", "work"),
    ("The hiring manager asked about my 'commitment' after I disclosed I have a chronic illness.", "work"),
]

# ---------------------------------------------------------------------------
#  SADNESS  (50 items)
# ---------------------------------------------------------------------------
SADNESS = [
    # loss / relationships
    ("The house sold last week. I drove past and saw they had cut down the oak tree in the front yard.", "relationships"),
    ("Her last message said she would call when she landed. That was four years ago today.", "relationships"),
    ("The restaurant where we had celebrated every anniversary closed during the pandemic and never reopened.", "relationships"),
    ("I found his handwriting in an old notebook and could not remember the sound of his voice.", "relationships"),
    ("The dog I grew up with died at sixteen, and my mother waited until after my exam to tell me.", "relationships"),
    ("My childhood home is now a parking lot. The street name is the only thing that remains.", "relationships"),
    ("The friendship ended with a text that said, 'I think we have both changed,' and nothing more.", "relationships"),
    ("I walked past the bench where we used to meet. Someone had carved our initials inside a heart.", "relationships"),
    ("The letter from my father arrived two months after his death. He had written it the week before.", "relationships"),
    ("She moved across the country and we promised to stay close. Now our conversations are weather reports.", "relationships"),
    # missed opportunities / regret
    ("I turned down the job in Berlin because I was afraid, and they hired someone I trained.", "existential"),
    ("The manuscript I abandoned in my twenties was published posthumously by someone else to acclaim.", "existential"),
    ("I spent my thirties proving myself to people who had already stopped watching.", "existential"),
    ("The language my grandmother spoke died with her because I never asked her to teach me.", "existential"),
    ("I realized the person I had been cruel to in high school had died five years ago.", "existential"),
    ("The photograph showed us laughing, and I could not remember what we were laughing about.", "existential"),
    # health / body
    ("The diagnosis explained the symptoms but also meant I would not be able to have children.", "health"),
    ("My hands have started to shake, and I can no longer draw the way I used to.", "health"),
    ("The medication works, but the side effects make me feel like I am watching myself from a distance.", "health"),
    ("My father's dementia progressed to the point where he no longer recognized my mother.", "health"),
    ("The physical therapy plateaued. The therapist said we had reached maximum medical improvement.", "health"),
    # work / professional disappointment
    ("The project I had led for four years was canceled the week before launch with no explanation.", "work"),
    ("My advisor told me I was not cut out for academia during the final year of my PhD.", "work"),
    ("The department eliminated the position I had been promised, effective the semester I graduated.", "work"),
    ("I watched my competitor win the prize for work I had started but never finished.", "work"),
    ("The layoff notice arrived on the same day as my daughter's birthday party.", "work"),
    ("My research specialty was declared obsolete by a consensus report published in my own journal.", "work"),
    # daily life / quiet losses
    ("The coffee shop where I wrote my dissertation became a vape store.", "daily_life"),
    ("The rain ruined the only photograph I had of my grandmother as a young woman.", "daily_life"),
    ("I finally learned to make his favorite dish perfectly, and there was no one left to cook it for.", "daily_life"),
    ("The tree I had planted as a sapling was uprooted by the storm last night.", "daily_life"),
    ("My favorite bookstore announced it was closing. I had meant to visit last month.", "daily_life"),
    ("The winter coat I had saved for was discontinued in my size the day before I decided to buy it.", "daily_life"),
    # creative / aesthetic
    ("The painting I had worked on for months was damaged in a studio flood beyond restoration.", "creative"),
    ("The film I had waited years to see was pulled from distribution the day before my screening.", "creative"),
    ("The musician whose work had carried me through my darkest years announced his retirement.", "creative"),
    ("I returned to the poem I had loved at twenty and found it no longer spoke to me.", "creative"),
    ("The museum closed the wing that held the one painting that had made me want to become an artist.", "creative"),
    # news / public
    ("The coral bleaching report confirmed that the reef I had dived on as a child is now functionally dead.", "news"),
    ("The war displaced two million people, and the headline was below the fold.", "news"),
    ("The last speaker of the language died, and no recordings exist.", "news"),
    ("The glacier I had hiked as a teenager has retreated beyond the ridgeline and will not return.", "news"),
    ("The school district cut the arts program that had been the reason I stayed in school.", "news"),
    ("The journalist who exposed the scandal was found dead. The investigation was closed within a week.", "news"),
    ("The friend I had tried to help through addiction relapsed on the anniversary of his mother's death.", "relationships"),
    ("The bookshop where I had worked in college burned down last night, and no one was injured but nothing was saved.", "daily_life"),
    ("I walked past the playground where we had met, and the swings had been removed for safety concerns.", "relationships"),
    ("The scholarship fund I had established in his name had zero applicants this year.", "existential"),
    ("The city renamed the street after her, but no one who knew her still lives there.", "news"),
    ("I found the mixtape she made me in a box of old cables. The tape had degraded into silence.", "relationships"),
]

# ---------------------------------------------------------------------------
#  NEUTRAL  (50 items)
# ---------------------------------------------------------------------------
NEUTRAL = [
    # daily life / mundane
    ("The meeting was scheduled for 10 AM but started at 10:07.", "daily_life"),
    ("I bought a new toothbrush. It is the same brand as the old one.", "daily_life"),
    ("The train arrived at the platform three minutes late.", "daily_life"),
    ("I reorganized my desk and moved the lamp from the left side to the right.", "daily_life"),
    ("The grocery store was out of the specific type of rice I usually buy.", "daily_life"),
    ("I filled the car with gas and noted the price had increased by two cents per liter.", "daily_life"),
    ("The coffee was brewed at 7:15 AM and consumed by 7:32.", "daily_life"),
    ("I replaced the lightbulb in the hallway. It is a slightly different wattage.", "daily_life"),
    ("The package contained three items instead of the four listed on the invoice.", "daily_life"),
    ("I took a different route to work because of road construction on the usual street.", "daily_life"),
    ("The printer ran out of cyan ink. I ordered a replacement cartridge online.", "daily_life"),
    ("I set a reminder to water the plants on Tuesday and Thursday.", "daily_life"),
    ("The weather forecast predicts partly cloudy skies for the remainder of the week.", "daily_life"),
    ("I scheduled a dental appointment for the third week of next month.", "daily_life"),
    ("The library book was due yesterday. I renewed it for another two weeks.", "daily_life"),
    # work / factual
    ("The dataset contains 14,732 rows and 18 columns.", "work"),
    ("Version 3.2 of the software was released on March 15 with updated dependencies.", "work"),
    ("The committee meets on the first Tuesday of each month in Conference Room B.", "work"),
    ("The server migration is planned for the weekend of the 24th.", "work"),
    ("I reviewed the appendix and found three typographical errors.", "work"),
    ("The budget spreadsheet uses a currency conversion rate of 1.12.", "work"),
    ("The citation style was changed from APA to Chicago for the final submission.", "work"),
    ("The lab equipment was calibrated according to the manufacturer's protocol.", "work"),
    ("The participant demographics table is on page 47 of the protocol document.", "work"),
    ("The backup system runs automatically at 2 AM and retains files for ninety days.", "work"),
    # instructional / procedural
    ("To reset the router, unplug it for thirty seconds and then reconnect the power.", "daily_life"),
    ("The recipe calls for two cups of flour, one teaspoon of salt, and three eggs.", "daily_life"),
    ("Fold the document in thirds before inserting it into the envelope.", "daily_life"),
    ("The password must contain at least eight characters including one numeral.", "daily_life"),
    ("Press the button labeled 'Submit' after verifying all required fields are completed.", "daily_life"),
    # descriptive / observational
    ("The building has twelve floors and two elevators on the north side.", "daily_life"),
    ("The painting depicts a landscape with a river, three trees, and a small stone bridge.", "creative"),
    ("The train station has four platforms, a ticket office, and a small cafe.", "daily_life"),
    ("The document is printed in Times New Roman, 12-point, double-spaced.", "work"),
    ("The garden contains roses, lavender, and a small vegetable patch in the corner.", "daily_life"),
    ("The coat has four buttons, two pockets, and a detachable lining.", "daily_life"),
    ("The city has a population of approximately 450,000 according to the most recent census.", "news"),
    # abstract / definitional
    ("A triangle is a polygon with three edges and three vertices.", "existential"),
    ("The second law of thermodynamics states that entropy in an isolated system tends to increase.", "existential"),
    ("Photosynthesis is the process by which plants convert light energy into chemical energy.", "existential"),
    ("The median is the value separating the higher half from the lower half of a data sample.", "work"),
    ("A haiku traditionally consists of three phrases with a 5-7-5 syllable pattern.", "creative"),
    ("The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.", "news"),
    ("The standard unit of electric current in the International System of Units is the ampere.", "existential"),
    ("Mitosis is a type of cell division that results in two daughter cells with identical nuclei.", "existential"),
    ("The Treaty of Westphalia established the principle of state sovereignty in 1648.", "news"),
    ("In computer science, a binary search algorithm operates on a sorted list by repeatedly dividing the interval.", "work"),
    ("The elevator in the south wing services floors two through fourteen and stops at the lobby.", "daily_life"),
    ("The document specifies a font size of eleven points and margins of one inch on all sides.", "work"),
    ("The museum's operating hours are Tuesday through Sunday, ten AM to six PM, closed Mondays.", "daily_life"),
]


# ---------------------------------------------------------------------------
#  Helper to build a split DataFrame
# ---------------------------------------------------------------------------
def _build(
    label: str,
    items: list[tuple[str, str]],
    train_n: int,
) -> pd.DataFrame:
    """Shuffle and assign train/test split."""
    assert len(items) == train_n + (len(items) - train_n)
    assert train_n > 0

    shuffled = items.copy()
    random.shuffle(shuffled)

    rows = []
    for i, (text, cat) in enumerate(shuffled, start=1):
        split = "train" if i <= train_n else "test"
        rows.append(
            {
                "id": f"{label}_{i:03d}",
                "prompt": text,
                "emotion_label": label,
                "split": split,
                "category": cat,
                "length_words": len(text.split()),
                "source": "hand_authored",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train_per_emotion = 35
    test_per_emotion = 15
    train_neutral = 35
    test_neutral = 15

    dfs = [
        _build("joy", JOY, train_per_emotion),
        _build("fear", FEAR, train_per_emotion),
        _build("anger", ANGER, train_per_emotion),
        _build("sadness", SADNESS, train_per_emotion),
        _build("neutral", NEUTRAL, train_neutral),
    ]

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["emotion_label", "id"]).reset_index(drop=True)

    out_path = "data/public/emotion_prompts.parquet"
    df.to_parquet(out_path, index=False)

    # Sanity checks
    print(f"Saved {len(df)} rows to {out_path}")
    print("\nSplit counts:")
    print(df.groupby(["emotion_label", "split"]).size().unstack(fill_value=0))
    print("\nCategory distribution:")
    print(df.groupby(["emotion_label", "category"]).size().unstack(fill_value=0))
    print("\nLength stats (words):")
    print(df.groupby("emotion_label")["length_words"].describe().round(1))

    # Check for explicit emotion words in non-neutral prompts
    emotion_words = {
        "joy": {"happy", "joy", "joyful", "elated", "ecstatic", "delighted", "cheerful"},
        "fear": {"afraid", "scared", "terrified", "fear", "frightened", "panicked", "anxious"},
        "anger": {"angry", "furious", "rage", "irritated", "annoyed", "livid", "outraged"},
        "sadness": {"sad", "depressed", "grief", "miserable", "sorrowful", "melancholy", "heartbroken"},
    }
    for label, words in emotion_words.items():
        subset = df[df["emotion_label"] == label]
        flagged = subset[subset["prompt"].str.lower().apply(lambda s: any(w in s for w in words))]
        print(f"\n{label}: {len(flagged)}/{len(subset)} prompts contain explicit emotion words")


if __name__ == "__main__":
    main()
