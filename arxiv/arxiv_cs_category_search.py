import arxiv
import time
from datetime import datetime

def current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create an instance of the Client
client = arxiv.Client()

# List of cs categories from the Arxiv taxonomy
cs_categories = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

# Define the additional search terms
additional_terms = ["machine learning", "", "", "warfare"]

# Construct the search query with multiple terms
# This will search for papers that contain all the additional terms
search_query = " AND ".join(f'"{term}"' for term in additional_terms)

# Loop through each category and perform the search
for category in cs_categories:
    print(f"{current_time()} - Starting search in category: {category}")
    search = arxiv.Search(
      query = f"cat:{category} AND ({search_query})",
      max_results = 100,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )

    try:
        # Fetch the results using the client for each category
        for result in client.results(search):
            print(f"{current_time()} - Category: {category}, Title: {result.title}")
            time.sleep(1)  # Sleep for a second between each result to avoid hitting rate limits
    except Exception as e:
        print(f"{current_time()} - An error occurred while processing category {category}: {e}")
        time.sleep(10)  # Sleep for some time before continuing to the next category
        continue  # Continue with the next category
    print(f"{current_time()} - Completed search in category: {category}")

print(f"{current_time()} - All searches completed.")
