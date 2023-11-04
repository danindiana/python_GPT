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
search_query = " AND ".join(f'"{term}"' for term in additional_terms)

# Generate a timestamped filename
filename = f"arxiv_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Open the file for writing
with open(filename, 'w') as file:
    # Loop through each category and perform the search
    for category in cs_categories:
        print(f"{current_time()} - Starting search in category: {category}")
        file.write(f"{current_time()} - Starting search in category: {category}\n")
        
        search = arxiv.Search(
          query = f"cat:{category} AND ({search_query})",
          max_results = 10000,
          sort_by = arxiv.SortCriterion.SubmittedDate
        )

        try:
            # Fetch the results using the client for each category
            for result in client.results(search):
                output = f"{current_time()} - Category: {category}, Title: {result.title}\n"
                print(output, end='')
                file.write(output)
                time.sleep(1)  # Sleep for a second between each result to avoid hitting rate limits
        except Exception as e:
            error_message = f"{current_time()} - An error occurred while processing category {category}: {e}\n"
            print(error_message, end='')
            file.write(error_message)
            time.sleep(10)  # Sleep for some time before continuing to the next category
            continue  # Continue with the next category
        completion_message = f"{current_time()} - Completed search in category: {category}\n"
        print(completion_message, end='')
        file.write(completion_message)

    completion_message = f"{current_time()} - All searches completed.\n"
    print(completion_message, end='')
    file.write(completion_message)

# Suggested file name for the script: `arxiv_cs_search_with_timestamp.py`
