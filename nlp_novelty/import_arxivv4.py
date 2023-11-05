import arxiv
import time
from datetime import datetime

# Create an instance of the Client
client = arxiv.Client()

# List of cs categories
cs_categories = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]
# Get the current date and time to use in the filename
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"arxiv_cs_categories_{formatted_time}.txt"

# Open the file for writing
with open(filename, 'w') as file:
    # Loop through each category and perform the search
    for category in cs_categories:
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=50000,  # Reduced number of results for quicker execution
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        try:
            # Fetch the results using the client for each category
            for result in client.results(search):
                output = f"Category: {category}, Title: {result.title}\n"
                print(output, end='')  # Print to console
                file.write(output)  # Write to file
                time.sleep(0.2)  # Sleep to avoid hitting rate limits
        except Exception as e:
            error_message = f"An error occurred while processing category {category}: {e}\n"
            print(error_message)
            file.write(error_message)  # Write error message to file
            time.sleep(10)  # Sleep before continuing to the next category
            continue  # Continue with the next category
