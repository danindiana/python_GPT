python programs related to calculating Jensen-Shannon Divergence of documents. 

calculate_js_divergence.py

This Python program performs the following tasks:

Reading Files:

It defines a read_file function that takes a file path as input, opens the file in read mode ('r'), reads its content, and returns the content as a string.
Tokenization:

It defines a tokenize function that takes a text string as input and splits it into tokens (words) based on whitespace. It returns a list of tokens.
Combining Tokens:

The get_combined_tokens function takes a list of file paths as input.
It initializes an empty set called all_tokens to store unique tokens across all files.
It iterates through each file, reads its content, tokenizes it, and updates the all_tokens set with the unique tokens found in that file.
It returns the set of all unique tokens.
Probability Distribution:

The get_probability_distribution function calculates the probability distribution of tokens in a file.
It takes two arguments: the list of tokens in the file (tokens) and the set of all unique tokens across all files (all_tokens).
It calculates the frequency of each token in the file using Counter.
It calculates the probability of each token by dividing its frequency by the total number of tokens in the file.
It returns a list representing the probability distribution of tokens in the file.
Jensen-Shannon Divergence:

The calculate_js_divergence function calculates the Jensen-Shannon Divergence (JSD) between two probability distributions.
It takes two probability distributions (dist1 and dist2) as input.
It uses the jensenshannon function from the scipy.spatial.distance library to compute the JSD. The JSD measures the similarity between two probability distributions.
It returns the JSD score.
Main Function:

The main function is the entry point of the program.
It takes a directory path as input.
It obtains a list of file paths in the specified directory and initializes an empty dictionary called distributions to store probability distributions for each file.
It iterates through each file in the directory:
Reads the file content.
Tokenizes the content.
Calculates the probability distribution of tokens.
Stores the distribution in the distributions dictionary.
It generates an output file name based on the current date and time.
It iterates through all pairs of files, calculates the JSD between their probability distributions, and writes the results to the output file.
Script Execution:

The program checks if it is being run as the main script (if __name__ == "__main__":) and then specifies the directory to process. In this case, it processes the specified directory /home/walter/Bsh_PostProcpdf_pdfmine_buk_QAULTITY.
In summary, this program reads text files in a specified directory, tokenizes their content, calculates the probability distribution of tokens in each file, and computes the Jensen-Shannon Divergence between pairs of files based on their token distributions. The results are written to an output file. This script is used to analyze the textual similarity between files in the specified directory.

