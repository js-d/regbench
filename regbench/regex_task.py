import csv
import re
from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset, FieldSpec
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template

# Define the prompt template
PROMPT_TEMPLATE = """
Given the following input, predict the match:

{prompt}

The ground truth answer is computed by searching for the regular expression in the string.
If a match is found, the first match is returned; otherwise, <NO_MATCH> is returned.
That is, the expected match is obtained by running the following code:
    match = re.search(regex, string)
    expected_match = match.group(0) if match else "<NO_MATCH>"

Please end your answer with Match: YOUR_GUESS_HERE
This should be the final text of your response.
"""


def process_csv(input_csv, output_csv):
    with open(input_csv, mode="r", newline="") as infile, open(
        output_csv, mode="w", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["expected_match", "question"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in reader:
            string = row["string"]
            regex = row["regex"]
            print(regex)
            print(string)
            print()
            match = re.search(regex, string)
            row["expected_match"] = match.group(0) if match else "<NO_MATCH>"
            row["question"] = f"String: {string}\nRegular Expression: {regex}"
            writer.writerow(row)


@task
def regex_prediction():
    # Preprocess the dataset
    input_csv = "../data/regex_dataset.csv"
    output_csv = "../data/regex_dataset_processed.csv"

    process_csv(input_csv, output_csv)

    ds = csv_dataset(
        csv_file=output_csv,
        sample_fields=FieldSpec(input="question", target="expected_match"),
        shuffle=True,
    )

    # Debug: Print the first sample of the dataset
    print("First sample of the dataset:", ds[0])

    return Task(
        dataset=ds,
        plan=[
            prompt_template(PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=match(),
    )
