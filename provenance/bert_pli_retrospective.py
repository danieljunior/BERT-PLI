from typing import List
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
import time
import os
import random
import string

os.environ["DFA_URL"] = "http://dfanalyzer:22000/"


def add_n_elements(task: Task, dataset_name: str, elements: List[Element]):
    for element in elements:
        task.add_dataset(DataSet(dataset_name, [element]))
        task.save()

if __name__ == "__main__":
    dataflow_tag = "bert-pli"
    shift = 22
    # Retrospective provenance
    # Task 1: docs_pairs_generation
    t1 = Task(1 + shift, dataflow_tag, "docs_pairs_generation")
    t1_input = DataSet("coliee_file_input", [Element(["/path/to/coliee/data", "train"])])
    t1.add_dataset(t1_input)
    t1.begin()

    # Simulated output data
    filepath_pair = "/path/to/processed/doc_pairs.json"
    split_type = "train"
    count = 1000

    t1_output = DataSet("doc_pair_dataset", [Element([filepath_pair, split_type, count])])
    t1.add_dataset(t1_output)
    t1.end()
    time.sleep(1)

    # Task 2: docs_texts_splitting
    t2 = Task(2 + shift, dataflow_tag, "docs_texts_splitting", dependency=t1)
    t2.begin()

    filepath_split = "/path/to/processed/doc_pairs_split.json"
    split_method = "sentence"
    split_level = "paragraph"
    split_count = 1500

    t2_output = DataSet(
        "splitted_doc_pair_dataset",
        [Element([filepath_split, split_type, split_count, split_method, split_level])],
    )
    t2.add_dataset(t2_output)
    t2.end()
    time.sleep(1)

    # Task 3: get_doc1_example
    t3 = Task(3 + shift, dataflow_tag, "get_doc1_example", dependency=t2)
    t3.begin()

    doc1_filepath = "/path/to/processed/doc1_segments.json"
    doc1_seq = 1500

    t3_output_elements = [
            Element(
                [
                    "".join(random.choices(string.ascii_letters + string.digits, k=10))+ ".json",
                    doc1_filepath,
                    "The court ruled in favor of the plaintiff based on evidence presented.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "Legal precedents from previous cases were cited during the hearing.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "The defendant argued that the contract was void due to misrepresentation.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "Witness testimony corroborated the timeline of events.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "Statutory interpretation played a key role in the judgment.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "The appeal was denied on grounds of insufficient new evidence.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "Jurisdiction was established through the location of the incident.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "Damages were awarded for both compensatory and punitive reasons.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "The settlement agreement included confidentiality clauses.",
                    doc1_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc1_filepath,
                    "Expert opinions differed on the causation of the injury.",
                    doc1_seq,
                ]
            ),
        ]
    add_n_elements(t3, "doc1_segment", t3_output_elements)
    t3.end()
    time.sleep(1)

    doc2_seq = 1500
    doc2_filepath = "/path/to/processed/doc2_segments.json"
    t4 = Task(4 + shift, dataflow_tag, "get_doc2_example", dependency=t2)
    t4.begin()
    t4_output_elements = [
            Element(
                [
                    "".join(random.choices(string.ascii_letters + string.digits, k=10))+ ".json",
                    doc2_filepath,
                    "The defendant filed a motion to dismiss the case.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Counterclaims were raised regarding breach of contract.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Discovery revealed inconsistencies in the plaintiff's testimony.".replace("'", "''"),
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "The judge granted a summary judgment in favor of the defense.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Appeals court overturned the lower court's decision.".replace("'", "''"),
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Settlement negotiations failed due to irreconcilable differences.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Expert witnesses provided conflicting reports on liability.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Punitive damages were sought but not awarded.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "Jurisdictional issues complicated the proceedings.",
                    doc2_seq,
                ]
            ),
            Element(
                [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.json',
                    doc2_filepath,
                    "The verdict was appealed on procedural grounds.",
                    doc2_seq,
                ]
            ),
        ]
    add_n_elements(t4, "doc2_segment", t4_output_elements)
    t4.end()
    time.sleep(1)

    label_value = 1
    t5 = Task(5 + shift, dataflow_tag, "get_label_example", dependency=t2)
    t5.begin()
    t5_output3 = DataSet("label", [Element([label_value])])
    t5.add_dataset(t5_output3)
    t5.end()
    time.sleep(1)

    # Task 6: doc1_relevant_segments_selection
    t6 = Task(6 + shift, dataflow_tag, "doc1_relevant_segments_selection", dependency=t3)
    t6.begin()

    doc1_id = "doc1_001"
    doc1_text = "relevant text from doc1"
    doc1_seq_rel = 50
    epoch_t4 = 1

    t6_output_elements = [
            Element([element.values[0], element.values[2], i, 'TextRank', epoch_t4])
            for i, element in enumerate(random.sample(t3_output_elements, 5))
        ]

    add_n_elements(t6, "doc1_relevant_segment", t6_output_elements)
    t6.end()
    time.sleep(1)

    # Task 7: doc2_relevant_segments_selection
    t7 = Task(7 + shift, dataflow_tag, "doc2_relevant_segments_selection", dependency=t4)
    t7.begin()

    doc2_id = "doc2_001"
    doc2_text = "relevant text from doc2"
    doc2_rel_idx = 50
    epoch_t7 = 1

    t7_output_elements = [
            Element([element.values[0], element.values[2], i, 'TextRank', epoch_t7])
            for i, element in enumerate(random.sample(t4_output_elements, 5))
        ]
    add_n_elements(t7, "doc2_relevant_segment", t7_output_elements)
    t7.end()
    time.sleep(1)

    # Task 8: bert_scores_calculation
    t8 = Task(8 + shift, dataflow_tag, "bert_scores_calculation", dependency=[t6, t7])
    t8.begin()

    row_idx = 0
    epoch_t8 = 1
    t8_output_elements = [
            Element([0, "[0.95, 0.87, 0.92, 0.78, 0.83]", epoch_t8]),
            Element([1, "[0.88, 0.91, 0.85, 0.79, 0.94]", epoch_t8]),
            Element([2, "[0.93, 0.86, 0.89, 0.82, 0.76]", epoch_t8]),
            Element([3, "[0.90, 0.84, 0.96, 0.77, 0.89]", epoch_t8]),
            Element([4, "[0.87, 0.92, 0.88, 0.81, 0.95]", epoch_t8]),
        ]
    add_n_elements(t8, "interaction_map", t8_output_elements)

    time.sleep(1)

    # Task 9: max_pooling
    t9 = Task(9 + shift, dataflow_tag, "max_pooling", dependency=t8)
    t9.begin()

    cell_idx = 0
    epoch_t9 = 1

    t9_output_elements = [
            Element([i, random.random(), epoch_t9]) for i in range(5)
        ]
    add_n_elements(t9, "feature_vector", t9_output_elements)
    t9.end()
    time.sleep(1)

    # Task 10: classification
    t10 = Task(10 + shift, dataflow_tag, "classification", dependency=t9)
    t10.begin()

    predicted_label = 1
    loss = 0.234
    epoch_t10 = 1
    t10_output = DataSet("output", [Element([predicted_label, 'CrossEntropyLoss', loss, epoch_t10])])
    t10.add_dataset(t10_output)
    t10.end()
    time.sleep(1)

    # Task 11: evaluating
    t11 = Task(11 + shift, dataflow_tag, "evaluating", dependency=[t10, t5])
    t11.begin()

    metric_type = "accuracy"
    metric_value = 0.92

    t11_output = DataSet("metric", [Element([metric_type, metric_value])])
    t11.add_dataset(t11_output)
    t11.end()

    print(f"Retrospective provenance for dataflow '{dataflow_tag}' completed successfully!")
