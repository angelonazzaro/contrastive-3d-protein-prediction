import os
from typing import Tuple

import torch
from tqdm import tqdm
from torchmetrics.functional import f1_score, accuracy, precision, recall


def get_model_basename(model):
    """
    Extracts the base name of a model file.

    Args:
        model (str): The file path to the mode

    Returns:
        str: The base name of the model file.
    """
    return '.'.join(model.split(os.sep)[-1].split(".")[:-1])


def predict_image(img_file, model, classes, imgs_dir):
    """
    Predicts classes for an input image.

    Args:
        img_file (str): The filename of the image.
        model (RSDClip): The pre-trained CLIP model.
        imgs_dir (str): Directory containing the evaluation images.

    Returns:
        str: The true label of the image.
        list of tuple: A list of top-K predicted class-label and probability pairs.
        """

    if os.sep in img_file:
        label = img_file.split(os.sep)[0]
    else:
        parts = img_file.split("_")
        label = "_".join(parts[:-1])

    label = label.lower()

    img_file = os.path.join(imgs_dir, img_file)

    # move inputs to current device
    for key in inputs.keys():
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(model.device)

    outputs = model(inputs, return_loss=False)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().detach().numpy()
    probs_np = np.asarray(probs)[0]
    probs_npi = np.argsort(-probs_np)
    predictions = [(classes_names[i], probs_np[i]) for i in probs_npi[0:k]]

    return label, predictions


def predict(model, eval_images, classes, model_scores_file):
    """
    Predicts classes for a list of evaluation images using a CLIP model and computes scores.

    Args:
        model (RSDClip): The pre-trained CLIP model.
        eval_images (list of str): List of image filenames to evaluate.
        model_scores_file (str): Path to the file to store prediction scores.
    """
    print("Generating predictions...")
    images_predicted = 0

    with open(model_scores_file, "w") as msf:
        for eval_image in tqdm(eval_images):
            label, predictions = predict_image(eval_image, model, processor, eval_sentences,
                                               classes_names, max(K_VALUES), imgs_dir)

            msf.write("{:s}\t{:s}\t{:s}\n".format(eval_image, label, "\t".join(["{:s}\t{:.5f}".format(c, p)
                                                                                for c, p in predictions])))
            images_predicted += 1

    print(f"{images_predicted} images evaluated, COMPLETED!")


def compute_scores(scores_file, model_scores_file, model_basename):
    """
    Computes final accuracy scores based on prediction results.

    Args:
        scores_file (str): Path to the file to store final scores.
        model_scores_file (str): Path to the file containing prediction scores.
        model_basename (str): Basename of the model being evaluated.
    """
    print("Computing final scores...")
    num_examples = 0
    correct_k = [0]

    with open(model_scores_file, "r") as msf:
        for line in msf:
            cols = line.strip().split('\t')
            label = cols[1]
            preds = []
            for i in range(2, min(len(cols), 22), 2):
                preds.append(cols[i])
                if label in preds:
                    correct_k[0] += 1
            num_examples += 1

    scores_k = [ck / num_examples for ck in correct_k]

    with open(scores_file, "a") as sf:
        sf.write("{:s}\t{:s}\n".format(model_basename, "\t".join(["{:.3f}".format(s) for s in scores_k])))


def compute_metrics(logits: torch.Tensor, ground_truth: torch.Tensor, n_classes: int, batch_size: int) \
        -> Tuple[float, float, float, float]:
    graph_probs = torch.softmax(logits, dim=-1)
    sequence_probs = torch.softmax(logits, dim=0)

    graph_acc = accuracy(preds=graph_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                         average="macro")
    sequence_acc = accuracy(preds=sequence_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                            average="macro")

    graph_prec = precision(preds=graph_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                           average="macro")
    sequence_prec = precision(preds=sequence_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                              average="macro")
    graph_rec = recall(preds=graph_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                       average="macro")
    sequence_rec = recall(preds=sequence_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                          average="macro")
    graph_f1 = f1_score(preds=graph_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                        average="macro")
    sequence_f1 = f1_score(preds=sequence_probs, target=ground_truth, task='multiclass', num_classes=n_classes,
                           average="macro")

    acc = graph_acc + sequence_acc / 2 / batch_size
    prec = graph_prec + sequence_prec / 2 / batch_size
    rec = graph_rec + sequence_rec / 2 / batch_size
    f1 = graph_f1 + sequence_f1 / 2 / batch_size

    return acc, prec, rec, f1
