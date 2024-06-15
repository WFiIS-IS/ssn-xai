import pickle

import SALib
import SALib.analyze.dgsm
import SALib.sample.finite_diff
import torch
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "shuffle_train_valid_test_split",
    "model_train",
    "plot_metrics",
    "plot_dropping_metrics",
    "analyze_dgsm",
    "analyze_fast",
    "prune",
    "reduce_datasets",
    "reduce_linear",
    "save_models",
]


def shuffle_train_valid_test_split(dataset: TensorDataset,
                                   valid_p=0.1, test_p=0.3):
    random_shuffle_state = 2024

    dataset = TensorDataset(
        *shuffle(*dataset.tensors, random_state=random_shuffle_state))

    train_offset = int(len(dataset) * (1 - (test_p + valid_p)))
    valid_offset = int(len(dataset) * valid_p) + train_offset

    return {
        "train": TensorDataset(*dataset[:train_offset]),
        "valid": TensorDataset(*dataset[train_offset:valid_offset]),
        "test":  TensorDataset(*dataset[valid_offset:])
    }


def model_train(datasets, model, optimizer, criterion, scoring, epochs=20):
    losses = []
    val_scores = []
    val_losses = []

    train_samples = len(datasets["train"])
    val_samples = len(datasets["valid"])

    train_loader = DataLoader(datasets["train"], batch_size=8, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        losses.append(running_loss / train_samples)

        model.eval()
        with torch.no_grad():
            inputs, targets = datasets["valid"][:]
            outputs = model(inputs)

            score = scoring(outputs, targets)
            loss = criterion(outputs, targets)

            val_scores.append(score)
            val_losses.append(loss / val_samples)

    return {
        "losses": losses,
        "val_losses": val_losses,
        "val_scores": val_scores
    }


def plot_metrics(dataset_name, metrics, labels, title=None, filename=None):
    plt.subplot(1, 2, 1)
    plt.plot(metrics["losses"], label="loss")
    plt.plot(metrics["val_losses"], label="validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["val_scores"], label=labels)
    plt.title("validation metrics")
    plt.legend()

    plt.gcf().set_size_inches(12, 4)
    plt.suptitle(dataset_name + "" if title is None else title)
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_dropping_metrics(metrics, labels):
    plt.plot(range(0, len(metrics)), metrics, label=labels)
    plt.title("average test metrics vs num of dropped features/inputs")
    plt.gcf().set_figwidth(10)
    plt.legend()
    plt.show()


def wrap_model(model):
    def wrapped_model(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float)
        with torch.no_grad():
            return model(inputs).detach().numpy()

    return wrapped_model


def analyze_dgsm(headers, model):
    problem = SALib.ProblemSpec({
        "num_vars": len(headers),
        "names": headers,
        "bounds": [[-1., 1.]] * len(headers),
    })

    problem.sample(SALib.sample.finite_diff.sample, N=1024)
    problem.evaluate(wrap_model(model))
    problem.analyze(SALib.analyze.dgsm.analyze)

    min_index = sum(problem.to_df())['dgsm'].argmin()

    return min_index


def analyze_fast(headers, model):
    problem = SALib.ProblemSpec({
        "num_vars": len(headers),
        "names": headers,
        "bounds": [[-1., 1.]] * len(headers),
    })

    problem.sample(SALib.sample.fast_sampler.sample, N=1024, seed=2024)
    problem.evaluate(wrap_model(model))
    problem.analyze(SALib.analyze.fast.analyze)

    # sum analysis over target classes
    min_index = sum(problem.to_df())['ST'].argmin()

    return min_index


def prune(
    datasets, headers, analyzer,
    model_factory, optimizer_factory, criterion, scoring, epochs,
    labels, dataset_name, title=None
):
    headers = headers.copy()

    dropped = []
    test_metrics = []
    models = []

    # removing features loop
    while len(headers) > 0:
        # averaging
        test_scores = []

        for s in range(5):
            model = model_factory(len(headers))
            optimizer = optimizer_factory(model)

            metrics = model_train(datasets, model, optimizer,
                                  criterion, scoring, epochs)

            with torch.no_grad():
                inputs, targets = datasets["test"][:]
                outputs = model(inputs)

                test_score = scoring(outputs, targets)
                test_loss = criterion(outputs, targets)

            test_scores.append(test_score)

        plot_metrics(dataset_name, metrics, labels)
        avg_test_score = tuple(map(lambda x: sum(x)/5, zip(*test_scores)))

        test_metrics.append(avg_test_score)
        models.append(model)

        print(f"Test: loss: {test_loss}, avg metrics: {avg_test_score}")

        # sensitive analysis
        if len(headers) <= 1:
            break

        min_index = analyzer(headers, model)

        indexes = torch.arange(len(headers)) != min_index
        datasets = {
            key: TensorDataset(
                dataset.tensors[0][:, indexes],
                dataset.tensors[1]
            )
            for key, dataset in datasets.items()
        }

        dropped.append((min_index, headers[min_index]))
        print(f"dropping feature: {dropped[-1]}")
        del headers[min_index]

        print("="*50)

    return test_metrics, dropped, models


def reduce_datasets(
    datasets, headers, dropped, num_to_drop
):
    reduced_headers = headers.copy()
    indexes = list(range(len(reduced_headers)))
    for idx, _ in dropped[:num_to_drop]:
        del indexes[idx]
        del reduced_headers[idx]

    reduced_datasets = {
        key: TensorDataset(
            dataset.tensors[0][:, indexes],
            dataset.tensors[1]
        )
        for key, dataset in datasets.items()
    }

    print(f"features left [{len(reduced_headers)}]: ", reduced_headers)

    return reduced_datasets, reduced_headers


def reduce_linear(linear: nn.Linear, dropped, num_drop_neurons):
    indexes = list(range(linear.out_features))
    for idx, _ in dropped[:num_drop_neurons]:
        del indexes[idx]

    new_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=len(indexes)
    )

    state_dict = linear.state_dict()
    state_dict["bias"] = state_dict["bias"][indexes]
    state_dict["weight"] = state_dict["weight"][indexes, :]

    new_linear.load_state_dict(state_dict)

    return new_linear


def save_models(dataset, layer, models, dropped):
    with open(f"{dataset}/{layer}.pickle", "wb") as file:
        pickle.dump({
            "models": [m.state_dict() for m in models],
            "dropped": dropped,
        }, file)
