from argparse import ArgumentParser
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import configs
from models.Devign2 import Devign2
from models.LMGNN import BertGGCN
from test import test
from utils.data.helper import check_split_exists, loads
from utils.data.input import train_val_test_split, load_split_datasets
from utils.figure.plot import plot_validation_loss

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model using the provided data.

    :param model: The model to be trained.
    :param device: The device to perform training on (e.g., 'cpu' or 'cuda').
    :param train_loader: The data loader containing the training data.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch number.
    :return: None
    """
    
    train_loss = 0.0

    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch}")
    for batch_idx, batch in progress_bar:
        batch.to(device)

        y_pred = model(batch)
        model.zero_grad()
        batch.y = batch.y.squeeze().float()
        loss = F.binary_cross_entropy(y_pred.squeeze(-1), batch.y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{train_loss:.6f}"})

    train_loss /= len(train_loader)
    print(f"Training Epoch {epoch} finished. Loss: {train_loss:.6f}")

def validate(model, device, test_loader, epoch):
    """
    Validates the model using the provided test data.

    :param model: The model to be validated.
    :param device: The device to perform validation on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Validating Epoch {epoch}")
    for batch_idx, batch in progress_bar:
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch)

        batch.y = batch.y.squeeze().float()
        # BCE on probabilities
        test_loss += F.binary_cross_entropy(y_.squeeze(-1), batch.y).item()
        probs = y_.squeeze(-1)
        pred = (probs > 0.5).long()

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        progress_bar.set_postfix({"loss": f"{test_loss:.4f}"})
    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print('Valid set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))
    print()

    return test_loss, accuracy, precision, recall, f1

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-m', '--modeLM', action='store_true', help='Specify the mode for LMTrain')
    parser.add_argument('-p', '--path', default="data/model/model.pth", help='Specify the path for the model')

    args = parser.parse_args()

    context = configs.Process()
    split_dir = PATHS.split
    input_dataset = loads(PATHS.input)
    
    if check_split_exists(split_dir):
        train_dataset, val_dataset, test_dataset, test_short_dataset, test_long_dataset = load_split_datasets(split_dir, input_dataset)
    else:
        train_dataset, val_dataset, test_dataset, test_short_dataset, test_long_dataset = train_val_test_split(input_dataset, shuffle=context.shuffle, save_path=split_dir)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader, test_short_loader, test_long_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            [train_dataset, val_dataset, test_dataset, test_short_dataset, test_long_dataset]))

    Bertggnn = configs.BertGGNN()
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select model based on the modeLM argument
    if args.modeLM:
        finetune_file = "data/model/graphcodebert_finetune.pt"
        hugging_path = "data/model/graphcodebert_finetune_hf"
        model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, hugging_path=hugging_path, finetune_file=finetune_file).to(device)
        model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, hugging_path=hugging_path, finetune_file=finetune_file).to(device)
    else:
        autoencoder_path = os.path.join(PATHS.model, 'autoencoder.pt')
        model = Devign2(gated_graph_conv_args, conv_args, emb_size, device, autoencoder_path).to(device)
        model_test = Devign2(gated_graph_conv_args, conv_args, emb_size, device, autoencoder_path).to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), 
        lr=Bertggnn.learning_rate, 
        weight_decay=Bertggnn.weight_decay
    )

    best_f1 = 0.0
    NUM_EPOCHS = context.epochs
    PATH = args.path
    losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        loss, acc, precision, recall, f1 = validate(model, device, val_loader, epoch)
        losses.append(loss)
        if f1 > best_f1:
            best_f1 = f1
            if PATH:
                torch.save(model.state_dict(), PATH)
    print(f"Training finished. Best F1: {best_f1:.4f}")
    plot_validation_loss(losses, 'workspace/validation_loss.png')

    # Load best model and test
    if PATH:
        model_test.load_state_dict(torch.load(PATH))
        print("Testing full")
        test(model_test, device, test_loader, 'workspace/')
        print("Testing short")
        test(model_test, device, test_short_loader, 'workspace/')
        print("Testing long")
        test(model_test, device, test_long_loader, 'workspace/')




