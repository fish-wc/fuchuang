# 本文件主要是用于模型管理的

import os
import torch


def save_model_with_unique_name(model, directory, base_filename, extension=".pth"):
    """
    Save a PyTorch model with a unique filename in the specified directory.

    Parameters:
    - model: The PyTorch model to save.
    - directory: The directory where the model should be saved.
    - base_filename: The base name for the file, without the extension.
    - extension: The file extension. Default is ".pt" for PyTorch models.

    This function checks if a file with the base_filename exists in the directory.
    If it does, it appends a number to the base_filename to make it unique.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = base_filename + extension
    filepath = os.path.join(directory, filename)

    # Check if the file exists and create a unique filename if necessary
    counter = 1
    while os.path.isfile(filepath):
        filename = f"{base_filename}_{counter}{extension}"
        filepath = os.path.join(directory, filename)
        counter += 1

    # Save the model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved as {filename}")

# 函数目的，用来保存模型，如果名字一样，那么就覆盖处理
def save_model_with_same_name(model, directory, base_filename, extension=".pth"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = base_filename + extension
    filepath = os.path.join(directory, filename)

    # Save the model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved as {filename}")

def load_model(model_path):
    """
    Load a PyTorch model from a file.

    Parameters:
    - model: An instance of the PyTorch model class you want to load.
             Make sure it has the same architecture as the saved model.
    - model_path: The path to the model file.

    Returns:
    - The model with loaded parameters.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:",device)
    model= torch.load(model_path,map_location=device)
    return model

if __name__=="__main__":
    model_path='../mymodel/resnet50.pth'
    model=load_model(model_path)
    print('顺利导出模型')
    name='resnet50'
    save_model_with_unique_name(model,'../mymodel',name)
    print("顺利保存")
    model2=load_model("../mymodel/resnet18.pth")
    print("模型又顺利导出了")

