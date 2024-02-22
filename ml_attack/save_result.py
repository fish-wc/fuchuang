import os
def savere(result_path,report):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    filename = 'classification_report.txt'
    with open(result_path+filename, 'w') as f:
        f.write(report)