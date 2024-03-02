import json
import os
import time

from flask import Flask, request, Response, render_template
from flask_socketio import SocketIO, emit
import subprocess

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return render_template('index.html')


import subprocess
import os
import json
from flask_socketio import emit


@socketio.on('attack')
def attack(data_1):
    data = data_1['params']
    attack_type = data['attack_type']
    choice = data['choice']
    print(data)
    # 构建攻击命令
    command = f"python main_attack.py --attack_type {attack_type} --choice {choice}"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

    for line in process.stdout:
        emit('attack_output', {'output': line.strip()})

    # 等待1分钟
    time.sleep(60)

    # 终止子进程
    process.terminate()

    result = {}

    if attack_type == 0:
        # 读取指定文件夹下的所有 PNG 文件并添加到 pic 列表中
        pic_folder = r'C:\Users\Zherui\Desktop\fuchuang\dlg_attack\no_noisy_dlg_attack'  # 指定文件夹路径
        pic_files = [os.path.join(pic_folder, f) for f in os.listdir(pic_folder) if f.endswith('.png')][:10]
        result['pic'] = pic_files
    elif attack_type == 1:
        # 读取指定地址下的 TXT 文件并解析分类报告信息
        txt_file = r'C:\Users\Zherui\Desktop\fuchuang\ml_attack\mla_result\classification_report1.txt'  # 指定 txt 文件路径
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        metrics = {}
        labels = ['precision', 'recall', 'f1-score', 'support']
        for line in lines[2:4]:  # 跳过前两行
            values = line.split()
            label = values[0]
            metrics[label] = {k: float(v) for k, v in zip(labels, values[1:])}

        result['metrics'] = metrics
    elif attack_type == 2:
        # 将指定图片添加到 pic 列表中
        pic_file = r'C:\Users\Zherui\Desktop\fuchuang\inversion_attack\norm_attack_out\best_test.png'
        result['pic'] = [pic_file]

    # 将数据以 JSON 格式通过 emit 发送给前端
    emit('attack_result', result)


@socketio.on('train')
def train(data_1):
    print(data_1)
    data = data_1['params']
    choice = data['choice']

    command = f"python main.py --choice {choice}"

    # 根据前端传递的参数来决定是否将其添加到命令中
    if 'no_models' in data:
        no_models_value = data['no_models']
        command += f" --no_models {no_models_value}"

    if 'model_name' in data:
        model_name_value = data['model_name']
        command += f" --model_name {model_name_value}"

    if 'type' in data:
        type_value = data['type']
        command += f" --type {type_value}"

    if 'global_epochs' in data:
        global_epochs_value = data['global_epochs']
        command += f" --global_epochs {global_epochs_value}"

    if 'local_epochs' in data:
        local_epochs_value = data['local_epochs']
        command += f" --local_epochs {local_epochs_value}"

    if 'k' in data:
        k_value = data['k']
        command += f" --k {k_value}"

    if 'batch_size' in data:
        batch_size_value = data['batch_size']
        command += f" --batch_size {batch_size_value}"

    if 'lr' in data:
        lr_value = data['lr']
        command += f" --lr {lr_value}"

    if 'momentum' in data:
        momentum_value = data['momentum']
        command += f" --momentum {momentum_value}"

    if 'lambda_' in data:
        lambda_value = data['lambda_']
        command += f" --lambda_ {lambda_value}"

    if 'dp' in data:
        dp_value = data['dp']
        command += f" --dp {dp_value}"

    if 'C' in data:
        C_value = data['C']
        command += f" --C {C_value}"

    if 'sigma' in data:
        sigma_value = data['sigma']
        command += f" --sigma {sigma_value}"

    if 'q' in data:
        q_value = data['q']
        command += f" --q {q_value}"

    if 'w' in data:
        w_value = data['w']
        command += f" --w {w_value}"

    if 'feature_num' in data:
        feature_num_value = data['feature_num']
        command += f" --feature_num {feature_num_value}"

    if 'eta' in data:
        eta_value = data['eta']
        command += f" --eta {eta_value}"

    if 'alpha' in data:
        alpha_value = data['alpha']
        command += f" --alpha {alpha_value}"

    if 'poison_label' in data:
        poison_label_value = data['poison_label']
        command += f" --poison_label {poison_label_value}"

    if 'poisoning_per_batch' in data:
        poisoning_per_batch_value = data['poisoning_per_batch']
        command += f" --poisoning_per_batch {poisoning_per_batch_value}"

    if 'prop' in data:
        prop_value = data['prop']
        command += f" --prop {prop_value}"

    if 'root' in data:
        root_value = data['root']
        command += f" --root {root_value}"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

    print(data)
    for line in process.stdout:
        emit('train_output', {'output': line.strip()})

    # 从文件中读取acc和loss数据
    with open('acc.txt', 'r') as f:
        acc_data = json.load(f)

    with open('loss.txt', 'r') as f:
        loss_data = json.load(f)

    # 准备要传递的数据字典
    data = {'acc': acc_data, 'loss': loss_data}
    emit('train_result', data)


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5000)
