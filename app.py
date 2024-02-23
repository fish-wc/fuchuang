from flask import Flask, request, jsonify, render_template, Response
from flask_socketio import SocketIO, emit
from flask import Flask, jsonify, stream_with_context
import subprocess
import threading
import queue

app = Flask(__name__)
# socketio = SocketIO(app)
#
# # 创建一个队列来存储子进程的输出
# output_queue = queue.Queue()
#
#
# def capture_output(process):
#     """捕获子进程的输出并将其放入队列中"""
#     for line in iter(process.stdout.readline, b''):
#         output_queue.put(line.decode('utf-8').rstrip())


@app.route('/attack', methods=['POST'])
def attack():
    data = request.form
    attack_type = data['attack_type']
    choice = data['choice']

    # 构建攻击命令
    command = f"python main_attack.py --attack_type {attack_type} --choice {choice}"

    # 执行攻击命令，并捕获输出
    completed_process = subprocess.run(command.split(), capture_output=True, text=True, check=True)

    # 输出子进程的标准输出和标准错误
    app.logger.info("Attack Command Output:")
    app.logger.info(completed_process.stdout)
    app.logger.error("Attack Command Error Output:")
    app.logger.error(completed_process.stderr)

    return jsonify({'message': 'Attack command executed successfully'}), 200


@app.route('/train', methods=['POST'])
def train():
    data = request.form
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

    # command += f" > output.log 2>&1 &"

    # 清空output.log文件
    # with open('output.log', 'w') as f:
    #     f.truncate(0)  # 清空文件内容

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

    #
    # # 使用线程来异步捕获输出，这样不会阻塞Flask的主线程
    # threading.Thread(target=capture_output, args=(process,)).start()
    #
    # # 使用生成器函数来流式传输输出到前端
    # def generate():
    #     while not output_queue.empty():
    #         yield output_queue.get() + '\n'
    #
    # # 实时打印 main.py 的输出到 Flask 应用的控制台
    # for line in iter(process.stdout.readline, b''):
    #     print(line.strip())  # 输出每一行，可根据需要进行格式化
    #
    # process.wait()
    #
    # return stream_with_context(generate())

    def generate_output():
        for line in process.stdout:
            yield line.strip() + '\n'

    return Response(generate_output(), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
