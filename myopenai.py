from openai import OpenAI

def readtxt():

    # 假设文件名为 example.txt，位于当前目录下
    file_path = './data/brochure.txt'

    # 使用 with 语句打开文件，这样可以确保文件在使用后会被正确关闭
    with open(file_path, 'r',encoding='utf-8') as file:
        # 读取文件全部内容
        content = file.read()
        # 打印内容
    return content


def myChatGPT(content):
    ### 需要在这个位置加一个模块，初始化GPT
    init=readtxt()

    end='请你根据我所提供的关于联邦学习的内容，回答我的问题'
    # end="请你根据我关于我们web系统的使用说明，回答我的一些问题。"
    content=init+content+end

    client = OpenAI(api_key='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')

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": content},
      ]
    )
    return response.choices[0].message

if __name__ == "__main__":
    content='请问，你知道联邦学习吗'
    respone=myChatGPT(content)
    print(respone)
