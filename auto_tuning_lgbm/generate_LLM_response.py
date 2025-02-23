import os
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルを読み込む（同じディレクトリにある場合）
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MAX_NUM_TOKENS = 4096

# 現状、OpenAIのみ対応
client = OpenAI()
model = "gpt-4o-mini"    # モデルを選択
default_system_message = "You are a helpful assistant"


# 会話履歴とメッセージを入力し、LLMから返答と全会話履歴を取得する関数
def get_response_from_llm(
        msg,
        client=client,
        model=model,
        system_message=default_system_message,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user","content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system","content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=1,
        stop=None,
        seed=0,
        response_format={"type":"json_object"}, # json形式で出力
    )

    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant","content": content}]

    return content, new_msg_history


# 動作確認用
msg="あなたの自己紹介をJSON形式でしてください"
print(get_response_from_llm(msg))


