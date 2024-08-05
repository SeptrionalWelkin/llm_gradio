import gradio as gr

from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_KEY)


# Define function to interact with OpenAI API
def openai_interact(prompt, mode, temperature, top_p, chat_history):
    if mode == "Instruct":
        model = "gpt-4o-mini"
    else:  # Chat mode
        model = "gpt-4o-mini"

    if mode == "Chat" and chat_history:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += chat_history + [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]


    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p
        # stream=True
    )

    answer = completion.choices[0].message.content
    yield answer

    # answer = ""
    # for chunk in completion:
    #     answer += chunk.choices[0].delta.get("content", "")
    #     yield answer


# Define Gradio app
def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# OpenAI GPT-4 Chat App")

        with gr.Row():
            mode = gr.Dropdown(choices=["Instruct", "Chat"], value="Instruct", label="Mode")
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="Temperature")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, label="Top P")

        prompt = gr.Textbox(lines=5, placeholder="Enter your prompt here...", label="Prompt")
        chat_history = gr.State([])

        output = gr.Markdown()

        def add_chat_history(history, mode, user_input, assistant_response):
            if mode == "Chat":
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": assistant_response})
            return history

        def reset_history():
            return []

        with gr.Row():
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear Chat History")

        submit_btn.click(
            openai_interact,
            inputs=[prompt, mode, temperature, top_p, chat_history],
            outputs=[output]
            #js='() => { document.querySelector(".gr-loading").classList.remove("gr-loading"); }'
        )
        submit_btn.click(
            add_chat_history,
            inputs=[chat_history, mode, prompt, output],
            outputs=[chat_history]
        )
        clear_btn.click(
            reset_history,
            outputs=[chat_history]
        )

    return demo


app = build_app()

if __name__ == "__main__":
    app.launch()
