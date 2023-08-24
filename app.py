import gradio as gr
from llm import llm_status, get_knowledge_based_answer


def predict(input,
            top_k,
            history_len,
            temperature,
            top_p,
            history=None):
    if history == None:
        history = []

    resp = get_knowledge_based_answer(
        query=input,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history)

    history.append((input, resp['result']))
    return '', history, history


def clear_session():
    return '', None


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("<h1><center>LangChain-ChatLLM</center></h1>")
        model_status = gr.State(llm_status())
        with gr.Row():
            with gr.Column(scale=1):
                model_argument = gr.Accordion("Model Config")
                with model_argument:

                    top_k = gr.Slider(1,
                                      10,
                                      value=10,
                                      step=1,
                                      label="vector search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=3,
                                            step=1,
                                            label="history len",
                                            interactive=True)

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

            with gr.Column(scale=4):
                chatbot = gr.Chatbot([[None, model_status.value]],
                                     label='ChatLLM').style(height=750)
                message = gr.Textbox(label='Please type in question:')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("clean up history")
                    send = gr.Button("send")

            send.click(predict,
                       inputs=[
                           message, top_k, history_len, temperature,
                           top_p, state
                       ],
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

            message.submit(predict,
                           inputs=[
                               message, top_k, history_len,
                               temperature, top_p, state
                           ],
                           outputs=[message, chatbot, state])
    # threads to consume the request
    demo.queue(concurrency_count=3) \
        .launch(server_name='0.0.0.0',
                server_port=7860,
                show_api=True,
                share=True,
                inbrowser=False)
