import gradio as gr
import vector_store
from llm import Llm


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
        top_p=top_p)

    history.append((input, resp))
    return '', history, history


def clear_session():
    return '', None


def llm_status():
    try:
        llm = Llm()
        llm._call('Human: hello')
        return """The initial model has loaded successfully and you can start a conversation\n\nExample question: Can you list me some inactive placement? \n\nCan you list all info about 'Programmatic Guaranteed Campaign for Push'?"""
    except Exception as e:
        print(e)
        return """The model did not load successfully"""


llm = Llm()
vector_s = vector_store.get_vector_store(
    "./data/advertising.json", "advertising")
system_prompt = "System: You are a helpful, respectful and honest assistant. When you answer user's question, please list the information by format. Please provide as much information as possible based on the chat history, rather than the context. If you can't get an answer from it, return user 'Not enough relevant information is provided'."


def load_demo_data(demo, history):
    llm.history = []
    global vector_s
    global system_prompt
    if demo == "APISearch":
        vector_s = vector_store.get_vector_store(
            "./data/api.json", "api")
        message = "APISearch Demo loaded!\n\nExample question: I want to update video group relations. Which API should I use?"
        system_prompt = "System: You are a helpful, respectful and honest assistant. When you answer user's question, please list the information by format. Please provide as much information as possible based on the chat history, rather than the context. If you can't get an answer from it, return user 'Not enough relevant information is provided'."
    else:
        vector_s = vector_store.get_vector_store(
            "./data/advertising.json", "advertising")
        message = "GenericSearch Demo loaded!\n\nExample question: Can you list me some inactive placement? \n\nCan you list all info about 'Programmatic Guaranteed Campaign for Push'"
        system_prompt = "System: You are a helpful, respectful and honest assistant. When you answer user's question, please list the information by format. Please provide as much information as possible based on the chat history, rather than the context. If you can't get an answer from it, return user 'Not enough relevant information is provided'."
    history = []
    return [[None, message]], history


# def build_llama2_prompt(system, user, history, history_len):
#     messages = [system]
#     length = len(history)
#     if length > history_len:
#         length = history_len
#     for combined_message in history[-length:]:
#         for message in combined_message:
#             messages.append(message)
#     messages.append(user)
#     return [messages]


def build_claude2_prompt(system, user, history, history_len):
    messages = system
    length = len(history)
    if length > history_len:
        length = history_len
    for combined_message in history[-length:]:
        for message in combined_message:
            messages = messages + "\n" + message
    messages = messages + "\n" + user
    return messages


def get_knowledge_based_answer(query, top_k: int = 5, history_len: int = 3, temperature: float = 0.01, top_p: float = 0.1):

    llm.temperature = temperature
    llm.top_p = top_p

    retriever = vector_s.as_retriever(search_kwargs={"k": top_k})
    document_list = retriever.get_relevant_documents(query)
    context = ""
    i = 1
    for doc in document_list:
        context += doc.page_content
        if i != len(document_list):
            context += ","
        i = i + 1
    user_prompt = f"\n\nHuman: Known content: {context}.\n\nQuestion:{query}\n"
    prompt = build_claude2_prompt(
        system_prompt, user_prompt, llm.history, history_len)
    response = llm._call(prompt)
    print(response)
    ai_prompt = f"Assistant: {response}"
    llm.history.append([user_prompt, ai_prompt])
    return response


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("<h1><center>LangChain-ChatLLM</center></h1>")
        model_status = gr.State(llm_status())
        with gr.Row():
            with gr.Column(scale=1):
                demo_choose = gr.Accordion("demo choose")
                with demo_choose:
                    demo_name = gr.Dropdown(
                        ["GenericSearch", "APISearch"], label="Demo choose", value="GenericSearch")
                    load_demo_button = gr.Button("Load Demo Data")
                model_argument = gr.Accordion("Model Config")
                with model_argument:

                    top_k = gr.Slider(1,
                                      10,
                                      value=5,
                                      step=1,
                                      label="vector search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=2,
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

            load_demo_button.click(
                load_demo_data,
                show_progress=True,
                inputs=[demo_name, state],
                outputs=[chatbot, state]
            )

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
