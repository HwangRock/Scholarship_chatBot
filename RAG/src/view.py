import gradio as gr

class ScholarshipView:
    def __init__(self, chatbot_callback):
        self.chatbot_callback = chatbot_callback

    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## 장학금 추천 및 상담 챗봇")
            gr.Markdown("학생 정보를 순차적으로 입력하고, 추천받은 장학금에 대한 추가 정보를 질문할 수 있습니다.")
            
            chatbot = gr.Chatbot(label="장학금 추천 및 상담", show_label=False)
            state = gr.State([])
            sequence_state = gr.State(1)
            
            input_box = gr.Textbox(label="입력", placeholder="여기에 입력하세요.")
            
            input_box.submit(
                self.chatbot_callback,
                inputs=[state, input_box, sequence_state],
                outputs=[chatbot, state, sequence_state]
            )
            input_box.submit(lambda: "", inputs=None, outputs=input_box)
        
        return demo
