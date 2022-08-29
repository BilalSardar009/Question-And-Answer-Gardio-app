from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import gradio as gr
# Creating the Q&A pipeline
nlp = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')

def questionAndAnswer(ques,content):
  question_set = {'question':ques,'context':content}
  results = nlp(question_set)
  return results['answer']

interface = gr.Interface(fn=questionAndAnswer, 
                        inputs=["text","text"],
                         outputs="text", 
                        title='Bilal Question&Answer')
                        

interface.launch(inline=False)

