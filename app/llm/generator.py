import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def call_llm(prompt: str, rag: bool) -> str:
    try:
        if rag:
            content = "You are a helpful assistant for a RAG system for employees to ask questions about their company, Trevor Tan Incorporated. Use only the provided context."
        else:
            content = "You are a helpful assistant."

        response = client.chat.completions.create(model='gpt-4o-mini',
                                                  messages=[{
                                                      'role': 'system',
                                                      'content': content
                                                  }, {
                                                      'role': 'user',
                                                      'content': prompt
                                                  }],
                                                  temperature=0.2)

        return response.choices[0].message.content

    except Exception as e:
        return f'LLM_ERROR: {str(e)}'
