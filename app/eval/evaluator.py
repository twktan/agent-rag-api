from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def evaluate_rag(query, answer, retrieved_chunks):

    context = '\n\n'.join([c['text'] for c in retrieved_chunks])

    judge_prompt = f'''
You are an evaluator for a RAG system.

Question: {query}

Context:
{context}

Answer:
{answer}

Rate the answer from 1 to 5 based on:
1. correctness
2. groundedness in context
3. clarity

Return ONLY a number.
'''

    try:
        response = client.chat.completions.create(model='gpt-4o-mini',
                                                  messages=[{
                                                      'role':
                                                      'user',
                                                      'content':
                                                      judge_prompt
                                                  }],
                                                  temperature=0)

        score = float(response.choices[0].message.content.strip())

        return score

    except Exception as e:
        print(f'[EVAL ERROR] {e}')
        return None
