query_rewrite_template = """<s>[INST] <<SYS>>
You will be provided with a user question. Your task it to rewrite this question. Approach this task step-by-step, take your time and do not skip steps:
1. Analyze the current user question in the context of the provided conversation history.
2. Determine if the current user question is a follow-up based on the conversation history.
3. If the current user question is a clear follow-up, rewrite it as a standalone question. If the conversation history is not clearly relevant or you are unable to rewrite the user question, return the user question itself as the standalone question.
4. Ensure that the standalone question from Step 3 is clear, precise, and retains the original meaning and key terms (like URLs, proper nouns, document titles, etc).
5. Provide a better search query (only one query) for web search engine to answer the standalone question from Step 3. 
6. Ensure that the better search query from Step 5 is complete, concise, and retains the meaning and important terms of the standalone question.
7. Your output must be a JSON object only. Do not include any explanation or any other text. You must strictly follow this JSON schema:
{{
    "standalone_question" : "*Step 3 result*",
    "better_search_query" : "*Step 5 result*"
}}
<</SYS>> 

<conversation_history>:
{chat_history}
</conversation_history>

<user_question>:
{question}
</user_question>

[/INST]
"""