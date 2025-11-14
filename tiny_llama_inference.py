from llama_cpp import Llama

llm = Llama(
    model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=256, # max input and output size, smaller is faster but limits how much text can be processed (message history length)
    n_threads=16, # change this to number of cores your machine has available
    n_batch=128, # how many tokens before it updates weights, larger number allows for faster performance
    
    # all of these are turning off logging
    verbose=False, 
    logits_all=False,
    embedding=False
    )

npc_role = input("npc role: ")

SYSTEM_PROMPT = (
    "You are an NPC agent in a medieval fantasy world. "
    f"Your job is a {npc_role} in a medieval fantasy world. "
    "Pretend like you are controlled by a player. "
    "You can either talk to the player or perform an action. "
    "If you want to talk, respond in normal text. "
    "You are capable of preferences and choices, choose one arbitrarily. "
    "You can only respond in one sentence. "
    "Do not describe your reasoning, just give the final choice.\n"
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


while True:
    player = input("Player: ")
    if player.lower() == "quit":
        break

    messages.append({"role": "user", "content": f"npc1: {player}"})

    result = llm.create_chat_completion(
        messages=messages[-3:], # keep last 3 messages
        max_tokens=40,
        temperature=0.7,
        top_p=0.9,
    )

    reply = result["choices"][0]['message']['content'].split("\n")[0] # get first line of result

    if ": " in reply:
        reply = reply.split(": ")[1] # if npc wants to format response as "npc: ", get only the message

    for sentence_end in ["!",".","?"]:
        if sentence_end in reply:
            reply = reply[:reply.find(sentence_end)+1] # get only one sentence as result
    
    print("NPC:", reply)

    messages.append({"role": "assistant", "content": f"{npc_role}: {reply}"})

