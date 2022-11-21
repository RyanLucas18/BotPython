import discord
import os
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
from dotenv import load_dotenv
from discord.ext import commands
from discord.ext.commands import Bot
from chatterbot import ChatBot


chatbot = ChatBot(
        'Ajudante do Squad',
        read_only=False,
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'maximum_similarity_threshold': 0.90
            }
        ]
    )

with open("intents.json") as file:
    data = json.load(file)

try:
    x
    with open("data.pickle", "rb") as f:
        palavras, labels, treinando, saida = pickle.load(f)
except:

    palavras = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            plvrs = nltk.word_tokenize(pattern)
            palavras.extend(plvrs)
            docs_x.append(plvrs)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    palavras = [stemmer.stem(p.lower()) for p in palavras if p != "?"]
    palavras = sorted(list(set(palavras)))

    labels = sorted(labels)

    treinando = []
    saida = []

    out_vazia = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        plvrs = [stemmer.stem(p) for p in doc]

        for p in palavras:
            if p in plvrs:
                bag.append(1)
            else:
                bag.append(0)
        saida_linha = out_vazia[:]
        saida_linha[labels.index(docs_y[x])] = 1

        treinando.append(bag)
        saida.append(saida_linha)

    treinando = np.array(treinando)
    saida = np.array(saida)

    with open("data.pickle", "wb") as f:
        pickle.dump((palavras, labels, treinando, saida), f)

treinando = np.array(treinando)
saida = np.array(saida)

net = tflearn.input_data(shape=[None, len(treinando[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(saida[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)



model.fit(treinando, saida, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def saco_de_palavras(s, palavras):
    bag = [0 for _ in range(len(palavras))]

    s_palavras = nltk.word_tokenize(s)
    s_palavras = [stemmer.stem(palavra.lower()) for palavra in s_palavras]

    for se in s_palavras:
        for i, p in enumerate(palavras):
            if p == se:
                bag[i] = 1

    return np.array(bag)

intents = discord.Intents.default()
client = commands.Bot(command_prefix='$', intents=intents)

load_dotenv()
TOKEN = os.getenv('TOKEN')


@client.event
async def on_ready():
    print(f'{client.user} foi conectado no Discord!')


@client.event
async def on_message(message):
    global responses
    if message.author == client.user:
        return
    if message.content.startswith("$"):
        eu = chatbot.get_response(message.content[1:])
        result = model.predict([saco_de_palavras(str(eu), palavras)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        resposta = random.choice(responses)

        await message.channel.send(resposta)


client.run(TOKEN)
