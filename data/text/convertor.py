import json
import os

import nltk

cmu = nltk.corpus.cmudict.dict()
punctuation = """;:,.!?¡¿—…"«»”"""
vowel_length = "ː"
primary_stress, secondary_stress = "ˈ", "ˌ"
stresses = {primary_stress: "1", secondary_stress: "2"}
with open(os.path.join(os.path.dirname(__file__), "phonemes_mapping.json"), 'r') as f:
    phonemes_mapping = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "diphthongs_mapping.json"), 'r') as f:
    diphthongs_mapping = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "combined_words_of_espeak.json"), 'r') as f:
    combined_transcriptions = json.load(f)


def ipa2arpabet(ipa_transcription):
    transcription = ipa_transcription.replace("m̩", "əm")
    transcription = transcription.replace("n̩", "ən")
    transcription = transcription.replace("l̩", "əl")
    transcription = transcription.replace("ɜː", "#ER0#")
    transcription = transcription.replace(vowel_length, "")
    transcription = transcription.replace(" ", "*")
    transcription = transcription.replace(primary_stress, "#"+primary_stress+"#")
    transcription = transcription.replace(secondary_stress, "#"+secondary_stress+"#")
    for phoneme in diphthongs_mapping.keys():
        transcription = transcription.replace(phoneme, "#"+diphthongs_mapping[phoneme]+"#")
    for phoneme in phonemes_mapping.keys():
        transcription = transcription.replace(phoneme, "#"+phonemes_mapping[phoneme]+"#")
    transcription = transcription.replace("#", " ")
    transcription_splitted = transcription.split()
    for i in range(len(transcription_splitted)):
        if transcription_splitted[i] == "*"*len(transcription_splitted[i]):
            transcription_splitted[i] = " "
    return transcription_splitted


def handle_stresses(transcription_splitted, verbose=False):
    if verbose:
        print("Before stress handling:", transcription_splitted)
    idxs_to_del = []
    if transcription_splitted[-1] in stresses.keys():
        del transcription_splitted[-1]
    for i in range(len(transcription_splitted) - 1):
        if transcription_splitted[i] in stresses.keys():
            stress = transcription_splitted[i]
            j = i + 1
            while not transcription_splitted[j][-1].isnumeric():
                j += 1
            if j < len(transcription_splitted):
                transcription_splitted[j] = transcription_splitted[j][:-1] + stresses[stress]
            idxs_to_del.append(i)
    if idxs_to_del:
        step = 0
        for idx in idxs_to_del:
            del transcription_splitted[idx - step]
            step += 1
    if verbose:
        print("After stress handling:", transcription_splitted)
    return transcription_splitted


def isolate_punctuation(text: str, transcription: str):
    for punct in punctuation:
        text = text.replace(punct, " " + punct + " ")
        transcription = transcription.replace(punct, " " + punct + " ")
    return text, transcription


def handle_combined_transcriptions(text: str, transcription_splitted: list):
    for word_pron in transcription_splitted:
        if word_pron in combined_transcriptions.keys():
            for pair in combined_transcriptions[word_pron]:
                text = text.replace(pair[0], "#" + pair[1] + "#")
    return text


def convert(espeak_prepended_result, verbose=False):
    result = []
    text, transcription = espeak_prepended_result[0]
    text, transcription = isolate_punctuation(text, transcription)
    transcription_splitted = transcription.split()
    text = handle_combined_transcriptions(text, transcription_splitted)
    text_splitted = text.split()
    for i in range(len(text_splitted)):
        word = text_splitted[i]
        if word in punctuation:
            result.append(word)
        elif word in cmu.keys() and len(cmu[word]) == 1:
            result.extend(cmu[word][0])
        else:
            arp = ipa2arpabet(transcription_splitted[i])
            result.extend(handle_stresses(arp))
        result.append(" ")
    if result[-1] == " ":
        del result[-1] 
    to_del_idxs = []
    for i in range(len(result) - 1):
        if result[i] == " " and result[i + 1] in punctuation:
            to_del_idxs.append(i)
    step = 0
    for i in to_del_idxs:
        del result[i - step]
        step += 1
    return result
